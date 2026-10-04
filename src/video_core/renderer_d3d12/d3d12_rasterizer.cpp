// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include <type_traits>

#include "common/alignment.h"
#include "video_core/control/channel_state.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_rasterizer.h"

namespace D3D12 {

namespace {
// Draws re-enabled, guards active (no-vertex skip, depth-RTV skip, all-slots-bound guard, RT-format and index/vertex range validation).
constexpr bool kSkipDraws = false;
} // namespace

using Tegra::Texture::TexturePair;

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_) : buffer_cache{buffer_cache_} {}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

bool AccelerateDMA::BufferClear(GPUVAddr dst_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(dst_address, amount, value);
}

RasterizerD3D12::RasterizerD3D12(Tegra::GPU& gpu,
                                 Tegra::MaxwellDeviceMemoryManager& device_memory, Device& device,
                                 ShaderCompiler& shader_compiler)
    : m_gpu{gpu}, m_device{device}, m_staging_pool{device}, m_command_list{device.GetDevice()},
      m_runtime{device, m_command_list, m_staging_pool}, m_buffer_cache{device_memory, m_runtime},
      m_texture_runtime{device, m_command_list, m_staging_pool},
      m_texture_cache{m_texture_runtime, device_memory},
      m_query_runtime{device, m_command_list, device_memory},
      m_query_cache{gpu, *this, device_memory, m_query_runtime},
      m_pipeline_cache{device_memory, device, shader_compiler},
      m_res_heap{device.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096, true},
      m_sampler_heap{device.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1024, true},
      m_accelerate_dma{m_buffer_cache} {
    // SlotVector moves elements on growth; these pointers are cached long-term in
    // Framebuffers, so capacity must cover boot-time peak to keep them stable.
    m_texture_cache.slot_image_views.Reserve(8192);
    m_texture_cache.slot_images.Reserve(4096);
    m_buffer_cache.slot_buffers.Reserve(4096);
    // CommandList is created closed; open it for recording. FlushCommands re-opens it
    // after each execute, so every record goes to a live list.
    m_command_list.Reset();
    // DEFAULT-heap scratch for unaligned uniform realigns (see CBV_SCRATCH_SIZE).
    D3D12_HEAP_PROPERTIES scratch_heap{};
    scratch_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    scratch_heap.CreationNodeMask = 1;
    scratch_heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC scratch_desc{};
    scratch_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    scratch_desc.Width = CBV_SCRATCH_SIZE;
    scratch_desc.Height = 1;
    scratch_desc.DepthOrArraySize = 1;
    scratch_desc.MipLevels = 1;
    scratch_desc.SampleDesc.Count = 1;
    scratch_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device.GetDevice()->CreateCommittedResource(
            &scratch_heap, D3D12_HEAP_FLAG_NONE, &scratch_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&m_cbv_scratch)))) {
        LOG_ERROR(Render_D3D12, "CBV scratch creation failed");
    }
    // Zero page for unbound push-constant / runtime-data root CBVs.
    D3D12_HEAP_PROPERTIES zero_heap{};
    zero_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    zero_heap.CreationNodeMask = 1;
    zero_heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC zero_desc{};
    zero_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    zero_desc.Width = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    zero_desc.Height = 1;
    zero_desc.DepthOrArraySize = 1;
    zero_desc.MipLevels = 1;
    zero_desc.SampleDesc.Count = 1;
    zero_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (SUCCEEDED(device.GetDevice()->CreateCommittedResource(
            &zero_heap, D3D12_HEAP_FLAG_NONE, &zero_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&m_zero_buffer)))) {
        m_zero_address = m_zero_buffer->GetGPUVirtualAddress();
        void* mapped = nullptr;
        if (SUCCEEDED(m_zero_buffer->Map(0, nullptr, &mapped)) && mapped) {
            std::memset(mapped, 0, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            m_zero_buffer->Unmap(0, nullptr);
        }
    }
}
RasterizerD3D12::~RasterizerD3D12() = default;

void RasterizerD3D12::Draw(bool is_indexed, u32 instance_count) {
    if (kSkipDraws) {
        return;
    }
    m_query_cache.NotifySegment(true);
    PipelineCache::StoredPipeline* stored = m_pipeline_cache.CurrentStoredPipeline();
    if (!stored || !stored->pipeline || !stored->pipeline->IsValid()) {
        return;
    }
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    ConfigureDraw(is_indexed, *stored, instance_count);
    // NOTE: no m_gpu.TickWork() here. The GPU worker ticks independently, and TickWork
    // from inside Draw hung the boot on console right after the first draw (TBD why).
}
namespace {

struct DrawParams {
    u32 base_instance{};
    u32 num_instances{};
    u32 base_vertex{};
    u32 num_vertices{};
    u32 first_index{};
    bool is_indexed{};
};

DrawParams MakeDrawParams(const Tegra::Engines::DrawManager::State& draw_state, u32 num_instances,
                          bool is_indexed) {
    using Topology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    if (draw_state.topology == Topology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Topology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

D3D_PRIMITIVE_TOPOLOGY ToD3DTopology(Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology,
                                    bool& supported) {
    using Topology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    supported = true;
    switch (topology) {
    case Topology::Points:
        return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    case Topology::Lines:
    case Topology::LineLoop:
    case Topology::LineStrip:
        return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case Topology::LinesAdjacency:
    case Topology::LineStripAdjacency:
        return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ;
    case Topology::Triangles:
    case Topology::TriangleStrip:
        return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    case Topology::TrianglesAdjacency:
    case Topology::TriangleStripAdjacency:
        return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ;
    case Topology::TriangleFan:
    case Topology::Quads:
    case Topology::QuadStrip:
    case Topology::Polygon:
    default:
        supported = false;
        return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    }
}

D3D12_VIEWPORT MakeViewport(const Tegra::Engines::Maxwell3D::Regs& regs) {
    const auto& src = regs.viewport_transform[0];
    float x = src.translate_x - src.scale_x;
    float width = src.scale_x * 2.0f;
    float y = src.translate_y - src.scale_y;
    float height = src.scale_y * 2.0f;
    using WindowOrigin = Tegra::Engines::Maxwell3D::Regs::WindowOrigin::Mode;
    using Swizzle = Tegra::Engines::Maxwell3D::Regs::ViewportSwizzle;
    const bool lower_left = regs.window_origin.mode != WindowOrigin::UpperLeft;
    if (lower_left) {
        y += static_cast<f32>(regs.surface_clip.height);
        height = -height;
    }
    if (src.swizzle.y == Swizzle::NegativeY) {
        y += height;
        height = -height;
    }
    const bool minus_one_to_one =
        regs.depth_mode == Tegra::Engines::Maxwell3D::Regs::DepthMode::MinusOneToOne;
    const float reduce_z = minus_one_to_one ? 1.0f : 0.0f;
    D3D12_VIEWPORT viewport{};
    viewport.TopLeftX = x;
    viewport.TopLeftY = y;
    viewport.Width = width != 0.0f ? width : 1.0f;
    viewport.Height = height != 0.0f ? height : 1.0f;
    viewport.MinDepth = std::clamp(src.translate_z - src.scale_z * reduce_z, 0.0f, 1.0f);
    viewport.MaxDepth = std::clamp(src.translate_z + src.scale_z, 0.0f, 1.0f);
    return viewport;
}

D3D12_RECT MakeScissor(const Tegra::Engines::Maxwell3D::Regs& regs) {
    const auto& src = regs.scissor_test[0];
    D3D12_RECT rect{};
    if (!src.enable) {
        rect.right = std::numeric_limits<LONG>::max();
        rect.bottom = std::numeric_limits<LONG>::max();
        return rect;
    }
    using WindowOrigin = Tegra::Engines::Maxwell3D::Regs::WindowOrigin::Mode;
    const bool lower_left = regs.window_origin.mode != WindowOrigin::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();
    min_y = std::max(min_y, 0);
    max_y = std::max(max_y, 0);
    rect.left = src.min_x;
    rect.top = min_y;
    rect.right = src.max_x;
    rect.bottom = max_y;
    return rect;
}

} // Anonymous namespace

void RasterizerD3D12::ConfigureDraw(bool is_indexed, PipelineCache::StoredPipeline& stored,
                                    u32 instance_count) {
    GraphicsPipeline* pipeline = stored.pipeline.get();
    const auto& regs = maxwell3d->regs;

    m_runtime.ClearDrawBindings();

    // Validate BEFORE recording any uploads: draws the backend cannot represent must
    // return here, before the buffer-binding section below records staging COPY commands
    // for garbage draw state (those copies execute at flush and fault the GPU on console,
    // removing the device even for draws every later guard would skip).
    m_texture_cache.UpdateRenderTargets(false);
    Framebuffer* framebuffer = m_texture_cache.GetFramebuffer();

    const auto& draw_state = maxwell3d->draw_manager->GetDrawState();
    const DrawParams params = MakeDrawParams(draw_state, instance_count, is_indexed);
    // Guard rails: never submit draws the backend cannot represent yet. An unbound
    // render target, an unexpanded index format, or an empty draw would otherwise go
    // to the GPU as undefined commands.
    ImageView* depth_view = framebuffer ? framebuffer->DepthBuffer() : nullptr;
    u32 num_rtvs = 0;
    for (size_t i = 0; i < VideoCommon::NUM_RT; ++i) {
        const auto format =
            static_cast<Tegra::RenderTargetFormat>(stored.pipeline->State().color_formats[i]);
        if (format != Tegra::RenderTargetFormat::NONE) {
            num_rtvs = static_cast<u32>(i) + 1;
        }
    }
    if (params.num_vertices == 0 || params.num_instances == 0) {
        return;
    }
    if (num_rtvs == 0 && (!depth_view || !depth_view->RenderTarget().ptr)) {
        return;
    }
    for (size_t i = 0; i < num_rtvs; ++i) {
        const auto format =
            static_cast<Tegra::RenderTargetFormat>(stored.pipeline->State().color_formats[i]);
        const bool is_none = (format == Tegra::RenderTargetFormat::NONE);
        if (!is_none) {
            // Garbage guest render-target state must never reach OMSetRenderTargets or PSO
            // creation: an out-of-range, depth/stencil, or otherwise unrepresentable format
            // builds a degenerate pipeline that removes the device at execute time.
            const u32 raw_format = static_cast<u32>(format);
            const VideoCore::Surface::PixelFormat pixel_format =
                VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
            const char* skip_reason = nullptr;
            if (raw_format <
                    static_cast<u32>(Tegra::RenderTargetFormat::R32G32B32A32_FLOAT) ||
                raw_format > static_cast<u32>(Tegra::RenderTargetFormat::X8B8G8R8_SRGB)) {
                skip_reason = "out-of-range";
            } else if (pixel_format == VideoCore::Surface::PixelFormat::Invalid) {
                skip_reason = "Invalid mapping";
            } else if (m_texture_runtime.IsDepthStencilFormat(pixel_format)) {
                skip_reason = "depth-stencil";
            } else if (!m_texture_runtime.IsRepresentableRenderTarget(pixel_format)) {
                skip_reason = "unrepresentable";
            }
            if (skip_reason != nullptr) {
                if (!logged_rt_hole) {
                    logged_rt_hole = true;
                    LOG_ERROR(Render_D3D12,
                              "Draw with unrepresentable render target slot {} (color_formats "
                              "value {}, reason: {}) skipped",
                              i, raw_format, skip_reason);
                }
                return;
            }
        }
        const ImageView* view = framebuffer ? framebuffer->ColorBuffers()[i] : nullptr;
        if (!view || !view->RenderTarget().ptr) {
            // OMSetRenderTargets receives handles for all num_rtvs slots, and a null CPU
            // handle anywhere in the array is an execution-time device-removal on Xbox
            // that Close cannot catch.
            if (!logged_rt_hole) {
                logged_rt_hole = true;
                LOG_ERROR(Render_D3D12,
                          "Draw with unbound render target slot {} (pipeline format NONE={}) skipped",
                          i, is_none ? 1 : 0);
            }
            return;
        }
    }

    // NOTE: the index/vertex-range guards stay below, AFTER the buffer-binding section.
    // ClearDrawBindings() above wipes the index/vertex bindings and only
    // UpdateGraphicsBuffers/BindHostGeometryBuffers repopulate them, so those guards would
    // read cleared state (and skip every indexed draw) if moved up here. They still run
    // before any command_list recording below (descriptor tables, RT transitions, OMSet,
    // input assembly, Draw).
    // The enabled-uniform/storage masks must be registered BEFORE UpdateGraphicsBuffers
    // resolves binding buffer ids (matching vk's Configure: SetUniformBuffersState, then
    // UpdateGraphicsBuffers). With the old order the first draw resolved nothing and the
    // bind step dereferenced slot_buffers[BufferId{}] (an invalid slot id) — the silent
    // wedge behind the first-draw freeze; later draws only recovered because the masks
    // leaked in from the previous call.
    std::array<u32, VideoCommon::NUM_STAGES> masks{};
    VideoCommon::UniformBufferSizes sizes{};
    for (size_t stage = 0; stage < VideoCommon::NUM_STAGES; ++stage) {
        const Shader::Info* info = stored.infos[stage];
        if (!info) {
            continue;
        }
        masks[stage] = info->constant_buffer_mask;
        sizes[stage] = info->constant_buffer_used_sizes;
    }
    m_buffer_cache.SetUniformBuffersState(masks, &sizes);
    for (size_t stage = 0; stage < VideoCommon::NUM_STAGES; ++stage) {
        const Shader::Info* info = stored.infos[stage];
        if (!info) {
            continue;
        }
        m_buffer_cache.UnbindGraphicsStorageBuffers(stage);
        m_buffer_cache.UnbindGraphicsTextureBuffers(stage);
        u32 ssbo_index = 0;
        for (const auto& desc : info->storage_buffers_descriptors) {
            m_buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                     desc.cbuf_offset, desc.is_written);
            ++ssbo_index;
        }
    }
    m_buffer_cache.UpdateGraphicsBuffers(is_indexed);
    m_buffer_cache.BindHostGeometryBuffers(is_indexed);
    m_texture_cache.SynchronizeGraphicsDescriptors();

    // Index/vertex-range guards. These read the index/vertex bindings populated by
    // UpdateGraphicsBuffers/BindHostGeometryBuffers above, and they still precede every
    // command_list recording below (descriptor tables, RT transitions, OMSet, input
    // assembly, Draw).
    const auto& index_binding = m_runtime.GetIndexBinding();
    if (params.is_indexed && (!index_binding.valid || !index_binding.supported)) {
        return;
    }

    // Out-of-bounds index/vertex ranges fault the GPU at execute time (device removal
    // that Close cannot catch). Guest draw state can arrive uninitialized on console,
    // so first_index/base_vertex may exceed the bound buffers; skip such draws here.
    if (params.is_indexed) {
        u32 index_elem_size = 0;
        if (index_binding.format == DXGI_FORMAT_R16_UINT) {
            index_elem_size = 2;
        } else if (index_binding.format == DXGI_FORMAT_R32_UINT) {
            index_elem_size = 4;
        } else {
            return;
        }
        const u64 index_required =
            (static_cast<u64>(params.first_index) + static_cast<u64>(params.num_vertices)) *
            static_cast<u64>(index_elem_size);
        if (index_required > static_cast<u64>(index_binding.size)) {
            static bool logged_index_range{};
            if (!logged_index_range) {
                logged_index_range = true;
                LOG_ERROR(Render_D3D12,
                          "Draw with out-of-bounds index range (first_index={} num_vertices={} "
                          "index_size={}) skipped",
                          params.first_index, params.num_vertices, index_binding.size);
            }
            return;
        }
    }
    {
        // Only slots the vertex shader actually reads can fault the GPU fetching
        // attributes; unread slots are routinely left unbound and must be ignored.
        // The pipeline input layout is built from the enabled fixed attributes
        // (see GraphicsPipeline ctor), so the read set is the distinct
        // State().attributes[i].buffer over enabled attributes.
        const auto& vertex_bindings = m_runtime.GetVertexBindings();
        const auto& pipe_state = stored.pipeline->State();
        bool slot_read[32] = {};
        u32 num_read_slots = 0;
        for (u32 i = 0; i < pipe_state.attributes.size(); ++i) {
            if (pipe_state.attributes[i].enabled == 0) {
                continue;
            }
            const u32 read_slot = pipe_state.attributes[i].buffer.Value();
            if (read_slot >= vertex_bindings.size() || slot_read[read_slot]) {
                continue;
            }
            slot_read[read_slot] = true;
            ++num_read_slots;
        }
        if (num_read_slots == 0) {
            if (params.num_vertices > 0) {
                static bool logged_empty_vs_input{};
                if (!logged_empty_vs_input) {
                    logged_empty_vs_input = true;
                    LOG_ERROR(Render_D3D12,
                              "Draw with empty vertex-shader input (num_vertices={}) skipped",
                              params.num_vertices);
                }
                return;
            }
        }
        for (u32 slot = 0; slot < vertex_bindings.size(); ++slot) {
            if (!slot_read[slot]) {
                continue;
            }
            const D3D12_VERTEX_BUFFER_VIEW& view = vertex_bindings[slot];
            const u64 vertex_required =
                (static_cast<u64>(params.base_vertex) + static_cast<u64>(params.num_vertices)) *
                static_cast<u64>(view.StrideInBytes);
            if (view.BufferLocation == 0 || view.SizeInBytes == 0 || view.StrideInBytes == 0 ||
                vertex_required > static_cast<u64>(view.SizeInBytes)) {
                static bool logged_vertex_range{};
                if (!logged_vertex_range) {
                    logged_vertex_range = true;
                    LOG_ERROR(Render_D3D12,
                              "Draw with degenerate vertex view on read slot={} stride={} "
                              "vertex_size={} skipped",
                              slot, view.StrideInBytes, view.SizeInBytes);
                }
                return;
            }
        }
    }

    // Descriptor slots in shader-register order. The shader_recompiler SPIR-V backend
    // allocates set-0 CBV bindings and set-1 SRV/UAV/sampler bindings from counters
    // that accumulate across the pipeline's stages (per stage: SSBO -> texture buffer
    // -> image buffer -> sampled texture -> image), and Mesa maps those to DXIL
    // space 0/1 registers with sampled images split into SRV tN + sampler sN. The
    // tables are therefore filled by global binding index: CBVs at heap [0, num_cbv),
    // SRVs at [num_cbv, +num_res), UAVs at [num_cbv + num_res, +num_res), samplers at
    // [0, num_res) of the sampler heap. Every table position gets a valid descriptor
    // (null descriptors / default sampler where a slot has no host binding yet).
    enum class SlotKind : u8 {
        Storage,     // SSBO: SRV when read-only, UAV when written
        TexelBuffer, // texture buffer (null placeholder for now)
        ImageBuffer, // image buffer (null placeholder for now)
        Sampled,     // sampled texture from the TIC walk
        Image,       // storage image (null placeholder for now)
    };
    struct DrawSlot {
        SlotKind kind{};
        bool is_written{};
        u32 record{};                     // index into GetResourceBindings()
        VideoCommon::ImageViewId view{};  // Sampled
        VideoCommon::SamplerId sampler{}; // Sampled
    };
    std::vector<u32> cbv_records;    // global CBV binding -> record index
    std::vector<DrawSlot> res_slots; // global set-1 binding -> slot
    const bool via_header = regs.sampler_binding ==
                            Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding;
    for (size_t stage = 0; stage < VideoCommon::NUM_STAGES; ++stage) {
        const Shader::Info* info = stored.infos[stage];
        if (!info) {
            continue;
        }
        // Texel buffers are bound through the buffer cache (see the BindGraphicsTextureBuffer
        // walk below); image-buffer/image descriptors still bind as null placeholders so the
        // table order matches the shader's binding allocation.
        const u32 texel_count = Shader::NumDescriptors(info->texture_buffer_descriptors);
        if ((!info->image_buffer_descriptors.empty() || !info->image_descriptors.empty()) &&
            !logged_texel_buffers) {
            logged_texel_buffers = true;
            LOG_ERROR(Render_D3D12, "Image-buffer/image bindings bind as null for now");
        }
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        std::vector<VideoCommon::ImageViewInOut> views;
        std::vector<VideoCommon::SamplerId> sampler_ids;
        const auto read_handle = [&](const auto& desc, u32 index) {
            const u32 index_offset = index << desc.size_shift;
            const u32 offset = desc.cbuf_offset + index_offset;
            const GPUVAddr addr = cbufs[desc.cbuf_index].address + offset;
            if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                          std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
                if (desc.has_secondary) {
                    const u32 second_offset = desc.secondary_cbuf_offset + index_offset;
                    const GPUVAddr separate_addr =
                        cbufs[desc.secondary_cbuf_index].address + second_offset;
                    const u32 lhs_raw = gpu_memory->Read<u32>(addr) << desc.shift_left;
                    const u32 rhs_raw =
                        gpu_memory->Read<u32>(separate_addr) << desc.secondary_shift_left;
                    const u32 raw = lhs_raw | rhs_raw;
                    return TexturePair(raw, via_header);
                }
            }
            return TexturePair(gpu_memory->Read<u32>(addr), via_header);
        };
        const auto add_image = [&](const auto& desc, bool blacklist) {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle = read_handle(desc, index);
                views.push_back({
                    .index = handle.first,
                    .blacklist = blacklist,
                    .id = {},
                });
            }
        };
        for (const auto& desc : info->texture_buffer_descriptors) {
            add_image(desc, false);
        }
        for (const auto& desc : info->image_buffer_descriptors) {
            add_image(desc, false);
        }
        const size_t sampled_views_begin = views.size();
        for (const auto& desc : info->texture_descriptors) {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle = read_handle(desc, index);
                views.push_back(VideoCommon::ImageViewInOut{.index = handle.first});
                sampler_ids.push_back(handle.first == 0 ? VideoCommon::NULL_SAMPLER_ID
                                                         : m_texture_cache.GetGraphicsSamplerId(
                                                               handle.second));
            }
        }
        const size_t sampled_views_end = views.size();
        for (const auto& desc : info->image_descriptors) {
            add_image(desc, desc.is_written);
        }
        m_texture_cache.FillGraphicsImageViews<false>(
            std::span(views.data(), views.size()));
        size_t tbo_index = 0;
        VideoCommon::ImageViewInOut* texture_buffer_it = views.data();
        const auto add_buffer = [&](const auto& desc, bool is_image) {
            bool is_written = false;
            if constexpr (std::is_same_v<decltype(desc), const Shader::ImageBufferDescriptor&>) {
                is_written = desc.is_written;
            }
            for (u32 i = 0; i < desc.count; ++i) {
                const ImageView& view =
                    m_texture_cache.GetImageView(texture_buffer_it->id);
                m_buffer_cache.BindGraphicsTextureBuffer(stage, tbo_index, view.GpuAddr(),
                                                         view.BufferSize(), view.format,
                                                         is_written, is_image);
                ++tbo_index;
                ++texture_buffer_it;
            }
        };
        for (const auto& desc : info->texture_buffer_descriptors) {
            add_buffer(desc, false);
        }
        for (const auto& desc : info->image_buffer_descriptors) {
            add_buffer(desc, true);
        }
        const size_t rec_begin = m_runtime.GetResourceBindings().size();
        m_buffer_cache.BindHostStageBuffers(stage);
        const auto& records = m_runtime.GetResourceBindings();
        std::vector<u32> texel_records;
        for (size_t i = rec_begin; i < records.size(); ++i) {
            const auto& record = records[i];
            if (record.kind == BufferCacheRuntime::BindingKind::Uniform) {
                cbv_records.push_back(static_cast<u32>(i));
            } else if (record.kind == BufferCacheRuntime::BindingKind::Storage) {
                res_slots.push_back(DrawSlot{
                    .kind = SlotKind::Storage,
                    .is_written = record.is_written,
                    .record = static_cast<u32>(i),
                });
            } else {
                texel_records.push_back(static_cast<u32>(i));
            }
        }
        for (u32 i = 0; i < texel_count; ++i) {
            res_slots.push_back(DrawSlot{
                .kind = SlotKind::TexelBuffer,
                .record = i < texel_records.size() ? texel_records[i] : ~u32{0},
            });
        }
        for (const auto& desc : info->image_buffer_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                res_slots.push_back(DrawSlot{
                    .kind = SlotKind::ImageBuffer,
                    .is_written = desc.is_written,
                });
            }
        }
        for (size_t i = sampled_views_begin; i < sampled_views_end; ++i) {
            res_slots.push_back(DrawSlot{
                .kind = SlotKind::Sampled,
                .view = views[i].id,
                .sampler = sampler_ids[i - sampled_views_begin],
            });
        }
        for (const auto& desc : info->image_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                res_slots.push_back(DrawSlot{
                    .kind = SlotKind::Image,
                    .is_written = desc.is_written,
                });
            }
        }
    }
    static u32 traced_draws = 0;
    const auto& trace_vertex_bindings = m_runtime.GetVertexBindings();
    const u32 trace_ib_size = index_binding.valid ? index_binding.size : 0;
    const u32 trace_vb0_size =
        m_runtime.MaxVertexSlot() > 0 ? trace_vertex_bindings[0].SizeInBytes : 0;
    const u32 trace_vb0_stride =
        m_runtime.MaxVertexSlot() > 0 ? trace_vertex_bindings[0].StrideInBytes : 0;
    if (traced_draws < 30) {
        if (num_rtvs > 4) {
            LOG_DEBUG(Render_D3D12,
                      "Draw {}: num_rtvs={} fmt=[{},{},{},{},{},{},{},{}] dsv={} topo={} verts={} "
                      "instances={} indexed={} first_idx={} base_vtx={} base_inst={} ib_size={} "
                      "vb0_size={} vb0_stride={}",
                      traced_draws, num_rtvs,
                      static_cast<u32>(stored.pipeline->State().color_formats[0]),
                      static_cast<u32>(stored.pipeline->State().color_formats[1]),
                      static_cast<u32>(stored.pipeline->State().color_formats[2]),
                      static_cast<u32>(stored.pipeline->State().color_formats[3]),
                      static_cast<u32>(stored.pipeline->State().color_formats[4]),
                      static_cast<u32>(stored.pipeline->State().color_formats[5]),
                      static_cast<u32>(stored.pipeline->State().color_formats[6]),
                      static_cast<u32>(stored.pipeline->State().color_formats[7]),
                      depth_view ? 1 : 0,
                      static_cast<u32>(stored.pipeline->State().topology.Value()),
                      params.num_vertices, params.num_instances, params.is_indexed ? 1 : 0,
                      params.first_index, params.base_vertex, params.base_instance, trace_ib_size,
                      trace_vb0_size, trace_vb0_stride);
        } else {
            LOG_DEBUG(Render_D3D12,
                      "Draw {}: num_rtvs={} fmt=[{},{},{},{}] dsv={} topo={} verts={} instances={} "
                      "indexed={} first_idx={} base_vtx={} base_inst={} ib_size={} vb0_size={} "
                      "vb0_stride={}",
                      traced_draws, num_rtvs,
                      static_cast<u32>(stored.pipeline->State().color_formats[0]),
                      static_cast<u32>(stored.pipeline->State().color_formats[1]),
                      static_cast<u32>(stored.pipeline->State().color_formats[2]),
                      static_cast<u32>(stored.pipeline->State().color_formats[3]),
                      depth_view ? 1 : 0,
                      static_cast<u32>(stored.pipeline->State().topology.Value()),
                      params.num_vertices, params.num_instances, params.is_indexed ? 1 : 0,
                      params.first_index, params.base_vertex, params.base_instance, trace_ib_size,
                      trace_vb0_size, trace_vb0_stride);
        }
        ++traced_draws;
    }

    m_command_list.SetRootSignature(pipeline->GetRootSignature().Get());
    m_command_list.SetPipelineState(pipeline->Get());
    ID3D12DescriptorHeap* heaps[] = {m_res_heap.Get(), m_sampler_heap.Get()};
    m_command_list.Get()->SetDescriptorHeaps(2, heaps);

    const RootSignature& root_sig = pipeline->GetRootSignature();
    const auto& bindings = m_runtime.GetResourceBindings();
    const u32 num_cbv = static_cast<u32>(cbv_records.size());
    const u32 num_res = static_cast<u32>(res_slots.size());
    // Contiguous table regions: [0, num_cbv) CBV, [num_cbv, +num_res) SRV,
    // [num_cbv + num_res, +num_res) UAV. The sampler table is [0, num_res).
    const u32 heap_total = num_cbv + num_res * 2;
    const u32 heap_base = heap_total != 0 ? m_res_heap.Allocate(heap_total) : 0;
    const u32 sampler_base = num_res != 0 ? m_sampler_heap.Allocate(num_res) : 0;
    if ((heap_total != 0 && heap_base == DescriptorHeap::INVALID_INDEX) ||
        (num_res != 0 && sampler_base == DescriptorHeap::INVALID_INDEX)) {
        if (!logged_heap_exhausted) {
            logged_heap_exhausted = true;
            LOG_ERROR(Render_D3D12, "Descriptor heap exhausted (cbv={} res={}); draw skipped",
                      num_cbv, num_res);
        }
        return;
    }
    const u32 cbv_slot_base = heap_base;
    const u32 srv_slot_base = heap_base + num_cbv;
    const u32 uav_slot_base = heap_base + num_cbv + num_res;
    ID3D12Device* const d3d = m_device.GetDevice();

    for (u32 slot = 0; slot < num_cbv; ++slot) {
        const auto& record = bindings[cbv_records[slot]];
        D3D12_GPU_VIRTUAL_ADDRESS address = record.address;
        u64 cbv_size = record.size != 0 ? record.size : u64{256};
        if (m_runtime.IsNullResource(record.resource) || record.size == 0) {
            // Unbound/disabled uniform: read zeros.
            address = m_zero_address;
            cbv_size = 256;
        } else if ((address & (D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1)) != 0) {
            // Root/table CBVs require 256-byte aligned addresses; copy the range into
            // the aligned DEFAULT-heap scratch. (The old UPLOAD staging target was
            // illegal: UPLOAD heap can never be a copy destination.)
            const u64 aligned_size =
                Common::AlignUp<u64>(record.size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            const u64 scratch_offset =
                Common::AlignUp<u64>(m_cbv_scratch_used,
                                     D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (!m_cbv_scratch || scratch_offset + aligned_size > CBV_SCRATCH_SIZE) {
                if (!logged_unaligned_uniform) {
                    logged_unaligned_uniform = true;
                    LOG_ERROR(Render_D3D12, "CBV scratch exhausted; uniform reads as zero");
                }
                address = m_zero_address;
                cbv_size = 256;
            } else {
                const std::array copies{VideoCommon::BufferCopy{
                    .src_offset = record.offset,
                    .dst_offset = scratch_offset,
                    .size = record.size,
                }};
                m_runtime.CopyBuffer(m_cbv_scratch.Get(), record.resource, copies, true);
                address = m_cbv_scratch->GetGPUVirtualAddress() + scratch_offset;
                m_cbv_scratch_used = scratch_offset + aligned_size;
            }
        }
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{};
        cbv.BufferLocation = address;
        cbv.SizeInBytes = static_cast<UINT>(Common::AlignUp<u64>(
            std::min<u64>(cbv_size, 64 * 1024), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT));
        d3d->CreateConstantBufferView(&cbv, m_res_heap.CpuHandle(cbv_slot_base + slot));
    }

    for (u32 k = 0; k < num_res; ++k) {
        const DrawSlot& slot = res_slots[k];
        ID3D12Resource* resource = nullptr;
        u64 offset = 0;
        u64 size = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE prebuilt{};
        bool have_prebuilt = false;
        bool want_uav = false;
        switch (slot.kind) {
        case SlotKind::Storage: {
            const auto& record = bindings[slot.record];
            if (!m_runtime.IsNullResource(record.resource) && (record.offset % 4) == 0 &&
                record.size >= 4) {
                resource = record.resource;
                offset = record.offset;
                size = record.size;
            }
            want_uav = slot.is_written;
            break;
        }
        case SlotKind::TexelBuffer:
            if (slot.record < bindings.size() && slot.record != ~u32{0} &&
                bindings[slot.record].kind == BufferCacheRuntime::BindingKind::Texel &&
                bindings[slot.record].view.ptr != 0) {
                prebuilt = bindings[slot.record].view;
                have_prebuilt = true;
            }
            break;
        case SlotKind::ImageBuffer:
        case SlotKind::Image:
            // No host path yet: null descriptors below.
            want_uav = slot.is_written;
            break;
        case SlotKind::Sampled:
            if (slot.view != VideoCommon::NULL_IMAGE_VIEW_ID) {
                prebuilt = m_texture_cache.GetImageView(slot.view).Sampled();
                have_prebuilt = prebuilt.ptr != 0;
            }
            break;
        }
        if (have_prebuilt) {
            d3d->CopyDescriptorsSimple(1, m_res_heap.CpuHandle(srv_slot_base + k), prebuilt,
                                       D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        } else if (resource) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R32_TYPELESS;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Buffer.FirstElement = offset / 4;
            srv.Buffer.NumElements = static_cast<UINT>(std::max<u64>(size / 4, 1));
            srv.Buffer.StructureByteStride = 0;
            srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            d3d->CreateShaderResourceView(resource, &srv, m_res_heap.CpuHandle(srv_slot_base + k));
        } else {
            D3D12_SHADER_RESOURCE_VIEW_DESC null_srv{};
            null_srv.Format = DXGI_FORMAT_R32_UINT;
            null_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            null_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            null_srv.Buffer.FirstElement = 0;
            null_srv.Buffer.NumElements = 1;
            d3d->CreateShaderResourceView(nullptr, &null_srv, m_res_heap.CpuHandle(srv_slot_base + k));
        }
        if (want_uav && resource) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format = DXGI_FORMAT_R32_TYPELESS;
            uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uav.Buffer.FirstElement = offset / 4;
            uav.Buffer.NumElements = static_cast<UINT>(std::max<u64>(size / 4, 1));
            uav.Buffer.StructureByteStride = 0;
            uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            d3d->CreateUnorderedAccessView(resource, nullptr, &uav,
                                           m_res_heap.CpuHandle(uav_slot_base + k));
        } else {
            D3D12_UNORDERED_ACCESS_VIEW_DESC null_uav{};
            null_uav.Format = DXGI_FORMAT_R32_UINT;
            null_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            null_uav.Buffer.FirstElement = 0;
            null_uav.Buffer.NumElements = 1;
            d3d->CreateUnorderedAccessView(nullptr, nullptr, &null_uav,
                                           m_res_heap.CpuHandle(uav_slot_base + k));
        }
        // Samplers: one per set-1 binding so sN always pairs with tN. Unbound
        // textures and non-texture slots get a default sampler instead of leaving
        // garbage in the table.
        if (slot.kind == SlotKind::Sampled && slot.sampler != VideoCommon::NULL_SAMPLER_ID) {
            const Sampler& sampler = m_texture_cache.GetSampler(slot.sampler);
            d3d->CreateSampler(&sampler.Desc(), m_sampler_heap.CpuHandle(sampler_base + k));
        } else {
            D3D12_SAMPLER_DESC default_sampler{};
            default_sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            default_sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            default_sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            default_sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            default_sampler.MaxLOD = D3D12_FLOAT32_MAX;
            d3d->CreateSampler(&default_sampler, m_sampler_heap.CpuHandle(sampler_base + k));
        }
    }
    if (num_cbv > 0 && root_sig.GetCbvTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetGraphicsRootDescriptorTable(root_sig.GetCbvTableIndex(),
                                                      m_res_heap.GpuHandle(cbv_slot_base));
    }
    if (num_res > 0 && root_sig.GetSrvTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetGraphicsRootDescriptorTable(root_sig.GetSrvTableIndex(),
                                                      m_res_heap.GpuHandle(srv_slot_base));
    }
    if (num_res > 0 && root_sig.GetUavTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetGraphicsRootDescriptorTable(root_sig.GetUavTableIndex(),
                                                      m_res_heap.GpuHandle(uav_slot_base));
    }
    if (num_res > 0 && root_sig.GetSamplerTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetGraphicsRootDescriptorTable(root_sig.GetSamplerTableIndex(),
                                                      m_sampler_heap.GpuHandle(sampler_base));
    }
    if (root_sig.GetPushConstantIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetGraphicsRootConstantBufferView(root_sig.GetPushConstantIndex(),
                                                         m_zero_address);
    }
    if (root_sig.GetRuntimeDataIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetGraphicsRootConstantBufferView(root_sig.GetRuntimeDataIndex(),
                                                         m_zero_address);
    }

    // Render targets from the texture cache's current framebuffer. Transitions are
    // deduplicated: one resource may back several slots and a second COMMON->RT
    // barrier on it would claim the wrong before-state.
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[VideoCommon::NUM_RT]{};
    std::array<ID3D12Resource*, VideoCommon::NUM_RT + 1> rt_resources{};
    std::array<D3D12_RESOURCE_STATES, VideoCommon::NUM_RT + 1> rt_states{};
    u32 rt_resource_count = 0;
    const auto transition_rt = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES state) {
        if (!resource) {
            return;
        }
        for (u32 i = 0; i < rt_resource_count; ++i) {
            if (rt_resources[i] == resource) {
                return;
            }
        }
        rt_resources[rt_resource_count] = resource;
        rt_states[rt_resource_count] = state;
        ++rt_resource_count;
        m_command_list.Transition(resource, D3D12_RESOURCE_STATE_COMMON, state);
    };
    for (size_t i = 0; i < num_rtvs; ++i) {
        ImageView* view = framebuffer ? framebuffer->ColorBuffers()[i] : nullptr;
        rtvs[i] = view ? view->RenderTarget() : D3D12_CPU_DESCRIPTOR_HANDLE{0};
        transition_rt(view ? view->Resource() : nullptr, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv =
        depth_view ? depth_view->RenderTarget() : D3D12_CPU_DESCRIPTOR_HANDLE{0};
    transition_rt(depth_view ? depth_view->Resource() : nullptr,
                  D3D12_RESOURCE_STATE_DEPTH_WRITE);
    if (num_rtvs > 0 || dsv.ptr != 0) {
        m_command_list.Get()->OMSetRenderTargets(num_rtvs, num_rtvs > 0 ? rtvs : nullptr,
                                                 FALSE, dsv.ptr != 0 ? &dsv : nullptr);
    }

    // Input assembly.
    bool topology_supported = true;
    const D3D_PRIMITIVE_TOPOLOGY topology = ToD3DTopology(
        static_cast<Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology>(
            stored.pipeline->State().topology.Value()),
        topology_supported);
    if (!topology_supported && !logged_topology) {
        logged_topology = true;
        LOG_ERROR(Render_D3D12, "Topology without a D3D12 equivalent draws as a triangle list");
    }
    m_command_list.SetPrimitiveTopology(topology);
    const u32 vertex_count = m_runtime.MaxVertexSlot();
    if (vertex_count > 0) {
        m_command_list.Get()->IASetVertexBuffers(
            0, vertex_count, m_runtime.GetVertexBindings().data());
    }
    if (params.is_indexed && index_binding.valid) {
        D3D12_INDEX_BUFFER_VIEW ib_view{};
        ib_view.BufferLocation = index_binding.address;
        ib_view.SizeInBytes = index_binding.size;
        ib_view.Format = index_binding.format;
        m_command_list.Get()->IASetIndexBuffer(&ib_view);
    }

    m_command_list.SetViewport(MakeViewport(regs));
    m_command_list.SetScissorRect(MakeScissor(regs));

    m_query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                regs.zpass_pixel_count_enable != 0);
    if (params.is_indexed && index_binding.valid) {
        m_command_list.DrawIndexed(params.num_vertices, params.num_instances, params.first_index,
                                   static_cast<s32>(params.base_vertex), params.base_instance);
    } else {
        m_command_list.Draw(params.num_vertices, params.num_instances, params.base_vertex,
                            params.base_instance);
    }

    // Restore render targets to the home state for later copies/blits.
    for (u32 i = 0; i < rt_resource_count; ++i) {
        m_command_list.Transition(rt_resources[i], rt_states[i], D3D12_RESOURCE_STATE_COMMON);
    }
}

void RasterizerD3D12::DrawTexture() {}

std::optional<AccelerateDisplayInfo> RasterizerD3D12::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{m_texture_cache.mutex};
    auto [image_view, scaled] =
        m_texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    {
        // Temporary display-lookup diagnostic: what address/format is queried and hit/miss.
        static u32 accelerate_calls = 0;
        if (++accelerate_calls % 20 == 0) {
            LOG_DEBUG(Render_D3D12,
                      "AccelerateDisplay #{}: fb_addr={:#x} queried={:#x} {}x{} stride={} "
                      "pixfmt={} -> {}",
                      accelerate_calls, static_cast<u64>(config.address),
                      static_cast<u64>(framebuffer_addr), config.width, config.height,
                      config.stride, static_cast<u32>(config.pixel_format),
                      image_view ? "HIT" : "MISS");
        }
    }
    if (!image_view) {
        // Miss fallback: CPU-blitted scanout buffers are only registered in the texture
        // cache after the GPU renders to them, so the exact-address lookup above misses
        // while triple-buffered surfaces rotate. Register the CPU surface as a linear
        // image through the same find-or-insert path blits use, upload its bytes, then
        // retry the lookup so all format/view logic below is reused.
        const auto pixfmt = config.pixel_format;
        const bool sane_config =
            config.width != 0 && config.height != 0 && config.stride != 0 &&
            config.width < 8192 && config.height < 8192 && config.stride < 8192 &&
            (pixfmt == Service::android::PixelFormat::Rgba8888 ||
             pixfmt == Service::android::PixelFormat::Rgb565 ||
             pixfmt == Service::android::PixelFormat::Bgra8888);
        if (sane_config && gpu_memory != nullptr) {
            // Same mapping TryFindFramebufferImageView uses for the view format.
            const VideoCore::Surface::PixelFormat view_format =
                pixfmt == Service::android::PixelFormat::Rgb565
                    ? VideoCore::Surface::PixelFormat::R5G6B5_UNORM
                    : (pixfmt == Service::android::PixelFormat::Bgra8888
                           ? VideoCore::Surface::PixelFormat::B8G8R8A8_UNORM
                           : VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM);
            const u32 bytes_per_pixel =
                view_format == VideoCore::Surface::PixelFormat::R5G6B5_UNORM ? 2U : 4U;
            // Framebuffer stride is in pixels; tolerate byte strides just in case.
            const u32 stride_pixels = config.stride >= config.width * bytes_per_pixel
                                          ? config.stride / bytes_per_pixel
                                          : config.stride;
            const u32 surface_width =
                stride_pixels >= config.width ? stride_pixels : config.width;
            const u32 pitch_bytes = surface_width * bytes_per_pixel;
            // The find-or-insert path derives the CPU address from a GPU address; CPU-only
            // scanout surfaces are not GPU-mapped, so fall back to a CPU-address-direct
            // JoinImages insert that keeps both page tables keyed at the framebuffer address.
            const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(framebuffer_addr);
            {
                static bool logged_gate = false;
                if (!logged_gate) {
                    logged_gate = true;
                    if (cpu_addr) {
                        LOG_DEBUG(Render_D3D12,
                                  "Display surface gate: addr={:#x} derived cpu_addr={:#x} {}",
                                  static_cast<u64>(framebuffer_addr),
                                  static_cast<u64>(*cpu_addr),
                                  (*cpu_addr == framebuffer_addr) ? "identity" : "remapped");
                    } else {
                        LOG_DEBUG(Render_D3D12,
                                  "Display surface gate: addr={:#x} derived cpu_addr=unmapped",
                                  static_cast<u64>(framebuffer_addr));
                    }
                }
            }
            VideoCommon::ImageInfo info{};
            info.type = VideoCommon::ImageType::Linear;
            info.format = view_format;
            info.size.width = surface_width;
            info.size.height = config.height;
            info.size.depth = 1;
            info.pitch = pitch_bytes;
            VideoCommon::ImageId image_id{};
            if (cpu_addr && *cpu_addr == framebuffer_addr) {
                image_id = m_texture_cache.FindOrInsertImage(
                    info, static_cast<GPUVAddr>(framebuffer_addr));
            } else {
                // Overlap-aware find-first (mirrors GetBlitImages' FindImage-before-insert):
                // JoinImages always allocates a new image, so a repeated miss for an already
                // registered scanout address would leak a duplicate ~pitch*height image every
                // frame. Reuse the live image covering this range when one exists.
                const size_t range_bytes = static_cast<size_t>(pitch_bytes) * config.height;
                VideoCommon::ImageId existing_id{};
                m_texture_cache.ForEachImageInRegion(
                    static_cast<DAddr>(framebuffer_addr), range_bytes,
                    [&](VideoCommon::ImageId candidate_id, auto& candidate) {
                        if (candidate.gpu_addr != static_cast<GPUVAddr>(framebuffer_addr) ||
                            candidate.cpu_addr != static_cast<DAddr>(framebuffer_addr)) {
                            return false;
                        }
                        if (candidate.info.type != VideoCommon::ImageType::Linear ||
                            candidate.info.pitch != pitch_bytes ||
                            candidate.info.format != view_format) {
                            return false;
                        }
                        existing_id = candidate_id;
                        return true;
                    });
                image_id = existing_id ? existing_id
                                       : m_texture_cache.JoinImages(
                                             info, static_cast<GPUVAddr>(framebuffer_addr),
                                             static_cast<DAddr>(framebuffer_addr));
            }
            {
                static bool logged_insert = false;
                if (!logged_insert) {
                    logged_insert = true;
                    LOG_DEBUG(Render_D3D12, "Display surface insert: addr={:#x} image_id={} {}",
                              static_cast<u64>(framebuffer_addr), image_id.index,
                              image_id ? "ok" : "null");
                }
            }
            if (image_id) {
                // Fresh images start CpuModified; this uploads the CPU bytes.
                m_texture_cache.PrepareImage(image_id, false, false);
                // Freshly inserted images have no views yet, so a TryFind retry would
                // filter them out (empty view list) before creating one. Create the
                // view directly, mirroring TryFindFramebufferImageView's view block.
                VideoCommon::ImageViewInfo retry_info{VideoCommon::ImageViewType::e2D,
                                                       view_format};
                if (config.blending == Tegra::BlendMode::Opaque) {
                    retry_info.x_source =
                        static_cast<u8>(Tegra::Texture::SwizzleSource::R);
                    retry_info.y_source =
                        static_cast<u8>(Tegra::Texture::SwizzleSource::G);
                    retry_info.z_source =
                        static_cast<u8>(Tegra::Texture::SwizzleSource::B);
                    retry_info.w_source =
                        static_cast<u8>(Tegra::Texture::SwizzleSource::OneFloat);
                }
                const VideoCommon::ImageViewId retry_view_id =
                    m_texture_cache.FindOrEmplaceImageView(image_id, retry_info);
                image_view = &m_texture_cache.GetImageView(retry_view_id);
                scaled = false;
                {
                    static bool logged_retry = false;
                    if (!logged_retry) {
                        logged_retry = true;
                        LOG_DEBUG(Render_D3D12, "Display surface retry: addr={:#x} -> {}",
                                  static_cast<u64>(framebuffer_addr),
                                  image_view ? "HIT" : "MISS");
                    }
                }
                if (image_view) {
                    static bool logged_registered = false;
                    if (!logged_registered) {
                        logged_registered = true;
                        LOG_DEBUG(Render_D3D12,
                                  "Display surface registered as linear image: addr={:#x} "
                                  "{}x{} pixfmt={}",
                                  static_cast<u64>(framebuffer_addr), surface_width,
                                  config.height, static_cast<u32>(pixfmt));
                    }
                }
            }
        }
    }
    if (!image_view) {
        return {};
    }
    m_query_cache.NotifySegment(false);
    (void)scaled; // Resolution scaling is disabled on the D3D12 backend for now.
    return AccelerateDisplayInfo{
        .view = image_view,
        .width = image_view->size.width,
        .height = image_view->size.height,
    };
}

void RasterizerD3D12::Clear(u32 layer_count) {}
void RasterizerD3D12::DispatchCompute() {}
void RasterizerD3D12::ResetCounter(VideoCommon::QueryType type) {
    if (type != VideoCommon::QueryType::ZPassPixelCount64) {
        return;
    }
    m_query_cache.CounterReset(type);
}
void RasterizerD3D12::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (!gpu_memory) {
        return;
    }
    m_query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}
void RasterizerD3D12::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {
    std::scoped_lock lock{m_buffer_cache.mutex};
    m_buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}
void RasterizerD3D12::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    std::scoped_lock lock{m_buffer_cache.mutex};
    m_buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}
void RasterizerD3D12::FlushAll() {}
void RasterizerD3D12::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{m_buffer_cache.mutex};
        m_buffer_cache.DownloadMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{m_texture_cache.mutex};
        m_texture_cache.DownloadMemory(addr, size);
    }
}
bool RasterizerD3D12::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{m_buffer_cache.mutex};
        if (m_buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{m_texture_cache.mutex};
        if (m_texture_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    return false;
}
void RasterizerD3D12::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{m_buffer_cache.mutex};
        m_buffer_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{m_texture_cache.mutex};
        m_texture_cache.WriteMemory(addr, size);
    }
}
bool RasterizerD3D12::OnCPUWrite(DAddr addr, u64 size) {
    std::scoped_lock lock{m_buffer_cache.mutex};
    return m_buffer_cache.OnCPUWrite(addr, size);
}
void RasterizerD3D12::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_buffer_cache.WriteMemory(addr, size);
    m_texture_cache.WriteMemory(addr, size);
}
VideoCore::RasterizerDownloadArea RasterizerD3D12::GetFlushArea(PAddr addr, u64 size) {
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}
void RasterizerD3D12::InvalidateGPUCache() {}
void RasterizerD3D12::UnmapMemory(DAddr addr, u64 size) {
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_buffer_cache.WriteMemory(addr, size);
    m_texture_cache.WriteMemory(addr, size);
}
void RasterizerD3D12::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {}
void RasterizerD3D12::SignalFence(std::function<void()>&& func) {
    func();
}
void RasterizerD3D12::SyncOperation(std::function<void()>&& func) {
    func();
}
void RasterizerD3D12::SignalSyncPoint(u32 value) {
    auto& syncpoint_manager = m_gpu.Host1x().GetSyncpointManager();
    syncpoint_manager.IncrementGuest(value);
    syncpoint_manager.IncrementHost(value);
}
void RasterizerD3D12::SignalReference() {}
void RasterizerD3D12::ReleaseFences(bool) {}
void RasterizerD3D12::FlushAndInvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    FlushRegion(addr, size, which);
    InvalidateRegion(addr, size, which);
}
void RasterizerD3D12::WaitForIdle() {
    m_runtime.Finish();
}
void RasterizerD3D12::FragmentBarrier() {}
void RasterizerD3D12::TiledCacheBarrier() {}
void RasterizerD3D12::FlushCommands() {
    // Serializes with the cache's recorders (the DMA pusher calls this from channel threads)
    // and waits out the GPU before recycling the allocator, as CommandList requires.
    // Descriptor tables only live as long as the flushed work, so the per-draw heaps and
    // the CBV scratch ring reset here too.
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_command_list.Execute(m_device);
    m_device.WaitForIdle();
    m_command_list.Reset();
    m_res_heap.Reset();
    m_sampler_heap.Reset();
    m_cbv_scratch_used = 0;
}
void RasterizerD3D12::TickFrame() {
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_buffer_cache.TickFrame();
    m_texture_cache.TickFrame();
}
void RasterizerD3D12::LogMemoryStats() {
    std::scoped_lock lock{m_texture_cache.mutex};
    const auto vram = m_texture_cache.GetVRAMStats();
    LOG_INFO(Render_D3D12,
             "Memory stats: textures={} images={} used={:.0f} MB staging={:.0f} MB "
             "gpu_usage={:.0f} MB evicted_total={:.0f} MB",
             vram.texture_count, vram.sparse_texture_count,
             static_cast<f64>(vram.total_used_bytes) / (1024.0 * 1024.0),
             static_cast<f64>(m_staging_pool.TotalBytes()) / (1024.0 * 1024.0),
             static_cast<f64>(m_texture_runtime.GetDeviceMemoryUsage()) / (1024.0 * 1024.0),
             static_cast<f64>(vram.evicted_total) / (1024.0 * 1024.0));
}
Tegra::Engines::AccelerateDMAInterface& RasterizerD3D12::AccessAccelerateDMA() {
    return m_accelerate_dma;
}
bool RasterizerD3D12::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{m_texture_cache.mutex};
    return m_texture_cache.BlitImage(dst, src, copy_config);
}
void RasterizerD3D12::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {
    std::scoped_lock lock{m_buffer_cache.mutex};
    const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) {
        return;
    }
    if (!m_buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
        m_buffer_cache.WriteMemory(*cpu_addr, copy_size);
    }
}
void RasterizerD3D12::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                        const VideoCore::DiskResourceLoadCallback& callback) {}
void RasterizerD3D12::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_buffer_cache.CreateChannel(channel);
    m_texture_cache.CreateChannel(channel);
    m_query_cache.CreateChannel(channel);
    m_pipeline_cache.CreateChannel(channel);
}
void RasterizerD3D12::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_buffer_cache.BindToChannel(channel_id);
    m_texture_cache.BindToChannel(channel_id);
    m_query_cache.BindToChannel(channel_id);
    m_pipeline_cache.BindToChannel(channel_id);
}
void RasterizerD3D12::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_buffer_cache.EraseChannel(channel_id);
    m_texture_cache.EraseChannel(channel_id);
    m_query_cache.EraseChannel(channel_id);
    m_pipeline_cache.EraseChannel(channel_id);
}

} // namespace D3D12