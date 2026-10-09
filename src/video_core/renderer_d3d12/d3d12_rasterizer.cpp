// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <type_traits>

#include <fmt/format.h>

#include "common/alignment.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_d3d12/d3d12_capture.h"
#include "video_core/renderer_d3d12/d3d12_cmd_ring.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_rasterizer.h"

namespace D3D12 {

namespace {
// Draws re-enabled, guards active (no-vertex skip, depth-RTV skip, all-slots-bound guard, RT-format and index/vertex range validation).
constexpr bool kSkipDraws = false;

// Temporary diagnostic: localize the device removal to a single draw by flushing after the
// first few draws and checking GetDeviceRemovedReason, dumping full bindings for each.
std::atomic<u32> g_diag_draws{0};

// TEMP DIAGNOSTIC (session 12): resources of the images the frontend presents; ConfigureDraw
// tags draws that render into them so the composition path is unmissable in the log.
std::array<ID3D12Resource*, 6> g_display_resources{};

// TEMP DIAGNOSTIC (session 11): split ConfigureDraw's state recording from the rest of the
// frame. When true, ConfigureDraw returns right after the upload/binding stage walk, before
// any state command (root signature, PSO, descriptor heaps/tables, RT transitions, OMSet,
// input assembly, viewport/scissor, CounterEnable) and before the draw itself is recorded.
// Uploads/buffer binding above the return point still run, so if the device still dies the
// trigger is outside ConfigureDraw's state commands; if it survives, they are implicated.
constexpr bool kSkipConfigureDrawState = false;
} // namespace

using Tegra::Texture::TexturePair;

// TEMP DIAGNOSTIC (session 13): color render target of the first depth-tested 3D draw, set in
// ConfigureDraw and probed by the present path (declared in d3d12_texture_cache.h).
Microsoft::WRL::ComPtr<ID3D12Resource> g_probe_scene3d;

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
    : m_gpu{gpu}, m_device_memory{device_memory}, m_device{device}, m_staging_pool{device},
      m_command_list{device.GetDevice()},
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
        return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
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
    // Xbox D3D12 rasterizes nothing with a negative-height viewport, so the Y flip is
    // applied by the translated shaders (DXIL_SPIRV_Y_FLIP_UNCONDITIONAL). Mirror the
    // viewport to a positive height; the combined screen mapping is unchanged.
    if (viewport.Height < 0.0f) {
        viewport.TopLeftY += viewport.Height;
        viewport.Height = -viewport.Height;
    }
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

    // TEMP DIAGNOSTIC: submit everything recorded before this draw and check the device,
    // to split "interleaved work" from "this draw's own commands".
    {
        static std::atomic<u32> g_diag_preflush{0};
        const u32 pre_index = g_diag_preflush.fetch_add(1, std::memory_order_relaxed);
        if (pre_index < 4) {
            m_command_list.Execute(m_device);
            m_device.WaitForIdle();
            m_command_list.Reset();
            m_res_heap.Reset();
            m_sampler_heap.Reset();
            m_cbv_scratch_used = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const HRESULT pre_reason = m_device.GetDevice()->GetDeviceRemovedReason();
            LOG_ERROR(Render_D3D12, "DIAG pre-flush {}: device reason {:#x}", pre_index,
                      static_cast<u32>(pre_reason));
            if (FAILED(pre_reason)) {
                Diag::Dump();
            }
        }
    }

    m_runtime.ClearDrawBindings();

    // Validate BEFORE recording any uploads: draws the backend cannot represent must
    // return here, before the buffer-binding section below records staging COPY commands
    // for garbage draw state (those copies execute at flush and fault the GPU on console,
    // removing the device even for draws every later guard would skip).
    m_texture_cache.UpdateRenderTargets(false);
    Framebuffer* framebuffer = m_texture_cache.GetFramebuffer();

    // TEMP DIAGNOSTIC: flush right after the RT-cache update so its recorded work can be
    // separated from the draw-state commands that follow.
    {
        static std::atomic<u32> g_diag_midflush{0};
        const u32 mid_index = g_diag_midflush.fetch_add(1, std::memory_order_relaxed);
        if (mid_index < 6) {
            m_command_list.Execute(m_device);
            m_device.WaitForIdle();
            m_command_list.Reset();
            m_res_heap.Reset();
            m_sampler_heap.Reset();
            m_cbv_scratch_used = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const HRESULT mid_reason = m_device.GetDevice()->GetDeviceRemovedReason();
            LOG_ERROR(Render_D3D12, "DIAG mid-flush {}: device reason {:#x}", mid_index,
                      static_cast<u32>(mid_reason));
            if (FAILED(mid_reason)) {
                Diag::Dump();
            }
        }
    }

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

    bool flinger_draw = false;
    {
        static bool marker_logged = false;
        if (!marker_logged) {
            marker_logged = true;
            LOG_WARNING(Render_D3D12, "ConfigureDraw build marker: 174-flinger-dbg");
        }
        // Diagnostic (session 12): which images draws render into. The flinger display
        // buffer GPU VAs must appear here if the guest draws its composed frame straight
        // into the queued buffer.
        static u32 rt_logs = 0;
        bool renders_to_display = false;
        std::string rts;
        std::string rts_ids;
        std::string rts_att;
        // TEMP DIAGNOSTIC (session 14): track the color target of the latest depth-tested wide
        // 3D draw (the scene behind the UI) so the scene3d BMP captures can prove whether that
        // scene RT receives content, independent of the composition/FLINGER layers. Session 13
        // armed this only once, which latched an auxiliary RT (0x5109f0000) that is black during
        // the style screen; the visible scene draws into 0x5103e0000 (960x540) continuously.
        const D3D12_VIEWPORT probe_vp = MakeViewport(maxwell3d->regs);
        const bool probe_wide_scene = probe_vp.Height > 0.0f && probe_vp.Width >= 640.0f &&
                                      probe_vp.Width / probe_vp.Height > 1.6f &&
                                      probe_vp.Width / probe_vp.Height < 1.85f;
        static u32 scene3d_probe_switches = 0;
        for (size_t i = 0; i < num_rtvs; ++i) {
            const ImageView* view = framebuffer ? framebuffer->ColorBuffers()[i] : nullptr;
            const u64 addr = view ? static_cast<u64>(view->GpuAddr()) : 0;
            ID3D12Resource* const res = view ? view->Resource() : nullptr;
            bool is_display = false;
            for (ID3D12Resource* const display_res : g_display_resources) {
                if (display_res != nullptr && display_res == res) {
                    is_display = true;
                }
            }
            if (is_display) {
                renders_to_display = true;
            } else if (res != nullptr && view != nullptr && view->size.width >= 1920 &&
                       view->size.height >= 1080) {
                g_probe_scene = res;
            }
            if (probe_wide_scene && res != nullptr && view != nullptr &&
                depth_view != nullptr && depth_view->RenderTarget().ptr &&
                g_probe_scene3d.Get() != res) {
                g_probe_scene3d = res;
                if (scene3d_probe_switches < 16) {
                    ++scene3d_probe_switches;
                    const D3D12_RESOURCE_DESC scene3d_desc = res->GetDesc();
                    ID3D12Resource* const scene3d_depth_res = depth_view->Resource();
                    const u32 dsv_fmt = scene3d_depth_res
                                            ? static_cast<u32>(scene3d_depth_res->GetDesc().Format)
                                            : 0;
                    LOG_WARNING(Render_D3D12,
                                "Scene3D probe RT: gpu={:#x} fmt={:#x} {}x{} dsv_fmt={:#x} "
                                "vp={}x{} aspect={:.3f}",
                                addr, static_cast<u32>(scene3d_desc.Format), scene3d_desc.Width,
                                scene3d_desc.Height, dsv_fmt, probe_vp.Width, probe_vp.Height,
                                probe_vp.Width / probe_vp.Height);
                }
            }
            // TEMP DIAGNOSTIC (session 14): the 6-vertex 1920x1080 fullscreen draws are the
            // compositor passes; track their render target so the mid-frame capture can dump it.
            if (res != nullptr && params.num_vertices == 6 && probe_vp.Width >= 1900.0f &&
                probe_vp.Height >= 1060.0f) {
                g_probe_ui = res;
            }
            rts += fmt::format("{}{:#x}", i == 0 ? "" : ",", addr);
            rts_ids += fmt::format("{}{}:{}", i == 0 ? "" : ",", view ? view->image_id.index : 0,
                                   static_cast<const void*>(res));
            rts_att += fmt::format("{}{:#x}", i == 0 ? "" : ",",
                                   stored.pipeline->State().attachments[i].raw);
        }
        flinger_draw = renders_to_display;
        {
            static u32 dbg_fd = 0;
            if (dbg_fd < 3 || dbg_fd % 500 == 0) {
                LOG_WARNING(Render_D3D12, "DBG-FD: fd={} rtd={}", flinger_draw ? 1 : 0,
                            renders_to_display ? 1 : 0);
            }
            ++dbg_fd;
        }
        // TEMP DIAGNOSTIC (session 13): stop logging every composition draw; the per-draw
        // log volume on the console is large enough to slow the emulator down.
        if (rt_logs < 400 || rt_logs % 1000 == 0) {
            const D3D12_VIEWPORT rt_vp = MakeViewport(maxwell3d->regs);
            const D3D12_RECT rt_sc = MakeScissor(maxwell3d->regs);
            LOG_WARNING(Render_D3D12,
                        "Draw RTs{}: gpu=[{}] ids=[{}] att=[{}] verts={} indexed={} "
                        "vp=({},{},{},{}) sc=({},{},{},{}) depth={} dfunc={} dsv={} cull_en={} "
                        "cull_face={} poly={} fd={} ff={}",
                        renders_to_display ? " (FLINGER)" : "", rts, rts_ids, rts_att,
                        params.num_vertices, params.is_indexed ? 1 : 0, rt_vp.TopLeftX,
                        rt_vp.TopLeftY, rt_vp.Width, rt_vp.Height, rt_sc.left, rt_sc.top,
                        rt_sc.right, rt_sc.bottom, stored.pipeline->State().depth_test_enable.Value(),
                        stored.pipeline->State().depth_test_func.Value(),
                        depth_view && depth_view->RenderTarget().ptr ? 1 : 0,
                        stored.pipeline->State().cull_enable.Value(),
                        static_cast<u32>(stored.pipeline->State().CullFaceMode()),
                        static_cast<u32>(stored.pipeline->State().PolygonModeMode()),
                        static_cast<u32>(stored.pipeline->State().FrontFaceMode()),
                        flinger_draw ? 1 : 0);
        }
        ++rt_logs;
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

    {
        static u32 dbg_a = 0;
        static u32 fd_dbg_a = 0;
        const auto& dbg_ib = m_runtime.GetIndexBinding();
        const bool log_dbg_a = dbg_a < 4 || (flinger_draw && (fd_dbg_a % 256) == 0);
        ++dbg_a;
        if (flinger_draw) {
            ++fd_dbg_a;
        }
        if (log_dbg_a) {
            LOG_WARNING(Render_D3D12,
                        "DBG-A: fd={} indexed={} ib_valid={} ib_supported={} ib_size={} fmt={:#x}",
                        flinger_draw ? 1 : 0, params.is_indexed ? 1 : 0, dbg_ib.valid,
                        dbg_ib.supported, dbg_ib.size, static_cast<u32>(dbg_ib.format));
        }
    }

    // Index/vertex-range guards. These read the index/vertex bindings populated by
    // UpdateGraphicsBuffers/BindHostGeometryBuffers above, and they still precede every
    // command_list recording below (descriptor tables, RT transitions, OMSet, input
    // assembly, Draw).
    const auto& index_binding = m_runtime.GetIndexBinding();
    if (params.is_indexed && (!index_binding.valid || !index_binding.supported)) {
        if (flinger_draw) {
            static u32 silent_guard_logs = 0;
            if (silent_guard_logs < 8) {
                ++silent_guard_logs;
                LOG_WARNING(Render_D3D12,
                            "FLINGER draw skipped: index binding invalid/unsupported (valid={} "
                            "supported={})",
                            index_binding.valid, index_binding.supported);
            }
        }
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
            if (flinger_draw) {
                static u32 silent_fmt_logs = 0;
                if (silent_fmt_logs < 8) {
                    ++silent_fmt_logs;
                    LOG_WARNING(Render_D3D12,
                                "FLINGER draw skipped: unsupported index format {:#x}",
                                static_cast<u32>(index_binding.format));
                }
            }
            return;
        }
        const u64 index_required =
            (static_cast<u64>(params.first_index) + static_cast<u64>(params.num_vertices)) *
            static_cast<u64>(index_elem_size);
        if (index_required > static_cast<u64>(index_binding.size)) {
            static u32 index_range_skips{};
            if ((index_range_skips++ % 256) == 0) {
                LOG_ERROR(Render_D3D12,
                          "Draw with out-of-bounds index range (first_index={} num_vertices={} "
                          "index_size={}) skipped (count {})",
                          params.first_index, params.num_vertices, index_binding.size,
                          index_range_skips);
            }
            if (flinger_draw) {
                static u32 fd_idx_logs = 0;
                if (fd_idx_logs < 4) {
                    ++fd_idx_logs;
                    LOG_WARNING(Render_D3D12,
                                "FLINGER skip: index range first_index={} verts={} size={}",
                                params.first_index, params.num_vertices, index_binding.size);
                }
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
                static u32 empty_vs_input_skips{};
                if ((empty_vs_input_skips++ % 256) == 0) {
                    LOG_ERROR(Render_D3D12,
                              "Draw with empty vertex-shader input (num_vertices={}) skipped "
                              "(count {})",
                              params.num_vertices, empty_vs_input_skips);
                }
                if (flinger_draw) {
                    static u32 fd_vin_logs = 0;
                    if (fd_vin_logs < 4) {
                        ++fd_vin_logs;
                        LOG_WARNING(Render_D3D12,
                                    "FLINGER skip: empty vertex-shader input verts={}",
                                    params.num_vertices);
                    }
                }
                return;
            }
        }
        for (u32 slot = 0; slot < vertex_bindings.size(); ++slot) {
            if (!slot_read[slot]) {
                continue;
            }
            const D3D12_VERTEX_BUFFER_VIEW& view = vertex_bindings[slot];
            // The vertex fetch range is governed by the largest index in the draw, not by
            // the index count: a fullscreen quad is 6 indices over 4 vertices, so using
            // the index count rejected every composition draw (black TV). Read the (small)
            // index range from guest memory and bound by the actual maximum index.
            constexpr u32 kMaxGuardIndices = 4096;
            u64 vertex_required = 0;
            bool range_known = false;
            if (params.is_indexed && index_binding.device_addr != 0 &&
                params.num_vertices <= kMaxGuardIndices &&
                (index_binding.format == DXGI_FORMAT_R16_UINT ||
                 index_binding.format == DXGI_FORMAT_R32_UINT)) {
                const u32 index_bytes = index_binding.format == DXGI_FORMAT_R16_UINT ? 2 : 4;
                std::array<u8, kMaxGuardIndices * 4> index_data{};
                const DAddr read_addr = index_binding.device_addr +
                                        static_cast<DAddr>(params.first_index) * index_bytes;
                const auto read_result = m_device_memory.ReadBlockUnsafe(
                    read_addr, index_data.data(),
                    static_cast<size_t>(params.num_vertices) * index_bytes,
                    "RasterizerD3D12.VertexRangeGuard", false);
                if (read_result.fully_mapped) {
                    u64 max_index = 0;
                    for (u32 i = 0; i < params.num_vertices; ++i) {
                        u64 value = 0;
                        if (index_bytes == 2) {
                            u16 v{};
                            std::memcpy(&v, index_data.data() + i * 2, sizeof(v));
                            value = v;
                        } else {
                            u32 v{};
                            std::memcpy(&v, index_data.data() + i * 4, sizeof(v));
                            value = v;
                        }
                        max_index = std::max(max_index, value);
                    }
                    vertex_required = (static_cast<u64>(params.base_vertex) + max_index + 1) *
                                      static_cast<u64>(view.StrideInBytes);
                    range_known = true;
                }
            }
            bool enforce_range = true;
            if (!range_known) {
                if (params.is_indexed) {
                    // Indexed draw whose index data cannot be inspected (e.g. guest u8
                    // indices expanded into staging): the index count is not a vertex
                    // bound, so never reject it here; the index-range guard above still
                    // protects the index fetch itself.
                    enforce_range = false;
                } else {
                    vertex_required = (static_cast<u64>(params.base_vertex) +
                                       static_cast<u64>(params.num_vertices)) *
                                      static_cast<u64>(view.StrideInBytes);
                }
            }
            if (view.BufferLocation == 0 || view.SizeInBytes == 0 || view.StrideInBytes == 0 ||
                (enforce_range && vertex_required > static_cast<u64>(view.SizeInBytes))) {
                static u32 vertex_range_skips{};
                if ((vertex_range_skips++ % 256) == 0) {
                    LOG_ERROR(Render_D3D12,
                              "Draw with degenerate vertex view on read slot={} stride={} "
                              "vertex_size={} skipped (count {})",
                              slot, view.StrideInBytes, view.SizeInBytes, vertex_range_skips);
                }
                if (flinger_draw) {
                    static u32 fd_vb_logs = 0;
                    if (fd_vb_logs < 4) {
                        ++fd_vb_logs;
                        LOG_WARNING(Render_D3D12,
                                    "FLINGER skip: degenerate vb slot={} loc={:#x} stride={} "
                                    "size={} base_vtx={} verts={} range_known={} dev={:#x} req={}",
                                    slot, static_cast<u64>(view.BufferLocation),
                                    view.StrideInBytes, view.SizeInBytes, params.base_vertex,
                                    params.num_vertices, range_known ? 1 : 0,
                                    static_cast<u64>(index_binding.device_addr), vertex_required);
                    }
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
        // TEMP DIAGNOSTIC (session 13): the composition draws execute but cover no pixels.
        // Dump the guest constant buffers the translated stage declares and their raw bytes,
        // so the VS's expected transform constants can be correlated with what the guest wrote.
        if (flinger_draw) {
            static u32 fd_cbuf_logs = 0;
            if (fd_cbuf_logs < 12) {
                ++fd_cbuf_logs;
                std::string used;
                for (u32 ci = 0; ci < Shader::Info::MAX_CBUFS; ++ci) {
                    if (((info->constant_buffer_mask >> ci) & 1) != 0) {
                        used += fmt::format("{}:{} ", ci, info->constant_buffer_used_sizes[ci]);
                    }
                }
                LOG_WARNING(Render_D3D12,
                            "FLINGER stage {}: cbuf_mask={:#x} used=[{}] nvn_base={} "
                            "nvn_used={:#x}",
                            stage, info->constant_buffer_mask, used, info->nvn_buffer_base,
                            static_cast<u64>(info->nvn_buffer_used.to_ulong()));
                for (u32 ci = 0; ci < Shader::Info::MAX_CBUFS; ++ci) {
                    if (((info->constant_buffer_mask >> ci) & 1) == 0) {
                        continue;
                    }
                    const auto& cb = cbufs[ci];
                    const auto dev = gpu_memory->GpuToCpuAddress(cb.address);
                    std::string hex;
                    std::string flts;
                    if (dev) {
                        std::array<u8, 64> bytes{};
                        const u32 dump_size =
                            std::min<u32>(static_cast<u32>(bytes.size()), cb.size);
                        const auto read = m_device_memory.ReadBlockUnsafe(
                            *dev, bytes.data(), dump_size, "FLINGER.CBVDump", false);
                        if (read.fully_mapped) {
                            for (u32 i = 0; i < dump_size; ++i) {
                                hex += fmt::format("{:02x}", bytes[i]);
                            }
                            for (u32 i = 0; i + 3 < dump_size; i += 4) {
                                f32 v = 0.0f;
                                std::memcpy(&v, bytes.data() + i, sizeof(v));
                                flts += fmt::format("{} ", v);
                            }
                        } else {
                            flts = "unmapped";
                        }
                    } else {
                        flts = "no-dev";
                    }
                    LOG_WARNING(Render_D3D12,
                                "FLINGER cbuf s{} i{}: gpu={:#x} dev={:#x} size={} used={} "
                                "enabled={} hex={} f32=[{}]",
                                stage, ci, static_cast<u64>(cb.address),
                                dev ? static_cast<u64>(*dev) : 0, cb.size,
                                info->constant_buffer_used_sizes[ci], cb.enabled ? 1 : 0, hex,
                                flts);
                }
            }
        }
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
        // TEMP DIAGNOSTIC: flush right after the texture views/upload step.
        {
            static std::atomic<u32> g_diag_tex_flush{0};
            if (num_rtvs == 6 &&
                g_diag_tex_flush.fetch_add(1, std::memory_order_relaxed) < 4) {
                m_command_list.Execute(m_device);
                m_device.WaitForIdle();
                m_command_list.Reset();
                m_res_heap.Reset();
                m_sampler_heap.Reset();
                m_cbv_scratch_used = 0;
                Diag::Push(Diag::CmdKind::Marker, 200);
                const HRESULT tex_reason = m_device.GetDevice()->GetDeviceRemovedReason();
                LOG_ERROR(Render_D3D12, "DIAG tex-flush: device reason {:#x}",
                          static_cast<u32>(tex_reason));
                if (FAILED(tex_reason)) {
                    DumpRecentTextureUploads();
                    Diag::Dump();
                }
            }
        }
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

    const bool diag_fault_pipe = num_rtvs == 6 && cbv_records.size() == 3;

    // TEMP DIAGNOSTIC: flush after the stage/descriptor walk (texture and buffer uploads)
    // and before any pipeline/root-signature/descriptor-table state is recorded.
    {
        static std::atomic<u32> g_diag_stage_flush{0};
        if (diag_fault_pipe && g_diag_stage_flush.fetch_add(1, std::memory_order_relaxed) < 2) {
            m_command_list.Execute(m_device);
            m_device.WaitForIdle();
            m_command_list.Reset();
            m_res_heap.Reset();
            m_sampler_heap.Reset();
            m_cbv_scratch_used = 0;
            const HRESULT stage_reason = m_device.GetDevice()->GetDeviceRemovedReason();
            LOG_ERROR(Render_D3D12, "DIAG stage-flush: device reason {:#x}",
                      static_cast<u32>(stage_reason));
            if (FAILED(stage_reason)) {
                DumpRecentTextureUploads();
            }
        }
    }

    // TEMP DIAGNOSTIC (session 11): skip every state command from here on. The device
    // removal at ~211 s survives with no draw/upload/copy/blit command, so record nothing
    // from the state block (root sig, PSO, heaps, descriptor tables, RT barriers, OMSet,
    // IA, viewport/scissor, counter) and see whether the device survives. The removal
    // check is DELAYED (~50 ms after WaitForIdle) because the removal status latches late
    // on Xbox; an immediate 0x0 check is not trustworthy. Active for all draws so a fault
    // at ~211 s can still be attributed even if it happens after the first draw batch.
    if (kSkipConfigureDrawState) {
        static std::atomic<u32> g_diag_stateskip{0};
        const u32 skip_index = g_diag_stateskip.fetch_add(1, std::memory_order_relaxed);
        if (skip_index < 4) {
            LOG_ERROR(Render_D3D12,
                      "DIAG stateskip {}: indexed={} verts={} inst={} num_rtvs={} cbv={} res={} "
                      "zpass={}",
                      skip_index, params.is_indexed ? 1 : 0, params.num_vertices,
                      params.num_instances, num_rtvs, cbv_records.size(), res_slots.size(),
                      regs.zpass_pixel_count_enable);
            m_command_list.Execute(m_device);
            m_device.WaitForIdle();
            m_command_list.Reset();
            m_res_heap.Reset();
            m_sampler_heap.Reset();
            m_cbv_scratch_used = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            Diag::Push(Diag::CmdKind::Marker, 100 + skip_index);
            const HRESULT stateskip_reason = m_device.GetDevice()->GetDeviceRemovedReason();
            LOG_ERROR(Render_D3D12, "DIAG stateskip {}: delayed device reason {:#x}", skip_index,
                      static_cast<u32>(stateskip_reason));
            if (FAILED(stateskip_reason)) {
                Diag::Dump();
            }
        }
        return;
    }

    {
        // TEMP DIAGNOSTIC (session 14): count draws that passed the pre-state guards and are
        // about to be recorded against the tracked scene RT, to split "guards skip the scene"
        // from "the scene draws execute but write nothing".
        static std::atomic<u32> scene_draw_passed{0};
        ID3D12Resource* const scene_rt =
            framebuffer != nullptr && num_rtvs > 0 && framebuffer->ColorBuffers()[0] != nullptr
                ? framebuffer->ColorBuffers()[0]->Resource()
                : nullptr;
        if (scene_rt != nullptr && scene_rt == g_probe_scene3d.Get()) {
            const u32 passed = scene_draw_passed.fetch_add(1, std::memory_order_relaxed);
            if (passed < 8 || passed % 256 == 0) {
                LOG_WARNING(Render_D3D12, "Scene RT draw passed guards #{} verts={} indexed={}",
                            passed, params.num_vertices, params.is_indexed ? 1 : 0);
            }
        }
    }
    m_command_list.SetRootSignature(pipeline->GetRootSignature().Get());
    m_command_list.SetPipelineState(pipeline->Get());
    ID3D12DescriptorHeap* heaps[] = {m_res_heap.Get(), m_sampler_heap.Get()};
    m_command_list.Get()->SetDescriptorHeaps(2, heaps);
    // TEMP DIAGNOSTIC: flush after root signature/PSO/heaps only.
    {
        static std::atomic<u32> g_diag_pso_flush{0};
        if (diag_fault_pipe && g_diag_pso_flush.fetch_add(1, std::memory_order_relaxed) < 2) {
            m_command_list.Execute(m_device);
            m_device.WaitForIdle();
            m_command_list.Reset();
            m_res_heap.Reset();
            m_sampler_heap.Reset();
            m_cbv_scratch_used = 0;
            const HRESULT pso_reason = m_device.GetDevice()->GetDeviceRemovedReason();
            LOG_ERROR(Render_D3D12, "DIAG pso-flush: device reason {:#x}",
                      static_cast<u32>(pso_reason));
        }
    }

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
        if (flinger_draw) {
            static u32 fd_heap_logs = 0;
            if (fd_heap_logs < 4) {
                ++fd_heap_logs;
                LOG_WARNING(Render_D3D12,
                            "FLINGER skip: descriptor heap exhausted cbv={} res={}", num_cbv,
                            num_res);
            }
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
            // ComparisonFunc and MaxAnisotropy are validated enums on Xbox; D3D12's
            // zero-initialized values (0) are invalid and make the whole sampler
            // descriptor invalid, which faults the GPU when a shader uses it.
            default_sampler.MaxAnisotropy = 1;
            default_sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            default_sampler.MinLOD = 0.0f;
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

    // TEMP DIAGNOSTIC: flush after the PSO/root-signature/descriptor-table commands, before
    // the render-target/input-assembly state, to split the remaining state commands.
    {
        static std::atomic<u32> g_diag_tables_flush{0};
        if (num_rtvs == 6 && cbv_records.size() == 3 &&
            g_diag_tables_flush.fetch_add(1, std::memory_order_relaxed) < 2) {
            m_command_list.Execute(m_device);
            m_device.WaitForIdle();
            m_command_list.Reset();
            m_res_heap.Reset();
            m_sampler_heap.Reset();
            m_cbv_scratch_used = 0;
            const HRESULT tables_reason = m_device.GetDevice()->GetDeviceRemovedReason();
            LOG_ERROR(Render_D3D12, "DIAG tables-flush: device reason {:#x}",
                      static_cast<u32>(tables_reason));
        }
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
        Diag::Push(Diag::CmdKind::StateSignal, 10, num_rtvs, num_rtvs > 0 ? rtvs[0].ptr : 0,
                   dsv.ptr);
        m_command_list.Get()->OMSetRenderTargets(num_rtvs, num_rtvs > 0 ? rtvs : nullptr,
                                                 FALSE, dsv.ptr != 0 ? &dsv : nullptr);
    }

    // TEMP DIAGNOSTIC (session 13): coverage detector. Clear the composition target to
    // magenta right before the draw; if the probe/present still shows magenta afterwards
    // the draw covered no pixels, otherwise it wrote color. Revert with the investigation.
    if (flinger_draw) {
        static constexpr bool kMagentaCoverageDetector = false;
        static u32 magenta_logs = 0;
        if (kMagentaCoverageDetector) {
            const f32 magenta[4] = {1.0f, 0.0f, 1.0f, 1.0f};
            m_command_list.ClearRenderTargetView(rtvs[0], magenta);
            if (magenta_logs < 4) {
                ++magenta_logs;
                LOG_WARNING(Render_D3D12, "FLINGER magenta clear (coverage detector) rt0={:#x}",
                            rtvs[0].ptr);
            }
        }
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
        Diag::Push(Diag::CmdKind::StateSignal, 11, vertex_count,
                   reinterpret_cast<u64>(m_runtime.GetVertexBindings().data()));
        m_command_list.Get()->IASetVertexBuffers(
            0, vertex_count, m_runtime.GetVertexBindings().data());
    }
    if (params.is_indexed && index_binding.valid) {
        D3D12_INDEX_BUFFER_VIEW ib_view{};
        ib_view.BufferLocation = index_binding.address;
        ib_view.SizeInBytes = index_binding.size;
        ib_view.Format = index_binding.format;
        Diag::Push(Diag::CmdKind::StateSignal, 12, index_binding.address, index_binding.size,
                   static_cast<u64>(index_binding.format));
        m_command_list.Get()->IASetIndexBuffer(&ib_view);
    }

    m_command_list.SetViewport(MakeViewport(regs));
    m_command_list.SetScissorRect(MakeScissor(regs));

    m_query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                regs.zpass_pixel_count_enable != 0);
    {
        static u32 executing_logs = 0;
        static u32 fd_executing_logs = 0;
        const bool log_executing =
            executing_logs < 4 || (flinger_draw && (fd_executing_logs % 256) == 0);
        ++executing_logs;
        if (flinger_draw) {
            ++fd_executing_logs;
        }
        if (log_executing) {
            LOG_WARNING(Render_D3D12,
                        "DBG-B: fd={} indexed={} verts={} first_idx={} base_vtx={} ib_valid={}",
                        flinger_draw ? 1 : 0, params.is_indexed ? 1 : 0, params.num_vertices,
                        params.first_index, params.base_vertex, index_binding.valid ? 1 : 0);
        }
    }
    if (params.is_indexed && index_binding.valid) {
        m_command_list.DrawIndexed(params.num_vertices, params.num_instances, params.first_index,
                                   static_cast<s32>(params.base_vertex), params.base_instance);
    } else {
        m_command_list.Draw(params.num_vertices, params.num_instances, params.base_vertex,
                            params.base_instance);
    }

    {
        // Diagnostic (session 12): remember the first texture this draw samples, and log
        // the sampled-slot summary for composition draws.
        u32 sampled_slots = 0;
        for (const DrawSlot& slot : res_slots) {
            if (slot.kind != SlotKind::Sampled) {
                continue;
            }
            ++sampled_slots;
            ImageView& sampled_view = m_texture_cache.GetImageView(slot.view);
            if (sampled_view.Resource() != nullptr) {
                g_probe_any_sampled = sampled_view.Resource();
                if (flinger_draw && !g_probe_sampled) {
                    g_probe_sampled = sampled_view.Resource();
                }
            }
        }
        static u32 slot_logs = 0;
        if (slot_logs < 8) {
            ++slot_logs;
            LOG_WARNING(Render_D3D12, "Draw slots: total={} sampled={} flinger={}",
                        res_slots.size(), sampled_slots, flinger_draw ? 1 : 0);
        }
        // TEMP DIAGNOSTIC (session 13): which textures each composition layer samples, so a
        // missing/wrong layer (only one composition layer shows) is attributable.
        // TEMP DIAGNOSTIC (session 13): does any draw sample the F2D-blitted animation
        // texture (src 0x516170000 / dst 0x51b700000)? A missing sample means the splash
        // layer is never composited.
        {
            static u32 anim_logs = 0;
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind != SlotKind::Sampled) {
                    continue;
                }
                ImageView& sv = m_texture_cache.GetImageView(slot.view);
                const u64 gpu = static_cast<u64>(sv.GpuAddr());
                if ((gpu == 0x516170000ULL || gpu == 0x51b700000ULL) && anim_logs < 16) {
                    ++anim_logs;
                    const u64 rt_gpu = (framebuffer && framebuffer->ColorBuffers()[0])
                                           ? static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr())
                                           : 0;
                    LOG_WARNING(Render_D3D12,
                                "ANIM sample: rt={:#x} fd={} img={} gpu={:#x} {}x{} fmt={:#x} "
                                "blend={}",
                                rt_gpu, flinger_draw ? 1 : 0, sv.image_id.index, gpu,
                                sv.size.width, sv.size.height, static_cast<u32>(sv.format),
                                stored.pipeline->State().attachments[0].enable ? 1 : 0);
                }
            }
        }
        // TEMP DIAGNOSTIC (session 15): who samples the 960x540 scene RT and the 1080p
        // scene source, to locate where the visible 3D scene is lost.
        {
            static u32 s15_rt_logs = 0;
            static std::chrono::steady_clock::time_point s15_rt_start =
                std::chrono::steady_clock::now();
            static bool s15_src_dumped = false;
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind != SlotKind::Sampled) {
                    continue;
                }
                ImageView& sv = m_texture_cache.GetImageView(slot.view);
                const u64 gpu = static_cast<u64>(sv.GpuAddr());
                if (!((sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                       (sv.size.width == 960 || sv.size.width >= 1920)))) {
                    continue;
                }
                if (s15_rt_logs < 64) {
                    ++s15_rt_logs;
                    const u64 rt_gpu = (framebuffer && framebuffer->ColorBuffers()[0])
                                           ? static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr())
                                           : 0;
                    LOG_WARNING(Render_D3D12,
                                "S15 RT sample: rt={:#x} src_img={} src_gpu={:#x} {}x{} fmt={:#x} "
                                "depth={}",
                                rt_gpu, sv.image_id.index, gpu, sv.size.width, sv.size.height,
                                static_cast<u32>(sv.format), depth_view != nullptr ? 1 : 0);
                }
                // TEMP DIAGNOSTIC (session 15): the 1080p scene source is finally sampled only
                // once the game is fully loaded; force a dump and a mid-frame capture of it
                // after 20 minutes, when the under-lit frame is actually on screen.
                const bool s15_src_late =
                    std::chrono::steady_clock::now() - s15_rt_start > std::chrono::minutes(20);
                if (s15_src_late && !s15_src_dumped &&
                    sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                    sv.size.width >= 1920 && sv.Resource() != nullptr) {
                    s15_src_dumped = true;
                    (void)D3D12::RecordTextureContentDump(
                        m_device.GetDevice(), m_command_list, sv.Resource(),
                        static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                        sv.image_id.index, true);
                    D3D12::RecordScene3dCaptureMidFrame(m_device.GetDevice(), m_command_list,
                                                        sv.Resource());
                }
            }
        }
        // TEMP DIAGNOSTIC (session 15): the final compositor draw samples the 1080p
        // B10G11R11 scene source; log it (any vertex count) and dump that source once.
        {
            ImageView* const s15_comp_rt =
                framebuffer != nullptr ? framebuffer->ColorBuffers()[0] : nullptr;
            ID3D12Resource* const s15_comp_res =
                s15_comp_rt != nullptr ? s15_comp_rt->Resource() : nullptr;
            bool s15_has_scene_source = false;
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind != SlotKind::Sampled) {
                    continue;
                }
                ImageView& sv = m_texture_cache.GetImageView(slot.view);
                if (sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                    sv.size.width >= 1920) {
                    s15_has_scene_source = true;
                }
            }
            const bool s15_final_rt =
                s15_comp_res != nullptr && s15_comp_res->GetDesc().Width >= 1920 &&
                s15_comp_res->GetDesc().Format == DXGI_FORMAT_R10G10B10A2_TYPELESS;
            if (s15_comp_res != nullptr && (s15_final_rt || s15_has_scene_source)) {
                static u32 s15_comp_logs = 0;
                // TEMP DIAGNOSTIC (session 15): the early dumps below run in the first
                // frames, when the compositor may still sample the pre-style scene source;
                // the forced dump 20 minutes in re-reads whatever B10G11R11 source it
                // samples then, ignoring the per-image dedupe in RecordTextureContentDump.
                static std::chrono::steady_clock::time_point s15_comp_start =
                    std::chrono::steady_clock::now();
                static bool s15_comp_late_dumped = false;
                const u32 s15_comp_n = s15_comp_logs++;
                // Recording must not depend on the log budget below: the compositor's
                // rate limit (8 logs + every 4096th) can exhaust the budget long before
                // the scene source appears, so the 1080p B10G11R11 source is captured
                // unconditionally. The duplicate call inside the logging loop is harmless
                // because RecordTextureContentDump dedupes per image.
                u32 s15_comp_dumps = 0;
                for (const DrawSlot& slot : res_slots) {
                    if (slot.kind != SlotKind::Sampled || s15_comp_dumps >= 2) {
                        continue;
                    }
                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                    if (sv.Resource() != nullptr &&
                        sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                        sv.size.width >= 1920) {
                        const bool s15_comp_force =
                            !s15_comp_late_dumped &&
                            std::chrono::steady_clock::now() - s15_comp_start >
                                std::chrono::minutes(20);
                        if (s15_comp_force) {
                            s15_comp_late_dumped = true;
                        }
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            sv.image_id.index, s15_comp_force);
                    }
                }
                // TEMP DIAGNOSTIC (session 15): capture the 1080p scene source the compositor
                // samples itself, so its pixels can be inspected as a BMP independently of
                // the texture-content readback path above.
                static bool s15_comp_captured = false;
                if (!s15_comp_captured && s15_has_scene_source) {
                    for (const DrawSlot& slot : res_slots) {
                        if (slot.kind != SlotKind::Sampled) {
                            continue;
                        }
                        ImageView& sv = m_texture_cache.GetImageView(slot.view);
                        if (sv.Resource() != nullptr &&
                            sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                            sv.size.width >= 1920) {
                            s15_comp_captured = true;
                            D3D12::RecordScene3dCaptureMidFrame(m_device.GetDevice(),
                                                                m_command_list, sv.Resource());
                            break;
                        }
                    }
                }
                if (s15_comp_n < 8 || (s15_comp_n % 4096 == 0 && s15_comp_n < 32768)) {
                    std::string s15_texs;
                    for (const DrawSlot& slot : res_slots) {
                        if (slot.kind != SlotKind::Sampled) {
                            continue;
                        }
                        ImageView& sv = m_texture_cache.GetImageView(slot.view);
                        s15_texs += fmt::format("[img={} gpu={:#x} {}x{} fmt={:#x} res={}]",
                                                sv.image_id.index,
                                                static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                                sv.size.height, static_cast<u32>(sv.format),
                                                sv.Resource() ? 1 : 0);
                        if (sv.Resource() != nullptr &&
                            sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                            sv.size.width >= 1920) {
                            (void)D3D12::RecordTextureContentDump(
                                m_device.GetDevice(), m_command_list, sv.Resource(),
                                static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                                sv.image_id.index);
                        }
                    }
                    LOG_WARNING(Render_D3D12,
                                "S15 compositor draw: rt_gpu={:#x} rt_fmt={:#x} rt={}x{} verts={} "
                                "blend={} textures={}",
                                static_cast<u64>(s15_comp_rt->GpuAddr()),
                                static_cast<u32>(s15_comp_res->GetDesc().Format),
                                s15_comp_res->GetDesc().Width, s15_comp_res->GetDesc().Height,
                                params.num_vertices,
                                stored.pipeline->State().attachments[0].enable ? 1 : 0, s15_texs);
                }
            }
        }
        if (flinger_draw) {
            static u32 flinger_tex_logs = 0;
            const u32 flinger_tex_count = flinger_tex_logs++;
            if (flinger_tex_count < 10 ||
                (flinger_tex_count % 64 == 0 && flinger_tex_count < 400)) {
                std::string texs;
                for (const DrawSlot& slot : res_slots) {
                    if (slot.kind != SlotKind::Sampled) {
                        continue;
                    }
                    ImageView& sampled_view = m_texture_cache.GetImageView(slot.view);
                    texs += fmt::format("[img={} gpu={:#x} {}x{} fmt={:#x} res={} samp={}]",
                                        sampled_view.image_id.index,
                                        static_cast<u64>(sampled_view.GpuAddr()),
                                        sampled_view.size.width, sampled_view.size.height,
                                        static_cast<u32>(sampled_view.format),
                                        sampled_view.Resource() ? 1 : 0, slot.sampler.index);
                }
                const Shader::Info* ps_info = stored.infos[4];
                LOG_WARNING(Render_D3D12,
                            "FLINGER textures: {} (ps_tex={} ps_img={} ps_mask={:#x} blend={})",
                            texs, ps_info ? Shader::NumDescriptors(ps_info->texture_descriptors) : 0,
                            ps_info ? Shader::NumDescriptors(ps_info->image_descriptors) : 0,
                            ps_info ? ps_info->constant_buffer_mask : 0,
                            stored.pipeline->State().attachments[0].enable ? 1 : 0);
            }
        }
        // TEMP DIAGNOSTIC (session 13): the depth-tested 3D draws (the scene behind the UI)
        // with their render target format and the textures they sample, so a black scene is
        // attributable to the RT format/contents or to missing/wrong sampled textures.
        if (depth_view != nullptr && depth_view->RenderTarget().ptr) {
            static u32 scene3d_draw_logs = 0;
            const u32 scene3d_index = scene3d_draw_logs++;
            // TEMP DIAGNOSTIC (session 15): dump the sampled textures and constant buffers of the
            // first scene draw and of one late (style-phase) scene draw.
            static std::chrono::steady_clock::time_point s15_start =
                std::chrono::steady_clock::now();
            static bool s15_early_tex_dump_done = false;
            static bool s15_late_dump_done = false;
            // The slot counts must be computed before the dump decision: the late dump has to
            // latch onto a draw that actually samples several textures (the visible scene), not
            // onto the first depth-tested draw that happens after the 20-minute mark.
            u32 s15_sampled = 0;
            u32 s15_texel = 0;
            u32 s15_image_buffer = 0;
            u32 s15_storage = 0;
            u32 s15_image = 0;
            for (const DrawSlot& slot : res_slots) {
                switch (slot.kind) {
                case SlotKind::Sampled:
                    ++s15_sampled;
                    break;
                case SlotKind::TexelBuffer:
                    ++s15_texel;
                    break;
                case SlotKind::ImageBuffer:
                    ++s15_image_buffer;
                    break;
                case SlotKind::Storage:
                    ++s15_storage;
                    break;
                case SlotKind::Image:
                    ++s15_image;
                    break;
                }
            }
            const bool s15_late_phase =
                std::chrono::steady_clock::now() - s15_start > std::chrono::minutes(20);
            const bool s15_dump_now =
                scene3d_index == 0 || (s15_sampled >= 5 && !s15_early_tex_dump_done) ||
                (s15_late_phase && !s15_late_dump_done && s15_sampled >= 5);
            if (s15_dump_now) {
                if (scene3d_index != 0 && s15_sampled >= 5) {
                    s15_early_tex_dump_done = true;
                }
                if (scene3d_index != 0 && s15_late_phase) {
                    s15_late_dump_done = true;
                }
                LOG_WARNING(Render_D3D12,
                            "S15 scene slots: total={} sampled={} texbuf={} imgbuf={} storage={} "
                            "image={} late={}",
                            res_slots.size(), s15_sampled, s15_texel, s15_image_buffer,
                            s15_storage, s15_image, s15_late_phase ? 1 : 0);
                for (u32 s15_rt = 0; s15_rt < num_rtvs && s15_rt < 4; ++s15_rt) {
                    const auto& s15_att = stored.pipeline->State().attachments[s15_rt];
                    LOG_WARNING(Render_D3D12,
                                "S15 scene om rt{}: enable={} src={:#x} dst={:#x} eq={:#x} "
                                "srcA={:#x} dstA={:#x} eqA={:#x} mask={:x} raw={:#x}",
                                s15_rt, s15_att.enable ? 1 : 0,
                                static_cast<u32>(s15_att.SourceRGBFactor()),
                                static_cast<u32>(s15_att.DestRGBFactor()),
                                static_cast<u32>(s15_att.EquationRGB()),
                                static_cast<u32>(s15_att.SourceAlphaFactor()),
                                static_cast<u32>(s15_att.DestAlphaFactor()),
                                static_cast<u32>(s15_att.EquationAlpha()),
                                static_cast<u32>(s15_att.mask_r ? 1 : 0) |
                                    static_cast<u32>(s15_att.mask_g ? 2 : 0) |
                                    static_cast<u32>(s15_att.mask_b ? 4 : 0) |
                                    static_cast<u32>(s15_att.mask_a ? 8 : 0),
                                s15_att.raw);
                }
                // TEMP DIAGNOSTIC (session 15): the vertex attributes and the raw vertex bytes
                // of the dumped scene draw, so a wrong attribute stride/format/offset or a
                // zeroed vertex buffer is visible from the data (mirrors the FLINGER dump).
                {
                    const auto& s15_pipe = stored.pipeline->State();
                    std::string s15_attrs;
                    std::array<bool, 32> s15_slot_seen{};
                    std::array<u32, 2> s15_dump_slots{};
                    u32 s15_dump_count = 0;
                    for (u32 i = 0; i < s15_pipe.attributes.size(); ++i) {
                        if (s15_pipe.attributes[i].enabled == 0) {
                            continue;
                        }
                        const auto& attr = s15_pipe.attributes[i];
                        const u32 slot = attr.buffer.Value();
                        s15_attrs += fmt::format("[a{}:s{} off={} type={} size={} div={}]", i, slot,
                                                 attr.offset.Value(),
                                                 static_cast<u32>(attr.Type()),
                                                 static_cast<u32>(attr.Size()),
                                                 s15_pipe.binding_divisors[slot]);
                        if (slot < s15_slot_seen.size() && !s15_slot_seen[slot] &&
                            s15_dump_count < s15_dump_slots.size()) {
                            s15_slot_seen[slot] = true;
                            s15_dump_slots[s15_dump_count++] = slot;
                        }
                    }
                    LOG_WARNING(Render_D3D12, "S15 scene attrs: {} vb_slots={} instances={}",
                                s15_attrs, m_runtime.MaxVertexSlot(), params.num_instances);
                    for (u32 d = 0; d < s15_dump_count; ++d) {
                        const u32 slot = s15_dump_slots[d];
                        const auto& vb = m_runtime.GetVertexBindings()[slot];
                        const DAddr vb_addr = m_buffer_cache.GetVertexBufferDeviceAddress(slot);
                        std::array<u8, 32> vb_bytes{};
                        const u32 read_size =
                            std::min<u32>(static_cast<u32>(vb_bytes.size()), vb.SizeInBytes);
                        std::string hex;
                        std::string floats;
                        if (vb_addr != 0 && read_size >= 8) {
                            const auto read = m_device_memory.ReadBlockUnsafe(
                                vb_addr, vb_bytes.data(), read_size, "S15.SceneVB", false);
                            if (read.fully_mapped) {
                                for (u32 i = 0; i < read_size; ++i) {
                                    hex += fmt::format("{:02x}", vb_bytes[i]);
                                }
                                for (u32 i = 0; i + 3 < read_size; i += 4) {
                                    f32 v = 0.0f;
                                    std::memcpy(&v, vb_bytes.data() + i, sizeof(v));
                                    floats += fmt::format("{} ", v);
                                }
                            } else {
                                floats = "unmapped";
                            }
                        }
                        LOG_WARNING(Render_D3D12,
                                    "S15 scene vb s{}: dev={:#x} size={} stride={} hex={} "
                                    "f32=[{}]",
                                    slot, static_cast<u64>(vb_addr), vb.SizeInBytes,
                                    vb.StrideInBytes, hex, floats);
                    }
                }
                u32 s15_sampled_seen = 0;
                for (const DrawSlot& slot : res_slots) {
                    if (slot.kind != SlotKind::Sampled || s15_sampled_seen >= 8) {
                        continue;
                    }
                    ++s15_sampled_seen;
                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                    if (sv.Resource() != nullptr) {
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            sv.image_id.index);
                    }
                    const auto dev = gpu_memory->GpuToCpuAddress(static_cast<GPUVAddr>(sv.GpuAddr()));
                    std::array<u8, 64> bytes{};
                    u32 bytes_read = 0;
                    u32 nonzero_byte_count = 0;
                    std::string hex;
                    if (dev) {
                        const auto read = m_device_memory.ReadBlockUnsafe(
                            *dev, bytes.data(), bytes.size(), "S15.TexProbe", false);
                        if (read.fully_mapped) {
                            bytes_read = static_cast<u32>(bytes.size());
                            for (u32 i = 0; i < bytes_read; ++i) {
                                if (bytes[i] != 0) {
                                    ++nonzero_byte_count;
                                }
                                if (i < 32) {
                                    hex += fmt::format("{:02x}", bytes[i]);
                                }
                            }
                        }
                    }
                    const u32 s15_dxgi =
                        sv.Resource() ? static_cast<u32>(sv.Resource()->GetDesc().Format) : 0;
                    LOG_WARNING(Render_D3D12,
                                "S15 texprobe img={} gpu={:#x} fmt={:#x} dxgi={:#x} dev={:#x} "
                                "nz={}/{} hex={}",
                                sv.image_id.index, static_cast<u64>(sv.GpuAddr()),
                                static_cast<u32>(sv.format), s15_dxgi,
                                dev ? static_cast<u64>(*dev) : 0, nonzero_byte_count, bytes_read,
                                hex);
                }
                for (size_t stage = 0; stage < VideoCommon::NUM_STAGES; ++stage) {
                    const Shader::Info* s15_info = stored.infos[stage];
                    if (s15_info == nullptr) {
                        continue;
                    }
                    const auto& s15_cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
                    for (u32 ci = 0; ci < Shader::Info::MAX_CBUFS; ++ci) {
                        if (((s15_info->constant_buffer_mask >> ci) & 1) == 0) {
                            continue;
                        }
                        const auto& cb = s15_cbufs[ci];
                        const auto dev = gpu_memory->GpuToCpuAddress(cb.address);
                        std::array<u8, 64> bytes{};
                        u32 nonzero_byte_count = 0;
                        u32 bytes_read = 0;
                        std::string flts = "unmapped";
                        if (dev) {
                            const auto read = m_device_memory.ReadBlockUnsafe(
                                *dev, bytes.data(), bytes.size(), "S15.SceneCBV", false);
                            if (read.fully_mapped) {
                                bytes_read = static_cast<u32>(bytes.size());
                                for (u32 i = 0; i < bytes_read; ++i) {
                                    if (bytes[i] != 0) {
                                        ++nonzero_byte_count;
                                    }
                                }
                                flts.clear();
                                for (u32 i = 0; i + 3 < bytes_read; i += 4) {
                                    f32 v = 0.0f;
                                    std::memcpy(&v, bytes.data() + i, sizeof(v));
                                    flts += fmt::format("{} ", v);
                                }
                            }
                        }
                        LOG_WARNING(Render_D3D12,
                                    "S15 scene cbv s{} i{}: gpu={:#x} dev={:#x} size={} used={} "
                                    "enabled={} nz={} f32=[{}]",
                                    stage, ci, static_cast<u64>(cb.address),
                                    dev ? static_cast<u64>(*dev) : 0, cb.size,
                                    s15_info->constant_buffer_used_sizes[ci],
                                    cb.enabled ? 1 : 0, nonzero_byte_count, flts);
                    }
                }
            }
            // TEMP DIAGNOSTIC (session 15): the scene's lighting LUTs are small B10G11R11 /
            // R16G16_FLOAT textures; dump the first draw that samples them and probe their
            // guest memory, independent of the larger draw-latched dump above.
            {
                static bool s15_lut_dumped = false;
                u32 s15_lut_slots = 0;
                for (const DrawSlot& slot : res_slots) {
                    if (s15_lut_dumped || slot.kind != SlotKind::Sampled || s15_lut_slots >= 4) {
                        continue;
                    }
                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                    const bool is_lut =
                        (sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                         sv.size.width <= 512) ||
                        sv.format == VideoCore::Surface::PixelFormat::R16G16_FLOAT;
                    if (!is_lut) {
                        continue;
                    }
                    ++s15_lut_slots;
                    if (sv.Resource() != nullptr) {
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            sv.image_id.index, true);
                    }
                    const auto dev = gpu_memory->GpuToCpuAddress(static_cast<GPUVAddr>(sv.GpuAddr()));
                    std::array<u8, 64> lut_bytes{};
                    u32 lut_nonzero = 0;
                    u32 lut_read = 0;
                    std::string lut_hex;
                    if (dev) {
                        const auto read = m_device_memory.ReadBlockUnsafe(
                            *dev, lut_bytes.data(), lut_bytes.size(), "S15.LutProbe", false);
                        if (read.fully_mapped) {
                            lut_read = static_cast<u32>(lut_bytes.size());
                            for (u32 i = 0; i < lut_read; ++i) {
                                if (lut_bytes[i] != 0) {
                                    ++lut_nonzero;
                                }
                                if (i < 32) {
                                    lut_hex += fmt::format("{:02x}", lut_bytes[i]);
                                }
                            }
                        }
                    }
                    LOG_WARNING(Render_D3D12,
                                "S15 LUT probe: img={} gpu={:#x} {}x{} fmt={:#x} dev={:#x} "
                                "nz={}/{} hex={}",
                                sv.image_id.index, static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                sv.size.height, static_cast<u32>(sv.format),
                                dev ? static_cast<u64>(*dev) : 0, lut_nonzero, lut_read, lut_hex);
                }
                if (!s15_lut_dumped && s15_lut_slots > 0) {
                    s15_lut_dumped = true;
                }
            }
            if (scene3d_index < 8 ||
                (scene3d_index % 256 == 0 && scene3d_index < 8192)) {
                std::string texs;
                for (const DrawSlot& slot : res_slots) {
                    if (slot.kind != SlotKind::Sampled) {
                        continue;
                    }
                    ImageView& sampled_view = m_texture_cache.GetImageView(slot.view);
                    texs += fmt::format("[img={} gpu={:#x} {}x{} fmt={:#x} res={}]",
                                        sampled_view.image_id.index,
                                        static_cast<u64>(sampled_view.GpuAddr()),
                                        sampled_view.size.width, sampled_view.size.height,
                                        static_cast<u32>(sampled_view.format),
                                        sampled_view.Resource() ? 1 : 0);
                }
                const ImageView* scene3d_rt = nullptr;
                for (size_t i = 0; i < num_rtvs; ++i) {
                    const ImageView* cand = framebuffer ? framebuffer->ColorBuffers()[i] : nullptr;
                    if (cand != nullptr && cand->Resource() != nullptr) {
                        scene3d_rt = cand;
                        break;
                    }
                }
                const u64 rt_gpu = scene3d_rt ? static_cast<u64>(scene3d_rt->GpuAddr()) : 0;
                const D3D12_VIEWPORT scene3d_vp = MakeViewport(maxwell3d->regs);
                ID3D12Resource* const dsv_res = depth_view->Resource();
                LOG_WARNING(Render_D3D12,
                            "D3D draw: rt_gpu={:#x} fmt={:#x} {}x{} dsv_fmt={:#x} cull_en={} "
                            "cull_face={} ff={} verts={} indexed={} textures={} "
                            "vp=({},{},{},{})",
                            rt_gpu,
                            scene3d_rt
                                ? static_cast<u32>(scene3d_rt->Resource()->GetDesc().Format)
                                : 0,
                            scene3d_rt ? scene3d_rt->Resource()->GetDesc().Width : 0,
                            scene3d_rt ? scene3d_rt->Resource()->GetDesc().Height : 0,
                            dsv_res ? static_cast<u32>(dsv_res->GetDesc().Format) : 0,
                            stored.pipeline->State().cull_enable.Value(),
                            static_cast<u32>(stored.pipeline->State().CullFaceMode()),
                            static_cast<u32>(stored.pipeline->State().FrontFaceMode()),
                            params.num_vertices, params.is_indexed ? 1 : 0, texs,
                            scene3d_vp.TopLeftX, scene3d_vp.TopLeftY, scene3d_vp.Width,
                            scene3d_vp.Height);
            }
        }
        if (flinger_draw) {
            // Diagnostic (session 12): attribute descriptors + raw vertex bytes of the
            // composition quad, so a wrong slot/offset/format is visible from the data.
            static u32 vb_dump_logs = 0;
            if (vb_dump_logs < 3) {
                ++vb_dump_logs;
                const auto& pipe_state = stored.pipeline->State();
                std::string attrs;
                std::array<bool, 32> slot_seen{};
                std::array<u32, 4> dump_slots{};
                u32 dump_count = 0;
                for (u32 i = 0; i < pipe_state.attributes.size(); ++i) {
                    if (pipe_state.attributes[i].enabled == 0) {
                        continue;
                    }
                    const u32 slot = pipe_state.attributes[i].buffer.Value();
                    const auto& attr = pipe_state.attributes[i];
                    attrs += fmt::format("[a{}:s{} off={} type={} size={} div={}]", i, slot,
                                         attr.offset.Value(), static_cast<u32>(attr.Type()),
                                         static_cast<u32>(attr.Size()),
                                         pipe_state.binding_divisors[slot]);
                    if (slot < slot_seen.size() && !slot_seen[slot] &&
                        dump_count < dump_slots.size()) {
                        slot_seen[slot] = true;
                        dump_slots[dump_count++] = slot;
                    }
                }
                LOG_WARNING(Render_D3D12, "FLINGER attrs: {} vb_slots={} instances={}", attrs,
                            m_runtime.MaxVertexSlot(), params.num_instances);
                // TEMP DIAGNOSTIC (session 13): the composition quad's index buffer, so the
                // actual vertex order / winding is visible from the data.
                if (index_binding.device_addr != 0 && index_binding.size != 0 &&
                    index_binding.size <= 64) {
                    std::array<u8, 64> ib_bytes{};
                    const auto ib_read = m_device_memory.ReadBlockUnsafe(
                        index_binding.device_addr, ib_bytes.data(), index_binding.size,
                        "FLINGER.IBDump", false);
                    if (ib_read.fully_mapped) {
                        std::string ib_hex;
                        std::string ib_vals;
                        for (u32 i = 0; i < index_binding.size; ++i) {
                            ib_hex += fmt::format("{:02x}", ib_bytes[i]);
                        }
                        if (index_binding.format == DXGI_FORMAT_R16_UINT) {
                            for (u32 i = 0; i + 1 < index_binding.size; i += 2) {
                                u16 v = 0;
                                std::memcpy(&v, ib_bytes.data() + i, sizeof(v));
                                ib_vals += fmt::format("{} ", v);
                            }
                        } else if (index_binding.format == DXGI_FORMAT_R32_UINT) {
                            for (u32 i = 0; i + 3 < index_binding.size; i += 4) {
                                u32 v = 0;
                                std::memcpy(&v, ib_bytes.data() + i, sizeof(v));
                                ib_vals += fmt::format("{} ", v);
                            }
                        }
                        LOG_WARNING(Render_D3D12,
                                    "FLINGER ib: dev={:#x} size={} fmt={} hex={} indices=[{}]",
                                    static_cast<u64>(index_binding.device_addr),
                                    index_binding.size, static_cast<u32>(index_binding.format),
                                    ib_hex, ib_vals);
                    }
                }
                const auto half_to_float = [](u16 h) -> f32 {
                    const u32 sign = (h >> 15) & 1U;
                    const u32 exp = (h >> 10) & 0x1FU;
                    const u32 man = h & 0x3FFU;
                    u32 bits = 0;
                    if (exp == 0) {
                        bits = sign << 31;
                    } else if (exp == 0x1F) {
                        bits = (sign << 31) | 0x7F800000U | (man << 13);
                    } else {
                        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
                    }
                    f32 out = 0.0f;
                    std::memcpy(&out, &bits, sizeof(out));
                    return out;
                };
                for (u32 d = 0; d < dump_count; ++d) {
                    const u32 slot = dump_slots[d];
                    const auto& vb = m_runtime.GetVertexBindings()[slot];
                    const DAddr vb_addr = m_buffer_cache.GetVertexBufferDeviceAddress(slot);
                    std::array<u8, 32> vb_bytes{};
                    const u32 read_size = std::min<u32>(static_cast<u32>(vb_bytes.size()),
                                                        vb.SizeInBytes);
                    std::string hex;
                    std::string halfs;
                    if (vb_addr != 0 && read_size >= 8) {
                        const auto read = m_device_memory.ReadBlockUnsafe(
                            vb_addr, vb_bytes.data(), read_size, "FLINGER.VBDump", false);
                        if (read.fully_mapped) {
                            for (u32 i = 0; i < read_size; ++i) {
                                hex += fmt::format("{:02x}", vb_bytes[i]);
                            }
                            for (u32 i = 0; i + 1 < read_size; i += 2) {
                                u16 h = 0;
                                std::memcpy(&h, vb_bytes.data() + i, sizeof(h));
                                halfs += fmt::format("{} ", half_to_float(h));
                            }
                        } else {
                            halfs = "unmapped";
                        }
                    }
                    LOG_WARNING(Render_D3D12,
                                "FLINGER vb s{}: dev={:#x} size={} stride={} hex={} halves=[{}]",
                                slot, static_cast<u64>(vb_addr), vb.SizeInBytes, vb.StrideInBytes,
                                hex, halfs);
                    // TEMP DIAGNOSTIC (session 13): read back the host (uploaded) vertex
                    // buffer, to prove whether the IA actually fetches the guest bytes.
                    ID3D12Resource* const vb_res = m_runtime.GetVertexBindingResource(slot);
                    if (vb_res != nullptr && vb.BufferLocation != 0) {
                        D3D12_HEAP_PROPERTIES heap_props{};
                        heap_props.Type = D3D12_HEAP_TYPE_READBACK;
                        D3D12_RESOURCE_DESC res_desc{};
                        res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                        res_desc.Width = 64;
                        res_desc.Height = 1;
                        res_desc.DepthOrArraySize = 1;
                        res_desc.MipLevels = 1;
                        res_desc.SampleDesc.Count = 1;
                        res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                        ComPtr<ID3D12Resource> readback;
                        if (SUCCEEDED(m_device.GetDevice()->CreateCommittedResource(
                                &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc,
                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                IID_PPV_ARGS(&readback)))) {
                            const u64 rel =
                                static_cast<u64>(vb.BufferLocation) -
                                static_cast<u64>(vb_res->GetGPUVirtualAddress());
                            const u32 copy_bytes = std::min<u32>(64, vb.SizeInBytes);
                            m_command_list.Transition(vb_res, D3D12_RESOURCE_STATE_COMMON,
                                                      D3D12_RESOURCE_STATE_COPY_SOURCE);
                            m_command_list.Get()->CopyBufferRegion(readback.Get(), 0, vb_res, rel,
                                                                   copy_bytes);
                            m_command_list.Transition(vb_res, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                      D3D12_RESOURCE_STATE_COMMON);
                            m_command_list.Execute(m_device);
                            m_device.WaitForIdle();
                            m_command_list.Reset();
                            m_res_heap.Reset();
                            m_sampler_heap.Reset();
                            m_cbv_scratch_used = 0;
                            void* mapped = nullptr;
                            if (SUCCEEDED(readback->Map(0, nullptr, &mapped))) {
                                std::string host_hex;
                                const u8* const host_bytes = static_cast<const u8*>(mapped);
                                for (u32 i = 0; i < copy_bytes; ++i) {
                                    host_hex += fmt::format("{:02x}", host_bytes[i]);
                                }
                                LOG_WARNING(Render_D3D12,
                                            "FLINGER host vb s{}: rel={:#x} bytes={} hex={}", slot,
                                            rel, copy_bytes, host_hex);
                                readback->Unmap(0, nullptr);
                            }
                        }
                    }
                }
            }
        }
    }

    // Restore render targets to the home state for later copies/blits.
    for (u32 i = 0; i < rt_resource_count; ++i) {
        m_command_list.Transition(rt_resources[i], rt_states[i], D3D12_RESOURCE_STATE_COMMON);
    }

    // Temporary diagnostic: dump the bindings of the first executing draws and flush after
    // each so the async device removal can be attributed to a single draw.
    const u32 diag_index = g_diag_draws.fetch_add(1, std::memory_order_relaxed);
    if (diag_index < 4) {
        LOG_ERROR(Render_D3D12,
                  "DIAG draw {}: indexed={} verts={} inst={} first_idx={} base_vertex={} "
                  "num_rtvs={} max_vb_slot={} zpass={}",
                  diag_index, params.is_indexed, params.num_vertices, params.num_instances,
                  params.first_index, params.base_vertex, num_rtvs, m_runtime.MaxVertexSlot(),
                  regs.zpass_pixel_count_enable);
        for (u32 i = 0; i < num_rtvs; ++i) {
            ImageView* view = framebuffer ? framebuffer->ColorBuffers()[i] : nullptr;
            const u64 gpu_addr = view ? static_cast<u64>(view->GpuAddr()) : 0;
            const u32 res_fmt =
                (view && view->Resource())
                    ? static_cast<u32>(view->Resource()->GetDesc().Format)
                    : 0;
            const u32 pso_fmt = static_cast<u32>(SurfaceFormat(
                VideoCore::Surface::PixelFormatFromRenderTargetFormat(
                    static_cast<Tegra::RenderTargetFormat>(
                        stored.pipeline->State().color_formats[i]))));
            LOG_ERROR(Render_D3D12,
                      "DIAG draw {} rt{}: pso_fmt={:#x} view_res_fmt={:#x} gpu_addr={:#x}",
                      diag_index, i, pso_fmt, res_fmt, gpu_addr);
            if (view && view->Resource()) {
                const auto diag_desc = view->Resource()->GetDesc();
                LOG_ERROR(Render_D3D12,
                          "DIAG draw {} rt{} desc: dim={} {}x{} mips={} fmt={:#x} flags={:#x} "
                          "samples={}",
                          diag_index, i, static_cast<u32>(diag_desc.Dimension), diag_desc.Width,
                          diag_desc.Height, diag_desc.MipLevels,
                          static_cast<u32>(diag_desc.Format), static_cast<u32>(diag_desc.Flags),
                          diag_desc.SampleDesc.Count);
            }
        }
        if (depth_view) {
            LOG_ERROR(Render_D3D12, "DIAG draw {} depth: gpu_addr={:#x} res_fmt={:#x}",
                      diag_index, static_cast<u64>(depth_view->GpuAddr()),
                      depth_view->Resource()
                          ? static_cast<u32>(depth_view->Resource()->GetDesc().Format)
                          : 0);
        }
        const auto& diag_vbs = m_runtime.GetVertexBindings();
        for (u32 s = 0; s < m_runtime.MaxVertexSlot(); ++s) {
            LOG_ERROR(Render_D3D12, "DIAG draw {} vb{}: loc={:#x} size={} stride={}", diag_index,
                      s, static_cast<u64>(diag_vbs[s].BufferLocation),
                      diag_vbs[s].SizeInBytes, diag_vbs[s].StrideInBytes);
        }
        if (params.is_indexed) {
            const auto& ib = m_runtime.GetIndexBinding();
            LOG_ERROR(Render_D3D12, "DIAG draw {} ib: loc={:#x} size={} fmt={} valid={}",
                      diag_index, static_cast<u64>(ib.address), ib.size,
                      static_cast<u32>(ib.format), ib.valid);
        }
        // Descriptor bindings with an alias check against this draw's RTV/depth addresses.
        u64 diag_rt_addrs[8]{};
        for (u32 i = 0; i < num_rtvs && i < 8; ++i) {
            ImageView* view = framebuffer ? framebuffer->ColorBuffers()[i] : nullptr;
            diag_rt_addrs[i] = view ? static_cast<u64>(view->GpuAddr()) : 0;
        }
        const u64 diag_depth_addr = depth_view ? static_cast<u64>(depth_view->GpuAddr()) : 0;
        const auto diag_aliases_rt = [&](u64 addr) {
            if (addr == 0) {
                return false;
            }
            for (u32 i = 0; i < num_rtvs && i < 8; ++i) {
                if (diag_rt_addrs[i] != 0 && addr == diag_rt_addrs[i]) {
                    return true;
                }
            }
            return diag_depth_addr != 0 && addr == diag_depth_addr;
        };
        const D3D12_VIEWPORT diag_vp = MakeViewport(regs);
        const D3D12_RECT diag_sc = MakeScissor(regs);
        LOG_ERROR(Render_D3D12,
                  "DIAG draw {} viewport: x={} y={} w={} h={} minz={} maxz={} scissor: "
                  "l={} t={} r={} b={} clip={}x{}",
                  diag_index, diag_vp.TopLeftX, diag_vp.TopLeftY, diag_vp.Width,
                  diag_vp.Height, diag_vp.MinDepth, diag_vp.MaxDepth, diag_sc.left, diag_sc.top,
                  diag_sc.right, diag_sc.bottom, regs.surface_clip.width,
                  regs.surface_clip.height);
        const auto& diag_bindings = m_runtime.GetResourceBindings();
        const RootSignature& diag_rs = stored.pipeline->GetRootSignature();
        LOG_ERROR(Render_D3D12,
                  "DIAG draw {} tables: cbv={} res={} heap_base={} sampler_base={} cbv_tbl={} "
                  "srv_tbl={} uav_tbl={} smp_tbl={}",
                  diag_index, cbv_records.size(), res_slots.size(), heap_base, sampler_base,
                  diag_rs.GetCbvTableIndex(), diag_rs.GetSrvTableIndex(),
                  diag_rs.GetUavTableIndex(), diag_rs.GetSamplerTableIndex());
        for (u32 i = 0; i < cbv_records.size(); ++i) {
            const auto& rec = diag_bindings[cbv_records[i]];
            LOG_ERROR(Render_D3D12,
                      "DIAG draw {} cbv{}: addr={:#x} size={} view={:#x} alias_rt={}", diag_index,
                      i, static_cast<u64>(rec.address), rec.size,
                      static_cast<u64>(rec.view.ptr), diag_aliases_rt(rec.address));
        }
        for (u32 i = 0; i < res_slots.size(); ++i) {
            const DrawSlot& slot = res_slots[i];
            switch (slot.kind) {
            case SlotKind::Storage:
            case SlotKind::TexelBuffer:
            case SlotKind::ImageBuffer: {
                if (slot.record < diag_bindings.size()) {
                    const auto& rec = diag_bindings[slot.record];
                    LOG_ERROR(Render_D3D12,
                              "DIAG draw {} res{} kind={} addr={:#x} size={} view={:#x} "
                              "alias_rt={}",
                              diag_index, i, static_cast<u32>(slot.kind),
                              static_cast<u64>(rec.address), rec.size,
                              static_cast<u64>(rec.view.ptr), diag_aliases_rt(rec.address));
                } else {
                    LOG_ERROR(Render_D3D12, "DIAG draw {} res{} kind={} unbound", diag_index, i,
                              static_cast<u32>(slot.kind));
                }
                break;
            }
            case SlotKind::Sampled: {
                const ImageView& img_view = m_texture_cache.GetImageView(slot.view);
                LOG_ERROR(Render_D3D12,
                          "DIAG draw {} res{} kind=Sampled addr={:#x} alias_rt={} default_sampler={}",
                          diag_index, i, static_cast<u64>(img_view.GpuAddr()),
                          diag_aliases_rt(static_cast<u64>(img_view.GpuAddr())),
                          slot.sampler == VideoCommon::NULL_SAMPLER_ID);
                break;
            }
            case SlotKind::Image: {
                LOG_ERROR(Render_D3D12, "DIAG draw {} res{} kind=Image placeholder", diag_index,
                          i);
                break;
            }
            }
        }
        m_command_list.Execute(m_device);
        m_device.WaitForIdle();
        m_command_list.Reset();
        m_res_heap.Reset();
        m_sampler_heap.Reset();
        m_cbv_scratch_used = 0;
        const HRESULT diag_reason = m_device.GetDevice()->GetDeviceRemovedReason();
        LOG_ERROR(Render_D3D12, "DIAG draw {}: post-flush device reason {:#x}", diag_index,
                  static_cast<u32>(diag_reason));
    }

    {
        // TEMP DIAGNOSTIC (session 14): record a scene3d readback copy right after a scene draw
        // (mid-frame), so the copy executes before any post-composite clear of the scene RT.
        // Alternate between the scene RT and the compositor's 1080p target.
        static u32 scene3d_midframe_tick = 0;
        static u32 scene3d_midframe_alt = 0;
        ID3D12Resource* const midframe_scene_rt =
            framebuffer != nullptr && num_rtvs > 0 && framebuffer->ColorBuffers()[0] != nullptr
                ? framebuffer->ColorBuffers()[0]->Resource()
                : nullptr;
        if (midframe_scene_rt != nullptr && midframe_scene_rt == g_probe_scene3d.Get()) {
            if ((scene3d_midframe_tick++ % 65536) == 0) {
                ID3D12Resource* const target =
                    (scene3d_midframe_alt++ % 2) == 0 ? midframe_scene_rt : g_probe_ui.Get();
                if (target != nullptr) {
                    RecordScene3dCaptureMidFrame(m_device.GetDevice(), m_command_list, target);
                }
            }
        }
    }
}

void RasterizerD3D12::DrawTexture() {
    // The guest presents a composed texture with DrawTexture (mirrors the Vulkan path):
    // the source image is copied into the current render target, i.e. the queued flinger
    // buffer. This was an empty stub, so those display buffers were never written.
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_texture_cache.UpdateRenderTargets(false);
    m_texture_cache.SynchronizeGraphicsDescriptors();
    const auto& state = maxwell3d->draw_manager->GetDrawTextureState();
    Framebuffer* const framebuffer = m_texture_cache.GetFramebuffer();
    ImageView* const dst_view = framebuffer ? framebuffer->ColorBuffers()[0] : nullptr;
    if (!dst_view || !dst_view->RenderTarget().ptr) {
        return;
    }
    ImageView& src_view = m_texture_cache.GetImageView(state.src_texture);
    if (src_view.IsBuffer()) {
        return;
    }
    const VideoCommon::Region2D dst_region{
        VideoCommon::Offset2D{static_cast<s32>(state.dst_x0), static_cast<s32>(state.dst_y0)},
        VideoCommon::Offset2D{static_cast<s32>(state.dst_x1), static_cast<s32>(state.dst_y1)}};
    const VideoCommon::Region2D src_region{
        VideoCommon::Offset2D{static_cast<s32>(state.src_x0), static_cast<s32>(state.src_y0)},
        VideoCommon::Offset2D{static_cast<s32>(state.src_x1), static_cast<s32>(state.src_y1)}};
    const bool accelerated = m_texture_runtime.BlitImage(
        framebuffer, *dst_view, src_view, dst_region, src_region,
        Tegra::Engines::Fermi2D::Filter::Bilinear, Tegra::Engines::Fermi2D::Operation::SrcCopy);
    if (!accelerated) {
        m_texture_runtime.ConvertImage(framebuffer, *dst_view, src_view);
    }
    static u32 draw_texture_logs = 0;
    if (draw_texture_logs < 20 || draw_texture_logs % 300 == 0) {
        LOG_WARNING(Render_D3D12,
                    "DrawTexture #{}: src_idx={} src={:#x} {}x{} dst=({},{})-({},{}) "
                    "src_region=({},{})-({},{}) accelerated={}",
                    draw_texture_logs, state.src_texture,
                    static_cast<u64>(src_view.GpuAddr()), src_view.size.width, src_view.size.height,
                    state.dst_x0, state.dst_y0, state.dst_x1, state.dst_y1, state.src_x0,
                    state.src_y0, state.src_x1, state.src_y1, accelerated ? 1 : 0);
    }
    ++draw_texture_logs;
}

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
    {
        // Diagnostic (session 12): does the guest's DRAM backing of the presented device
        // address actually contain the frame? (The texture-cache image can be stale.)
        static u32 cpu_probe_logs = 0;
        if (cpu_probe_logs < 24 || cpu_probe_logs % 300 == 0) {
            const u8* const cpu_bytes =
                gpu_memory ? gpu_memory->GetPointer<u8>(static_cast<VAddr>(framebuffer_addr))
                           : nullptr;
            if (cpu_bytes != nullptr) {
                const size_t row_pitch = static_cast<size_t>(config.stride) * 4;
                u32 min_value = 255;
                u32 max_value = 0;
                u64 sum = 0;
                u32 nonzero = 0;
                for (u32 y = 0; y < 64; ++y) {
                    const u8* const row = cpu_bytes + y * row_pitch;
                    for (u32 x = 0; x < 64 * 4; ++x) {
                        const u32 value = row[x];
                        min_value = value < min_value ? value : min_value;
                        max_value = value > max_value ? value : max_value;
                        sum += value;
                        if (value != 0) {
                            ++nonzero;
                        }
                    }
                }
                LOG_WARNING(Render_D3D12,
                            "Display CPU probe: addr={:#x} min={} max={} sum={} nonzero={}/{}",
                            static_cast<u64>(framebuffer_addr), min_value, max_value, sum, nonzero,
                            64 * 64 * 4);
            } else {
                LOG_WARNING(Render_D3D12, "Display CPU probe: addr={:#x} unmapped",
                            static_cast<u64>(framebuffer_addr));
            }
        }
        ++cpu_probe_logs;
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
    {
        // Diagnostic (session 12): remember the resources the frontend presents.
        static u32 display_register = 0;
        g_display_resources[display_register % g_display_resources.size()] =
            image_view->Resource();
        ++display_register;
    }
    m_query_cache.NotifySegment(false);
    (void)scaled; // Resolution scaling is disabled on the D3D12 backend for now.
    return AccelerateDisplayInfo{
        .view = image_view,
        .width = image_view->size.width,
        .height = image_view->size.height,
    };
}

void RasterizerD3D12::Clear(u32 layer_count) {
    const auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    // TEMP DIAGNOSTIC (session 13): bounded proof of whether the guest actually clears and
    // with which values, since a black screen can mean "no clear ever arrived" just as well
    // as "the clear landed on the wrong target".
    static u64 clear_calls = 0;
    const u64 clear_index = clear_calls++;
    if (clear_index < 16 || (clear_index % 4096) == 0) {
        LOG_WARNING(Render_D3D12,
                    "D3D12 Clear #{}: color={} depth={} stencil={} rt={} layer={} count={} "
                    "scissor={} color0=({},{},{},{}) depth_val={}",
                    clear_index, use_color, use_depth, use_stencil,
                    regs.clear_surface.RT.Value(), regs.clear_surface.layer.Value(), layer_count,
                    regs.clear_control.use_scissor ? 1 : 0, regs.clear_color[0],
                    regs.clear_color[1], regs.clear_color[2], regs.clear_color[3],
                    regs.clear_depth);
    }

    // The render-target walk resolves guest memory, so it must be coherent first
    // (RasterizerVulkan::Clear calls gpu_memory->FlushCaching() here; the D3D12
    // rasterizer reaches it through the same shared Tegra::MemoryManager pointer
    // that ConfigureDraw and friends use, so guard for a missing GPU memory
    // manager before flushing).
    if (gpu_memory != nullptr) {
        gpu_memory->FlushCaching();
    }
    m_query_cache.NotifySegment(true);
    m_query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                regs.zpass_pixel_count_enable != 0);

    // Same cache locking as ConfigureDraw/DrawTexture.
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    m_texture_cache.UpdateRenderTargets(true);
    Framebuffer* const framebuffer = m_texture_cache.GetFramebuffer();
    if (!framebuffer) {
        return;
    }
    const VideoCommon::Extent2D render_area = framebuffer->RenderArea();
    if (render_area.width == 0 || render_area.height == 0) {
        return;
    }

    // Full framebuffer area, or the guest scissor when the clear is scissor controlled.
    D3D12_RECT clear_rect;
    if (regs.clear_control.use_scissor) {
        const D3D12_RECT scissor = MakeScissor(regs);
        clear_rect.left = scissor.left;
        clear_rect.top = scissor.top;
        clear_rect.right = scissor.right;
        clear_rect.bottom = scissor.bottom;
    } else {
        clear_rect.left = 0;
        clear_rect.top = 0;
        clear_rect.right = static_cast<s32>(render_area.width);
        clear_rect.bottom = static_cast<s32>(render_area.height);
    }
    // Vulkan only clamps the extent to the render area; D3D12 additionally rejects rects
    // reaching outside the cleared resource, so clamp the offsets as well.
    clear_rect.left = std::max<s32>(clear_rect.left, 0);
    clear_rect.top = std::max<s32>(clear_rect.top, 0);
    clear_rect.right = std::min<s32>(clear_rect.right, static_cast<s32>(render_area.width));
    clear_rect.bottom = std::min<s32>(clear_rect.bottom, static_cast<s32>(render_area.height));
    if (clear_rect.right <= clear_rect.left || clear_rect.bottom <= clear_rect.top) {
        return;
    }
    // D3D12 takes an array of clear rects; a full-area clear needs none.
    D3D12_RECT clear_rects[1] = {clear_rect};
    const D3D12_RECT* p_rects = regs.clear_control.use_scissor ? clear_rects : nullptr;
    const UINT rect_count = regs.clear_control.use_scissor ? 1U : 0U;

    // Array layers: the texture cache builds the framebuffer's RTV/DSV for the whole
    // image and has no per-layer-slice views, so a clear that names a layer sub-range
    // (regs.clear_surface.layer / layer_count) clears every layer of the view. Stated
    // here rather than silently ignored.
    if (regs.clear_surface.layer != 0 || layer_count != 1) {
        static bool logged_layer_range{};
        if (!logged_layer_range) {
            logged_layer_range = true;
            LOG_WARNING(Render_D3D12,
                        "Clear with array layer range (layer={} count={}): the whole view is "
                        "cleared (no per-layer RTV/DSV)",
                        regs.clear_surface.layer.Value(), layer_count);
        }
    }

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && color_attachment < VideoCommon::NUM_RT) {
        const ImageView* const view = framebuffer->ColorBuffers()[color_attachment];
        ID3D12Resource* const resource = view ? view->Resource() : nullptr;
        if (view != nullptr && resource != nullptr && view->RenderTarget().ptr != 0) {
            // Clear-value conversion mirrors RasterizerVulkan::Clear: regs.clear_color is
            // a std::array<f32, 4>, so a non-integer render target takes its components
            // verbatim while an integer target scales into the component range (uint:
            // 2 * bit_size, sint: 2 * (bit_size - 1) minus a half step). D3D12 reads an
            // integer RTV's clear color as raw u32/s32 lanes, so the integer result is
            // reinterpreted into the f32 slot. A partial R/G/B/A mask has no D3D12
            // equivalent (Vulkan does a masked blit here) and clears all four channels.
            const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(
                regs.rt[color_attachment].format);
            f32 color[4]{};
            if (!VideoCore::Surface::IsPixelFormatInteger(format)) {
                for (u32 i = 0; i < 4; ++i) {
                    color[i] = regs.clear_color[i];
                }
            } else if (!VideoCore::Surface::IsPixelFormatSignedInteger(format)) {
                const f32 scale = static_cast<f32>(
                    static_cast<u64>(VideoCore::Surface::PixelComponentSizeBitsInteger(format)) <<
                    1U);
                for (u32 i = 0; i < 4; ++i) {
                    const u32 value = static_cast<u32>(scale * regs.clear_color[i]);
                    std::memcpy(&color[i], &value, sizeof(value));
                }
            } else {
                const size_t bits = VideoCore::Surface::PixelComponentSizeBitsInteger(format);
                const f32 scale = static_cast<f32>(static_cast<s64>(bits - 1) << 1);
                for (u32 i = 0; i < 4; ++i) {
                    const s32 value = static_cast<s32>(scale * (regs.clear_color[i] - 0.5f));
                    std::memcpy(&color[i], &value, sizeof(value));
                }
            }
            // COMMON -> RENDER_TARGET excursion, same as ConfigureDraw's transition_rt.
            m_command_list.Transition(resource, D3D12_RESOURCE_STATE_COMMON,
                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
            Diag::Push(Diag::CmdKind::ClearRtv, view->RenderTarget().ptr, rect_count);
            m_command_list.Get()->ClearRenderTargetView(view->RenderTarget(), color,
                                                        rect_count, p_rects);
            m_command_list.Transition(resource, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                      D3D12_RESOURCE_STATE_COMMON);
        }
    }

    ImageView* const depth_view = framebuffer->DepthBuffer();
    ID3D12Resource* const depth_resource = depth_view ? depth_view->Resource() : nullptr;
    if ((use_depth || use_stencil) && depth_view != nullptr && depth_resource != nullptr &&
        depth_view->RenderTarget().ptr != 0) {
        // Only a DSV whose format carries a stencil plane accepts the stencil flag.
        const VideoCore::Surface::PixelFormat depth_format = depth_view->format;
        const bool has_stencil = depth_format == VideoCore::Surface::PixelFormat::D24_UNORM_S8_UINT ||
                                 depth_format == VideoCore::Surface::PixelFormat::S8_UINT_D24_UNORM ||
                                 depth_format == VideoCore::Surface::PixelFormat::D32_FLOAT_S8_UINT;
        D3D12_CLEAR_FLAGS flags{};
        if (use_depth) {
            flags |= D3D12_CLEAR_FLAG_DEPTH;
        }
        if (use_stencil && has_stencil) {
            flags |= D3D12_CLEAR_FLAG_STENCIL;
        }
        if (flags != 0) {
            // regs.clear_depth is already an f32 in [0, 1]; the stencil clear value is
            // narrowed to the UINT8 ClearDepthStencilView takes.
            m_command_list.Transition(depth_resource, D3D12_RESOURCE_STATE_COMMON,
                                      D3D12_RESOURCE_STATE_DEPTH_WRITE);
            Diag::Push(Diag::CmdKind::ClearDsv, depth_view->RenderTarget().ptr,
                       static_cast<u64>(flags));
            m_command_list.Get()->ClearDepthStencilView(depth_view->RenderTarget(), flags,
                                                        regs.clear_depth,
                                                        static_cast<u8>(regs.clear_stencil),
                                                        rect_count, p_rects);
            m_command_list.Transition(depth_resource, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                      D3D12_RESOURCE_STATE_COMMON);
        }
    }
}
void RasterizerD3D12::DispatchCompute() {
    // TEMP DIAGNOSTIC (session 13): compute dispatches are currently dropped entirely (the
    // D3D12 backend has no compute pipeline path). The missing 3D scene on the style screen
    // is the prime suspect for guest compute usage, so log the dispatch rate and shape.
    if (!kepler_compute) {
        return;
    }
    const auto& qmd = kepler_compute->launch_description;
    static u64 dispatch_total = 0;
    const u64 dispatch_index = dispatch_total++;
    if (dispatch_index < 8 || (dispatch_index % 1000) == 0) {
        LOG_WARNING(Render_D3D12,
                    "D3D12 compute dispatch #{}: program={:#x} grid=({},{},{}) "
                    "cbuf_mask={:#x}",
                    dispatch_index, qmd.program_start, qmd.grid_dim_x.Value(),
                    qmd.grid_dim_y.Value(), qmd.grid_dim_z.Value(),
                    qmd.const_buffer_enable_mask.Value());
    }

    StoredComputePipeline* const pipeline = m_pipeline_cache.CurrentComputePipeline();
    if (!pipeline) {
        static bool logged_no_compute_pipeline = false;
        if (!logged_no_compute_pipeline) {
            logged_no_compute_pipeline = true;
            LOG_ERROR(Render_D3D12,
                      "D3D12 compute dispatch dropped: no pipeline (translation/PSO failed)");
        }
        return;
    }
    if (kepler_compute->GetIndirectComputeAddress()) {
        static bool logged_indirect_compute = false;
        if (!logged_indirect_compute) {
            logged_indirect_compute = true;
            LOG_ERROR(Render_D3D12, "D3D12 indirect compute dispatch not implemented; dropped");
        }
        return;
    }

    const Shader::Info& info = pipeline->info;
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};

    m_runtime.ClearDrawBindings();
    m_texture_cache.SynchronizeComputeDescriptors();

    // Compute binding mirrors Vulkan::ComputePipeline::Configure: CBVs + storage buffers
    // (raw SRV/UAV), texture/image buffers through the buffer cache, sampled textures from
    // the compute TIC walk and storage images, so the tables line up with the shader's
    // set-1 register allocation.
    VideoCommon::ComputeUniformBufferSizes sizes{};
    for (size_t i = 0; i < sizes.size(); ++i) {
        sizes[i] = info.constant_buffer_used_sizes[i];
    }
    m_buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask, &sizes);

    m_buffer_cache.UnbindComputeStorageBuffers();
    u32 ssbo_index = 0;
    for (const auto& desc : info.storage_buffers_descriptors) {
        m_buffer_cache.BindComputeStorageBuffer(ssbo_index++, desc.cbuf_index, desc.cbuf_offset,
                                                desc.is_written);
    }

    // Set-1 resource walk. The texture handles are read from the COMPUTE const buffers
    // (the launch description's const buffer config, not the graphics shader stages),
    // resolved into image views, and the texture/image buffers are bound through the
    // buffer cache BEFORE the host-stage walk so their Texel records are appended in
    // the shader's binding order.
    const auto& cbufs = qmd.const_buffer_config;
    const bool via_header = qmd.linked_tsc != 0;
    const auto read_handle = [&](const auto& desc, u32 index) {
        // The QMD only carries eight const buffers; a descriptor past that reads as an
        // unbound handle instead of walking off the launch description.
        if (desc.cbuf_index >= cbufs.size()) [[unlikely]] {
            return TexturePair(u32{0}, via_header);
        }
        const u32 index_offset = index << desc.size_shift;
        const u32 offset = desc.cbuf_offset + index_offset;
        const GPUVAddr addr = cbufs[desc.cbuf_index].Address() + offset;
        if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                      std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
            if (desc.has_secondary) {
                if (desc.secondary_cbuf_index >= cbufs.size()) [[unlikely]] {
                    return TexturePair(u32{0}, via_header);
                }
                const u32 second_offset = desc.secondary_cbuf_offset + index_offset;
                const GPUVAddr separate_addr =
                    cbufs[desc.secondary_cbuf_index].Address() + second_offset;
                const u32 lhs_raw = gpu_memory->Read<u32>(addr) << desc.shift_left;
                const u32 rhs_raw =
                    gpu_memory->Read<u32>(separate_addr) << desc.secondary_shift_left;
                const u32 raw = lhs_raw | rhs_raw;
                return TexturePair(raw, via_header);
            }
        }
        return TexturePair(gpu_memory->Read<u32>(addr), via_header);
    };
    std::vector<VideoCommon::ImageViewInOut> views;
    std::vector<VideoCommon::SamplerId> sampler_ids;
    u32 texbuf_total = 0;
    u32 texbuf_resolved = 0;
    u32 imgbuf_total = 0;
    u32 imgbuf_resolved = 0;
    u32 tex_total = 0;
    u32 tex_resolved = 0;
    u32 img_total = 0;
    u32 img_resolved = 0;
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
    for (const auto& desc : info.texture_buffer_descriptors) {
        texbuf_total += desc.count;
        add_image(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        imgbuf_total += desc.count;
        add_image(desc, false);
    }
    const size_t sampled_views_begin = views.size();
    for (const auto& desc : info.texture_descriptors) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle = read_handle(desc, index);
            views.push_back(VideoCommon::ImageViewInOut{.index = handle.first});
            sampler_ids.push_back(handle.first == 0
                                       ? VideoCommon::NULL_SAMPLER_ID
                                       : m_texture_cache.GetComputeSamplerId(handle.second));
            ++tex_total;
        }
    }
    const size_t sampled_views_end = views.size();
    for (const auto& desc : info.image_descriptors) {
        img_total += desc.count;
        add_image(desc, desc.is_written);
    }
    m_texture_cache.FillComputeImageViews(std::span(views.data(), views.size()));
    {
        // Per-category resolved counts for the bind diagnostic below.
        const auto count_resolved = [&](size_t begin, size_t end) {
            u32 resolved = 0;
            for (size_t i = begin; i < end && i < views.size(); ++i) {
                if (views[i].id != VideoCommon::NULL_IMAGE_VIEW_ID) {
                    ++resolved;
                }
            }
            return resolved;
        };
        texbuf_resolved = count_resolved(0, texbuf_total);
        imgbuf_resolved = count_resolved(texbuf_total, texbuf_total + imgbuf_total);
        tex_resolved = count_resolved(sampled_views_begin, sampled_views_end);
        img_resolved = count_resolved(sampled_views_end, views.size());
    }

    m_buffer_cache.UnbindComputeTextureBuffers();
    size_t tbo_index = 0;
    const auto add_buffer = [&](const auto& desc, bool is_image) {
        bool is_written = false;
        if constexpr (std::is_same_v<decltype(desc), const Shader::ImageBufferDescriptor&>) {
            is_written = desc.is_written;
        }
        for (u32 i = 0; i < desc.count; ++i) {
            const ImageView& buffer_view = m_texture_cache.GetImageView(views[tbo_index].id);
            m_buffer_cache.BindComputeTextureBuffer(tbo_index, buffer_view.GpuAddr(),
                                                   buffer_view.BufferSize(), buffer_view.format,
                                                   is_written, is_image);
            ++tbo_index;
        }
    };
    for (const auto& desc : info.texture_buffer_descriptors) {
        add_buffer(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add_buffer(desc, true);
    }

    m_buffer_cache.UpdateComputeBuffers();
    m_buffer_cache.BindHostComputeBuffers();

    const auto& bindings = m_runtime.GetResourceBindings();
    std::vector<u32> cbv_records;
    std::vector<u32> storage_records;
    std::vector<bool> storage_written;
    std::vector<u32> texel_records;
    for (size_t i = 0; i < bindings.size(); ++i) {
        const auto& record = bindings[i];
        if (record.kind == BufferCacheRuntime::BindingKind::Uniform) {
            cbv_records.push_back(static_cast<u32>(i));
        } else if (record.kind == BufferCacheRuntime::BindingKind::Storage) {
            storage_records.push_back(static_cast<u32>(i));
            storage_written.push_back(record.is_written);
        } else {
            texel_records.push_back(static_cast<u32>(i));
        }
    }

    // Set-1 slot order, following the shader's binding allocation: storage buffers first
    // (records in bind order), then texture buffers (buffer-cache Texel records), image
    // buffers, sampled textures (views + samplers from the compute TIC/TSC walk) and
    // storage images. This is the order CreateComputePipeline built the root signature
    // from, so the SRV and UAV tables line up with the shader's space-1 registers.
    enum class ResKind : u8 {
        Storage,     // SSBO: SRV when read-only, UAV when written
        TexelBuffer, // texture buffer bound through the buffer cache
        ImageBuffer, // image buffer bound through the buffer cache
        Sampled,     // sampled texture from the compute TIC walk
        Image,       // storage image
    };
    struct ResSlot {
        ResKind kind{};
        u32 record{0xFFFFFFFFu}; // index into bindings for storage/texel; 0xFFFFFFFF = null
        bool is_written{};
        VideoCommon::ImageViewId view{VideoCommon::NULL_IMAGE_VIEW_ID};
        VideoCommon::SamplerId sampler{VideoCommon::NULL_SAMPLER_ID};
    };
    std::vector<ResSlot> res_slots;
    for (size_t i = 0; i < storage_records.size(); ++i) {
        res_slots.push_back(ResSlot{
            .kind = ResKind::Storage,
            .record = storage_records[i],
            .is_written = storage_written[i],
        });
    }
    // BindHostComputeTextureBuffers appends the Texel records in const-buffer order:
    // texture buffers first, then image buffers.
    size_t texel_cursor = 0;
    for (const auto& desc : info.texture_buffer_descriptors) {
        for (u32 i = 0; i < desc.count; ++i) {
            res_slots.push_back(ResSlot{
                .kind = ResKind::TexelBuffer,
                .record = texel_cursor < texel_records.size() ? texel_records[texel_cursor]
                                                             : 0xFFFFFFFFu,
            });
            ++texel_cursor;
        }
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        for (u32 i = 0; i < desc.count; ++i) {
            res_slots.push_back(ResSlot{
                .kind = ResKind::ImageBuffer,
                .record = texel_cursor < texel_records.size() ? texel_records[texel_cursor]
                                                             : 0xFFFFFFFFu,
                .is_written = desc.is_written,
            });
            ++texel_cursor;
        }
    }
    for (size_t i = sampled_views_begin; i < sampled_views_end; ++i) {
        res_slots.push_back(ResSlot{
            .kind = ResKind::Sampled,
            .view = views[i].id,
            .sampler = sampler_ids[i - sampled_views_begin],
        });
    }
    u32 image_cursor = 0;
    for (const auto& desc : info.image_descriptors) {
        for (u32 i = 0; i < desc.count; ++i) {
            res_slots.push_back(ResSlot{
                .kind = ResKind::Image,
                .is_written = desc.is_written,
                .view = image_cursor < views.size() - sampled_views_end
                            ? views[sampled_views_end + image_cursor].id
                            : VideoCommon::NULL_IMAGE_VIEW_ID,
            });
            ++image_cursor;
        }
    }

    const RootSignature& root_sig = pipeline->root_signature;
    const u32 num_cbv = static_cast<u32>(cbv_records.size());
    const u32 num_res = static_cast<u32>(res_slots.size());
    // Contiguous table regions, same layout as ConfigureDraw: [0, num_cbv) CBV,
    // [num_cbv, +num_res) SRV, [num_cbv + num_res, +num_res) UAV, samplers [0, num_res).
    const u32 heap_total = num_cbv + num_res * 2;
    const u32 heap_base = heap_total != 0 ? m_res_heap.Allocate(heap_total) : 0;
    const u32 sampler_base = num_res != 0 ? m_sampler_heap.Allocate(num_res) : 0;
    if ((heap_total != 0 && heap_base == DescriptorHeap::INVALID_INDEX) ||
        (num_res != 0 && sampler_base == DescriptorHeap::INVALID_INDEX)) {
        static bool logged_compute_heap_exhausted = false;
        if (!logged_compute_heap_exhausted) {
            logged_compute_heap_exhausted = true;
            LOG_ERROR(Render_D3D12,
                      "Compute descriptor heap exhausted (cbv={} res={}); dispatch dropped",
                      num_cbv, num_res);
        }
        return;
    }
    const u32 cbv_slot_base = heap_base;
    const u32 srv_slot_base = heap_base + num_cbv;
    const u32 uav_slot_base = heap_base + num_cbv + num_res;
    ID3D12Device* const d3d = m_device.GetDevice();

    // UAV state transitions: every real storage-buffer resource that is written, and
    // every storage image whose host resource is known, goes COMMON -> UNORDERED_ACCESS
    // before the dispatch, and back afterwards.
    const auto slot_resource = [&](const ResSlot& slot) -> ID3D12Resource* {
        if (slot.kind == ResKind::Image) {
            return slot.view != VideoCommon::NULL_IMAGE_VIEW_ID
                       ? m_texture_cache.GetImageView(slot.view).Resource()
                       : nullptr;
        }
        if (slot.record == 0xFFFFFFFFu) {
            return nullptr;
        }
        const auto& record = bindings[slot.record];
        return m_runtime.IsNullResource(record.resource) ? nullptr : record.resource;
    };
    const auto wants_uav_access = [&](const ResSlot& slot) {
        if (slot.kind == ResKind::Image) {
            return slot.view != VideoCommon::NULL_IMAGE_VIEW_ID;
        }
        return slot.is_written && slot.record != 0xFFFFFFFFu;
    };
    const auto transition_uav = [&](D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        // Deduplicate: several slots can alias one resource and a second
        // COMMON -> UNORDERED_ACCESS barrier would use the wrong before-state.
        ID3D12Resource* done[32]{};
        size_t done_count = 0;
        for (size_t i = 0; i < res_slots.size(); ++i) {
            const ResSlot& slot = res_slots[i];
            if (!wants_uav_access(slot)) {
                continue;
            }
            ID3D12Resource* resource = slot_resource(slot);
            if (!resource) {
                continue;
            }
            bool seen = false;
            for (size_t d = 0; d < done_count; ++d) {
                if (done[d] == resource) {
                    seen = true;
                    break;
                }
            }
            if (seen) {
                continue;
            }
            if (done_count < 32) {
                done[done_count++] = resource;
            }
            m_command_list.Transition(resource, before, after);
        }
    };

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
        const ResSlot& slot = res_slots[k];
        ID3D12Resource* resource = nullptr;
        u64 offset = 0;
        u64 size = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE prebuilt{};
        D3D12_CPU_DESCRIPTOR_HANDLE uav_prebuilt{};
        bool have_prebuilt = false;
        bool have_uav_prebuilt = false;
        bool want_uav = false;
        switch (slot.kind) {
        case ResKind::Storage: {
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
        case ResKind::TexelBuffer:
        case ResKind::ImageBuffer:
            // Texture/image buffers bind through the buffer cache: the Texel record's
            // prebuilt SRV is copied into the table like SlotKind::TexelBuffer in
            // ConfigureDraw. Image-buffer writes stay null UAVs (no write path yet).
            if (slot.record < bindings.size() && slot.record != 0xFFFFFFFFu &&
                bindings[slot.record].kind == BufferCacheRuntime::BindingKind::Texel &&
                bindings[slot.record].view.ptr != 0) {
                prebuilt = bindings[slot.record].view;
                have_prebuilt = true;
            }
            break;
        case ResKind::Sampled:
            if (slot.view != VideoCommon::NULL_IMAGE_VIEW_ID) {
                prebuilt = m_texture_cache.GetImageView(slot.view).Sampled();
                have_prebuilt = prebuilt.ptr != 0;
            }
            break;
        case ResKind::Image:
            // Storage image: copy the resolved image view's prebuilt storage (UAV)
            // descriptor into the UAV table slot; null UAV when the image view has no
            // UAV-capable storage view.
            if (slot.view != VideoCommon::NULL_IMAGE_VIEW_ID) {
                uav_prebuilt = m_texture_cache.GetImageView(slot.view).Storage();
                have_uav_prebuilt = uav_prebuilt.ptr != 0;
            }
            want_uav = true;
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
            d3d->CreateShaderResourceView(nullptr, &null_srv,
                                          m_res_heap.CpuHandle(srv_slot_base + k));
        }
        if (have_uav_prebuilt) {
            d3d->CopyDescriptorsSimple(1, m_res_heap.CpuHandle(uav_slot_base + k), uav_prebuilt,
                                       D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        } else if (want_uav && resource) {
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
        // Samplers: one per set-1 binding so sN always pairs with tN. Slots without a
        // resolved sampler get a default sampler instead of leaving garbage in the table.
        if (slot.kind == ResKind::Sampled && slot.sampler != VideoCommon::NULL_SAMPLER_ID) {
            const Sampler& sampler = m_texture_cache.GetSampler(slot.sampler);
            d3d->CreateSampler(&sampler.Desc(), m_sampler_heap.CpuHandle(sampler_base + k));
        } else {
            D3D12_SAMPLER_DESC default_sampler{};
            default_sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            default_sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            default_sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            default_sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            // ComparisonFunc and MaxAnisotropy are validated enums on Xbox; D3D12's
            // zero-initialized values (0) are invalid and make the whole sampler
            // descriptor invalid, which faults the GPU when a shader uses it.
            default_sampler.MaxAnisotropy = 1;
            default_sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            default_sampler.MinLOD = 0.0f;
            default_sampler.MaxLOD = D3D12_FLOAT32_MAX;
            d3d->CreateSampler(&default_sampler, m_sampler_heap.CpuHandle(sampler_base + k));
        }
    }

    // UAV transitions before the dispatch so the resources are writable.
    transition_uav(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_command_list.SetComputeRootSignature(root_sig.Get());
    m_command_list.SetPipelineState(pipeline->pipeline.Get());
    ID3D12DescriptorHeap* compute_heaps[] = {m_res_heap.Get(), m_sampler_heap.Get()};
    m_command_list.Get()->SetDescriptorHeaps(2, compute_heaps);
    if (num_cbv > 0 && root_sig.GetCbvTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetComputeRootDescriptorTable(root_sig.GetCbvTableIndex(),
                                                    m_res_heap.GpuHandle(cbv_slot_base));
    }
    if (num_res > 0 && root_sig.GetSrvTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetComputeRootDescriptorTable(root_sig.GetSrvTableIndex(),
                                                    m_res_heap.GpuHandle(srv_slot_base));
    }
    if (num_res > 0 && root_sig.GetUavTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetComputeRootDescriptorTable(root_sig.GetUavTableIndex(),
                                                    m_res_heap.GpuHandle(uav_slot_base));
    }
    if (num_res > 0 && root_sig.GetSamplerTableIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetComputeRootDescriptorTable(root_sig.GetSamplerTableIndex(),
                                                    m_sampler_heap.GpuHandle(sampler_base));
    }
    if (root_sig.GetPushConstantIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetComputeRootConstantBufferView(root_sig.GetPushConstantIndex(),
                                                        m_zero_address);
    }
    if (root_sig.GetRuntimeDataIndex() != RootSignature::INVALID_PARAMETER) {
        m_command_list.SetComputeRootConstantBufferView(root_sig.GetRuntimeDataIndex(),
                                                        m_zero_address);
    }

    const u32 gx = qmd.grid_dim_x.Value();
    const u32 gy = qmd.grid_dim_y.Value();
    const u32 gz = qmd.grid_dim_z.Value();
    if (dispatch_index < 8) {
        LOG_WARNING(Render_D3D12,
                    "D3D12 compute bind #{}: cbv={} res={} ssbo={} null_slots={} "
                    "texbuf={}/{} img={}/{} tex={}/{} imgdesc={}/{} grid=({},{},{})",
                    dispatch_index, num_cbv, num_res, storage_records.size(),
                    res_slots.size() - storage_records.size(), texbuf_resolved, texbuf_total,
                    imgbuf_resolved, imgbuf_total, tex_resolved, tex_total, img_resolved,
                    img_total, gx, gy, gz);
    }
    m_command_list.Dispatch(gx, gy, gz);

    // Make the UAV writes visible before the resources return to COMMON.
    for (size_t i = 0; i < res_slots.size(); ++i) {
        const ResSlot& slot = res_slots[i];
        if (!wants_uav_access(slot)) {
            continue;
        }
        ID3D12Resource* resource = slot_resource(slot);
        if (resource) {
            m_command_list.UAVBarrier(resource);
        }
    }
    transition_uav(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
}
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
    if (addr == 0 || size == 0) {
        return false;
    }
    {
        std::scoped_lock lock{m_buffer_cache.mutex};
        if (m_buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{m_texture_cache.mutex};
        m_texture_cache.WriteMemory(addr, size);
    }
    return false;
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
    // The frontend only ticks the caches on present (RendererD3D12::Composite), so a guest
    // that stops presenting (loading screens, long non-rendering phases) would starve the
    // texture-cache GC while uploads keep growing it. Tick here too, throttled so normal
    // play does not advance the LRU model faster than the present path.
    static u64 last_cache_tick = 0;
    const u64 now = GetTickCount64();
    if (now - last_cache_tick >= 100) {
        last_cache_tick = now;
        m_buffer_cache.TickFrame();
        m_texture_cache.TickFrame();
    }
}
void RasterizerD3D12::TickFrame() {
    // Detect a GPU hang/TDR before it can silently kill the process: D3D12 exposes the
    // removal reason here, and this log is the only in-app trace (fail-fast paths never
    // reach the crash/terminate handlers).
    static u32 remove_check_counter = 0;
    if ((++remove_check_counter % 256) == 0) {
        static bool logged_removed = false;
        const HRESULT reason = m_device.GetDevice()->GetDeviceRemovedReason();
        if (FAILED(reason) && !logged_removed) {
            logged_removed = true;
            LOG_CRITICAL(Render_D3D12, "Device removed: reason {:#x}", static_cast<u32>(reason));
        }
    }
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    // Flush before ticking: the cache GC can delete images the caches own, and Xbox's
    // command-list Close validates resource references. Destroying a resource still
    // referenced by a recorded-but-unexecuted command makes Close fail with E_INVALIDARG.
    // Executing and waiting for idle first guarantees nothing pending references them.
    m_command_list.Execute(m_device);
    m_device.WaitForIdle();
    m_command_list.Reset();
    m_res_heap.Reset();
    m_sampler_heap.Reset();
    m_cbv_scratch_used = 0;
    m_buffer_cache.TickFrame();
    m_texture_cache.TickFrame();
}
void RasterizerD3D12::LogMemoryStats() {
    std::scoped_lock lock{m_texture_cache.mutex};
    const auto vram = m_texture_cache.GetVRAMStats();
    const auto buffers = m_buffer_cache.GetMemoryUsage();
    LOG_INFO(Render_D3D12,
             "Memory stats: textures={} images={} used={:.0f} MB staging={:.0f} MB "
             "gpu_usage={:.0f} MB evicted_total={:.0f} MB buffers={} buffer_used={:.0f} MB "
             "buffer_large={:.0f} MB buffer_evicted={:.0f} MB",
             vram.texture_count, vram.sparse_texture_count,
             static_cast<f64>(vram.total_used_bytes) / (1024.0 * 1024.0),
             static_cast<f64>(m_staging_pool.TotalBytes()) / (1024.0 * 1024.0),
             static_cast<f64>(m_texture_runtime.GetDeviceMemoryUsage()) / (1024.0 * 1024.0),
             static_cast<f64>(vram.evicted_total) / (1024.0 * 1024.0), buffers.buffers,
             static_cast<f64>(buffers.total_bytes) / (1024.0 * 1024.0),
             static_cast<f64>(buffers.large_bytes) / (1024.0 * 1024.0),
             static_cast<f64>(buffers.evicted_bytes) / (1024.0 * 1024.0));
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
    if (gpu_memory == nullptr || copy_size == 0 || memory.empty()) {
        return;
    }
    // Mirror RasterizerVulkan: the uploaded bytes must land in guest memory; the
    // buffer/texture cache hooks below only refresh host-side cached copies.
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) {
        return;
    }
    {
        std::scoped_lock lock{m_buffer_cache.mutex};
        if (!m_buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            m_buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    {
        std::scoped_lock lock{m_texture_cache.mutex};
        m_texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
}
void RasterizerD3D12::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                        const VideoCore::DiskResourceLoadCallback& callback) {}
void RasterizerD3D12::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    // Initialize the Maxwell dirty-flag tables exactly like the Vulkan state tracker does.
    // The shared buffer cache consumes these flags to update and bind vertex/index buffers;
    // without this setup the tables stay all zero and every attributed draw is skipped
    // (degenerate vertex view), leaving the screen black.
    VideoCommon::Dirty::SetupDirtyFlags(channel.maxwell_3d->dirty.tables);
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