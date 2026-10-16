// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstring>
#include <iterator>
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

// TEMP DIAGNOSTIC (session 15, per-draw flush attribution): armed by the first render target
// bound into the wiper window [0x517000000, 0x517200000) that ConfigureDraw's window tracer
// sees. While armed, RasterizerD3D12::Draw submits and waits out the command list after every
// recorded draw (exactly what FlushCommands does), so the draw that hangs the GPU is the one
// between the last clean "S15 per-draw flush" line and the next one. Cleared as soon as the
// device reports a removal reason. Single-process flag: only ever touched from the GPU thread
// (ConfigureDraw/Draw), so no synchronisation is needed for correctness of the gate itself.
bool g_s15_flush_every_draw = false;

// Bounded log budget for the attribution lines; 2048 entries in total, shared by the
// pre-draw line in ConfigureDraw and the post-flush line in Draw (a draw consumes at most
// two lines), so logging can never dominate the log.
constexpr u32 kS15FlushLogCap = 2048;
std::atomic<u32> g_s15_flush_logs{0};

// Draw identity captured in ConfigureDraw next to the recorded draw. Draw() cannot reach the
// framebuffer, the pipeline state, or the guest draw parameters, so they are parked here for
// the post-flush line.
std::atomic<u64> g_s15_flush_rt0{0};
std::atomic<u32> g_s15_flush_rts{0};
std::atomic<u32> g_s15_flush_att{0};
std::atomic<u32> g_s15_flush_verts{0};
std::atomic<u32> g_s15_flush_draws{0};

// TEMP DIAGNOSTIC (session 15, style-screen chain): set by the S15 compositor block in
// ConfigureDraw once the scene-composite draw's sampled textures have been dumped
// (980021..980023), and consumed by the post-draw section of the same function, which dumps
// that draw's composite target once (980024) and clears the flag again. Only ever touched
// from the GPU thread (ConfigureDraw), like the flush flags above.
bool g_s15_composite_target_dump = false;

// TEMP DIAGNOSTIC (S18 cycle 3, Phase 0.2.3): repeated post-draw dumps of the composite target.
// `g_s18_target_res` non-null = armed for that exact resource (set every 60 s by the composite
// branch); the post-draw probe dumps it once and clears it. `g_s18_target_used` counts dumps
// (ids 990201+, cap 8). The 1.0.272 build armed a plain counter and dumped the next 8 draws'
// targets indiscriminately (a burst of 8 different RTs in 200 ms) - the resource key fixes that.
ID3D12Resource* g_s18_target_res = nullptr;
u32 g_s18_target_used = 0;

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

// TEMP DIAGNOSTIC (S17): GPU-VA ranges of the 27 MB lighting-LUT bank (nvmap handle 0x370,
// dev 0x29426000, mapped at 0x5e9900000 / 0x5eb340000 / 0x5ecd80000, 0x1a34000 bytes each)
// and of the R8 mask family (0x510fa0000 sampled member / 0x510f30000 written member). Used
// by the pre-guard draw producer tracers; the wide window matches the existing S15 LUT RT
// tracer and the per-VA extent covers the third mapping's tail (0x5ecd80000 + 0x1a34000).
constexpr bool S17ProducerRange(u64 gpu) {
    // Bank window CORRECTED after the 1.0.266 harvest: the previous bound 0x5ed00000 is a
    // 1.6 GB address (a typo for 0x5ed000000) and made every bank test in the S15/S17 draw
    // tracers FALSE - the source of the long-standing "no bank RT bind, trustworthy" artefact
    // that hid the real producer (a 6-vertex draw into 0x5eb340000). Covers all three VAs.
    // Mask windows tightened to 1 MB: the previous 4 MB reached the 0x511270000 mip chain and
    // ate the whole draw budget at t=741-746.
    return (gpu >= 0x5e9000000ull && gpu < 0x5ee7b4000ull) ||
           (gpu >= 0x510fa0000ull && gpu < 0x510fa0000ull + 0x100000ull) ||
           (gpu >= 0x510f30000ull && gpu < 0x510f30000ull + 0x100000ull);
}
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

ID3D12Resource* RasterizerD3D12::GetFeedbackSnapshot(ID3D12Resource* src) {
    if (src == nullptr) {
        return nullptr;
    }
    const D3D12_RESOURCE_DESC desc = src->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width == 0 ||
        desc.Height == 0 || desc.Width > 4096 || desc.Height > 4096) {
        return nullptr;
    }
    // Keyed by source pointer AND desc shape: the texture cache can destroy an image and
    // reuse its address for a differently-shaped resource, so a pointer-only match hit a
    // stale entry, skipped creation, and refreshed/sampled a wrongly-shaped snapshot
    // (mismatched-desc CopyResource/SRV => device removal). When full, the oldest entry is
    // evicted (see below) so a fresh snapshot is always created.
    FeedbackSnapshot* entry = nullptr;
    for (FeedbackSnapshot& candidate : m_feedback_snapshots) {
        const D3D12_RESOURCE_DESC& cached = candidate.desc;
        if (candidate.source == src && cached.Width == desc.Width &&
            cached.Height == desc.Height && cached.Format == desc.Format &&
            cached.DepthOrArraySize == desc.DepthOrArraySize &&
            cached.MipLevels == desc.MipLevels &&
            cached.SampleDesc.Count == desc.SampleDesc.Count) {
            entry = &candidate;
            break;
        }
    }
    if (entry == nullptr) {
        if (m_feedback_snapshots.size() >= 24) {
            static bool logged_evict = false;
            if (!logged_evict) {
                logged_evict = true;
                LOG_WARNING(Render_D3D12,
                            "Feedback snapshot cache full ({} entries); evicting oldest "
                            "(res=0x{:x})",
                            m_feedback_snapshots.size(),
                            reinterpret_cast<u64>(m_feedback_snapshots.front().source));
            }
            // Eviction is safe: the backend executes and waits for GPU idle at command-list
            // boundaries (TickFrame/FlushCommands), so the current command list's snapshots
            // are the newest entries and the oldest belongs to an already-completed
            // command list.
            m_feedback_snapshots.erase(m_feedback_snapshots.begin());
        }
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;
        ComPtr<ID3D12Resource> snapshot;
        const HRESULT hr = m_device.GetDevice()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&snapshot));
        if (FAILED(hr)) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                LOG_ERROR(Render_D3D12, "Feedback snapshot creation failed ({}x{}) hr={:#x}",
                          desc.Width, desc.Height, static_cast<u32>(hr));
            }
            return nullptr;
        }
        m_feedback_snapshots.push_back(FeedbackSnapshot{src, desc, std::move(snapshot)});
        entry = &m_feedback_snapshots.back();
    }
    // Refresh the snapshot with the target's current contents. The target is in COMMON
    // here: the RT transition to RENDER_TARGET is recorded after this descriptor fill.
    // (In-order execution on one queue makes reusing one scratch per shape safe: the
    // previous draw that sampled it has already executed by the time the next copy runs.)
    m_command_list.Transition(src, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_command_list.Transition(entry->snapshot.Get(), D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_STATE_COPY_DEST);
    m_command_list.Get()->CopyResource(entry->snapshot.Get(), src);
    m_command_list.Transition(entry->snapshot.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_STATE_COMMON);
    m_command_list.Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE,
                              D3D12_RESOURCE_STATE_COMMON);
    static u32 s15_fb_refresh_logs = 0;
    if (s15_fb_refresh_logs < 24) {
        ++s15_fb_refresh_logs;
        LOG_WARNING(Render_D3D12,
                    "S15 feedback snapshot refresh #{}: src=0x{:x} snap=0x{:x} {}x{} fmt={:#x}",
                    s15_fb_refresh_logs, reinterpret_cast<u64>(src),
                    reinterpret_cast<u64>(entry->snapshot.Get()), desc.Width, desc.Height,
                    static_cast<u32>(desc.Format));
    }
    return entry->snapshot.Get();
}

void RasterizerD3D12::Draw(bool is_indexed, u32 instance_count) {
    if (kSkipDraws) {
        return;
    }
    m_query_cache.NotifySegment(true);
    PipelineCache::StoredPipeline* stored = m_pipeline_cache.CurrentStoredPipeline();
    if (!stored || !stored->pipeline || !stored->pipeline->IsValid()) {
        // TEMP DIAGNOSTIC (S17): the 1.0.266 harvest proved the lighting LUT is produced by a
        // 6-vertex draw into 0x5eb340000, and that draw's PSO compile was skipped at exactly
        // that moment ("commit headroom below 1150 MB"). Every draw with an uncompiled PSO is
        // dropped HERE and retried when the guest re-issues it - so count and identify the
        // drops whose guest RT0 is a bank VA. If this stays near zero, the PSO gate is not the
        // limiter for the LUT pass.
        static u32 s17_drop_bank_logs = 0;
        static u32 s17_drop_total = 0;
        static u32 s17_drop_no_stored = 0;
        static u32 s17_drop_no_pipeline = 0;
        static u32 s17_drop_invalid = 0;
        ++s17_drop_total;
        if (!stored) {
            ++s17_drop_no_stored;
        } else if (!stored->pipeline) {
            ++s17_drop_no_pipeline;
        } else {
            ++s17_drop_invalid;
        }
        // S17 cycle 3/4: quantify the drop pressure across the whole run (the bank-only lines
        // are capped at 24) and split it by cause. One line per 5000 drops keeps the log
        // bounded. nostored = no channel/pipeline key at all (incl. no vertex program);
        // nopipeline = key valid but CreatePipeline returned nullptr; invalid = failed PSO.
        if ((s17_drop_total % 5000) == 0) {
            LOG_WARNING(Render_D3D12,
                        "S17 LUT producer: drops_total={} (nostored={} nopipeline={} invalid={})",
                        s17_drop_total, s17_drop_no_stored, s17_drop_no_pipeline,
                        s17_drop_invalid);
        }
        if (s17_drop_bank_logs < 24 && maxwell3d != nullptr &&
            S17ProducerRange(static_cast<u64>(maxwell3d->regs.rt[0].Address()))) {
            ++s17_drop_bank_logs;
            LOG_WARNING(Render_D3D12,
                        "S17 LUT producer: draw rt=0x{:x} DROPPED (stored={} pipeline={} valid={}) "
                        "drops_total={}",
                        static_cast<u64>(maxwell3d->regs.rt[0].Address()), stored ? 1 : 0,
                        (stored && stored->pipeline) ? 1 : 0,
                        (stored && stored->pipeline) ? (stored->pipeline->IsValid() ? 1 : 0) : 0,
                        s17_drop_total);
        }
        // S18 cycle 4 (Phase 0.2.4): the exposure chain's tiny passes are dropped HERE when
        // their PSO is missing/uncompiled - the bank-only counter above cannot see them. The
        // 1.0.273 harvest read the 128x1 histogram array as uniformly (0,0,0,1) (its clear
        // value) even though a 2048-vertex histogram draw binds it every frame, so "the
        // histogram pass is dropped" is the prime remaining suspect. Keyed on the tiny VAs.
        {
            static u32 s18_drop_tiny_logs = 0;
            const u64 s18_drop_rt =
                maxwell3d != nullptr ? static_cast<u64>(maxwell3d->regs.rt[0].Address()) : 0;
            if (s18_drop_tiny_logs < 12 &&
                (s18_drop_rt == 0x5d5445180ull || s18_drop_rt == 0x5d5445100ull ||
                 s18_drop_rt == 0x5d5445080ull)) {
                ++s18_drop_tiny_logs;
                LOG_WARNING(Render_D3D12,
                            "S18 tiny drop: rt=0x{:x} DROPPED (stored={} pipeline={} valid={}) "
                            "drops_total={}",
                            s18_drop_rt, stored ? 1 : 0, (stored && stored->pipeline) ? 1 : 0,
                            (stored && stored->pipeline)
                                ? (stored->pipeline->IsValid() ? 1 : 0)
                                : 0,
                            s17_drop_total);
            }
        }
        return;
    }
    std::scoped_lock lock{m_buffer_cache.mutex, m_texture_cache.mutex};
    ConfigureDraw(is_indexed, *stored, instance_count);
    // TEMP DIAGNOSTIC (session 15): per-draw flush attribution. Armed by the first draw into
    // the wiper window [0x517000000, 0x517200000) (ConfigureDraw), so every draw after the
    // first "S15 window bind" is submitted on its own and waited out. The last clean line
    // before a removal reason names the draw that hung the GPU; its inputs are on the
    // matching "S15 pre-draw" line ConfigureDraw logged just before recording it.
    if (g_s15_flush_every_draw) {
        // Same sequence FlushCommands()/TickFrame() use mid-frame (Execute -> WaitForIdle ->
        // Reset, which re-opens the command list, plus the per-draw descriptor-heap and CBV
        // scratch resets), reused verbatim instead of hand-rolling it: both cache mutexes are
        // recursive and Draw already holds them, so calling the existing helper here is safe.
        FlushCommands();
        const HRESULT removed_reason = m_device.GetDevice()->GetDeviceRemovedReason();
        const u32 draws = g_s15_flush_draws.load(std::memory_order_relaxed);
        const u32 flush_index = g_s15_flush_logs.fetch_add(1, std::memory_order_relaxed);
        if (flush_index < kS15FlushLogCap) {
            LOG_WARNING(Render_D3D12,
                        "S15 per-draw flush #{}: draw={} rt0=0x{:x} rts={} att={:#x} verts={} "
                        "hr={:#x}",
                        flush_index, draws,
                        g_s15_flush_rt0.load(std::memory_order_relaxed),
                        g_s15_flush_rts.load(std::memory_order_relaxed),
                        g_s15_flush_att.load(std::memory_order_relaxed),
                        g_s15_flush_verts.load(std::memory_order_relaxed),
                        static_cast<u32>(removed_reason));
        }
        if (FAILED(removed_reason)) {
            g_s15_flush_every_draw = false;
            LOG_ERROR(Render_D3D12, "S15 hang attributed to draw #{} (hr={:#x})", draws,
                      static_cast<u32>(removed_reason));
        }
    }
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

    // ClearDrawBindings() above deliberately wipes the runtime's index/vertex views so
    // stale host pointers can never leak into a draw, but the shared cache re-binds a
    // vertex slot only while its per-slot dirty flag is set (BindHostVertexBuffers). A
    // draw whose guest vertex-stream registers were not rewritten since the previous
    // draw (e.g. the 3D color pass reusing the depth prepass's streams) would leave the
    // wiped views all-zero (BufferLocation == 0) and be skipped by the vertex-range
    // guard below, so force the flags of the streams this pipeline actually reads. Read
    // slots mirror the guard's derivation (distinct State().attributes[i].buffer over
    // enabled attributes); unread slots are never fetched by the input layout. The
    // master Dirty::VertexBuffers flag is deliberately not forced: the update phase
    // re-resolves a slot's persistent channel binding whenever the guest or the
    // deleted-buffer path re-dirty it, so the persistent bindings are valid to re-bind
    // from as-is. BindHostIndexBuffer rebuilds the index view unconditionally, but
    // UpdateIndexBuffer sizes the persistent index binding from the CURRENT draw's
    // first/count only while its flag is set, so force it for indexed draws too.
    {
        auto& flags = maxwell3d->dirty.flags;
        const auto& pipe_state = stored.pipeline->State();
        for (u32 i = 0; i < pipe_state.attributes.size(); ++i) {
            if (pipe_state.attributes[i].enabled == 0) {
                continue;
            }
            const u32 read_slot = pipe_state.attributes[i].buffer.Value();
            if (read_slot < VideoCommon::NUM_VERTEX_BUFFERS) {
                flags[VideoCommon::Dirty::VertexBuffer0 + read_slot] = true;
            }
        }
        if (is_indexed) {
            flags[VideoCommon::Dirty::IndexBuffer] = true;
        }
    }

    // Validate BEFORE recording any uploads: draws the backend cannot represent must
    // return here, before the buffer-binding section below records staging COPY commands
    // for garbage draw state (those copies execute at flush and fault the GPU on console,
    // removing the device even for draws every later guard would skip).
    // TEMP FIX CANDIDATE (S17): Vulkan calls gpu_memory->FlushCaching() before every draw and
    // dispatch (vk_rasterizer.cpp:276 / :589) because KeplerMemory's block-linear inline-upload
    // path writes guest memory with CacheType::None and relies on this flush to invalidate the
    // caches (memory_manager.cpp:482-485). D3D12 only called it from Clear / Fermi2D / MaxwellDMA,
    // so such an upload could leave the texture cache stale indefinitely - a candidate cause of
    // the all-zero 27 MB LUT bank. No-op when nothing is pending.
    gpu_memory->FlushCaching();
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
    // TEMP DIAGNOSTIC (S15): depth-side half of the wiper-window RT tracer; a depth view
    // can land in [0x517000000, 0x517200000) too, and this runs before the skip guards so
    // draws rejected below still log. One shared static cap with the color-slot tracer in
    // the validation loop.
    static u32 s15_window_bind_logs = 0;
    if (depth_view != nullptr && s15_window_bind_logs < 24) {
        const u64 s15_depth_gpu = static_cast<u64>(depth_view->GpuAddr());
        if (s15_depth_gpu >= 0x517000000ull && s15_depth_gpu < 0x517200000ull) {
            ++s15_window_bind_logs;
            LOG_WARNING(Render_D3D12,
                        "S15 window bind: gpu={:#x} res={:#x} {}x{} slot={} raw_fmt={:#x} "
                        "unrepresentable={} unbound={} att={:#x} verts={}",
                        s15_depth_gpu, reinterpret_cast<u64>(depth_view->Resource()),
                        depth_view->size.width, depth_view->size.height, VideoCommon::NUM_RT,
                        static_cast<u32>(depth_view->format), 0,
                        depth_view->RenderTarget().ptr == 0 ? 1 : 0,
                        stored.pipeline->State().attachments[0].raw, params.num_vertices);
        }
    }
    // TEMP DIAGNOSTIC (S17): pre-guard half of the LUT/mask producer tracer. The post-guard
    // 'S15 LUT RT' block cannot see a draw its guards later reject (unrepresentable format,
    // unbound RTV), so a skipped producer bind was invisible to every previous probe. Shared
    // 32-draw budget with the color-slot half in the validation loop below.
    static u32 s17_producer_bind_logs = 0;
    if (depth_view != nullptr && s17_producer_bind_logs < 32 &&
        S17ProducerRange(static_cast<u64>(depth_view->GpuAddr()))) {
        ++s17_producer_bind_logs;
        LOG_WARNING(Render_D3D12,
                    "S17 LUT producer: draw depth gpu={:#x} res={:#x} {}x{} fmt={:#x} bank={} "
                    "att={:#x} verts={}",
                    static_cast<u64>(depth_view->GpuAddr()),
                    reinterpret_cast<u64>(depth_view->Resource()), depth_view->size.width,
                    depth_view->size.height, static_cast<u32>(depth_view->format),
                    static_cast<u64>(depth_view->GpuAddr()) >= 0x5e9000000ull ? 1 : 0,
                    stored.pipeline->State().attachments[0].raw, params.num_vertices);
    }
    if (num_rtvs == 0 && (!depth_view || !depth_view->RenderTarget().ptr)) {
        return;
    }
    for (size_t i = 0; i < num_rtvs; ++i) {
        const auto format =
            static_cast<Tegra::RenderTargetFormat>(stored.pipeline->State().color_formats[i]);
        const bool is_none = (format == Tegra::RenderTargetFormat::NONE);
        const u32 raw_format = static_cast<u32>(format);
        const char* skip_reason = nullptr;
        if (!is_none) {
            // Garbage guest render-target state must never reach OMSetRenderTargets or PSO
            // creation: an out-of-range, depth/stencil, or otherwise unrepresentable format
            // builds a degenerate pipeline that removes the device at execute time.
            const VideoCore::Surface::PixelFormat pixel_format =
                VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
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
        }
        const ImageView* view = framebuffer ? framebuffer->ColorBuffers()[i] : nullptr;
        const bool unbound = !view || !view->RenderTarget().ptr;
        // TEMP DIAGNOSTIC (S15): wiper-window RT tracer, evaluated before the skip returns
        // below so a producer draw this loop rejects (unrepresentable format or unbound
        // slot) is still logged - the previous after-loop tracer never saw skipped draws.
        if (s15_window_bind_logs < 24 && view != nullptr) {
            const u64 s15_gpu = static_cast<u64>(view->GpuAddr());
            if (s15_gpu >= 0x517000000ull && s15_gpu < 0x517200000ull) {
                ++s15_window_bind_logs;
                // TEMP DIAGNOSTIC (session 15): per-draw flush attribution was armed here; the
                // 1.0.243 hang is fixed (1.0.244 ran clean) and the mode costs a full GPU sync
                // per draw, so it stays disarmed and only the log lines above remain.
                // g_s15_flush_every_draw = true;
                LOG_WARNING(Render_D3D12,
                            "S15 window bind: gpu={:#x} res={:#x} {}x{} slot={} raw_fmt={:#x} "
                            "unrepresentable={} unbound={} att={:#x} verts={}",
                            s15_gpu, reinterpret_cast<u64>(view->Resource()), view->size.width,
                            view->size.height, i, raw_format, skip_reason != nullptr ? 1 : 0,
                            unbound ? 1 : 0, stored.pipeline->State().attachments[0].raw,
                            params.num_vertices);
            }
        }
        // TEMP DIAGNOSTIC (S17): color-slot half of the pre-guard producer tracer (see the
        // depth-side block above); runs before the skip returns below so a rejected producer
        // bind still logs. 32-draw budget shared with the depth half.
        if (view != nullptr && s17_producer_bind_logs < 32 &&
            S17ProducerRange(static_cast<u64>(view->GpuAddr()))) {
            ++s17_producer_bind_logs;
            LOG_WARNING(Render_D3D12,
                        "S17 LUT producer: draw rt slot={} gpu={:#x} res={:#x} {}x{} raw_fmt={:#x} "
                        "bank={} unrepresentable={} unbound={} att={:#x} verts={}",
                        i, static_cast<u64>(view->GpuAddr()),
                        reinterpret_cast<u64>(view->Resource()), view->size.width, view->size.height,
                        raw_format, static_cast<u64>(view->GpuAddr()) >= 0x5e9000000ull ? 1 : 0,
                        skip_reason != nullptr ? 1 : 0, unbound ? 1 : 0,
                        stored.pipeline->State().attachments[0].raw, params.num_vertices);
        }
        // S18 (Phase 0.2.1) - replaces the S17 "EXPO writer" draw tracer, which required
        // `format == R32G32B32A32_FLOAT` AND a 1x1 view size and NEVER fired in the 1.0.270
        // run even though the "Draw RTs" decimation shows a recurring 3-vertex draw with
        // vp=(0,0,1,1) rendering into gpu=[0x5d5445100] (ids=[3961:0x182e9d0fda0]) - the
        // 1x1-viewport final reduction pass that produces the exposure texel. Log ANY tiny
        // color RT bind (<= 2x2) plus the two 1x1 exposure VAs regardless of size/format, so
        // the writer is named even when the RT view's PixelFormat differs from the sampled
        // view's. Pre-guard placement: a draw the guards reject below still logs. Cap 24.
        static u32 s18_tiny_rt_logs = 0;
        if (view != nullptr && s18_tiny_rt_logs < 24) {
            const u64 s18_rt_gpu = static_cast<u64>(view->GpuAddr());
            const bool s18_tiny =
                (view->size.width <= 2 && view->size.height <= 2) ||
                (view->size.width <= 128 && view->size.height <= 2);
            const bool s18_expo_va = s18_rt_gpu == 0x5d5445100ull ||
                                     s18_rt_gpu == 0x5d5445180ull ||
                                     s18_rt_gpu == 0x5e8728480ull;
            if (s18_tiny || s18_expo_va) {
                ++s18_tiny_rt_logs;
                // S18 cycle 4: the guards' verdict and the attachment's blend state for this
                // draw. The 1.0.273 harvest showed the 128x1 histogram array reads back as its
                // clear value (0,0,0,1) although a 2048-vertex draw binds it with att=0x4240900f
                // every frame - so either the draw is rejected below, or its blend writes
                // nothing (the factors/equations are now visible here).
                const auto& s18_att = stored.pipeline->State().attachments[0];
                LOG_WARNING(Render_D3D12,
                            "S18 tiny RT: slot={} img={} gpu={:#x} res=0x{:x} {}x{} fmt={:#x} "
                            "att={:#x} verts={} vp=({},{},{},{}) skip={} unbound={} "
                            "om=({:#x},{:#x},{:#x},{:#x}) mask={:x}",
                            i, view->image_id.index, s18_rt_gpu,
                            reinterpret_cast<u64>(view->Resource()), view->size.width,
                            view->size.height, static_cast<u32>(view->format), s18_att.raw,
                            params.num_vertices, MakeViewport(maxwell3d->regs).TopLeftX,
                            MakeViewport(maxwell3d->regs).TopLeftY,
                            MakeViewport(maxwell3d->regs).Width,
                            MakeViewport(maxwell3d->regs).Height,
                            skip_reason != nullptr ? 1 : 0, unbound ? 1 : 0,
                            static_cast<u32>(s18_att.SourceRGBFactor()),
                            static_cast<u32>(s18_att.DestRGBFactor()),
                            static_cast<u32>(s18_att.EquationRGB()),
                            static_cast<u32>(s18_att.EquationAlpha()),
                            (static_cast<u32>(s18_att.mask_r ? 1 : 0)) |
                                (static_cast<u32>(s18_att.mask_g ? 2 : 0)) |
                                (static_cast<u32>(s18_att.mask_b ? 4 : 0)) |
                                (static_cast<u32>(s18_att.mask_a ? 8 : 0)));
            }
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
        if (unbound) {
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

    // TEMP DIAGNOSTIC (S15, producer tracer): the wiper-input window
    // [0x517000000, 0x517200000) never appears in the decimated "Draw RTs" log; it is now
    // traced inside the validation loop above and at the depth view, so skipped draws log
    // too. The block below covers the dynamic-res family window [0x5103e0000, 0x510640000)
    // minus the 960x540 member itself: settles whether the scene ever renders into the
    // members the composite samples. Late-gated like the small-writer tracer so the boot
    // mip-chain writes into 0x510590000 cannot consume the cap before the style phase.
    {
        static u32 s15_family_writer_logs = 0;
        static std::chrono::steady_clock::time_point s15_family_writer_start =
            std::chrono::steady_clock::now();
        const bool s15_family_writer_late =
            std::chrono::steady_clock::now() - s15_family_writer_start >
            std::chrono::minutes(20);
        if (s15_family_writer_late && s15_family_writer_logs < 64 && num_rtvs > 0) {
            const ImageView* const s15_view = framebuffer->ColorBuffers()[0];
            const u64 s15_gpu = static_cast<u64>(s15_view->GpuAddr());
            if (s15_gpu >= 0x5103e0000ull && s15_gpu < 0x510640000ull &&
                s15_gpu != 0x5103e0000ull) {
                ++s15_family_writer_logs;
                LOG_WARNING(Render_D3D12,
                            "S15 family writer: gpu={:#x} res={:#x} {}x{} att={:#x} verts={} "
                            "blend={}",
                            s15_gpu, reinterpret_cast<u64>(s15_view->Resource()),
                            s15_view->size.width, s15_view->size.height,
                            stored.pipeline->State().attachments[0].raw, params.num_vertices,
                            stored.pipeline->State().attachments[0].enable.Value());
            }
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
            // TEMP DIAGNOSTIC (session 15): the 256x256 B10G11R11 lighting LUT is all-zero
            // while the scene samples it; dump the RT resource the first few times a small
            // R11G11B10 target is rendered into, to see whether the render lands in it.
            // TEMP DIAGNOSTIC (session 17): also catch every draw whose render target lies in
            // the 27 MB LUT bank (nvmap handle 0x370, mapped at 0x5e9900000 / 0x5eb340000 /
            // 0x5ecd80000, dev 0x29426000). This loop runs before the RT/vertex guards, so a
            // producer draw that is skipped later in ConfigureDraw still shows up here.
            // S17: bank bound corrected (0x5ed00000 -> 0x5ee7b4000; the old bound was a 1.6 GB
            // address and never matched, hiding the real bank RT binds).
            const bool s17_lut_bank = addr >= 0x5e9000000ull && addr < 0x5ee7b4000ull;
            if (res != nullptr) {
                const D3D12_RESOURCE_DESC s15_lut_desc = res->GetDesc();
                const bool s17_small_lut =
                    depth_view == nullptr &&
                    s15_lut_desc.Format == DXGI_FORMAT_R11G11B10_FLOAT &&
                    s15_lut_desc.Width <= 512 && s15_lut_desc.Height <= 512;
                if (s17_lut_bank || s17_small_lut) {
                    // Separate budgets: in the 1.0.252 run the small-LUT branch consumed the
                    // whole shared cap by t=220, so a later bank bind could never be observed.
                    static u32 s15_lut_rtv_dumps = 0;
                    static u32 s17_bank_dumps = 0;
                    const bool s17_allow_dump =
                        s17_lut_bank ? (s17_bank_dumps < 24) : (s15_lut_rtv_dumps < 24);
                    if (s17_allow_dump) {
                        if (s17_lut_bank) {
                            ++s17_bank_dumps;
                        } else {
                            ++s15_lut_rtv_dumps;
                        }
                        LOG_WARNING(Render_D3D12,
                                    "S15 LUT RT: gpu={:#x} res=0x{:x} {}x{} fmt={:#x} verts={} "
                                    "bank={} att={:#x} vp=({},{},{},{})",
                                    static_cast<u64>(view->GpuAddr()),
                                    reinterpret_cast<u64>(res), s15_lut_desc.Width,
                                    s15_lut_desc.Height, static_cast<u32>(s15_lut_desc.Format),
                                    params.num_vertices, static_cast<u32>(s17_lut_bank),
                                    static_cast<u32>(stored.pipeline->State().attachments[0].raw),
                                    MakeViewport(maxwell3d->regs).TopLeftX,
                                    MakeViewport(maxwell3d->regs).TopLeftY,
                                    MakeViewport(maxwell3d->regs).Width,
                                    MakeViewport(maxwell3d->regs).Height);
                        (void)D3D12::RecordTextureContentDump(m_device.GetDevice(),
                                                              m_command_list, res,
                                                              static_cast<u32>(
                                                                  VideoCore::Surface::PixelFormat::
                                                                      B10G11R11_FLOAT),
                                                              static_cast<u64>(view->GpuAddr()),
                                                              900000u + s15_lut_rtv_dumps, true);
                    }
                }
            }
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
            // Values as bound, captured before any in-place repair below.
            const u32 bound_stride = view.StrideInBytes;
            const u32 bound_size = view.SizeInBytes;
            // D3D12 legally accepts StrideInBytes == 0 (constant vertex) and any nonzero
            // SizeInBytes, while NVN often leaves the guest stream stride/size registers
            // stale-zero. So a zero stride alone no longer skips a draw, and a zero size is
            // repaired below whenever the vertex range is known; only a genuinely unbound
            // view (BufferLocation == 0) or an unbounded size keeps the old skip. The guest
            // meshes (Splatoon 3 style screen) hit this path with slot=0 stride=0 size=0.
            u64 effective_size = static_cast<u64>(bound_size);
            // The guest's stream size register is frequently stale or short: NVN leaves it at
            // whatever the last explicit bind set while later draws reuse the same stream with
            // a larger index range. Treat a too-short size exactly like a missing one - repair
            // it from the committed resource - instead of skipping the draw. Skipping here was
            // costing ~35k draws per run (S15 vb skip counters climbed every frame).
            const bool size_short = enforce_range && vertex_required > effective_size;
            if (view.BufferLocation != 0 && (bound_size == 0 || size_short)) {
                // The guest size is 0/stale (NVN leaves it at 0 for the big character meshes,
                // which also have >kMaxGuardIndices vertices so range_known is false). The
                // committed resource is the authority: bound the view by everything left in
                // it after the view offset, which can never fault (an out-of-range vertex
                // fetch removes the device and Close cannot catch it).
                if (ID3D12Resource* resource = m_runtime.GetVertexBindingResource(slot)) {
                    const u64 resource_size = resource->GetDesc().Width;
                    const u64 resource_base = resource->GetGPUVirtualAddress();
                    if (view.BufferLocation >= resource_base) {
                        const u64 offset = static_cast<u64>(view.BufferLocation) - resource_base;
                        effective_size = resource_size > offset ? resource_size - offset : u64{0};
                    }
                }
                if (bound_size == 0 && effective_size == 0 && range_known) {
                    // Only for the zero-size case: synthesise a size from the vertex range.
                    // A merely-short size never reaches here with a zero clamp result - in
                    // that path the committed resource stays the sole authority.
                    effective_size = std::max<u64>(
                        vertex_required + static_cast<u64>(bound_stride),
                        bound_stride != 0 ? static_cast<u64>(bound_stride) : u64{16});
                    effective_size = std::min<u64>(effective_size, 512u * 1024u * 1024u);
                }
                if (effective_size > 0) {
                    // Transient: BindVertexBuffers rewrites the whole array whenever the
                    // vertex bindings change, so the patch applies to this draw only.
                    m_runtime.SetVertexBindingSize(slot, static_cast<u32>(effective_size));
                }
            }
            if (view.BufferLocation == 0 || effective_size == 0 ||
                (enforce_range && vertex_required > effective_size)) {
                static u32 vertex_range_skips{};
                if ((vertex_range_skips++ % 256) == 0) {
                    LOG_ERROR(Render_D3D12,
                              "Draw with degenerate vertex view on read slot={} stride={} "
                              "vertex_size={} skipped (count {})",
                              slot, bound_stride, bound_size, vertex_range_skips);
                }
                // TEMP DIAGNOSTIC (session 15): full context for the first skips of the run
                // (the big character meshes are believed to die here): draw params, the host
                // resource behind the view, and the raw vertex-stream registers 0..3.
                static u32 s15_vb_skip_details = 0;
                if (s15_vb_skip_details < 64) {
                    ++s15_vb_skip_details;
                    ID3D12Resource* const s15_vb_res =
                        m_runtime.GetVertexBindingResource(slot);
                    fmt::memory_buffer s15_vb_streams;
                    for (u32 s = 0; s < 4; ++s) {
                        const auto& s15_vs = regs.vertex_streams[s];
                        fmt::format_to(std::back_inserter(s15_vb_streams),
                                       "[s{} str={} en={} addr={:#x}]", s,
                                       s15_vs.stride.Value(), s15_vs.enable.Value(),
                                       static_cast<u64>(s15_vs.Address()));
                    }
                    LOG_ERROR(Render_D3D12,
                              "S15 vb skip detail #{}: slot={} loc={:#x} stride={} size={} "
                              "verts={} indexed={} base={} first={} att={:#x} req={} "
                              "range_known={} res={:#x} res_base={:#x} res_size={:#x} {}",
                              s15_vb_skip_details, slot,
                              static_cast<u64>(view.BufferLocation), bound_stride, bound_size,
                              params.num_vertices, params.is_indexed ? 1 : 0, params.base_vertex,
                              params.first_index,
                              static_cast<u32>(pipe_state.attachments[0].raw), vertex_required,
                              range_known ? 1 : 0,
                              reinterpret_cast<u64>(s15_vb_res),
                              s15_vb_res ? s15_vb_res->GetGPUVirtualAddress() : 0,
                              s15_vb_res ? s15_vb_res->GetDesc().Width : 0,
                              fmt::to_string(s15_vb_streams));
                }
                if (flinger_draw) {
                    static u32 fd_vb_logs = 0;
                    if (fd_vb_logs < 4) {
                        ++fd_vb_logs;
                        LOG_WARNING(Render_D3D12,
                                    "FLINGER skip: degenerate vb slot={} loc={:#x} stride={} "
                                    "size={} base_vtx={} verts={} range_known={} dev={:#x} req={}",
                                    slot, static_cast<u64>(view.BufferLocation),
                                    bound_stride, bound_size, params.base_vertex,
                                    params.num_vertices, range_known ? 1 : 0,
                                    static_cast<u64>(index_binding.device_addr), vertex_required);
                    }
                }
                return;
            }
            if (bound_size == 0 || bound_stride == 0) {
                // The draw now proceeds where this guard used to skip it (zero bound
                // size/stride). TEMP DIAGNOSTIC (session 15), static cap.
                static u32 s15_vertex_guard_repair_logs = 0;
                if (s15_vertex_guard_repair_logs < 24) {
                    ++s15_vertex_guard_repair_logs;
                    LOG_WARNING(Render_D3D12,
                                "S15 vertex guard repair: slot={} stride={} size={} "
                                "repaired={} verts={} indexed={} base={} first={} att={:#x}",
                                slot, bound_stride, bound_size, effective_size, params.num_vertices,
                                params.is_indexed ? 1 : 0, params.base_vertex, params.first_index,
                                static_cast<u32>(pipe_state.attachments[0].raw));
                }
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

    // S18 cycle 2 (Phase 0.2.2): the tiny reduction draw's sampled inputs + the raw readback of
    // its RT. res_slots is only populated above this point (the pre-guard RT loop at the top of
    // the function runs before the fill), so this is the first place the reduction pass's input
    // textures can be listed - they name what the auto-exposure chain measures. The 8x1
    // R32G32B32A32 RT readback (id 990401+) holds eight raw float4 texels: the reduction's
    // output, of which the composite samples the copy in the 1x1. The copy is recorded before
    // the draw executes, so the values are the previous reduction (fine - it changes slowly).
    // Cap 8 logs / 6 dumps.
    {
        static u32 s18_rt_input_logs = 0;
        ImageView* const s18_rt_view =
            (framebuffer != nullptr && num_rtvs > 0) ? framebuffer->ColorBuffers()[0] : nullptr;
        if (s18_rt_view != nullptr && s18_rt_input_logs < 8) {
            const u64 s18_rt_gpu = static_cast<u64>(s18_rt_view->GpuAddr());
            const bool s18_rt_tiny =
                s18_rt_view->size.width <= 2 && s18_rt_view->size.height <= 2;
            const bool s18_rt_expo_va = s18_rt_gpu == 0x5d5445100ull ||
                                        s18_rt_gpu == 0x5d5445080ull ||
                                        s18_rt_gpu == 0x5d5445180ull ||
                                        s18_rt_gpu == 0x5e8728480ull ||
                                        s18_rt_gpu == 0x5e88a8480ull;
            if (s18_rt_tiny || s18_rt_expo_va) {
                ++s18_rt_input_logs;
                std::string s18_rt_inputs;
                u32 s18_rt_input_count = 0;
                for (const DrawSlot& s18_slot : res_slots) {
                    if (s18_slot.kind != SlotKind::Sampled || s18_rt_input_count >= 4) {
                        continue;
                    }
                    ImageView& s18_sv = m_texture_cache.GetImageView(s18_slot.view);
                    ++s18_rt_input_count;
                    fmt::format_to(std::back_inserter(s18_rt_inputs),
                                   "[img={} gpu={:#x} {}x{} fmt={:#x} res={}]",
                                   s18_sv.image_id.index, static_cast<u64>(s18_sv.GpuAddr()),
                                   s18_sv.size.width, s18_sv.size.height,
                                   static_cast<u32>(s18_sv.format), s18_sv.Resource() ? 1 : 0);
                }
                LOG_WARNING(Render_D3D12,
                            "S18 tiny RT inputs: img={} gpu={:#x} {}x{} verts={} inputs={}",
                            s18_rt_view->image_id.index, s18_rt_gpu, s18_rt_view->size.width,
                            s18_rt_view->size.height, params.num_vertices, s18_rt_inputs);
                // S18 cycle 4 (Phase 0.2.4): when the tiny RT is the 128x1 histogram array,
                // dump its INPUT (the reduced-scene source - img 3702, 192x108 B10G11R11 in the
                // 1.0.273 run). The array reads back as its clear value (0,0,0,1) and a
                // luminance-weighted histogram of a black input is black, so this dump separates
                // "the histogram pass is dropped" from "its input is empty". Id 990701+.
                static u32 s18_hist_in_dumps = 0;
                if (s18_rt_gpu == 0x5d5445180ull) {
                    for (const DrawSlot& s18_slot : res_slots) {
                        if (s18_slot.kind != SlotKind::Sampled) {
                            continue;
                        }
                        ImageView& s18_sv = m_texture_cache.GetImageView(s18_slot.view);
                        if (s18_sv.Resource() == nullptr || s18_hist_in_dumps >= 6) {
                            break;
                        }
                        ++s18_hist_in_dumps;
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, s18_sv.Resource(),
                            static_cast<u32>(s18_sv.format), static_cast<u64>(s18_sv.GpuAddr()),
                            990700u + s18_hist_in_dumps, /*force=*/true);
                        LOG_WARNING(Render_D3D12,
                                    "S18 hist input: id={} gpu={:#x} {}x{} fmt={:#x} res=0x{:x}",
                                    990700u + s18_hist_in_dumps,
                                    static_cast<u64>(s18_sv.GpuAddr()), s18_sv.size.width,
                                    s18_sv.size.height, static_cast<u32>(s18_sv.format),
                                    reinterpret_cast<u64>(s18_sv.Resource()));
                        break;
                    }
                }
            }
        }
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

    // Feedback-loop detection (session 15): a draw that samples a resource it also renders
    // into must not sample the live target - D3D12 leaves that read undefined (Xbox returns
    // zeros) and the draw then writes the zeros back over the whole target.
    const auto is_feedback_source = [&](const ImageView& view) {
        if (view.Resource() == nullptr) {
            return false;
        }
        for (size_t i = 0; i < num_rtvs; ++i) {
            ImageView* const rt = framebuffer != nullptr ? framebuffer->ColorBuffers()[i] : nullptr;
            if (rt != nullptr && (rt->Resource() == view.Resource() ||
                                  rt->GpuAddr() == view.GpuAddr())) {
                return true;
            }
        }
        return depth_view != nullptr && (depth_view->Resource() == view.Resource() ||
                                         depth_view->GpuAddr() == view.GpuAddr());
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
        const DrawSlot& slot = res_slots[k];
        ID3D12Resource* resource = nullptr;
        u64 offset = 0;
        u64 size = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE prebuilt{};
        bool have_prebuilt = false;
        bool want_uav = false;
        bool feedback_view_written = false;
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
        case SlotKind::Sampled: {
            if (slot.view != VideoCommon::NULL_IMAGE_VIEW_ID) {
                ImageView& image_view = m_texture_cache.GetImageView(slot.view);
                prebuilt = image_view.Sampled();
                have_prebuilt = prebuilt.ptr != 0;
                // Feedback loop: sample a snapshot copy of the target instead of the live
                // resource (undefined read on D3D12; written back as zeros on Xbox).
                if (is_feedback_source(image_view)) {
                    ID3D12Resource* const snapshot = GetFeedbackSnapshot(image_view.Resource());
                    static u32 s15_fb_slot_logs = 0;
                    if (s15_fb_slot_logs < 64) {
                        ++s15_fb_slot_logs;
                        LOG_WARNING(Render_D3D12,
                                    "S15 feedback slot #{}: k={} prebuilt=0x{:x} res=0x{:x} "
                                    "gpu=0x{:x} snap=0x{:x} srv_cpu=0x{:x} table=0x{:x} "
                                    "num_rtvs={}",
                                    s15_fb_slot_logs, k, static_cast<u64>(prebuilt.ptr),
                                    reinterpret_cast<u64>(image_view.Resource()),
                                    static_cast<u64>(image_view.GpuAddr()),
                                    reinterpret_cast<u64>(snapshot),
                                    static_cast<u64>(m_res_heap.CpuHandle(srv_slot_base + k).ptr),
                                    static_cast<u64>(m_res_heap.GpuHandle(srv_slot_base).ptr),
                                    num_rtvs);
                    }
                    if (have_prebuilt && snapshot != nullptr) {
                        image_view.CreateSampledViewFor(
                            snapshot, m_res_heap.CpuHandle(srv_slot_base + k));
                        have_prebuilt = false;
                        feedback_view_written = true;
                        // Verify what the snapshot actually contains (bounded).
                        static u32 s15_fb_dump_logs = 0;
                        if (s15_fb_dump_logs < 24) {
                            ++s15_fb_dump_logs;
                            (void)D3D12::RecordTextureContentDump(
                                m_device.GetDevice(), m_command_list, snapshot,
                                static_cast<u32>(image_view.format),
                                static_cast<u64>(image_view.GpuAddr()),
                                941000u + s15_fb_dump_logs,
                                /*force=*/true);
                        }
                    }
                }
            }
            break;
        }
        }
        if (feedback_view_written) {
            // SRV already created directly into the table (feedback snapshot substitute).
        } else if (have_prebuilt) {
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
    // TEMP DIAGNOSTIC (session 15): in-draw probe dumps must never read back a resource that is
    // currently bound as this draw's render target. RecordTextureContentDump records
    // COMMON -> COPY_SOURCE, but the RT stays in RENDER_TARGET in the command stream until the
    // restore loop at the end of ConfigureDraw; recording the dump mid-window produces an
    // invalid transition pair on Xbox and can fail Close (dropping the whole batch). Skip any
    // dump of a currently-bound RT/depth resource.
    const auto s15_is_current_rt = [&](ID3D12Resource* resource) {
        if (resource == nullptr) {
            return false;
        }
        for (u32 i = 0; i < rt_resource_count; ++i) {
            if (rt_resources[i] == resource) {
                return true;
            }
        }
        return false;
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
    // TEMP DIAGNOSTIC (session 15): per-draw flush attribution. Emitted for every draw while
    // g_s15_flush_every_draw is armed, so the hanging draw's inputs are on the line right
    // before its flush in RasterizerD3D12::Draw. Shares the 2048-entry cap with that line.
    if (g_s15_flush_every_draw) {
        const u32 s15_pre_index = g_s15_flush_logs.fetch_add(1, std::memory_order_relaxed);
        if (s15_pre_index < kS15FlushLogCap) {
            u32 s15_pre_sampled_total = 0;
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind == SlotKind::Sampled) {
                    ++s15_pre_sampled_total;
                }
            }
            fmt::memory_buffer s15_pre_textures;
            u32 s15_pre_shown = 0;
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind != SlotKind::Sampled || s15_pre_shown >= 3) {
                    continue;
                }
                if (s15_pre_shown != 0) {
                    fmt::format_to(std::back_inserter(s15_pre_textures), ",");
                }
                fmt::format_to(std::back_inserter(s15_pre_textures), "0x{:x}",
                               static_cast<u64>(m_texture_cache.GetImageView(slot.view).GpuAddr()));
                ++s15_pre_shown;
            }
            const u64 s15_pre_rt0 =
                (framebuffer != nullptr && framebuffer->ColorBuffers()[0] != nullptr)
                    ? static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr())
                    : 0;
            LOG_WARNING(Render_D3D12,
                        "S15 pre-draw: rt0=0x{:x} rts={} att={:#x} verts={} textures={} [{}]",
                        s15_pre_rt0, num_rtvs, stored.pipeline->State().attachments[0].raw,
                        params.num_vertices, s15_pre_sampled_total,
                        fmt::to_string(s15_pre_textures));
            // Park this draw's identity so Draw() can log it next to its removal reason.
            g_s15_flush_draws.fetch_add(1, std::memory_order_relaxed);
            g_s15_flush_rt0.store(s15_pre_rt0, std::memory_order_relaxed);
            g_s15_flush_rts.store(num_rtvs, std::memory_order_relaxed);
            g_s15_flush_att.store(static_cast<u32>(stored.pipeline->State().attachments[0].raw),
                                  std::memory_order_relaxed);
            g_s15_flush_verts.store(params.num_vertices, std::memory_order_relaxed);
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
        // TEMP DIAGNOSTIC (session 15): log each distinct (render target, sampled B10G11R11 RT)
        // signature once, so the whole style-phase scene->composite chain is visible regardless
        // of log caps or timing.
        {
            static std::vector<u64> s15_chain_seen;
            u64 s15_sig = 1469598103934665603ull;
            bool s15_relevant = false;
            std::string s15_chain;
            for (u32 s15_i = 0; s15_i < num_rtvs; ++s15_i) {
                ImageView* const s15_rt_view =
                    framebuffer != nullptr ? framebuffer->ColorBuffers()[s15_i] : nullptr;
                if (s15_rt_view == nullptr || s15_rt_view->Resource() == nullptr) {
                    continue;
                }
                const D3D12_RESOURCE_DESC s15_rt_desc = s15_rt_view->Resource()->GetDesc();
                s15_chain += fmt::format("[rt {:#x} {:#x} {}x{} res=0x{:x}]",
                                         static_cast<u64>(s15_rt_view->GpuAddr()),
                                         static_cast<u32>(s15_rt_desc.Format), s15_rt_desc.Width,
                                         s15_rt_desc.Height,
                                         reinterpret_cast<u64>(s15_rt_view->Resource()));
                s15_sig = s15_sig * 1099511628211ull ^ static_cast<u64>(s15_rt_view->GpuAddr());
                if (s15_rt_desc.Format == DXGI_FORMAT_R11G11B10_FLOAT &&
                    s15_rt_desc.Width >= 960) {
                    s15_relevant = true;
                }
            }
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind != SlotKind::Sampled) {
                    continue;
                }
                ImageView& sv = m_texture_cache.GetImageView(slot.view);
                if (sv.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
                    continue;
                }
                s15_chain += fmt::format("[tex {:#x} {}x{} res=0x{:x}]",
                                         static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                         sv.size.height, reinterpret_cast<u64>(sv.Resource()));
                s15_sig = s15_sig * 1099511628211ull ^ static_cast<u64>(sv.GpuAddr());
                if (sv.size.width >= 960) {
                    s15_relevant = true;
                }
            }
            if (s15_relevant &&
                std::find(s15_chain_seen.begin(), s15_chain_seen.end(), s15_sig) ==
                    s15_chain_seen.end()) {
                if (s15_chain_seen.size() < 256) {
                    s15_chain_seen.push_back(s15_sig);
                    LOG_WARNING(Render_D3D12, "S15 chain #{}: verts={} depth={} {}",
                                s15_chain_seen.size(), params.num_vertices,
                                depth_view != nullptr ? 1 : 0, s15_chain);
                }
            }
        }
        // TEMP DIAGNOSTIC (session 15): who samples the 960x540 scene RT and the 1080p
        // scene source, to locate where the visible 3D scene is lost.
        {
            static u32 s15_rt_logs = 0;
            static std::chrono::steady_clock::time_point s15_rt_start =
                std::chrono::steady_clock::now();
            static std::chrono::steady_clock::time_point s15_src_next = s15_rt_start;
            static u32 s15_src_dumps = 0;
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind != SlotKind::Sampled) {
                    continue;
                }
                ImageView& sv = m_texture_cache.GetImageView(slot.view);
                const u64 gpu = static_cast<u64>(sv.GpuAddr());
                if (sv.size.width < 960 || sv.size.height < 540) {
                    continue;
                }
                ++s15_rt_logs;
                if (s15_rt_logs <= 128 || (s15_rt_logs % 4096) == 0) {
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
                // repeatedly after 20 minutes (every ~4 min, cap 6), when the under-lit frame
                // is actually on screen, to pair it with the post-draw probe of the same frame.
                const bool s15_src_late =
                    std::chrono::steady_clock::now() - s15_rt_start > std::chrono::minutes(20);
                if (s15_src_late && std::chrono::steady_clock::now() >= s15_src_next &&
                    s15_src_dumps < 6 &&
                    sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                    sv.size.width >= 960 && sv.Resource() != nullptr &&
                    !s15_is_current_rt(sv.Resource())) {
                    s15_src_next = std::chrono::steady_clock::now() + std::chrono::minutes(4);
                    ++s15_src_dumps;
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
            bool s15_stage_samples_scene_rt = false;
            for (const DrawSlot& slot : res_slots) {
                if (slot.kind != SlotKind::Sampled) {
                    continue;
                }
                ImageView& sv = m_texture_cache.GetImageView(slot.view);
                if (sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                    sv.size.width >= 1920) {
                    s15_has_scene_source = true;
                }
                if (static_cast<u64>(sv.GpuAddr()) == 0x5103e0000ull) {
                    s15_stage_samples_scene_rt = true;
                }
            }
            const bool s15_final_rt =
                s15_comp_res != nullptr && s15_comp_res->GetDesc().Width >= 1920 &&
                s15_comp_res->GetDesc().Format == DXGI_FORMAT_R10G10B10A2_TYPELESS;
            // TEMP DIAGNOSTIC (session 15, style-screen chain): the scene composite is the
            // blend-off 3-vertex draw into the 1920x1080 target that samples the 960x540
            // B10G11R11 scene RT at gpu 0x5103e0000; every other draw into that target is a
            // UI quad, which must not consume the composite's stage-dump budget.
            const bool s15_stage_scene_composite =
                s15_comp_res != nullptr && s15_comp_res->GetDesc().Width == 1920 &&
                s15_comp_res->GetDesc().Height == 1080 && params.num_vertices == 3 &&
                !stored.pipeline->State().attachments[0].enable.Value() &&
                s15_stage_samples_scene_rt;
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
                // TEMP DIAGNOSTIC (session 15): also dump the style-phase scene source the
                // composite samples (B10G11R11, any width >= 960), bounded to ~once per 45 s:
                // this is the exact image the composite reads, so its content at composite
                // time decides whether the loss is the scene RT or the composite path.
                static std::chrono::steady_clock::time_point s15_comp_src_next =
                    std::chrono::steady_clock::now();
                static u32 s15_comp_src_dumps = 0;
                const bool s15_comp_src_due =
                    std::chrono::steady_clock::now() >= s15_comp_src_next &&
                    s15_comp_src_dumps < 24;
                for (const DrawSlot& slot : res_slots) {
                    if (slot.kind != SlotKind::Sampled || s15_comp_dumps >= 2) {
                        continue;
                    }
                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                    if (sv.Resource() != nullptr &&
                        sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                        ((sv.size.width >= 1920 && !s15_is_current_rt(sv.Resource())) ||
                         (s15_comp_src_due && sv.size.width >= 960 &&
                          !s15_is_current_rt(sv.Resource())))) {
                        const bool s15_comp_force =
                            !s15_comp_late_dumped &&
                            std::chrono::steady_clock::now() - s15_comp_start >
                                std::chrono::minutes(20);
                        if (s15_comp_force) {
                            s15_comp_late_dumped = true;
                        }
                        if (s15_comp_src_due && sv.size.width >= 960 && sv.size.width < 1920) {
                            s15_comp_src_next =
                                std::chrono::steady_clock::now() + std::chrono::seconds(45);
                            ++s15_comp_src_dumps;
                            LOG_WARNING(Render_D3D12,
                                        "S15 composite scene source: img={} gpu={:#x} {}x{} "
                                        "fmt={:#x} res=0x{:x}",
                                        sv.image_id.index, static_cast<u64>(sv.GpuAddr()),
                                        sv.size.width, sv.size.height,
                                        static_cast<u32>(sv.format),
                                        reinterpret_cast<u64>(sv.Resource()));
                        }
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            sv.image_id.index, s15_comp_force || s15_comp_src_due);
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
                // TEMP DIAGNOSTIC (session 15, style-screen chain): one-shot, late-gated stage
                // dumps for the scene composite. Its sampled textures are dumped in slot order
                // under unique ids (the 1x1 parameter texture, the 960x540 scene RT, the
                // 240x136 bloom) so a dark RT can be attributed to a stage instead of guessed,
                // and the flag for the post-draw dump of the composite target itself is armed
                // here. force=true so the per-image dedupe in RecordTextureContentDump cannot
                // skip a stage.
                static bool s15_stage_composite_done = false;
                static std::chrono::steady_clock::time_point s15_stage_start =
                    std::chrono::steady_clock::now();
                if (!s15_stage_composite_done && s15_stage_scene_composite &&
                    std::chrono::steady_clock::now() - s15_stage_start >
                        std::chrono::minutes(20)) {
                    s15_stage_composite_done = true;
                    u32 s15_stage_inputs = 0;
                    for (const DrawSlot& slot : res_slots) {
                        if (slot.kind != SlotKind::Sampled || s15_stage_inputs >= 3) {
                            continue;
                        }
                        ImageView& sv = m_texture_cache.GetImageView(slot.view);
                        if (sv.Resource() == nullptr) {
                            continue;
                        }
                        ++s15_stage_inputs;
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            980020u + s15_stage_inputs, /*force=*/true);
                    }
                    g_s15_composite_target_dump = true;
                    LOG_WARNING(Render_D3D12, "S15 composite stage dumps: ids=980021..980024");
                }
                // S18 cycle 2 (Phase 0.2.2): the composite is the ONLY consumer of the exposure
                // texel, and it is a NO-depth draw - the depth-gated S18 block can never see it
                // (1.0.271: the exposure timeline never fired; the only readback came from the
                // one-shot S15 stage dump). Arm the periodic dumps from here instead, every 60 s:
                //  (a) every sampled 1x1 R32G32B32A32 (the exposure texel), ids 990501+;
                //  (b) each cached image over the 1x1's guest bytes (the 8x1 reduction RT and its
                //      siblings) read back raw, ids 990601+;
                //  (c) the composite target post-draw, ids 990201+ - the 1.0.271 one-shot dump
                //      showed it nearly black (nonzero 6664 of 8294400) while the visible frame
                //      is bright, so the visible brightness must enter DOWNSTREAM of this pass.
                {
                    static std::chrono::steady_clock::time_point s18c_last =
                        std::chrono::steady_clock::now();
                    static u32 s18c_dumps = 0;
                    static u32 s18c_sib_dumps = 0;
                    static bool s18c_census_done = false;
                    if (s18c_dumps < 40 &&
                        std::chrono::steady_clock::now() - s18c_last >
                            std::chrono::seconds(60)) {
                        s18c_last = std::chrono::steady_clock::now();
                        for (const DrawSlot& slot : res_slots) {
                            if (slot.kind != SlotKind::Sampled) {
                                continue;
                            }
                            ImageView& xv = m_texture_cache.GetImageView(slot.view);
                            if (xv.size.width != 1 || xv.size.height != 1 ||
                                xv.format !=
                                    VideoCore::Surface::PixelFormat::R32G32B32A32_FLOAT ||
                                xv.Resource() == nullptr) {
                                continue;
                            }
                            ++s18c_dumps;
                            const u32 s18c_id = 990500u + s18c_dumps;
                            (void)D3D12::RecordTextureContentDump(
                                m_device.GetDevice(), m_command_list, xv.Resource(),
                                static_cast<u32>(xv.format), static_cast<u64>(xv.GpuAddr()),
                                s18c_id, /*force=*/true);
                            LOG_WARNING(Render_D3D12,
                                        "S18B EXPO dump: id={} img={} gpu={:#x} res=0x{:x}",
                                        s18c_id, xv.image_id.index,
                                        static_cast<u64>(xv.GpuAddr()),
                                        reinterpret_cast<u64>(xv.Resource()));
                            if (gpu_memory != nullptr) {
                                const std::optional<DAddr> s18c_cpu =
                                    gpu_memory->GpuToCpuAddress(
                                        static_cast<GPUVAddr>(xv.GpuAddr()));
                                if (!s18c_census_done && s18c_cpu) {
                                    s18c_census_done = true;
                                    LOG_WARNING(Render_D3D12,
                                                "S18B EXPO census: gpu={:#x} cpu={:#x}",
                                                static_cast<u64>(xv.GpuAddr()),
                                                static_cast<u64>(*s18c_cpu));
                                }
                                // S18 cycle 3: the window covers 1 KiB from the 1x1's guest base
                                // so the reduction draw's REAL input - the 128x1 R32G32B32A32
                                // luminance array at +0x80 (img 3700 in the 1.0.272 run, sampled by
                                // the 8x1 reduction) - is enumerated and dumped too. Cap 24 dumps
                                // (three images per 60 s round = ~8 rounds of timeline).
                                if (s18c_cpu && s18c_sib_dumps < 24) {
                                    m_texture_cache.ForEachImageInRegion(
                                        *s18c_cpu, 0x400,
                                        [&](VideoCommon::ImageId s18c_image_id, auto& s18c_image) {
                                            if (s18c_sib_dumps >= 24) {
                                                return;
                                            }
                                            ++s18c_sib_dumps;
                                            const u32 s18c_sib_id = 990600u + s18c_sib_dumps;
                                            if (s18c_image.Handle() != nullptr) {
                                                (void)D3D12::RecordTextureContentDump(
                                                    m_device.GetDevice(), m_command_list,
                                                    s18c_image.Handle(),
                                                    static_cast<u32>(s18c_image.info.format),
                                                    static_cast<u64>(s18c_image.gpu_addr),
                                                    s18c_sib_id, /*force=*/true);
                                            }
                                            LOG_WARNING(
                                                Render_D3D12,
                                                "S18B EXPO sibling: id={} img={} gpu={:#x} "
                                                "{}x{} fmt={:#x} flags={:#x} tick={}",
                                                s18c_sib_id, s18c_image_id.index,
                                                static_cast<u64>(s18c_image.gpu_addr),
                                                s18c_image.info.size.width,
                                                s18c_image.info.size.height,
                                                static_cast<u32>(s18c_image.info.format),
                                                static_cast<u32>(s18c_image.flags),
                                                s18c_image.modification_tick);
                                        });
                                }
                            }
                            break;
                        }
                        if (g_s18_target_res == nullptr && s15_comp_res != nullptr &&
                            g_s18_target_used < 40) {
                            g_s18_target_res = s15_comp_res;
                            LOG_WARNING(Render_D3D12, "S18B target dump armed: res=0x{:x}",
                                        reinterpret_cast<u64>(s15_comp_res));
                        }
                    }
                }
                // S18 cycle 5 (Phase 0.2.5): matched input set for every capture phase. The
                // composite's other two inputs - the 960x540 B10G11R11 scene RT and the 240x136
                // blur - are dumped on the same 60 s cadence as the exposure texel, so a phase's
                // visible frame (capture) can be compared directly against the exact inputs the
                // 3-vertex composite consumed. Ids 990801+ (scene) / 990901+ (blur); cap 30 each.
                {
                    static std::chrono::steady_clock::time_point s18d_last =
                        std::chrono::steady_clock::now();
                    static u32 s18d_scene_dumps = 0;
                    static u32 s18d_blur_dumps = 0;
                    if ((s18d_scene_dumps < 30 || s18d_blur_dumps < 30) &&
                        std::chrono::steady_clock::now() - s18d_last >
                            std::chrono::seconds(60)) {
                        s18d_last = std::chrono::steady_clock::now();
                        for (const DrawSlot& slot : res_slots) {
                            if (slot.kind != SlotKind::Sampled) {
                                continue;
                            }
                            ImageView& sv = m_texture_cache.GetImageView(slot.view);
                            if (sv.Resource() == nullptr ||
                                sv.format !=
                                    VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
                                continue;
                            }
                            if (sv.size.width == 960 && sv.size.height == 540 &&
                                s18d_scene_dumps < 30) {
                                ++s18d_scene_dumps;
                                const u32 s18d_id = 990800u + s18d_scene_dumps;
                                (void)D3D12::RecordTextureContentDump(
                                    m_device.GetDevice(), m_command_list, sv.Resource(),
                                    static_cast<u32>(sv.format),
                                    static_cast<u64>(sv.GpuAddr()), s18d_id, /*force=*/true);
                                LOG_WARNING(Render_D3D12,
                                            "S18D scene input: id={} img={} gpu={:#x} {}x{}",
                                            s18d_id, sv.image_id.index,
                                            static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                            sv.size.height);
                            } else if (sv.size.width == 240 && sv.size.height == 136 &&
                                       s18d_blur_dumps < 30) {
                                ++s18d_blur_dumps;
                                const u32 s18d_id = 990900u + s18d_blur_dumps;
                                (void)D3D12::RecordTextureContentDump(
                                    m_device.GetDevice(), m_command_list, sv.Resource(),
                                    static_cast<u32>(sv.format),
                                    static_cast<u64>(sv.GpuAddr()), s18d_id, /*force=*/true);
                                LOG_WARNING(Render_D3D12,
                                            "S18D blur input: id={} img={} gpu={:#x} {}x{}",
                                            s18d_id, sv.image_id.index,
                                            static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                            sv.size.height);
                            }
                        }
                    }
                }
                // S18 cycle 5 (Phase 0.2.5): PHASE-GATED SAME-RUN EXPOSURE A/B (the clean causal
                // test the plan asks for). The auto-exposure is display-referred and
                // self-referencing - a dark composite yields an empty histogram, which yields
                // ~zero exposure, which keeps the composite dark - and the settled phase sits at
                // a stable 3/64 that only leaves the brightest HDR objects (the fish fins)
                // visible. Once past guest frame 12500 (after the load, camera settled), force
                // the composite's sampled 1x1 to 1.0 for 150 s, then to 8.0 for 150 s, then stop
                // and let the game's loop resume: the captures before/during/after sit at the
                // SAME camera phase, so the difference is purely the exposure. Bounded: one
                // window, and the clears are logged at a 1-in-64 cadence.
                {
                    static bool s18ab_started = false;
                    static bool s18ab_second = false;
                    static bool s18ab_done = false;
                    static std::chrono::steady_clock::time_point s18ab_t0{};
                    static u32 s18ab_clears = 0;
                    if (!s18ab_started && g_s18_guest_frames > 12500) {
                        s18ab_started = true;
                        s18ab_t0 = std::chrono::steady_clock::now();
                        LOG_WARNING(Render_D3D12,
                                    "S18 EXPOSURE A/B: forcing the composite 1x1 to 1.0 for 150 s "
                                    "(guest_frame={})",
                                    g_s18_guest_frames);
                    }
                    const auto s18ab_elapsed =
                        std::chrono::steady_clock::now() - s18ab_t0;
                    if (s18ab_started && !s18ab_done && !s18ab_second &&
                        s18ab_elapsed > std::chrono::seconds(150)) {
                        s18ab_second = true;
                        LOG_WARNING(Render_D3D12,
                                    "S18 EXPOSURE A/B: switching to 8.0 for 150 s (clears={})",
                                    s18ab_clears);
                    }
                    if (s18ab_started && !s18ab_done && s18ab_second &&
                        s18ab_elapsed > std::chrono::seconds(300)) {
                        s18ab_done = true;
                        LOG_WARNING(Render_D3D12,
                                    "S18 EXPOSURE A/B: window closed after {} clears (game loop "
                                    "resumes)",
                                    s18ab_clears);
                    }
                    const bool s18ab_armed = s18ab_started && !s18ab_done;
                    if (s18ab_armed) {
                        const f32 s18ab_value = s18ab_second ? 8.0f : 1.0f;
                        for (const DrawSlot& slot : res_slots) {
                            if (slot.kind != SlotKind::Sampled) {
                                continue;
                            }
                            ImageView& xv = m_texture_cache.GetImageView(slot.view);
                            if (xv.size.width != 1 || xv.size.height != 1 ||
                                xv.format !=
                                    VideoCore::Surface::PixelFormat::R32G32B32A32_FLOAT ||
                                xv.Resource() == nullptr) {
                                continue;
                            }
                            ID3D12Resource* const s18ab_res = xv.Resource();
                            // Own 1-descriptor RTV heap (the runtime's is private).
                            static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> s18ab_heap;
                            if (!s18ab_heap) {
                                D3D12_DESCRIPTOR_HEAP_DESC s18ab_hd{};
                                s18ab_hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
                                s18ab_hd.NumDescriptors = 1;
                                s18ab_hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
                                (void)m_device.GetDevice()->CreateDescriptorHeap(
                                    &s18ab_hd, IID_PPV_ARGS(&s18ab_heap));
                            }
                            if (s18ab_heap) {
                                const D3D12_CPU_DESCRIPTOR_HANDLE s18ab_rtv =
                                    s18ab_heap->GetCPUDescriptorHandleForHeapStart();
                                D3D12_RENDER_TARGET_VIEW_DESC s18ab_rd{};
                                s18ab_rd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                                s18ab_rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                                m_device.GetDevice()->CreateRenderTargetView(
                                    s18ab_res, &s18ab_rd, s18ab_rtv);
                                m_command_list.Transition(s18ab_res,
                                                          D3D12_RESOURCE_STATE_COMMON,
                                                          D3D12_RESOURCE_STATE_RENDER_TARGET);
                                const f32 s18ab_color[4] = {s18ab_value, s18ab_value,
                                                            s18ab_value, 1.0f};
                                m_command_list.ClearRenderTargetView(s18ab_rtv, s18ab_color);
                                m_command_list.Transition(s18ab_res,
                                                          D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                          D3D12_RESOURCE_STATE_COMMON);
                                ++s18ab_clears;
                                if (s18ab_clears <= 4 || (s18ab_clears % 64) == 0) {
                                    LOG_WARNING(Render_D3D12,
                                                "S18 EXPOSURE A/B: forced {} on img={} gpu={:#x} "
                                                "(n={})",
                                                s18ab_value, xv.image_id.index,
                                                static_cast<u64>(xv.GpuAddr()), s18ab_clears);
                                }
                            }
                            break;
                        }
                    }
                }
                // TEMP DIAGNOSTIC (session 15, style-screen chain): the UI panel/icon textures
                // the compositor samples (small textures, and not the scene composite itself).
                // One unique id per resource so the missing dark panel's source can be
                // inspected; the loading-phase draws are excluded by the late gate and the
                // per-resource memory so repeated icons cannot consume the budget.
                static ID3D12Resource* s15_stage_ui_res[4] = {};
                static u32 s15_stage_ui_dumps = 0;
                if (!s15_stage_scene_composite && s15_stage_ui_dumps < 4 &&
                    std::chrono::steady_clock::now() - s15_stage_start >
                        std::chrono::minutes(20)) {
                    for (const DrawSlot& slot : res_slots) {
                        if (slot.kind != SlotKind::Sampled || s15_stage_ui_dumps >= 4) {
                            continue;
                        }
                        ImageView& sv = m_texture_cache.GetImageView(slot.view);
                        if (sv.Resource() == nullptr || sv.size.width > 256) {
                            continue;
                        }
                        bool s15_stage_ui_seen = false;
                        for (u32 s15_stage_ui_i = 0; s15_stage_ui_i < 4; ++s15_stage_ui_i) {
                            if (s15_stage_ui_res[s15_stage_ui_i] == sv.Resource()) {
                                s15_stage_ui_seen = true;
                                break;
                            }
                        }
                        if (s15_stage_ui_seen) {
                            continue;
                        }
                        s15_stage_ui_res[s15_stage_ui_dumps] = sv.Resource();
                        const u32 s15_stage_ui_id = 980031u + s15_stage_ui_dumps;
                        ++s15_stage_ui_dumps;
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            s15_stage_ui_id, /*force=*/true);
                        LOG_WARNING(Render_D3D12,
                                    "S15 ui texture dump: id={} img={} gpu={:#x} {}x{} fmt={:#x}",
                                    s15_stage_ui_id, sv.image_id.index,
                                    static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                    sv.size.height, static_cast<u32>(sv.format));
                    }
                }
                // TEMP DIAGNOSTIC (session 15): the early rate limit below suppressed the
                // style-phase compositor lines, so also log the compositor's full sampled
                // texture list a bounded number of times after 20 minutes, to see WHICH image
                // the final composite actually samples at style phase (the 960x540 scene RT
                // with content, or the dynamic-res ~1300x731 source).
                static std::chrono::steady_clock::time_point s15_comp_late_start =
                    std::chrono::steady_clock::now();
                static u32 s15_comp_late_logs = 0;
                const bool s15_comp_late_log =
                    std::chrono::steady_clock::now() - s15_comp_late_start >
                        std::chrono::minutes(20) &&
                    s15_comp_late_logs < 12;
                if (s15_comp_late_log) {
                    ++s15_comp_late_logs;
                }
                if (s15_comp_n < 8 || (s15_comp_n % 4096 == 0 && s15_comp_n < 32768) ||
                    s15_comp_late_log) {
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
                            sv.size.width >= 1920 && !s15_is_current_rt(sv.Resource())) {
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
            static bool s15_big_dump_done = false;
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
            const bool s15_big_dump_now = s15_late_phase && !s15_big_dump_done &&
                                          params.num_vertices >= 4000 && s15_sampled >= 5;
            const bool s15_dump_now =
                scene3d_index == 0 || (s15_sampled >= 5 && !s15_early_tex_dump_done) ||
                (s15_late_phase && !s15_late_dump_done && s15_sampled >= 5) || s15_big_dump_now;
            if (s15_dump_now) {
                if (scene3d_index != 0 && s15_sampled >= 5) {
                    s15_early_tex_dump_done = true;
                }
                if (scene3d_index != 0 && s15_late_phase) {
                    s15_late_dump_done = true;
                }
                if (s15_big_dump_now) {
                    s15_big_dump_done = true;
                }
                LOG_WARNING(Render_D3D12,
                            "S15 scene slots: total={} sampled={} texbuf={} imgbuf={} storage={} "
                            "image={} verts={} late={}",
                            res_slots.size(), s15_sampled, s15_texel, s15_image_buffer,
                            s15_storage, s15_image, params.num_vertices, s15_late_phase ? 1 : 0);
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
                    if (sv.Resource() != nullptr && !s15_is_current_rt(sv.Resource())) {
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            sv.image_id.index, true);
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
                static bool s15_lut_late_done = false;
                static std::chrono::steady_clock::time_point s15_lut_start =
                    std::chrono::steady_clock::now();
                const bool s15_lut_late =
                    std::chrono::steady_clock::now() - s15_lut_start > std::chrono::minutes(20);
                const bool s15_lut_run = !s15_lut_dumped || (s15_lut_late && !s15_lut_late_done);
                u32 s15_lut_slots = 0;
                for (const DrawSlot& slot : res_slots) {
                    if (slot.kind != SlotKind::Sampled) {
                        continue;
                    }
                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                    // The 256x256 bank sub-view must reach the probe (and the white-clear
                    // below) even after the one-shot LUT gate above has closed - in 1.0.253
                    // the gate fired on a DIFFERENT 128x128 LUT at t=594 and the bank view
                    // was therefore never probed, so the causal probe never ran. Everything
                    // else keeps the gate's original semantics.
                    const bool s17_bank_view =
                        static_cast<u64>(sv.GpuAddr()) == 0x5eb340000ull;
                    // S17 (1.0.266 harvest): the bank-view budget used to burn all 8 probes in
                    // 16 ms; it is now a 60-second cadence with 24 probes, so the LUT resource
                    // content gets a timeline across the whole run (did it populate? when? does
                    // it revert?). Non-bank LUTs keep the original one-shot/late semantics.
                    static u32 s17_bank_views = 0;
                    static std::chrono::steady_clock::time_point s17_bank_last_probe =
                        std::chrono::steady_clock::now();
                    const bool s17_bank_ready =
                        std::chrono::steady_clock::now() - s17_bank_last_probe >
                        std::chrono::seconds(60);
                    const bool s17_bank_probe =
                        s17_bank_view && s17_bank_views < 24 && s17_bank_ready;
                    if ((!s15_lut_run && !s17_bank_probe) || s15_lut_slots >= 4) {
                        continue;
                    }
                    if (s17_bank_probe) {
                        ++s17_bank_views;
                        s17_bank_last_probe = std::chrono::steady_clock::now();
                    }
                    const bool is_lut =
                        (sv.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                         sv.size.width <= 512) ||
                        sv.format == VideoCore::Surface::PixelFormat::R16G16_FLOAT;
                    if (!is_lut) {
                        continue;
                    }
                    ++s15_lut_slots;
                    if (sv.Resource() != nullptr && !s15_is_current_rt(sv.Resource())) {
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, sv.Resource(),
                            static_cast<u32>(sv.format), static_cast<u64>(sv.GpuAddr()),
                            sv.image_id.index, true);
                    }
                    // S17: read the sampled LUT bytes through the GPU-VA accessor. The previous
                    // GpuToCpuAddress() + ReadBlockUnsafe pair produced stale/zero readings for
                    // this bank (flagged in the S17 forensics), and a 64-byte prefix at the image
                    // base is weak evidence about the whole 256 KiB view - widen to 4 KiB. This
                    // makes 'S15 LUT probe nz' meaningful: nonzero here == the guest bytes behind
                    // the sampled view exist.
                    std::array<u8, 4096> lut_bytes{};
                    u32 lut_nonzero = 0;
                    u32 lut_read = 0;
                    std::string lut_hex;
                    if (gpu_memory != nullptr) {
                        gpu_memory->ReadBlock(static_cast<GPUVAddr>(sv.GpuAddr()), lut_bytes.data(),
                                              lut_bytes.size(),
                                              VideoCommon::CacheType::NoTextureCache);
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
                    // S17 cycle 4: `res=` is the sampled view's resource pointer. The 1.0.268
                    // run read nz=0 from every probe while the frame was lit - logging the
                    // pointer per probe shows whether the incarnation the dump reads changes
                    // (resource replaced/recreated) or stays constant (dump path suspect).
                    LOG_WARNING(Render_D3D12,
                                "S15 LUT probe: img={} gpu={:#x} {}x{} fmt={:#x} res=0x{:x} "
                                "nz={}/{} hex={} verts={}",
                                sv.image_id.index, static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                sv.size.height, static_cast<u32>(sv.format),
                                reinterpret_cast<u64>(sv.Resource()), lut_nonzero, lut_read,
                                lut_hex, params.num_vertices);
                    // TEMP DIAGNOSTIC (session 17) - CAUSAL PROBE: is the all-zero lighting LUT
                    // what keeps the 3D scene dark? One-shot, in-stream, before the draw that
                    // samples it: clear the bank sub-view to white so every lookup returns 1.0.
                    // If the character lights up, the LUT is causally responsible and the next
                    // step is finding its (currently unlogged) producer; if nothing changes, the
                    // LUT is irrelevant and the darkness is elsewhere. Diagnostic only - remove
                    // once the producer is found. Bounded: exactly one clear per process.
                    static bool s17_lut_white_done = false;
                    // The probe has served its purpose: with the FlushCaching() fix the LUT now
                    // receives real guest data, so clearing it would destroy correct content.
                    // Kept disabled (not deleted) until the whole S17 set is cleaned up.
                    static constexpr bool s17_enable_white_probe = false;
                    if (s17_enable_white_probe && !s17_lut_white_done &&
                        static_cast<u64>(sv.GpuAddr()) == 0x5eb340000ull &&
                        sv.Resource() != nullptr) {
                        s17_lut_white_done = true;
                        ID3D12Resource* const s17_lut_res = sv.Resource();
                        // Own 1-descriptor RTV heap: TextureCacheRuntime::rtv_heap is private and
                        // this probe must not consume the runtime's descriptor budget.
                        static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> s17_rtv_heap;
                        if (!s17_rtv_heap) {
                            D3D12_DESCRIPTOR_HEAP_DESC s17_heap_desc{};
                            s17_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
                            s17_heap_desc.NumDescriptors = 1;
                            s17_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
                            (void)m_device.GetDevice()->CreateDescriptorHeap(
                                &s17_heap_desc, IID_PPV_ARGS(&s17_rtv_heap));
                        }
                        if (s17_rtv_heap) {
                            const D3D12_CPU_DESCRIPTOR_HANDLE s17_rtv =
                                s17_rtv_heap->GetCPUDescriptorHandleForHeapStart();
                            D3D12_RENDER_TARGET_VIEW_DESC s17_rtv_desc{};
                            s17_rtv_desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
                            s17_rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                            m_device.GetDevice()->CreateRenderTargetView(s17_lut_res, &s17_rtv_desc,
                                                                         s17_rtv);
                            m_command_list.Transition(s17_lut_res, D3D12_RESOURCE_STATE_COMMON,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
                            const f32 s17_white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                            m_command_list.ClearRenderTargetView(s17_rtv, s17_white);
                            m_command_list.Transition(s17_lut_res,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                      D3D12_RESOURCE_STATE_COMMON);
                            LOG_WARNING(Render_D3D12,
                                        "S17 LUT white probe: cleared gpu=0x5eb340000 res=0x{:x}",
                                        reinterpret_cast<u64>(s17_lut_res));
                        }
                    }
                    // ONE-SHOT: enumerate every cached image registered over the 27 MB LUT
                    // bank (cpu 0x29426000) once the 256x256 LUT sub-view itself is the
                    // probed image, so aliasing/sibling coverage of that region is visible
                    // in one run.
                    static bool s15_lut_siblings_done = false;
                    if (!s15_lut_siblings_done &&
                        static_cast<u64>(sv.GpuAddr()) == 0x5eb340000ull) {
                        s15_lut_siblings_done = true;
                        LOG_WARNING(
                            Render_D3D12,
                            "S15 LUT region: cpu=0x29426000 size=0x1a40000 gpu_modified={}",
                            m_texture_cache.IsRegionGpuModified(0x29426000u, 0x1a40000u) ? 1
                                                                                          : 0);
                        u32 s15_sib_count = 0;
                        m_texture_cache.ForEachImageInRegion(
                            0x29426000u, 0x1a40000u,
                            [&](VideoCommon::ImageId image_id, auto& image) {
                                if (s15_sib_count >= 8) {
                                    return;
                                }
                                ++s15_sib_count;
                                LOG_WARNING(
                                    Render_D3D12,
                                    "S15 LUT sibling: id={} gpu={:#x} flags={:#x} size={} "
                                    "fmt={:#x} tick={}",
                                    image_id.index, static_cast<u64>(image.gpu_addr),
                                    static_cast<u32>(image.flags), image.guest_size_bytes,
                                    static_cast<u32>(image.info.format), image.modification_tick);
                            });
                    }
                }
                if (s15_lut_run && s15_lut_slots > 0) {
                    if (!s15_lut_dumped) {
                        s15_lut_dumped = true;
                    }
                    if (s15_lut_late) {
                        s15_lut_late_done = true;
                    }
                }
            }
            // S18 (Phase 0.2.1) - replaces the S17 "EXPOSURE value" guest read, which always
            // printed zeros (GPU-authored texture - the standing trap). The live value can only
            // come from a resource readback, so every 60 s force a texdump of EVERY distinct
            // sampled 1x1 R32G32B32A32 view (the composite samples one such texel; a second 1x1
            // is sampled by the big draws). The .bin sidecar and the "S18 EXPOSURE readback"
            // line in WriteTextureContentDumps carry the floats. Ids 990001+; cap 48 dumps
            // (2 per minute -> ~24 min of timeline). A one-shot census enumerates every cached
            // image registered over the first 1x1's guest bytes (the alias/incarnation set).
            {
                static std::chrono::steady_clock::time_point s18_expo_last =
                    std::chrono::steady_clock::now();
                static u32 s18_expo_dumps = 0;
                static bool s18_expo_census_done = false;
                if (s18_expo_dumps < 48 &&
                    std::chrono::steady_clock::now() - s18_expo_last >
                        std::chrono::seconds(60)) {
                    s18_expo_last = std::chrono::steady_clock::now();
                    for (const DrawSlot& slot : res_slots) {
                        if (slot.kind != SlotKind::Sampled) {
                            continue;
                        }
                        ImageView& xv = m_texture_cache.GetImageView(slot.view);
                        if (xv.size.width != 1 || xv.size.height != 1 ||
                            xv.format != VideoCore::Surface::PixelFormat::R32G32B32A32_FLOAT ||
                            xv.Resource() == nullptr) {
                            continue;
                        }
                        ++s18_expo_dumps;
                        const u32 s18_expo_id = 990000u + s18_expo_dumps;
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, xv.Resource(),
                            static_cast<u32>(xv.format), static_cast<u64>(xv.GpuAddr()),
                            s18_expo_id, /*force=*/true);
                        LOG_WARNING(Render_D3D12,
                                    "S18 EXPO dump: id={} img={} gpu={:#x} res=0x{:x}",
                                    s18_expo_id, xv.image_id.index,
                                    static_cast<u64>(xv.GpuAddr()),
                                    reinterpret_cast<u64>(xv.Resource()));
                        if (!s18_expo_census_done && gpu_memory != nullptr) {
                            s18_expo_census_done = true;
                            const std::optional<DAddr> s18_expo_cpu =
                                gpu_memory->GpuToCpuAddress(
                                    static_cast<GPUVAddr>(xv.GpuAddr()));
                            LOG_WARNING(Render_D3D12, "S18 EXPO census: gpu={:#x} cpu={:#x}",
                                        static_cast<u64>(xv.GpuAddr()),
                                        s18_expo_cpu ? static_cast<u64>(*s18_expo_cpu) : 0ull);
                            if (s18_expo_cpu) {
                                m_texture_cache.ForEachImageInRegion(
                                    *s18_expo_cpu, 16,
                                    [&](VideoCommon::ImageId s18_image_id, auto& s18_image) {
                                        LOG_WARNING(
                                            Render_D3D12,
                                            "S18 EXPO census: img={} gpu={:#x} size={} flags={:#x} "
                                            "{}x{} fmt={:#x} tick={}",
                                            s18_image_id.index,
                                            static_cast<u64>(s18_image.gpu_addr),
                                            s18_image.guest_size_bytes,
                                            static_cast<u32>(s18_image.flags),
                                            s18_image.info.size.width,
                                            s18_image.info.size.height,
                                            static_cast<u32>(s18_image.info.format),
                                            s18_image.modification_tick);
                                    });
                            }
                        }
                    }
                }
            }
            // S17 cycle 4: the MASK and EXPO white probes are DISABLED. Both served their
            // purpose (mask: no step change, addendum 23; exposure: inconclusive by design,
            // addendum 25) and the EXPO probe fired at t=766 in the 1.0.268 run, i.e. the lit
            // frame has not yet been proven WITHOUT probe assistance. Keep the code for
            // reference; do not re-enable without a same-run A/B.
            static constexpr bool s17_enable_causal_probes = false;
            // TEMP DIAGNOSTIC (S17): MASK causal probe. The R8 mask family's SAMPLED member
            // (gpu 0x510fa0000, 1038x584, dxgi 0x1b TYPELESS) reads all-zero while the 960x540
            // member (0x510f30000) IS written by 6-vertex draws - the dynamic-res family
            // incoherence. The big 3D colour draws sample the ZERO one, so if the mask is a
            // shading multiplier the scene is dark for a second, independent reason. One-shot
            // clear to white, the same methodology that proved the LUT gate. If the scene
            // brightens, the mask joins the LUT as a proven factor and the fix target is the
            // family incoherence rather than the bank.
            {
                static bool s17_mask_white_done = false;
                if (s17_enable_causal_probes && !s17_mask_white_done) {
                    for (const DrawSlot& slot : res_slots) {
                        if (slot.kind != SlotKind::Sampled) {
                            continue;
                        }
                        ImageView& mv = m_texture_cache.GetImageView(slot.view);
                        if (static_cast<u64>(mv.GpuAddr()) != 0x510fa0000ull ||
                            mv.Resource() == nullptr) {
                            continue;
                        }
                        s17_mask_white_done = true;
                        ID3D12Resource* const s17_mask_res = mv.Resource();
                        static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> s17_mask_heap;
                        if (!s17_mask_heap) {
                            D3D12_DESCRIPTOR_HEAP_DESC s17_mhd{};
                            s17_mhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
                            s17_mhd.NumDescriptors = 1;
                            s17_mhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
                            (void)m_device.GetDevice()->CreateDescriptorHeap(
                                &s17_mhd, IID_PPV_ARGS(&s17_mask_heap));
                        }
                        if (s17_mask_heap) {
                            const D3D12_CPU_DESCRIPTOR_HANDLE s17_mrtv =
                                s17_mask_heap->GetCPUDescriptorHandleForHeapStart();
                            D3D12_RENDER_TARGET_VIEW_DESC s17_mrd{};
                            s17_mrd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                            s17_mrd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                            m_device.GetDevice()->CreateRenderTargetView(s17_mask_res, &s17_mrd,
                                                                         s17_mrtv);
                            m_command_list.Transition(s17_mask_res,
                                                      D3D12_RESOURCE_STATE_COMMON,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
                            const f32 s17_mwhite[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                            m_command_list.ClearRenderTargetView(s17_mrtv, s17_mwhite);
                            m_command_list.Transition(s17_mask_res,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                      D3D12_RESOURCE_STATE_COMMON);
                            LOG_WARNING(Render_D3D12,
                                        "S17 MASK white probe: cleared gpu=0x510fa0000 res=0x{:x}",
                                        reinterpret_cast<u64>(s17_mask_res));
                        }
                        break;
                    }
                }
            }
            // TEMP DIAGNOSTIC (S17): EXPOSURE causal probe. The scene composite samples a 1x1
            // R32G32B32A32_FLOAT texture (img=980021, gpu 0x5d5445100) whose decoded values are
            // small (~0.05-0.87) - if it is an exposure/tonemap scale it is the last input that
            // can darken the WHOLE scene uniformly. One-shot: clear it to 1.0 and watch the frame.
            // Keyed on the view's signature (1x1 + R32G32B32A32_FLOAT) rather than a hardcoded
            // address, because ids and addresses drift between runs.
            {
                static bool s17_exp_done = false;
                if (s17_enable_causal_probes && !s17_exp_done) {
                    for (const DrawSlot& slot : res_slots) {
                        if (slot.kind != SlotKind::Sampled) {
                            continue;
                        }
                        ImageView& ev = m_texture_cache.GetImageView(slot.view);
                        if (ev.format != VideoCore::Surface::PixelFormat::R32G32B32A32_FLOAT ||
                            ev.size.width != 1 || ev.size.height != 1 ||
                            ev.Resource() == nullptr) {
                            continue;
                        }
                        s17_exp_done = true;
                        ID3D12Resource* const s17_exp_res = ev.Resource();
                        static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> s17_exp_heap;
                        if (!s17_exp_heap) {
                            D3D12_DESCRIPTOR_HEAP_DESC s17_ehd{};
                            s17_ehd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
                            s17_ehd.NumDescriptors = 1;
                            s17_ehd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
                            (void)m_device.GetDevice()->CreateDescriptorHeap(
                                &s17_ehd, IID_PPV_ARGS(&s17_exp_heap));
                        }
                        if (s17_exp_heap) {
                            const D3D12_CPU_DESCRIPTOR_HANDLE s17_ertv =
                                s17_exp_heap->GetCPUDescriptorHandleForHeapStart();
                            D3D12_RENDER_TARGET_VIEW_DESC s17_erd{};
                            s17_erd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                            s17_erd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                            m_device.GetDevice()->CreateRenderTargetView(s17_exp_res, &s17_erd,
                                                                         s17_ertv);
                            m_command_list.Transition(s17_exp_res, D3D12_RESOURCE_STATE_COMMON,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
                            const f32 s17_ewhite[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                            m_command_list.ClearRenderTargetView(s17_ertv, s17_ewhite);
                            m_command_list.Transition(s17_exp_res,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                      D3D12_RESOURCE_STATE_COMMON);
                            LOG_WARNING(Render_D3D12,
                                        "S17 EXPO probe: cleared 1x1 gpu={:#x} res=0x{:x} to 1.0",
                                        static_cast<u64>(ev.GpuAddr()),
                                        reinterpret_cast<u64>(s17_exp_res));
                        }
                        break;
                    }
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

    // TEMP DIAGNOSTIC (session 15): post-draw content probe of the R11G11B10 RTs (runs after
    // the draw command, unlike the early per-RT probe), to see whether the big scene draw's
    // own output and the small LUT renders actually land.
    {
        if (framebuffer != nullptr && num_rtvs > 0 && framebuffer->ColorBuffers()[0] != nullptr &&
            framebuffer->ColorBuffers()[0]->Resource() != nullptr) {
            ID3D12Resource* const s15_post_res = framebuffer->ColorBuffers()[0]->Resource();
            const D3D12_RESOURCE_DESC s15_post_desc = s15_post_res->GetDesc();
            if (s15_post_desc.Format == DXGI_FORMAT_R11G11B10_FLOAT) {
                // The scene-class probe must latch the COLOR pass, not the depth-only prepass
                // (att mask 0 writes no color, so its post-draw RT readback is legitimately
                // unchanged). Require at least one color channel to be written.
                const auto s15_post_mask = stored.pipeline->State().attachments[0].Mask();
                const bool s15_post_color =
                    s15_post_mask[0] || s15_post_mask[1] || s15_post_mask[2];
                const bool s15_post_scene = depth_view != nullptr && s15_post_color &&
                                            params.num_vertices >= 4000 &&
                                            s15_post_desc.Width >= 960 && s15_post_desc.Width <= 1400;
                const bool s15_post_lut = depth_view == nullptr && s15_post_desc.Width <= 512 &&
                                          s15_post_desc.Height <= 512;
                // TEMP DIAGNOSTIC (session 15): feedback-loop detector. A draw that both
                // samples and renders the SAME resource (rt == tex) is invalid on D3D12
                // (sampling a live RTV returns undefined data, often zeros) and a prime
                // suspect for the scene RT being wiped in place. Detect it and dump the RT
                // right after such a draw (940xxx) so the wipe is directly observable.
                bool s15_post_feedback = false;
                for (const DrawSlot& slot : res_slots) {
                    if (slot.kind != SlotKind::Sampled) {
                        continue;
                    }
                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                    if (sv.Resource() != nullptr && sv.Resource() == s15_post_res) {
                        s15_post_feedback = true;
                        break;
                    }
                }
                // Separate budgets so the boot LUT self-sampling draws (128 wide) cannot
                // starve the scene-RT ones (the earlier single cap was consumed at t=221).
                static u32 s15_post_feedback_dumps = 0;
                static u32 s15_post_feedback_scene_dumps = 0;
                u32& s15_fb_counter =
                    s15_post_desc.Width >= 512 ? s15_post_feedback_scene_dumps
                                               : s15_post_feedback_dumps;
                if (s15_post_feedback && s15_fb_counter < 24) {
                    ++s15_fb_counter;
                    LOG_WARNING(Render_D3D12,
                                "S15 feedback loop draw: gpu={:#x} res=0x{:x} {}x{} verts={} "
                                "color={} n={}",
                                static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()),
                                reinterpret_cast<u64>(s15_post_res), s15_post_desc.Width,
                                s15_post_desc.Height, params.num_vertices, s15_post_color ? 1 : 0,
                                s15_fb_counter);
                    (void)D3D12::RecordTextureContentDump(
                        m_device.GetDevice(), m_command_list, s15_post_res,
                        static_cast<u32>(VideoCore::Surface::PixelFormat::B10G11R11_FLOAT),
                        static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()),
                        940000u + s15_fb_counter +
                            (s15_post_desc.Width >= 512 ? 100u : 0u),
                        true);
                }
                // TEMP DIAGNOSTIC (session 15): sequence trace of EVERY writer to the 960x540
                // scene RT for the first draws after 20 minutes, so the pass that zeroes the
                // RT (or the last writer before the composite) is directly visible. Post-draw
                // state per writer, ids 960xxx, bounded.
                {
                    static u32 s15_scene_writer_dumps = 0;
                    static std::chrono::steady_clock::time_point s15_scene_writer_start =
                        std::chrono::steady_clock::now();
                    const bool s15_scene_writer_late =
                        std::chrono::steady_clock::now() - s15_scene_writer_start >
                        std::chrono::minutes(20);
                    if (s15_scene_writer_late && s15_scene_writer_dumps < 64 &&
                        s15_post_desc.Width >= 900 && s15_post_desc.Width <= 1000 &&
                        s15_post_desc.Height >= 500 && s15_post_desc.Height <= 600) {
                        ++s15_scene_writer_dumps;
                        u32 s15_writer_sampled = 0;
                        u32 s15_writer_feedback = 0;
                        for (const DrawSlot& slot : res_slots) {
                            if (slot.kind != SlotKind::Sampled) {
                                continue;
                            }
                            ++s15_writer_sampled;
                            ImageView& sv = m_texture_cache.GetImageView(slot.view);
                            if (sv.Resource() == s15_post_res) {
                                ++s15_writer_feedback;
                            }
                        }
                        LOG_WARNING(Render_D3D12,
                                    "S15 scene writer #{}: gpu={:#x} res=0x{:x} verts={} att={:#x} "
                                    "sampled={} self_sample={} depth={}",
                                    s15_scene_writer_dumps,
                                    static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()),
                                    reinterpret_cast<u64>(s15_post_res), params.num_vertices,
                                    stored.pipeline->State().attachments[0].raw, s15_writer_sampled,
                                    s15_writer_feedback, depth_view != nullptr ? 1 : 0);
                        (void)D3D12::RecordTextureContentDump(
                            m_device.GetDevice(), m_command_list, s15_post_res,
                            static_cast<u32>(VideoCore::Surface::PixelFormat::B10G11R11_FLOAT),
                            static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()),
                            960000u + s15_scene_writer_dumps, true);
                    }
                }
                // TEMP DIAGNOSTIC (session 15): the small writers (fullscreen/copy passes)
                // into the scene RT, logged with its own budget so they cannot be starved
                // by (or starve) the writer dump above. Late window + same RT size window,
                // small vertex counts only.
                {
                    static u32 s15_small_writer_logs = 0;
                    static std::chrono::steady_clock::time_point s15_small_writer_start =
                        std::chrono::steady_clock::now();
                    const bool s15_small_writer_late =
                        std::chrono::steady_clock::now() - s15_small_writer_start >
                        std::chrono::minutes(20);
                    if (s15_small_writer_late && s15_small_writer_logs < 16 &&
                        params.num_vertices <= 8 && s15_post_desc.Width >= 900 &&
                        s15_post_desc.Width <= 1000 && s15_post_desc.Height >= 500 &&
                        s15_post_desc.Height <= 600) {
                        ++s15_small_writer_logs;
                        fmt::memory_buffer s15_small_chain;
                        u32 s15_small_sampled = 0;
                        for (const DrawSlot& slot : res_slots) {
                            if (slot.kind != SlotKind::Sampled) {
                                continue;
                            }
                            ImageView& sv = m_texture_cache.GetImageView(slot.view);
                            if (s15_small_sampled < 8) {
                                fmt::format_to(
                                    std::back_inserter(s15_small_chain),
                                    "[img={} gpu={:#x} fmt={:#x} res=0x{:x} same_rt={}]",
                                    sv.image_id.index, static_cast<u64>(sv.GpuAddr()),
                                    static_cast<u32>(sv.format),
                                    reinterpret_cast<u64>(sv.Resource()),
                                    sv.Resource() == s15_post_res ? 1 : 0);
                            }
                            ++s15_small_sampled;
                        }
                        const D3D12_VIEWPORT s15_small_vp = MakeViewport(regs);
                        const D3D12_RECT s15_small_sc = MakeScissor(regs);
                        LOG_WARNING(Render_D3D12,
                                    "S15 small scene writer #{}: verts={} att={:#x} sampled={} "
                                    "vp=({},{},{},{}) sc=({},{},{},{}) blend={} {}",
                                    s15_small_writer_logs, params.num_vertices,
                                    stored.pipeline->State().attachments[0].raw,
                                    s15_small_sampled, s15_small_vp.TopLeftX,
                                    s15_small_vp.TopLeftY, s15_small_vp.Width,
                                    s15_small_vp.Height, s15_small_sc.left, s15_small_sc.top,
                                    s15_small_sc.right, s15_small_sc.bottom,
                                    stored.pipeline->State().attachments[0].enable.Value(),
                                    fmt::to_string(s15_small_chain));
                        // Post-draw content dump of the scene RT for every small writer
                        // event that logs (970xxx), so the wiper's effect on the RT is
                        // visible even if the 960xxx per-writer budget is exhausted:
                        // before-content in the 960xxx series, after-content in 970xxx.
                        static u32 s15_small_writer_dumps = 0;
                        if (s15_small_writer_dumps < 16) {
                            ++s15_small_writer_dumps;
                            (void)D3D12::RecordTextureContentDump(
                                m_device.GetDevice(), m_command_list, s15_post_res,
                                static_cast<u32>(
                                    VideoCore::Surface::PixelFormat::B10G11R11_FLOAT),
                                static_cast<u64>(
                                    framebuffer->ColorBuffers()[0]->GpuAddr()),
                                970000u + s15_small_writer_dumps, /*force=*/true);
                        }
                        // TEMP DIAGNOSTIC (session 15, style-screen chain): the self-sampler
                        // signature (<=4 verts, write mask 0xf, 4 sampled textures) is the
                        // draw that REPLACES the scene RT instead of blending into it. Dump
                        // every sampled slot in order on the first 2 such events, with unique
                        // ids, so the four inputs of the replacing pass are inspectable
                        // individually; force=true bypasses the per-image dedupe.
                        if (params.num_vertices <= 4 &&
                            (stored.pipeline->State().attachments[0].raw & 0xFu) == 0xFu &&
                            s15_small_sampled == 4) {
                            static bool s15_selfsampler_pending = true;
                            static u32 s15_selfsampler_events = 0;
                            if (s15_selfsampler_pending) {
                                ++s15_selfsampler_events;
                                const u32 s15_selfsampler_base =
                                    s15_selfsampler_events == 1 ? 980001u : 980011u;
                                u32 s15_selfsampler_dumped = 0;
                                for (const DrawSlot& slot : res_slots) {
                                    if (slot.kind != SlotKind::Sampled) {
                                        continue;
                                    }
                                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                                    if (sv.Resource() == nullptr) {
                                        continue;
                                    }
                                    ++s15_selfsampler_dumped;
                                    (void)D3D12::RecordTextureContentDump(
                                        m_device.GetDevice(), m_command_list, sv.Resource(),
                                        static_cast<u32>(sv.format),
                                        static_cast<u64>(sv.GpuAddr()),
                                        s15_selfsampler_base + s15_selfsampler_dumped,
                                        /*force=*/true);
                                }
                                LOG_WARNING(Render_D3D12,
                                            "S15 selfsampler dumps: verts={} rts={} inputs={}",
                                            params.num_vertices, num_rtvs,
                                            s15_selfsampler_dumped);
                                if (s15_selfsampler_events >= 2) {
                                    s15_selfsampler_pending = false;
                                }
                            }
                        }
                        // TEMP DIAGNOSTIC (session 15): the zeroing ("wiper") draw
                        // signature: <=4 verts, write-mask low nibble 0x7 (RGB only, no
                        // alpha) and <=2 sampled slots. The same-rt self-sampler (mask
                        // 0xf) is already covered by the feedback-loop dumps above and is
                        // excluded by the 0x7 check. Dump the wiper's INPUT textures
                        // (their own image ids, forced so the per-image dedupe cannot
                        // skip them) so a zeroed RT can be attributed to a zero input;
                        // bounded to the first 4 such events, up to 3 inputs each.
                        if (params.num_vertices <= 4 &&
                            (stored.pipeline->State().attachments[0].raw & 0xFu) == 0x7u &&
                            s15_small_sampled <= 2) {
                            static u32 s15_wiper_input_events = 0;
                            if (s15_wiper_input_events < 4) {
                                ++s15_wiper_input_events;
                                fmt::memory_buffer s15_wiper_chain;
                                u32 s15_wiper_dumped = 0;
                                for (const DrawSlot& slot : res_slots) {
                                    if (slot.kind != SlotKind::Sampled) {
                                        continue;
                                    }
                                    ImageView& sv = m_texture_cache.GetImageView(slot.view);
                                    fmt::format_to(
                                        std::back_inserter(s15_wiper_chain),
                                        "[img={} gpu={:#x} fmt={:#x} res=0x{:x} same_rt={}]",
                                        sv.image_id.index, static_cast<u64>(sv.GpuAddr()),
                                        static_cast<u32>(sv.format),
                                        reinterpret_cast<u64>(sv.Resource()),
                                        sv.Resource() == s15_post_res ? 1 : 0);
                                    if (sv.Resource() != nullptr && s15_wiper_dumped < 3) {
                                        ++s15_wiper_dumped;
                                        (void)D3D12::RecordTextureContentDump(
                                            m_device.GetDevice(), m_command_list, sv.Resource(),
                                            static_cast<u32>(sv.format),
                                            static_cast<u64>(sv.GpuAddr()), sv.image_id.index,
                                            /*force=*/true);
                                        // TEMP DIAGNOSTIC (session 15, style-screen chain): the
                                        // input above is dumped under its own image id, whose
                                        // filename collides with older runs; re-record the first
                                        // event's input under the unique 980005 id so it lands
                                        // in its own file.
                                        static bool s15_wiper_stable_dumped = false;
                                        if (!s15_wiper_stable_dumped) {
                                            s15_wiper_stable_dumped = true;
                                            (void)D3D12::RecordTextureContentDump(
                                                m_device.GetDevice(), m_command_list,
                                                sv.Resource(), static_cast<u32>(sv.format),
                                                static_cast<u64>(sv.GpuAddr()), 980005u,
                                                /*force=*/true);
                                        }
                                    }
                                }
                                LOG_WARNING(
                                    Render_D3D12,
                                    "S15 wiper inputs: verts={} att={:#x} inputs={} dumped={} {}",
                                    params.num_vertices,
                                    stored.pipeline->State().attachments[0].raw,
                                    s15_small_sampled, s15_wiper_dumped,
                                    fmt::to_string(s15_wiper_chain));
                            }
                            // TEMP DIAGNOSTIC (S15): the wiper's blend factors from the
                            // packed BlendingAttachment (own budget so the input dumps
                            // above cannot starve it).
                            {
                                static u32 s15_wiper_blend_logs = 0;
                                if (s15_wiper_blend_logs < 8) {
                                    ++s15_wiper_blend_logs;
                                    const auto& s15_att =
                                        stored.pipeline->State().attachments[0];
                                    LOG_WARNING(
                                        Render_D3D12,
                                        "S15 wiper blend: att={:#x} enable={} src={:#x} "
                                        "dst={:#x} op={:#x} asrc={:#x} adst={:#x} aop={:#x}",
                                        s15_att.raw, s15_att.enable.Value(),
                                        s15_att.factor_source_rgb.Value(),
                                        s15_att.factor_dest_rgb.Value(),
                                        s15_att.equation_rgb.Value(),
                                        s15_att.factor_source_a.Value(),
                                        s15_att.factor_dest_a.Value(),
                                        s15_att.equation_a.Value());
                                }
                            }
                            // ONE-SHOT (S15): enumerate every cached image registered over
                            // the wiper input's CPU memory (4 MB window) - a sibling image
                            // over the same guest memory is how the LUT bank received its
                            // content (JoinImages alias copy).
                            {
                                static bool s15_wiper_siblings_done = false;
                                if (!s15_wiper_siblings_done) {
                                    s15_wiper_siblings_done = true;
                                    ImageView* s15_wiper_in_view = nullptr;
                                    for (const DrawSlot& slot : res_slots) {
                                        if (slot.kind != SlotKind::Sampled) {
                                            continue;
                                        }
                                        s15_wiper_in_view =
                                            &m_texture_cache.GetImageView(slot.view);
                                        break;
                                    }
                                    if (s15_wiper_in_view != nullptr) {
                                        const std::optional<DAddr> s15_wiper_cpu =
                                            gpu_memory->GpuToCpuAddress(
                                                static_cast<GPUVAddr>(
                                                    s15_wiper_in_view->GpuAddr()));
                                        if (s15_wiper_cpu) {
                                            u32 s15_wiper_sib_total = 0;
                                            m_texture_cache.ForEachImageInRegion(
                                                *s15_wiper_cpu, 0x400000u,
                                                [&](VideoCommon::ImageId s15_id,
                                                    auto& s15_img) {
                                                    ++s15_wiper_sib_total;
                                                    if (s15_wiper_sib_total <= 8) {
                                                        LOG_WARNING(
                                                            Render_D3D12,
                                                            "S15 wiper-input siblings: "
                                                            "cpu=0x{:x} id={} gpu={:#x} "
                                                            "flags={:#x} size={} fmt={:#x} "
                                                            "tick={}",
                                                            static_cast<u64>(*s15_wiper_cpu),
                                                            s15_id.index,
                                                            static_cast<u64>(s15_img.gpu_addr),
                                                            static_cast<u32>(s15_img.flags),
                                                            s15_img.guest_size_bytes,
                                                            static_cast<u32>(
                                                                s15_img.info.format),
                                                            s15_img.modification_tick);
                                                    }
                                                });
                                            LOG_WARNING(
                                                Render_D3D12,
                                                "S15 wiper-input siblings total: cpu=0x{:x} "
                                                "count={}",
                                                static_cast<u64>(*s15_wiper_cpu),
                                                s15_wiper_sib_total);
                                        } else {
                                            LOG_WARNING(
                                                Render_D3D12,
                                                "S15 wiper-input siblings: gpu={:#x} unmapped",
                                                static_cast<u64>(
                                                    s15_wiper_in_view->GpuAddr()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                static u32 s15_post_scene_dumps = 0;
                static u32 s15_post_lut_dumps = 0;
                static std::chrono::steady_clock::time_point s15_post_start =
                    std::chrono::steady_clock::now();
                const bool s15_post_late =
                    std::chrono::steady_clock::now() - s15_post_start > std::chrono::minutes(20);
                const bool s15_post_scene_budget =
                    s15_post_scene && s15_post_late && s15_post_scene_dumps < 4;
                const bool s15_post_lut_budget = s15_post_lut && s15_post_lut_dumps < 8;
                if (s15_post_scene_budget || s15_post_lut_budget) {
                    const u32 s15_post_index =
                        s15_post_scene_budget ? (930000u + ++s15_post_scene_dumps)
                                              : (920000u + ++s15_post_lut_dumps);
                    LOG_WARNING(Render_D3D12,
                                "S15 post-draw RT: gpu={:#x} res=0x{:x} {}x{} depth={} verts={} "
                                "lut={}",
                                static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()),
                                reinterpret_cast<u64>(s15_post_res), s15_post_desc.Width,
                                s15_post_desc.Height, depth_view != nullptr ? 1 : 0,
                                params.num_vertices, s15_post_lut ? 1 : 0);
                    (void)D3D12::RecordTextureContentDump(
                        m_device.GetDevice(), m_command_list, s15_post_res,
                        static_cast<u32>(VideoCore::Surface::PixelFormat::B10G11R11_FLOAT),
                        static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()),
                        s15_post_index, true);
                    if (s15_post_scene_budget) {
                        D3D12::RecordScene3dCaptureMidFrame(m_device.GetDevice(), m_command_list,
                                                            s15_post_res);
                        // TEMP DIAGNOSTIC (session 15): dump the sampled inputs of the EXACT
                        // same scene draw as the 930xxx RT probe, so a black scene RT can be
                        // correlated with its (possibly zero) sampled textures. Bounded to a
                        // few probe events, up to 8 inputs each; force=true bypasses per-image
                        // dedupe so every probe event really lands.
                        static u32 s15_post_input_dumps = 0;
                        if (s15_post_input_dumps < 4) {
                            ++s15_post_input_dumps;
                            fmt::memory_buffer s15_input_chain;
                            u32 s15_input_count = 0;
                            u32 s15_input_dumped = 0;
                            for (const DrawSlot& slot : res_slots) {
                                if (slot.kind != SlotKind::Sampled) {
                                    continue;
                                }
                                ImageView& sv = m_texture_cache.GetImageView(slot.view);
                                if (sv.Resource() == nullptr) {
                                    continue;
                                }
                                ++s15_input_count;
                                fmt::format_to(std::back_inserter(s15_input_chain),
                                               "[img={} gpu={:#x} {}x{} fmt={:#x} res={}]",
                                               sv.image_id.index,
                                               static_cast<u64>(sv.GpuAddr()), sv.size.width,
                                               sv.size.height, static_cast<u32>(sv.format),
                                               sv.Resource() ? 1 : 0);
                                if (s15_input_dumped < 8) {
                                    ++s15_input_dumped;
                                    (void)D3D12::RecordTextureContentDump(
                                        m_device.GetDevice(), m_command_list, sv.Resource(),
                                        static_cast<u32>(sv.format),
                                        static_cast<u64>(sv.GpuAddr()), sv.image_id.index,
                                        /*force=*/true);
                                }
                            }
                            LOG_WARNING(Render_D3D12,
                                        "S15 scene-probe inputs: gpu={:#x} verts={} inputs={} "
                                        "dumped={} {}",
                                        static_cast<u64>(
                                            framebuffer->ColorBuffers()[0]->GpuAddr()),
                                        params.num_vertices, s15_input_count, s15_input_dumped,
                                        fmt::to_string(s15_input_chain));
                        }
                    }
                }
            }
            // TEMP DIAGNOSTIC (session 15, style-screen chain): the composite target of the very
            // scene-composite draw whose sampled textures were dumped above, recorded once under
            // a unique id now that the draw has executed and the render targets are back in
            // COMMON. The flag is cleared here so exactly one such dump can be armed.
            if (g_s15_composite_target_dump) {
                g_s15_composite_target_dump = false;
                (void)D3D12::RecordTextureContentDump(
                    m_device.GetDevice(), m_command_list, s15_post_res,
                    static_cast<u32>(VideoCore::Surface::PixelFormat::A2B10G10R10_UNORM),
                    static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()), 980024u,
                    /*force=*/true);
            }
            // S18 cycle 3 (Phase 0.2.3): post-draw dump of the composite target, exactly once per
            // arming and only for the armed resource (ids 990201+). The 1.0.271 one-shot showed
            // the target nearly black right after the 3-vertex composite (nonzero 6664 of
            // 8294400) while the visible frame is a bright smear; this turns that single
            // observation into a 60 s-cadence timeline paired with the exposure readbacks.
            if (g_s18_target_res != nullptr && s15_post_res == g_s18_target_res &&
                g_s18_target_used < 40) {
                ++g_s18_target_used;
                const u32 s18_target_id = 990200u + g_s18_target_used;
                (void)D3D12::RecordTextureContentDump(
                    m_device.GetDevice(), m_command_list, s15_post_res,
                    static_cast<u32>(VideoCore::Surface::PixelFormat::A2B10G10R10_UNORM),
                    static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()), s18_target_id,
                    /*force=*/true);
                LOG_WARNING(Render_D3D12, "S18B target dump: id={} res=0x{:x}", s18_target_id,
                            reinterpret_cast<u64>(s15_post_res));
                g_s18_target_res = nullptr;
            }
            // S18 cycle 5 (Phase 0.2.5): the DISPLAY buffer (the flinger's render target) dumped
            // on a 60 s cadence - the visible frame's own GPU content, so each capture can be
            // compared against the resource it was blitted from (and against the composite
            // target it was composed from). Ids 991001+; cap 30.
            {
                static std::chrono::steady_clock::time_point s18e_last =
                    std::chrono::steady_clock::now();
                static u32 s18e_dumps = 0;
                bool s18e_is_display = false;
                for (ID3D12Resource* const s18e_r : g_display_resources) {
                    if (s18e_r != nullptr && s18e_r == s15_post_res) {
                        s18e_is_display = true;
                    }
                }
                if (s18e_is_display && s18e_dumps < 30 &&
                    std::chrono::steady_clock::now() - s18e_last > std::chrono::seconds(60)) {
                    s18e_last = std::chrono::steady_clock::now();
                    ++s18e_dumps;
                    const u32 s18e_id = 991000u + s18e_dumps;
                    (void)D3D12::RecordTextureContentDump(
                        m_device.GetDevice(), m_command_list, s15_post_res,
                        static_cast<u32>(framebuffer->ColorBuffers()[0]->format),
                        static_cast<u64>(framebuffer->ColorBuffers()[0]->GpuAddr()), s18e_id,
                        /*force=*/true);
                    LOG_WARNING(Render_D3D12, "S18E display dump: id={} res=0x{:x}", s18e_id,
                                reinterpret_cast<u64>(s15_post_res));
                }
            }
        }
    }

    // TEMP DIAGNOSTIC (session 15): capture the final composite target (1920x1080
    // R10G10B10A2) once after 20 minutes, to localize whether the presented loss happens
    // before or after this surface.
    {
        static std::chrono::steady_clock::time_point s15_final_start =
            std::chrono::steady_clock::now();
        static bool s15_final_captured = false;
        if (!s15_final_captured &&
            std::chrono::steady_clock::now() - s15_final_start > std::chrono::minutes(20) &&
            framebuffer != nullptr && num_rtvs > 0 &&
            framebuffer->ColorBuffers()[0] != nullptr &&
            framebuffer->ColorBuffers()[0]->Resource() != nullptr) {
            ID3D12Resource* const s15_final_res = framebuffer->ColorBuffers()[0]->Resource();
            const D3D12_RESOURCE_DESC s15_final_desc = s15_final_res->GetDesc();
            if (s15_final_desc.Width >= 1920 &&
                s15_final_desc.Format == DXGI_FORMAT_R10G10B10A2_TYPELESS) {
                s15_final_captured = true;
                D3D12::RecordScene3dCaptureMidFrame(m_device.GetDevice(), m_command_list,
                                                    s15_final_res);
            }
        }
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
        // TEMP DIAGNOSTIC (session 15): log color clears that target the 960x540 scene RT,
        // to see whether (and when) the guest wipes the scene render target between the
        // scene draws and the composite.
        {
            static u32 s15_scene_clear_logs = 0;
            if (use_color && view != nullptr && resource != nullptr &&
                static_cast<u64>(view->GpuAddr()) == 0x5103e0000ULL &&
                s15_scene_clear_logs < 32) {
                ++s15_scene_clear_logs;
                LOG_WARNING(Render_D3D12,
                            "S15 scene RT clear: gpu={:#x} color0=({},{},{},{}) scissor={} "
                            "rect=({},{},{},{})",
                            static_cast<u64>(view->GpuAddr()), regs.clear_color[0],
                            regs.clear_color[1], regs.clear_color[2], regs.clear_color[3],
                            regs.clear_control.use_scissor ? 1 : 0, clear_rect.left,
                            clear_rect.top, clear_rect.right, clear_rect.bottom);
            }
        }
        {
            // TEMP DIAGNOSTIC (S15, producer tracer): bounded trace of a clear landing
            // inside the wiper-input window [0x517000000, 0x517200000) - a clear is a
            // candidate zero-producer for the 480x270 R16G16 input no draw renders into.
            static u32 s15_wiper_clear_logs = 0;
            if (use_color && view != nullptr && resource != nullptr &&
                s15_wiper_clear_logs < 8) {
                const u64 s15_wiper_clear_gpu = static_cast<u64>(view->GpuAddr());
                if (s15_wiper_clear_gpu >= 0x517000000ull &&
                    s15_wiper_clear_gpu < 0x517200000ull) {
                    ++s15_wiper_clear_logs;
                    LOG_WARNING(Render_D3D12,
                                "S15 wiper-range clear: gpu={:#x} color=({},{},{},{}) scissor={}",
                                s15_wiper_clear_gpu, regs.clear_color[0], regs.clear_color[1],
                                regs.clear_color[2], regs.clear_color[3],
                                regs.clear_control.use_scissor ? 1 : 0);
                }
            }
        }
        {
            // TEMP DIAGNOSTIC (S18, Phase 0.2.1): clear-path writer of the 1x1 R32G32B32A32
            // exposure textures. The S17 draw/compute tracer pairs saw 0 binds for a 1x1, so a
            // clear is a remaining candidate (the whole-image clear value would BE the texel).
            // Keyed on the two 1x1 exposure VAs and on any tiny RT. Cap 12.
            static u32 s18_tiny_clear_logs = 0;
            if (use_color && view != nullptr && resource != nullptr && s18_tiny_clear_logs < 12) {
                const u64 s18_clear_gpu = static_cast<u64>(view->GpuAddr());
                const bool s18_tiny = view->size.width <= 2 && view->size.height <= 2;
                const bool s18_expo_va = s18_clear_gpu == 0x5d5445100ull ||
                                         s18_clear_gpu == 0x5d5445180ull ||
                                         s18_clear_gpu == 0x5e8728480ull;
                if (s18_tiny || s18_expo_va) {
                    ++s18_tiny_clear_logs;
                    LOG_WARNING(Render_D3D12,
                                "S18 tiny clear: img={} gpu={:#x} res=0x{:x} {}x{} fmt={:#x} "
                                "color=({},{},{},{}) scissor={}",
                                view->image_id.index, s18_clear_gpu,
                                reinterpret_cast<u64>(resource), view->size.width,
                                view->size.height, static_cast<u32>(view->format),
                                regs.clear_color[0], regs.clear_color[1], regs.clear_color[2],
                                regs.clear_color[3], regs.clear_control.use_scissor ? 1 : 0);
                }
            }
        }
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
    // REVERTED (session 17, 1.0.256.0 -> 1.0.257.0): adding gpu_memory->FlushCaching() here
    // BLACKED OUT the 3D scene on console (captures mean 5.4/255 with only a fragment of the
    // character left, UI intact), where 1.0.255.0 was lit. Cause: D3D12 never marks
    // compute-written STORAGE IMAGES as GpuModified (Vulkan does it in pipeline_helper.h:305 for
    // desc.is_written), so flushing before a dispatch makes RasterizerD3D12::InvalidateRegion
    // mark those images CpuModified and the next sample re-uploads them from stale guest memory,
    // destroying GPU-authored content. Do NOT re-add this flush until compute storage-image
    // writes are marked GpuModified the way Vulkan marks them.
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

    // TEMP DIAGNOSTIC (S15, producer tracer): the wiper-input window [0x517000000,
    // 0x517200000) has no logged producer and compute dispatches log no addresses, so a
    // compute writer would be invisible. Bounded trace (cap 16) of compute bindings that
    // touch the window: SSBO guest addresses (read from the launch-description const
    // buffers, same convention as BufferCache::StorageBufferBinding) and image /
    // image-buffer views by guest GPU VA or CPU-translation overlap (JoinImages aliases).
    {
        static u32 s15_compute_wiper_logs = 0;
        // TEMP DIAGNOSTIC (S17): the same walk, but for the LUT bank / mask VAs. The S15
        // predicate below only covers the 0x517 window, so a compute writer of the bank was
        // invisible; this one names it. Own 16-entry budget, and the loop breaks require BOTH
        // budgets to be saturated so one category cannot starve the other.
        static u32 s17_compute_producer_logs = 0;
        if (s15_compute_wiper_logs < 16 || s17_compute_producer_logs < 16) {
            const std::optional<DAddr> s15_window_cpu = gpu_memory->GpuToCpuAddress(0x517000000ull);
            const auto s15_compute_hits = [&](GPUVAddr gpu, u64 size) {
                if (gpu >= 0x517000000ull && gpu < 0x517200000ull) {
                    return true;
                }
                const std::optional<DAddr> cpu = gpu_memory->GpuToCpuAddress(gpu);
                return cpu && s15_window_cpu &&
                       *cpu < *s15_window_cpu + 0x200000ull &&
                       *s15_window_cpu < *cpu + (size != 0 ? size : 0x400000ull);
            };
            const auto s15_compute_log = [&](const char* kind, u64 gpu, u64 size) {
                if (s15_compute_wiper_logs >= 16) {
                    return;
                }
                ++s15_compute_wiper_logs;
                LOG_WARNING(
                    Render_D3D12,
                    "S15 wiper-range compute: program={:#x} grid=({},{},{}) kind={} gpu={:#x} "
                    "size={}",
                    static_cast<u64>(qmd.program_start), qmd.grid_dim_x.Value(),
                    qmd.grid_dim_y.Value(), qmd.grid_dim_z.Value(), kind,
                    static_cast<u64>(gpu), size);
            };
            const auto s17_compute_hits = [&](GPUVAddr gpu, u64 size) {
                const u64 s17_end = gpu + (size != 0 ? size : 1);
                const auto s17_overlap = [&](u64 base, u64 length) {
                    return gpu < base + length && s17_end > base;
                };
                return s17_overlap(0x5e9900000ull, 0x1a34000ull) ||
                       s17_overlap(0x5eb340000ull, 0x1a34000ull) ||
                       s17_overlap(0x5ecd80000ull, 0x1a34000ull) ||
                       s17_overlap(0x510fa0000ull, 0x100000ull) ||
                       s17_overlap(0x510f30000ull, 0x100000ull);
            };
            const auto s17_compute_log = [&](const char* kind, u64 gpu, u64 size) {
                if (s17_compute_producer_logs >= 16) {
                    return;
                }
                ++s17_compute_producer_logs;
                LOG_WARNING(
                    Render_D3D12,
                    "S17 LUT producer: compute program={:#x} grid=({},{},{}) kind={} gpu={:#x} "
                    "size={}",
                    static_cast<u64>(qmd.program_start), qmd.grid_dim_x.Value(),
                    qmd.grid_dim_y.Value(), qmd.grid_dim_z.Value(), kind, static_cast<u64>(gpu),
                    size);
            };
            for (const auto& desc : info.storage_buffers_descriptors) {
                if (s15_compute_wiper_logs >= 16 && s17_compute_producer_logs >= 16) {
                    break;
                }
                if (desc.cbuf_index >= cbufs.size()) [[unlikely]] {
                    continue;
                }
                const GPUVAddr s15_ssbo_cbuf = cbufs[desc.cbuf_index].Address() + desc.cbuf_offset;
                const GPUVAddr s15_ssbo_gpu = gpu_memory->Read<u64>(s15_ssbo_cbuf);
                const u64 s15_ssbo_size =
                    desc.cbuf_index == 0 ? gpu_memory->Read<u32>(s15_ssbo_cbuf + 8) : 0;
                if (s15_compute_hits(s15_ssbo_gpu, s15_ssbo_size)) {
                    s15_compute_log("ssbo", s15_ssbo_gpu, s15_ssbo_size);
                }
                if (s17_compute_hits(s15_ssbo_gpu, s15_ssbo_size)) {
                    s17_compute_log("ssbo", s15_ssbo_gpu, s15_ssbo_size);
                }
            }
            for (size_t i = texbuf_total; i < texbuf_total + imgbuf_total && i < views.size();
                 ++i) {
                if (s15_compute_wiper_logs >= 16 && s17_compute_producer_logs >= 16) {
                    break;
                }
                if (views[i].id == VideoCommon::NULL_IMAGE_VIEW_ID) {
                    continue;
                }
                ImageView& s15_iv = m_texture_cache.GetImageView(views[i].id);
                const GPUVAddr s15_gpu = s15_iv.GpuAddr();
                if (s15_compute_hits(s15_gpu, s15_iv.BufferSize())) {
                    s15_compute_log("imgbuf", s15_gpu, s15_iv.BufferSize());
                }
                if (s17_compute_hits(s15_gpu, s15_iv.BufferSize())) {
                    s17_compute_log("imgbuf", s15_gpu, s15_iv.BufferSize());
                }
            }
            for (size_t i = sampled_views_end; i < views.size(); ++i) {
                if (s15_compute_wiper_logs >= 16 && s17_compute_producer_logs >= 16) {
                    break;
                }
                if (views[i].id == VideoCommon::NULL_IMAGE_VIEW_ID) {
                    continue;
                }
                ImageView& s15_iv = m_texture_cache.GetImageView(views[i].id);
                const GPUVAddr s15_gpu = s15_iv.GpuAddr();
                if (s15_compute_hits(s15_gpu, 0)) {
                    s15_compute_log("img", s15_gpu, 0);
                }
                if (s17_compute_hits(s15_gpu, 0)) {
                    s17_compute_log("img", s15_gpu, 0);
                }
                // S17 cycle 5 (Phase 0.2): name a compute writer of the 1x1 exposure input.
                // S18 cycle 3: widened from 1x1 to the whole exposure family - a compute pass
                // writing the 8x1 reduction RT or the 128x1 luminance array is equally a
                // candidate producer (the 1.0.272 harvest showed the reduction draw samples
                // img 3700 = 128x1 R32G32B32A32 at gpu +0x80).
                static u32 s17_expo_compute_logs = 0;
                const u32 s18_cw = s15_iv.size.width;
                const u32 s18_ch = s15_iv.size.height;
                const bool s18_compute_tiny =
                    (s18_cw <= 2 && s18_ch <= 2) || (s18_cw <= 128 && s18_ch <= 1) ||
                    (s18_cw <= 1 && s18_ch <= 128);
                if (s17_expo_compute_logs < 12 && s18_compute_tiny &&
                    s15_iv.format == VideoCore::Surface::PixelFormat::R32G32B32A32_FLOAT) {
                    ++s17_expo_compute_logs;
                    LOG_WARNING(Render_D3D12,
                                "S18 EXPO writer: compute program={:#x} img={} gpu={:#x} "
                                "{}x{} grid=({},{},{})",
                                static_cast<u64>(qmd.program_start), s15_iv.image_id.index,
                                static_cast<u64>(s15_gpu), s18_cw, s18_ch,
                                qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                qmd.grid_dim_z.Value());
                }
            }
        }
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
    // TEMP DIAGNOSTIC (S17): does a guest CPU store reach the rasterizer for the 27 MB LUT
    // bank's device range? The base is resolved once via GpuToCpuAddress (every harvested run
    // reported 0x29426000; the literal keeps the probe alive if that ever changes). The
    // buffer-cache early return swallows the texture invalidation whenever the range is a
    // registered GPU-modified buffer, so the swallow exit is logged explicitly - otherwise a
    // missing 'S17 write notify' line could mean either "no store ever arrived" or "the store
    // was eaten here". Bounded 16.
    static const DAddr s17_bank_dev = [this]() -> DAddr {
        if (gpu_memory != nullptr) {
            if (const auto cpu = gpu_memory->GpuToCpuAddress(0x5eb340000ull)) {
                return static_cast<DAddr>(*cpu);
            }
        }
        return static_cast<DAddr>(0x29426000);
    }();
    const bool s17_bank_write = addr < s17_bank_dev + 0x1a34000 && addr + size > s17_bank_dev;
    {
        std::scoped_lock lock{m_buffer_cache.mutex};
        if (m_buffer_cache.OnCPUWrite(addr, size)) {
            static u32 s17_cpuwrite_swallow_logs = 0;
            if (s17_bank_write && s17_cpuwrite_swallow_logs < 16) {
                ++s17_cpuwrite_swallow_logs;
                LOG_WARNING(Render_D3D12,
                            "S17 LUT producer: cpuwrite dev={:#x} size={} swallowed by buffer "
                            "cache (texture cache never notified)",
                            static_cast<u64>(addr), size);
            }
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
void RasterizerD3D12::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    // Mirror RasterizerVulkan (vk_rasterizer.cpp:809): a GMMU remap means the backing memory
    // changed, so mark every overlapping image CpuModified|Remapped. FindImage then stops serving
    // them and JoinImages deletes them, while the replacement image's RefreshContents uploads the
    // CURRENT guest bytes. Without this the 27 MB LUT bank - one dev range mapped at three GPU VAs
    // via MapBufferEx, which fires this on every re-map - keeps the image it created before the
    // guest filled the buffer, and is sampled with stale (zero) content for the whole run.
    std::scoped_lock lock{m_texture_cache.mutex};
    m_texture_cache.UnmapGPUMemory(as_id, addr, size);
}
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
    // TEMP DIAGNOSTIC (S18, Phase 0.2.1): guest-frame counter. RendererD3D12::Composite calls
    // this once per guest present request (not per host present - the S17 "perf" probe counts
    // those), so this is the guest's frame count; captures carry it as their phase fingerprint.
    ++g_s18_guest_frames;
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