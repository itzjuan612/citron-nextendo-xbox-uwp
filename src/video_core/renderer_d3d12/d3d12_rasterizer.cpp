// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "video_core/control/channel_state.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_d3d12/d3d12_rasterizer.h"

namespace D3D12 {

AccelerateDMA::AccelerateDMA() = default;

bool AccelerateDMA::BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) {
    return false;
}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    return false;
}

RasterizerD3D12::RasterizerD3D12(Tegra::GPU& gpu) : m_gpu{gpu} {}
RasterizerD3D12::~RasterizerD3D12() = default;

void RasterizerD3D12::Draw(bool is_indexed, u32 instance_count) {}
void RasterizerD3D12::DrawTexture() {}
void RasterizerD3D12::Clear(u32 layer_count) {}
void RasterizerD3D12::DispatchCompute() {}
void RasterizerD3D12::ResetCounter(VideoCommon::QueryType type) {}
void RasterizerD3D12::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (!gpu_memory) {
        return;
    }
    if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
        const u64 ticks = m_gpu.GetTicks();
        gpu_memory->Write<u64>(gpu_addr + 8, ticks);
        gpu_memory->Write<u64>(gpu_addr, static_cast<u64>(payload));
    } else {
        gpu_memory->Write<u32>(gpu_addr, payload);
    }
}
void RasterizerD3D12::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {}
void RasterizerD3D12::DisableGraphicsUniformBuffer(size_t stage, u32 index) {}
void RasterizerD3D12::FlushAll() {}
void RasterizerD3D12::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType) {}
bool RasterizerD3D12::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType) {
    return false;
}
void RasterizerD3D12::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType) {}
bool RasterizerD3D12::OnCPUWrite(PAddr addr, u64 size) {
    return false;
}
void RasterizerD3D12::OnCacheInvalidation(PAddr addr, u64 size) {}
VideoCore::RasterizerDownloadArea RasterizerD3D12::GetFlushArea(PAddr addr, u64 size) {
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}
void RasterizerD3D12::InvalidateGPUCache() {}
void RasterizerD3D12::UnmapMemory(DAddr addr, u64 size) {}
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
void RasterizerD3D12::FlushAndInvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType) {}
void RasterizerD3D12::WaitForIdle() {}
void RasterizerD3D12::FragmentBarrier() {}
void RasterizerD3D12::TiledCacheBarrier() {}
void RasterizerD3D12::FlushCommands() {}
void RasterizerD3D12::TickFrame() {}
Tegra::Engines::AccelerateDMAInterface& RasterizerD3D12::AccessAccelerateDMA() {
    return m_accelerate_dma;
}
bool RasterizerD3D12::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    return false;
}
void RasterizerD3D12::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {}
void RasterizerD3D12::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                        const VideoCore::DiskResourceLoadCallback& callback) {}
void RasterizerD3D12::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
}
void RasterizerD3D12::BindChannel(Tegra::Control::ChannelState& channel) {
    BindToChannel(channel.bind_id);
}
void RasterizerD3D12::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
}

} // namespace D3D12
