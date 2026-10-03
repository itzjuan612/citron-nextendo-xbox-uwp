// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <array>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/buffer_cache/usage_tracker.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_d3d12/d3d12_descriptor_heap.h"
#include "video_core/renderer_d3d12/d3d12_staging_buffer_pool.h"
#include "video_core/surface.h"

namespace D3D12 {

class BufferCacheRuntime;
class CommandList;
class Device;

using Microsoft::WRL::ComPtr;

/// A GPU buffer resource plus usage tracking and cached texel-buffer views.
class Buffer : public VideoCommon::BufferBase {
public:
    explicit Buffer(BufferCacheRuntime& runtime, VideoCommon::NullBufferParams null_params);
    explicit Buffer(BufferCacheRuntime& runtime, VAddr cpu_addr_, u64 size_bytes_);

    [[nodiscard]] ID3D12Resource* Handle() const noexcept {
        return buffer.Get();
    }

    operator ID3D12Resource*() const noexcept {
        return buffer.Get();
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS GpuAddr() const noexcept {
        return buffer ? buffer->GetGPUVirtualAddress() : D3D12_GPU_VIRTUAL_ADDRESS{0};
    }

    /// Cached SRV for a texel buffer range. The handle is zero when the format is not
    /// representable as a typed buffer view.
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE View(u32 offset, u32 size,
                                                   VideoCore::Surface::PixelFormat format);

    [[nodiscard]] bool IsRegionUsed(u64 offset, u64 size) const noexcept {
        return tracker.IsUsed(offset, size);
    }

    void MarkUsage(u64 offset, u64 size) noexcept {
        tracker.Track(offset, size);
    }

    void ResetUsageTracking() noexcept {
        tracker.Reset();
    }

private:
    struct BufferView {
        u32 offset;
        u32 size;
        VideoCore::Surface::PixelFormat format;
        D3D12_CPU_DESCRIPTOR_HANDLE handle;
    };

    Device* device{};
    DescriptorHeap* view_heap{};
    ComPtr<ID3D12Resource> buffer;
    std::vector<BufferView> views;
    VideoCommon::UsageTracker tracker;
    bool is_null{};
};

/// Binding state accumulated by the buffer cache and consumed by the draw translation
/// (graphics pipeline cache step). Index/vertex entries mirror what IASet* needs; resource
/// entries are uniform/storage/texel bindings in bind order (descriptor-table order).
class BufferCacheRuntime {
    friend Buffer;

public:
    using PrimitiveTopology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;

    static constexpr u32 MAX_VERTEX_BUFFERS = 32;

    struct IndexBinding {
        D3D12_GPU_VIRTUAL_ADDRESS address{};
        u32 size{};
        u32 first_index{};
        u32 num_indices{};
        DXGI_FORMAT format{DXGI_FORMAT_R32_UINT};
        bool valid{};
    };

    struct ResourceBinding {
        D3D12_GPU_VIRTUAL_ADDRESS address{};
        u64 offset{};
        u64 size{};
        D3D12_CPU_DESCRIPTOR_HANDLE view{};
    };

    explicit BufferCacheRuntime(Device& device, CommandList& command_list,
                                  StagingBufferPool& staging_pool);

    void TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept;

    void Finish();

    u64 GetDeviceLocalMemory() const;

    u64 GetDeviceMemoryUsage() const;

    void CleanupUnusedBuffers();

    bool CanReportMemoryUsage() const;

    u32 GetStorageBufferAlignment() const;

    [[nodiscard]] StagingBufferRef UploadStagingBuffer(size_t size);

    [[nodiscard]] StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);

    bool CanReorderUpload(const Buffer& buffer, std::span<const VideoCommon::BufferCopy> copies);

    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void PreCopyBarrier();

    void CopyBuffer(ID3D12Resource* dst_buffer, ID3D12Resource* src_buffer,
                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                    bool can_reorder = false);

    void PostCopyBarrier();

    void ClearBuffer(ID3D12Resource* dest_buffer, u32 offset, size_t size, u32 value);

    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 base_vertex,
                         u32 num_indices, ID3D12Resource* buffer, u32 offset, u32 size);

    void BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count);

    void BindVertexBuffer(u32 index, ID3D12Resource* buffer, u32 offset, u32 size, u32 stride);

    void BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    void BindTransformFeedbackBuffer(u32 index, ID3D12Resource* buffer, u32 offset, u32 size);

    void BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    std::span<u8> BindMappedUniformBuffer([[maybe_unused]] size_t stage,
                                          [[maybe_unused]] u32 binding_index, u32 size) {
        const StagingBufferRef ref = staging_pool.Request(size, StagingUsage::Upload);
        BindBuffer(ref.buffer, static_cast<u32>(ref.offset), size);
        return ref.mapped_span;
    }

    void BindUniformBuffer(ID3D12Resource* buffer, u32 offset, u32 size) {
        BindBuffer(buffer, offset, size);
    }

    void BindStorageBuffer(ID3D12Resource* buffer, u32 offset, u32 size,
                           [[maybe_unused]] bool is_written) {
        BindBuffer(buffer, offset, size);
    }

    void BindTextureBuffer(Buffer& buffer, u32 offset, u32 size,
                           VideoCore::Surface::PixelFormat format) {
        BindView(buffer.View(offset, size, format));
    }

    [[nodiscard]] const IndexBinding& GetIndexBinding() const noexcept {
        return index_binding;
    }

    [[nodiscard]] const std::array<D3D12_VERTEX_BUFFER_VIEW, MAX_VERTEX_BUFFERS>&
    GetVertexBindings() const noexcept {
        return vertex_bindings;
    }

    [[nodiscard]] const std::vector<ResourceBinding>& GetResourceBindings() const noexcept {
        return resource_bindings;
    }

    void ClearDrawBindings() noexcept;

private:
    void BindBuffer(ID3D12Resource* buffer, u32 offset, u32 size);
    void BindView(D3D12_CPU_DESCRIPTOR_HANDLE view);
    ID3D12Resource* ReserveNullBuffer();

    Device& device;
    CommandList& command_list;
    StagingBufferPool& staging_pool;
    DescriptorHeap view_heap;
    ComPtr<ID3D12Resource> null_buffer;

    IndexBinding index_binding;
    std::array<D3D12_VERTEX_BUFFER_VIEW, MAX_VERTEX_BUFFERS> vertex_bindings{};
    std::vector<ResourceBinding> resource_bindings;

    u64 tick{};
    bool logged_quad_index{};
    bool logged_u8_index{};
    bool logged_transform_feedback{};
};

struct BufferCacheParams {
    using Runtime = D3D12::BufferCacheRuntime;
    using Buffer = D3D12::Buffer;
    using Async_Buffer = D3D12::StagingBufferRef;
    using MemoryTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using BufferCache = VideoCommon::BufferCache<BufferCacheParams>;

} // namespace D3D12
