// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>

#include "common/alignment.h"
#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_buffer_cache.h"
#include "video_core/renderer_d3d12/d3d12_command_list.h"
#include "video_core/renderer_d3d12/d3d12_device.h"

namespace D3D12 {

namespace {

using PixelFormat = VideoCore::Surface::PixelFormat;

DXGI_FORMAT ToDxgiFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::R8_UNORM:
        return DXGI_FORMAT_R8_UNORM;
    case PixelFormat::R8_SNORM:
        return DXGI_FORMAT_R8_SNORM;
    case PixelFormat::R8_UINT:
        return DXGI_FORMAT_R8_UINT;
    case PixelFormat::R8_SINT:
        return DXGI_FORMAT_R8_SINT;
    case PixelFormat::R8G8_UNORM:
        return DXGI_FORMAT_R8G8_UNORM;
    case PixelFormat::R8G8_SNORM:
        return DXGI_FORMAT_R8G8_SNORM;
    case PixelFormat::R8G8_UINT:
        return DXGI_FORMAT_R8G8_UINT;
    case PixelFormat::R8G8_SINT:
        return DXGI_FORMAT_R8G8_SINT;
    case PixelFormat::R16_UNORM:
        return DXGI_FORMAT_R16_UNORM;
    case PixelFormat::R16_SNORM:
        return DXGI_FORMAT_R16_SNORM;
    case PixelFormat::R16_UINT:
        return DXGI_FORMAT_R16_UINT;
    case PixelFormat::R16_SINT:
        return DXGI_FORMAT_R16_SINT;
    case PixelFormat::R16_FLOAT:
        return DXGI_FORMAT_R16_FLOAT;
    case PixelFormat::R32_UINT:
        return DXGI_FORMAT_R32_UINT;
    case PixelFormat::R32_SINT:
        return DXGI_FORMAT_R32_SINT;
    case PixelFormat::R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case PixelFormat::A8B8G8R8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case PixelFormat::A8B8G8R8_SNORM:
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    case PixelFormat::A8B8G8R8_UINT:
        return DXGI_FORMAT_R8G8B8A8_UINT;
    case PixelFormat::A8B8G8R8_SINT:
        return DXGI_FORMAT_R8G8B8A8_SINT;
    case PixelFormat::A8B8G8R8_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case PixelFormat::B8G8R8A8_UNORM:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case PixelFormat::B8G8R8A8_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

u32 ElementSize(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SINT:
        return 1;
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R16_FLOAT:
        return 2;
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return 4;
    default:
        return 0;
    }
}

} // Anonymous namespace

Buffer::Buffer(BufferCacheRuntime& runtime, VideoCommon::NullBufferParams null_params)
    : VideoCommon::BufferBase{null_params}, device{&runtime.device}, view_heap{&runtime.view_heap},
      tracker{0} {
    is_null = true;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->GetDevice()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&buffer)))) {
        LOG_CRITICAL(Render_D3D12, "Null buffer creation failed");
    }
}

Buffer::Buffer(BufferCacheRuntime& runtime, VAddr cpu_addr_, u64 size_bytes_)
    : VideoCommon::BufferBase{cpu_addr_, size_bytes_}, device{&runtime.device},
      view_heap{&runtime.view_heap}, tracker{static_cast<size_t>(size_bytes_)} {
    const u64 size = size_bytes_ == 0 ? 4 : Common::AlignUp(size_bytes_, u64{4});
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->GetDevice()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&buffer)))) {
        LOG_CRITICAL(Render_D3D12, "Buffer creation failed ({} bytes)", size);
    }
}

D3D12_CPU_DESCRIPTOR_HANDLE Buffer::View(u32 offset, u32 size, PixelFormat format) {
    for (const BufferView& view : views) {
        if (view.offset == offset && view.size == size && view.format == format) {
            return view.handle;
        }
    }
    const DXGI_FORMAT dxgi_format = ToDxgiFormat(format);
    const u32 element_size = ElementSize(dxgi_format);
    if (element_size == 0 || (offset % element_size) != 0) {
        return D3D12_CPU_DESCRIPTOR_HANDLE{0};
    }
    const u32 index = view_heap->Allocate();
    if (index == DescriptorHeap::INVALID_INDEX) {
        LOG_ERROR(Render_D3D12, "Buffer view heap exhausted");
        return D3D12_CPU_DESCRIPTOR_HANDLE{0};
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE handle = view_heap->CpuHandle(index);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = dxgi_format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = offset / element_size;
    srv.Buffer.NumElements = size / element_size;
    srv.Buffer.StructureByteStride = 0;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    device->GetDevice()->CreateShaderResourceView(buffer.Get(), &srv, handle);
    views.push_back(BufferView{
        .offset = offset,
        .size = size,
        .format = format,
        .handle = handle,
    });
    return handle;
}

BufferCacheRuntime::BufferCacheRuntime(Device& device_, CommandList& command_list_)
    : device{device_}, command_list{command_list_}, staging_pool{device_},
      view_heap{device_.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096, false} {
    ReserveNullBuffer();
}

void BufferCacheRuntime::TickFrame([[maybe_unused]] Common::SlotVector<Buffer>& slot_buffers) noexcept {
    ++tick;
    staging_pool.TickFrame();
}

void BufferCacheRuntime::Finish() {
    device.WaitForIdle();
}

u64 BufferCacheRuntime::GetDeviceLocalMemory() const {
    return 0;
}

u64 BufferCacheRuntime::GetDeviceMemoryUsage() const {
    return 0;
}

void BufferCacheRuntime::CleanupUnusedBuffers() {}

bool BufferCacheRuntime::CanReportMemoryUsage() const {
    return false;
}

u32 BufferCacheRuntime::GetStorageBufferAlignment() const {
    return 4;
}

StagingBufferRef BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    return staging_pool.Request(size, StagingUsage::Upload);
}

StagingBufferRef BufferCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, StagingUsage::Download, deferred);
}

bool BufferCacheRuntime::CanReorderUpload(const Buffer&, std::span<const VideoCommon::BufferCopy>) {
    return false;
}

void BufferCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

void BufferCacheRuntime::PreCopyBarrier() {}

void BufferCacheRuntime::CopyBuffer(ID3D12Resource* dst_buffer, ID3D12Resource* src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies,
                                    [[maybe_unused]] bool barrier,
                                    [[maybe_unused]] bool can_reorder) {
    if (!command_list.IsValid() || !dst_buffer || !src_buffer) {
        return;
    }
    const bool same_resource = dst_buffer == src_buffer;
    command_list.Transition(dst_buffer, D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_COPY_DEST);
    if (!same_resource) {
        command_list.Transition(src_buffer, D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    for (const VideoCommon::BufferCopy& copy : copies) {
        command_list.CopyBufferRegion(dst_buffer, copy.dst_offset, src_buffer, copy.src_offset,
                                      copy.size);
    }
    command_list.Transition(dst_buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_COMMON);
    if (!same_resource) {
        command_list.Transition(src_buffer, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                D3D12_RESOURCE_STATE_COMMON);
    }
}

void BufferCacheRuntime::PostCopyBarrier() {}

void BufferCacheRuntime::ClearBuffer(ID3D12Resource* dest_buffer, u32 offset, size_t size,
                                     u32 value) {
    if (!command_list.IsValid() || !dest_buffer || size == 0) {
        return;
    }
    // Fill an upload buffer with the pattern and copy it over. A UAV clear would avoid the
    // copy but requires paired CPU/GPU descriptors; replace this when the descriptor heap
    // management is in place.
    static constexpr u64 clear_chunk = 64 * 1024;
    StagingBufferRef staging = staging_pool.Request(clear_chunk, StagingUsage::Upload);
    if (!staging.buffer) {
        return;
    }
    u32* pattern = reinterpret_cast<u32*>(staging.mapped_span.data());
    std::fill_n(pattern, clear_chunk / sizeof(u32), value);

    command_list.Transition(dest_buffer, D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_COPY_DEST);
    u64 remaining = size;
    u64 dst_offset = offset;
    while (remaining > 0) {
        const u64 chunk = std::min<u64>(remaining, clear_chunk);
        command_list.CopyBufferRegion(dest_buffer, dst_offset, staging.buffer, staging.offset,
                                      chunk);
        dst_offset += chunk;
        remaining -= chunk;
    }
    command_list.Transition(dest_buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_COMMON);
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 base_vertex, u32 num_indices, ID3D12Resource* buffer,
                                         u32 offset, u32 size) {
    if (topology == PrimitiveTopology::Quads || topology == PrimitiveTopology::QuadStrip) {
        if (!logged_quad_index) {
            logged_quad_index = true;
            LOG_ERROR(Render_D3D12, "Quad index expansion not implemented yet (topology={})",
                      static_cast<u32>(topology));
        }
    }
    DXGI_FORMAT dxgi_format = DXGI_FORMAT_R32_UINT;
    switch (index_format) {
    case IndexFormat::UnsignedByte:
        if (!logged_u8_index) {
            logged_u8_index = true;
            LOG_ERROR(Render_D3D12, "8-bit index expansion not implemented yet");
        }
        dxgi_format = DXGI_FORMAT_R16_UINT;
        break;
    case IndexFormat::UnsignedShort:
        dxgi_format = DXGI_FORMAT_R16_UINT;
        break;
    case IndexFormat::UnsignedInt:
    default:
        dxgi_format = DXGI_FORMAT_R32_UINT;
        break;
    }
    const D3D12_GPU_VIRTUAL_ADDRESS address =
        buffer ? buffer->GetGPUVirtualAddress() + offset : D3D12_GPU_VIRTUAL_ADDRESS{0};
    index_binding = IndexBinding{
        .address = address,
        .size = size,
        .first_index = base_vertex,
        .num_indices = num_indices,
        .format = dxgi_format,
        .valid = buffer != nullptr,
    };
}

void BufferCacheRuntime::BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count) {
    if (!logged_quad_index) {
        logged_quad_index = true;
        LOG_ERROR(Render_D3D12,
                  "Quad index generation not implemented yet (topology={} first={} count={})",
                  static_cast<u32>(topology), first, count);
    }
}

void BufferCacheRuntime::BindVertexBuffer(u32 index, ID3D12Resource* buffer, u32 offset, u32 size,
                                          u32 stride) {
    if (index >= MAX_VERTEX_BUFFERS) {
        return;
    }
    vertex_bindings[index] = D3D12_VERTEX_BUFFER_VIEW{
        .BufferLocation = buffer ? buffer->GetGPUVirtualAddress() + offset
                                 : D3D12_GPU_VIRTUAL_ADDRESS{0},
        .SizeInBytes = size,
        .StrideInBytes = stride,
    };
}

void BufferCacheRuntime::BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    for (u32 i = 0; i < bindings.buffers.size(); ++i) {
        const u32 index = bindings.min_index + i;
        if (index >= MAX_VERTEX_BUFFERS) {
            break;
        }
        const Buffer* buffer = bindings.buffers[i];
        vertex_bindings[index] = D3D12_VERTEX_BUFFER_VIEW{
            .BufferLocation = buffer->GpuAddr() + bindings.offsets[i],
            .SizeInBytes = static_cast<u32>(bindings.sizes[i]),
            .StrideInBytes = static_cast<u32>(bindings.strides[i]),
        };
    }
}

void BufferCacheRuntime::BindTransformFeedbackBuffer(u32 index, ID3D12Resource* buffer,
                                                     u32 offset, u32 size) {
    if (!logged_transform_feedback) {
        logged_transform_feedback = true;
        LOG_ERROR(Render_D3D12, "Transform feedback bindings not implemented yet ({} {} {})",
                  index, offset, size);
    }
    (void)buffer;
}

void BufferCacheRuntime::BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    if (!logged_transform_feedback) {
        logged_transform_feedback = true;
        LOG_ERROR(Render_D3D12, "Transform feedback bindings not implemented yet ({} buffers)",
                  bindings.buffers.size());
    }
}

void BufferCacheRuntime::BindBuffer(ID3D12Resource* buffer, u32 offset, u32 size) {
    ID3D12Resource* resource = buffer ? buffer : null_buffer.Get();
    const D3D12_GPU_VIRTUAL_ADDRESS base =
        resource ? resource->GetGPUVirtualAddress() : D3D12_GPU_VIRTUAL_ADDRESS{0};
    // Note: root CBVs require 256-byte aligned addresses; the draw translation copies
    // unaligned uniform ranges into an aligned staging region as needed.
    resource_bindings.push_back(ResourceBinding{
        .address = base + offset,
        .offset = offset,
        .size = size,
        .view = {},
    });
}

void BufferCacheRuntime::BindView(D3D12_CPU_DESCRIPTOR_HANDLE view) {
    resource_bindings.push_back(ResourceBinding{
        .address = 0,
        .offset = 0,
        .size = 0,
        .view = view,
    });
}

ID3D12Resource* BufferCacheRuntime::ReserveNullBuffer() {
    if (null_buffer) {
        return null_buffer.Get();
    }
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device.GetDevice()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&null_buffer)))) {
        LOG_CRITICAL(Render_D3D12, "Runtime null buffer creation failed");
    }
    return null_buffer.Get();
}

void BufferCacheRuntime::ClearDrawBindings() noexcept {
    index_binding = IndexBinding{};
    vertex_bindings = {};
    resource_bindings.clear();
}

} // namespace D3D12