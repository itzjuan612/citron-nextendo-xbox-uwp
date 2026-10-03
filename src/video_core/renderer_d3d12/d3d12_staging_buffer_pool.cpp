// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>

#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_staging_buffer_pool.h"

namespace D3D12 {

StagingBufferPool::StagingBufferPool(Device& device_) : device{device_} {}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(u64 size, StagingUsage usage, bool deferred) {
    if (size == 0) {
        size = 1;
    }
    // Find an expired entry of the same size class and usage.
    const u64 size_class = u64{1} << std::bit_width(size - 1);
    Entry* chosen = nullptr;
    u64 chosen_index = ~u64{0};
    for (u64 i = 0; i < entries.size(); ++i) {
        Entry& entry = entries[i];
        if (entry.deferred != deferred || entry.in_use || entry.usage != usage) {
            continue;
        }
        if (entry.size != size_class) {
            continue;
        }
        if (tick <= entry.last_used_tick + NUM_SYNCS) {
            continue;
        }
        chosen = &entry;
        chosen_index = i;
        break;
    }
    if (!chosen) {
        chosen = &CreateEntry(size_class, usage);
        chosen_index = entries.size() - 1;
    }
    chosen->in_use = true;
    chosen->deferred = deferred;
    chosen->last_used_tick = tick;
    return StagingBufferRef{
        .buffer = chosen->resource.Get(),
        .offset = 0,
        .mapped_span = chosen->mapped_span.subspan(0, size),
        .index = chosen_index,
    };
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    if (ref.index >= entries.size()) {
        return;
    }
    entries[ref.index].in_use = false;
    entries[ref.index].deferred = false;
    deferred_frees.emplace_back(ref.index, tick);
    ref = StagingBufferRef{};
}

void StagingBufferPool::TickFrame() {
    ++tick;
    while (!deferred_frees.empty() && deferred_frees.front().second + NUM_SYNCS <= tick) {
        deferred_frees.pop_front();
    }
}

void StagingBufferPool::Nuke() {
    entries.clear();
    deferred_frees.clear();
    total_bytes = 0;
}

u64 StagingBufferPool::GetMemoryUsage() const {
    return total_bytes;
}

StagingBufferPool::Entry& StagingBufferPool::CreateEntry(u64 size, StagingUsage usage) {
    ID3D12Device* d3d = device.GetDevice();
    Entry entry;
    entry.size = size;
    entry.usage = usage;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = usage == StagingUsage::Upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK;
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
    const D3D12_RESOURCE_STATES state =
        usage == StagingUsage::Upload ? D3D12_RESOURCE_STATE_GENERIC_READ
                                      : D3D12_RESOURCE_STATE_COPY_DEST;
    if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                            IID_PPV_ARGS(&entry.resource)))) {
        LOG_CRITICAL(Render_D3D12, "Staging buffer creation failed ({} bytes)", size);
        // Keep the pool usable with an empty entry; the caller will see a null buffer.
        entries.emplace_back(std::move(entry));
        return entries.back();
    }
    void* mapped = nullptr;
    if (FAILED(entry.resource->Map(0, nullptr, &mapped)) || !mapped) {
        LOG_CRITICAL(Render_D3D12, "Staging buffer Map failed ({} bytes)", size);
    }
    entry.mapped_span = std::span<u8>{static_cast<u8*>(mapped), static_cast<size_t>(size)};
    total_bytes += size;
    entries.emplace_back(std::move(entry));
    return entries.back();
}

} // namespace D3D12
