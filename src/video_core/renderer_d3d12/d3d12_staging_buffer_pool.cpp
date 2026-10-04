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
        // (b) Reclaim: the oldest expired entry of the same size class, usage and
        // deferred flag, even if still marked in_use. Safe because the pool is only
        // pumped after the GPU has gone idle (FlushCommands waits before recycling),
        // so an in_use entry here is no longer referenced by in-flight work.
        u64 best_tick = ~u64{0};
        for (u64 i = 0; i < entries.size(); ++i) {
            const Entry& entry = entries[i];
            if (entry.size != size_class || entry.usage != usage ||
                entry.deferred != deferred) {
                continue;
            }
            if (entry.last_used_tick + NUM_SYNCS > tick) {
                continue;
            }
            if (entry.last_used_tick < best_tick) {
                best_tick = entry.last_used_tick;
                chosen_index = i;
            }
        }
        if (chosen_index != ~u64{0}) {
            chosen = &entries[chosen_index];
        }
    }
    if (!chosen) {
        // (c) Overwrite: the oldest entry of the same size class, usage and deferred
        // flag regardless of age or in_use state, instead of allocating unboundedly.
        u64 best_tick = ~u64{0};
        for (u64 i = 0; i < entries.size(); ++i) {
            const Entry& entry = entries[i];
            if (entry.size != size_class || entry.usage != usage ||
                entry.deferred != deferred) {
                continue;
            }
            if (entry.last_used_tick < best_tick) {
                best_tick = entry.last_used_tick;
                chosen_index = i;
            }
        }
        if (chosen_index != ~u64{0}) {
            chosen = &entries[chosen_index];
        }
    }
    if (!chosen) {
        // Hard cap: evict LRU entries until the new entry fits (at least one
        // eviction attempt is bounded by the pool being non-empty).
        while (!entries.empty() && total_bytes + size_class > MAX_POOL_BYTES) {
            EvictLRU();
        }
        chosen = &CreateEntry(size_class, usage);
        chosen_index = entries.size() - 1;
    }
    if (!chosen->resource || chosen->mapped_span.size() < size) {
        // Allocation failed (memory pressure) or the entry is short: hand back an empty
        // ref so callers can no-op instead of dereferencing a null mapping.
        chosen->in_use = false;
        return StagingBufferRef{};
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
    Entry& entry = entries[ref.index];
    if (entry.resource.Get() != ref.buffer) {
        // Stale index: EvictLRU() erases entries via vector::erase, shifting later
        // indices, so the slot may now hold a different resource. Drop the ref
        // without touching the new occupant. (Reused-in-place entries keep their
        // resource, and TickFrame never touches entries, so no check is needed
        // for the reclaim/overwrite paths.)
        ref = StagingBufferRef{};
        return;
    }
    entry.in_use = false;
    entry.deferred = false;
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

std::span<u8> StagingBufferPool::MappedSpan(ID3D12Resource* resource) const {
    for (const Entry& entry : entries) {
        if (entry.resource.Get() == resource) {
            return entry.mapped_span;
        }
    }
    return {};
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
    HRESULT hr = d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state,
                                               nullptr, IID_PPV_ARGS(&entry.resource));
    if (FAILED(hr) && !entries.empty()) {
        // Memory pressure: evict the LRU entry and retry once before giving up.
        EvictLRU();
        hr = d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                          IID_PPV_ARGS(&entry.resource));
    }
    if (FAILED(hr)) {
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
    if (!logged_high_water && total_bytes > 128ull * 1024 * 1024) {
        logged_high_water = true;
        LOG_INFO(Render_D3D12, "Staging pool high-water {} bytes", total_bytes);
    }
    entries.emplace_back(std::move(entry));
    return entries.back();
}

void StagingBufferPool::EvictLRU() {
    if (entries.empty()) {
        return;
    }
    // Prefer an idle entry; only evict an in_use entry when every entry is busy.
    u64 victim = ~u64{0};
    u64 victim_tick = ~u64{0};
    for (u64 i = 0; i < entries.size(); ++i) {
        if (entries[i].in_use) {
            continue;
        }
        if (entries[i].last_used_tick < victim_tick) {
            victim_tick = entries[i].last_used_tick;
            victim = i;
        }
    }
    if (victim == ~u64{0}) {
        for (u64 i = 0; i < entries.size(); ++i) {
            if (entries[i].last_used_tick < victim_tick) {
                victim_tick = entries[i].last_used_tick;
                victim = i;
            }
        }
    }
    if (victim == ~u64{0}) {
        return;
    }
    // Only successful entries were charged to total_bytes; failed (null resource)
    // entries were emplaced without charging the pool.
    const u64 evicted_size = entries[victim].size;
    const bool evicted_charged = entries[victim].resource.Get() != nullptr;
    entries.erase(entries.begin() + static_cast<ptrdiff_t>(victim));
    if (evicted_charged) {
        total_bytes = total_bytes >= evicted_size ? total_bytes - evicted_size : 0;
    }
    // Note: erase shifts later indices, so deferred_frees may hold stale indices.
    // FreeDeferred validates bounds and resource identity before touching an entry.
}

} // namespace D3D12
