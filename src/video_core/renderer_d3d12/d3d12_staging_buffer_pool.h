// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <deque>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace D3D12 {

class Device;

using Microsoft::WRL::ComPtr;

enum class StagingUsage {
    Upload,   ///< CPU-writable, GENERIC_READ, source of GPU copies.
    Download, ///< GPU-writable, COPY_DEST, read back by the CPU.
};

/// Handle to a region of a staging buffer. The resource is kept alive by the pool until
/// the entry is recycled (see `StagingBufferPool` lifetime rules).
struct StagingBufferRef {
    ID3D12Resource* buffer{};
    u64 offset{};
    std::span<u8> mapped_span;
    u64 index{~u64{0}};
};

/// Pool of persistently-mapped staging buffers for GPU uploads and downloads.
///
/// Entries are rounded up to power-of-two size classes and recycled once the GPU has
/// retired the work that used them. Lifetime is currently modelled with a tick counter
/// (`NUM_SYNCS` frames of latency); replace this with fence ticks when the D3D12
/// scheduler lands. Deferred refs (async downloads) are only recycled after
/// `FreeDeferred` has been called and the latency window has passed.
class StagingBufferPool {
public:
    static constexpr u64 NUM_SYNCS = 32;

    explicit StagingBufferPool(Device& device);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    /// Requests `size` bytes of staging memory. Non-deferred refs are recycled
    /// automatically once the GPU is done with them.
    [[nodiscard]] StagingBufferRef Request(u64 size, StagingUsage usage, bool deferred = false);

    /// Marks a deferred ref as unused; it is recycled after the latency window.
    void FreeDeferred(StagingBufferRef& ref);

    /// Advances the lifetime model and recycles expired entries.
    void TickFrame();

    /// Drops every entry. Only safe once the GPU is idle.
    void Nuke();

    [[nodiscard]] u64 GetMemoryUsage() const;

private:
    struct Entry {
        ComPtr<ID3D12Resource> resource;
        std::span<u8> mapped_span;
        u64 size{};
        u64 last_used_tick{};
        StagingUsage usage{};
        bool in_use{};
        bool deferred{};
    };

    Entry& CreateEntry(u64 size, StagingUsage usage);

    Device& device;
    std::vector<Entry> entries;
    std::deque<std::pair<u64, u64>> deferred_frees; // (entry index, free tick)
    u64 tick{};
    u64 total_bytes{};
};

} // namespace D3D12
