// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>

#include <memory>
#include <span>

#include "common/common_types.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/query_cache/query_cache.h"
#include "video_core/query_cache/types.h"

namespace Tegra {
namespace Engines {
class Maxwell3D;
}
} // namespace Tegra

namespace D3D12 {

class CommandList;
class Device;

struct QueryCacheRuntimeImpl;

/// Host side of the Maxwell query/report mechanism.
///
/// Counter values are produced on the CPU for now (guest streamers): `SyncValues` writes
/// them straight into guest memory. GPU-produced queries (occlusion/timestamps via D3D12
/// query heaps) and predicated conditional rendering land with the real draw translation —
/// conditional rendering currently always executes.
class QueryCacheRuntime {
public:
    explicit QueryCacheRuntime(Device& device, CommandList& command_list,
                               Tegra::MaxwellDeviceMemoryManager& device_memory);
    ~QueryCacheRuntime();

    template <typename SyncValuesType>
    void SyncValues(std::span<SyncValuesType> values,
                    [[maybe_unused]] ID3D12Resource* base_src_buffer = nullptr) {
        for (const SyncValuesType& sync : values) {
            const auto size = static_cast<size_t>(sync.size);
            if (size == sizeof(u64)) {
                const u64 value = sync.value;
                device_memory.WriteBlockUnsafe(sync.address, &value, size);
            } else {
                const u32 value = static_cast<u32>(sync.value);
                device_memory.WriteBlockUnsafe(sync.address, &value, size);
            }
        }
    }

    void Barriers(bool is_prebarrier);

    void EndHostConditionalRendering();

    void PauseHostConditionalRendering();

    void ResumeHostConditionalRendering();

    bool HostConditionalRenderingCompareValue(VideoCommon::LookupData object_1, bool qc_dirty);

    bool HostConditionalRenderingCompareValues(VideoCommon::LookupData object_1,
                                               VideoCommon::LookupData object_2, bool qc_dirty,
                                               bool equal_check);

    VideoCommon::StreamerInterface* GetStreamerInterface(VideoCommon::QueryType query_type);

    void Bind3DEngine(Tegra::Engines::Maxwell3D* maxwell3d);

    template <typename Func>
    void View3DRegs(Func&& func);

private:
    friend struct QueryCacheRuntimeImpl;
    std::unique_ptr<QueryCacheRuntimeImpl> impl;
    Device& device;
    CommandList& command_list;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
};

struct QueryCacheParams {
    using RuntimeType = D3D12::QueryCacheRuntime;
};

using QueryCache = VideoCommon::QueryCacheBase<QueryCacheParams>;

} // namespace D3D12
