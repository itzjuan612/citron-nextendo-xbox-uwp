// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/d3d12_query_cache.h"

namespace D3D12 {

struct QueryCacheRuntimeImpl {
    explicit QueryCacheRuntimeImpl(QueryCacheRuntime& runtime)
        : payload_streamer(static_cast<size_t>(VideoCommon::QueryType::Payload), runtime),
          needed_minus_succeeded_streamer(
              static_cast<size_t>(
                  VideoCommon::QueryType::StreamingPrimitivesNeededMinusSucceeded),
              runtime, 0u),
          zpass_streamer(static_cast<size_t>(VideoCommon::QueryType::ZPassPixelCount64),
                         runtime, 1u),
          streaming_byte_streamer(
              static_cast<size_t>(VideoCommon::QueryType::StreamingByteCount), runtime,
              0u) {}

    VideoCommon::GuestStreamer<QueryCacheParams> payload_streamer;
    VideoCommon::StubStreamer<QueryCacheParams> needed_minus_succeeded_streamer;
    VideoCommon::StubStreamer<QueryCacheParams> zpass_streamer;
    VideoCommon::StubStreamer<QueryCacheParams> streaming_byte_streamer;
    Tegra::Engines::Maxwell3D* maxwell3d{};
};

QueryCacheRuntime::QueryCacheRuntime(Device& device_, CommandList& command_list_,
                                     Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : device{device_}, command_list{command_list_}, device_memory{device_memory_} {
    // Device/command list are used by the GPU-produced query paths that land with
    // the real draw translation; the CPU fallback paths below only need the impl.
    impl = std::make_unique<QueryCacheRuntimeImpl>(*this);
}

QueryCacheRuntime::~QueryCacheRuntime() = default;

void QueryCacheRuntime::Barriers([[maybe_unused]] bool is_prebarrier) {}

void QueryCacheRuntime::EndHostConditionalRendering() {}

void QueryCacheRuntime::PauseHostConditionalRendering() {}

void QueryCacheRuntime::ResumeHostConditionalRendering() {}

bool QueryCacheRuntime::HostConditionalRenderingCompareValue(
    [[maybe_unused]] VideoCommon::LookupData object_1, [[maybe_unused]] bool qc_dirty) {
    return true;
}

bool QueryCacheRuntime::HostConditionalRenderingCompareValues(
    [[maybe_unused]] VideoCommon::LookupData object_1,
    [[maybe_unused]] VideoCommon::LookupData object_2, [[maybe_unused]] bool qc_dirty,
    [[maybe_unused]] bool equal_check) {
    return true;
}

VideoCommon::StreamerInterface* QueryCacheRuntime::GetStreamerInterface(
    VideoCommon::QueryType query_type) {
    switch (query_type) {
    case VideoCommon::QueryType::Payload:
        return &impl->payload_streamer;
    case VideoCommon::QueryType::StreamingPrimitivesNeededMinusSucceeded:
        return &impl->needed_minus_succeeded_streamer;
    case VideoCommon::QueryType::ZPassPixelCount64:
        return &impl->zpass_streamer;
    case VideoCommon::QueryType::StreamingByteCount:
        return &impl->streaming_byte_streamer;
    default:
        // The template falls back to Payload with value 1 for unmapped types.
        return nullptr;
    }
}

void QueryCacheRuntime::Bind3DEngine(Tegra::Engines::Maxwell3D* maxwell3d_) {
    impl->maxwell3d = maxwell3d_;
}

template <typename Func>
void QueryCacheRuntime::View3DRegs(Func&& func) {
    if (impl->maxwell3d) {
        func(*impl->maxwell3d);
    }
}

} // namespace D3D12
