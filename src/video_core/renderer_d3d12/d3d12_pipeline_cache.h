// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_d3d12/d3d12_graphics_pipeline.h"
#include "video_core/renderer_d3d12/d3d12_shader_compiler.h"
#include "video_core/shader_cache.h"

namespace Tegra {
namespace Engines {
class Maxwell3D;
}
} // namespace Tegra

namespace D3D12 {

class Device;

using Microsoft::WRL::ComPtr;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

struct PipelineKeyHash {
    size_t operator()(const GraphicsPipelineCacheKey& key) const noexcept {
        return key.Hash();
    }
};

/// On-demand translator from Maxwell programs to signed D3D12 graphics pipelines.
///
/// Each draw refreshes the guest shader hashes and the fixed pipeline state; on a cache
/// miss the Maxwell programs are translated to IR, emitted as SPIR-V, converted to signed
/// DXIL and baked into a PSO. Synchronous builds only (no disk cache or worker threads
/// yet — a stall on first use of each unique pipeline).
class PipelineCache : public VideoCommon::ShaderCache {
public:
    /// A pipeline plus the translated `Shader::Info`s needed for descriptor setup.
    /// The translation-time IR (`Shader::IR::Program`) is intentionally dropped after
    /// each stage compiles: only `info` is read at draw time, so retaining the full IR
    /// per cached pipeline would pin the per-stage block/syntax vectors forever.
    struct StoredPipeline {
        std::unique_ptr<GraphicsPipeline> pipeline;
        std::array<Shader::Info, MAX_SHADER_STAGES> infos_storage;
        std::array<const Shader::Info*, MAX_SHADER_STAGES> infos{};
    };

    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory, Device& device,
                           ShaderCompiler& compiler);
    ~PipelineCache();

    /// Returns the pipeline for the currently bound Maxwell state, building it on a miss.
    /// Returns nullptr when shaders are invalid or translation/compilation fails.
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    /// Same as above but also exposes the stored shader infos for descriptor setup.
    [[nodiscard]] StoredPipeline* CurrentStoredPipeline();

private:
    [[nodiscard]] StoredPipeline* CurrentPipelineSlowPath();
    [[nodiscard]] std::unique_ptr<StoredPipeline> CreatePipeline(
        const GraphicsPipelineCacheKey& key);

    Device& device;
    ShaderCompiler& compiler;
    Shader::Profile profile{};
    Shader::HostTranslateInfo host_info{};
    ShaderPools pools;
    GraphicsPipelineCacheKey current_key{};
    StoredPipeline* current_pipeline{};
    size_t compiled_stage_count{};
    size_t compiled_stage_bytes{};
    std::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<StoredPipeline>, PipelineKeyHash>
        cache;
};

} // namespace D3D12
