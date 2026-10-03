// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_d3d12/d3d12_fixed_pipeline_state.h"
#include "video_core/renderer_d3d12/d3d12_root_signature.h"

namespace D3D12 {

using Microsoft::WRL::ComPtr;

/// Number of Maxwell shader stages tracked per graphics pipeline (excludes compute).
constexpr u32 MAX_SHADER_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

/// Cache key of a graphics pipeline: the guest shader hashes plus the packed fixed state.
struct GraphicsPipelineCacheKey {
    std::array<u64, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram> unique_hashes{};
    FixedPipelineState state{};

    [[nodiscard]] size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};

/// A D3D12 graphics pipeline: root signature, PSO and the static render target formats.
class GraphicsPipeline {
public:
    GraphicsPipeline() = default;

    /// Creates the root signature and PSO from DXIL blobs. `stages[i]` corresponds to the
    /// Maxwell stage order (`Tegra::Engines::Maxwell3D::Regs::ShaderType`: Vertex=1, ...).
    /// Empty blobs disable that stage.
    GraphicsPipeline(ID3D12Device* device, const GraphicsPipelineCacheKey& key,
                     const std::array<std::vector<u8>, MAX_SHADER_STAGES>& stages,
                     const std::array<const Shader::Info*, MAX_SHADER_STAGES>& infos,
                     std::array<bool, MAX_SHADER_STAGES> needs_runtime_data);
    ~GraphicsPipeline();

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(GraphicsPipeline&&) = default;
    GraphicsPipeline& operator=(GraphicsPipeline&&) = default;

    [[nodiscard]] bool IsValid() const {
        return pipeline != nullptr;
    }

    [[nodiscard]] ID3D12PipelineState* Get() const {
        return pipeline.Get();
    }

    [[nodiscard]] RootSignature& GetRootSignature() {
        return root_signature;
    }

    [[nodiscard]] const RootSignature& GetRootSignature() const {
        return root_signature;
    }

    [[nodiscard]] const FixedPipelineState& State() const {
        return key.state;
    }

    /// Computes the descriptor layout implied by a set of shader infos.
    static RootSignatureParams BuildRootSignatureParams(
        const std::array<const Shader::Info*, MAX_SHADER_STAGES>& infos,
        const std::array<bool, MAX_SHADER_STAGES>& needs_runtime_data);

private:
    GraphicsPipelineCacheKey key{};
    RootSignature root_signature;
    ComPtr<ID3D12PipelineState> pipeline;
};

/// Converts a guest pixel format to its DXGI equivalent (DXGI_FORMAT_UNKNOWN when unsupported).
[[nodiscard]] DXGI_FORMAT SurfaceFormat(VideoCore::Surface::PixelFormat format);

} // namespace D3D12
