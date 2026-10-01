// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <d3d12.h>
#include <wrl/client.h>

#include "common/common_types.h"

namespace D3D12 {

using Microsoft::WRL::ComPtr;

/// Descriptor layout of a pipeline, derived from `Shader::Info` binding assignment.
///
/// The `shader_recompiler` SPIR-V backend (with `unified_descriptor_binding = false`,
/// `has_split_descriptor_sets = true`) assigns uniforms to descriptor set 0 and all other
/// resources (storage buffers, textures, images) to descriptor set 1, using a binding counter
/// that accumulates across the stages of a pipeline. Mesa's `spirv_to_dxil` maps
/// `(set, binding)` to DXIL `(space, register)`, and splits combined image samplers into a
/// texture `tN` and a sampler `sN` at the same register. The root signature mirrors that:
/// CBVs in space 0 and SRV/Sampler/UAV tables in space 1, plus optional root CBVs for the
/// push-constant and runtime-data buffers at spaces 30/31.
struct RootSignatureParams {
    /// Total number of bindings used by the `b` register namespace (space 0).
    u32 num_cbv{};
    /// Total number of bindings used by the `t`/`u` register namespaces (space 1).
    u32 num_resources{};
    /// Push constants are lowered to a root CBV at space 30, binding 0.
    bool needs_push_constants{};
    /// Runtime-data CBV (draw/workgroup sysvals) at space 31, binding 0.
    bool needs_runtime_data{};

    bool operator==(const RootSignatureParams&) const = default;
};

/// Root signature corresponding to a `RootSignatureParams` layout.
class RootSignature {
public:
    RootSignature() = default;
    RootSignature(ID3D12Device* device, const RootSignatureParams& params);
    ~RootSignature();

    RootSignature(const RootSignature&) = delete;
    RootSignature& operator=(const RootSignature&) = delete;
    RootSignature(RootSignature&&) = default;
    RootSignature& operator=(RootSignature&&) = default;

    [[nodiscard]] bool IsValid() const {
        return root_signature != nullptr;
    }

    [[nodiscard]] ID3D12RootSignature* Get() const {
        return root_signature.Get();
    }

    /// Returns the root parameter index of the CBV table, or INVALID_PARAMETER when absent.
    [[nodiscard]] u32 GetCbvTableIndex() const {
        return cbv_table_index;
    }

    [[nodiscard]] u32 GetSrvTableIndex() const {
        return srv_table_index;
    }

    [[nodiscard]] u32 GetSamplerTableIndex() const {
        return sampler_table_index;
    }

    [[nodiscard]] u32 GetUavTableIndex() const {
        return uav_table_index;
    }

    [[nodiscard]] u32 GetPushConstantIndex() const {
        return push_constant_index;
    }

    [[nodiscard]] u32 GetRuntimeDataIndex() const {
        return runtime_data_index;
    }

    static constexpr u32 INVALID_PARAMETER = 0xFFFFFFFFu;

private:
    ComPtr<ID3D12RootSignature> root_signature;
    u32 cbv_table_index{INVALID_PARAMETER};
    u32 srv_table_index{INVALID_PARAMETER};
    u32 sampler_table_index{INVALID_PARAMETER};
    u32 uav_table_index{INVALID_PARAMETER};
    u32 push_constant_index{INVALID_PARAMETER};
    u32 runtime_data_index{INVALID_PARAMETER};
};

/// Register spaces reserved for the translator's synthetic buffers.
constexpr u32 RUNTIME_DATA_REGISTER_SPACE = 31;
constexpr u32 PUSH_CONSTANT_REGISTER_SPACE = 30;

} // namespace D3D12
