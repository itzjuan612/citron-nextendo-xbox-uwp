// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "common/dynamic_library.h"
#include "video_core/renderer_d3d12/mesa/spirv_to_dxil.h"

namespace D3D12 {

/// Shader stages as understood by Mesa's spirv_to_dxil.
using ShaderStage = dxil_spirv_shader_stage;

/// Extra information the DXIL translation reports to the caller.
struct ShaderMetadata {
    bool requires_runtime_data{};
    bool needs_draw_sysvals{};
};

/// Translates the SPIR-V produced by `shader_recompiler` into signed DXIL.
///
/// Uses Mesa's `spirv_to_dxil.dll` (MIT, the same translator Mesa's Dozen driver uses)
/// to emit DXIL from SPIR-V, then signs the container with Microsoft's DXIL validator
/// (`dxil.dll`) as required by the D3D12 runtime for SM6 shaders. Both DLLs are loaded
/// from the application/package directory; the resulting blobs are directly usable in
/// `D3D12_GRAPHICS_PIPELINE_STATE_DESC` / `D3D12_COMPUTE_PIPELINE_STATE_DESC`.
class ShaderCompiler {
public:
    ShaderCompiler();
    ~ShaderCompiler();

    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;

    [[nodiscard]] bool IsValid() const {
        return spirv_to_dxil_fn != nullptr && dxil_create_instance != nullptr;
    }

    /// Compiles a SPIR-V module (entry point "main") to signed DXIL.
    /// Returns an empty vector on failure; `GetLastError()` holds the reason.
    [[nodiscard]] std::vector<u8> Compile(std::span<const u32> spirv, ShaderStage stage,
                                          ShaderMetadata& metadata);

    [[nodiscard]] const std::string& GetLastError() const {
        return last_error;
    }

private:
    void LoadLibraries();

    std::vector<u8> Translate(std::span<const u32> spirv, ShaderStage stage,
                              ShaderMetadata& metadata);
    bool Sign(std::vector<u8>& dxil);

    Common::DynamicLibrary spirv_to_dxil_library;
    Common::DynamicLibrary dxil_library;

    decltype(&spirv_to_dxil) spirv_to_dxil_fn{};
    decltype(&spirv_to_dxil_free) spirv_to_dxil_free_fn{};
    void* dxil_create_instance{};
    u32 validator_version{};

    std::mutex mutex;
    std::string last_error;
};

} // namespace D3D12
