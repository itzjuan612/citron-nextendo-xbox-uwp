// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/d3d12_shader_compiler.h"

#include <algorithm>
#include <cstring>

// windows.h defines min/max macros that break std::min.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <unknwn.h>
#include <objidl.h>

#include <dxcapi.h>
#include <wrl/client.h>

#include "common/logging.h"
#include "video_core/renderer_d3d12/mesa/dxil_versions.h"

namespace D3D12 {

using Microsoft::WRL::ComPtr;

namespace {

/// Minimal `IDxcBlob` implementation used to hand DXIL bytes to the validator.
/// Mirrors Mesa's `ShaderBlob` in src/microsoft/compiler/dxil_validator.cpp.
class ShaderBlob final : public IDxcBlob {
public:
    ShaderBlob(void* data, size_t size) : data{data}, size{size} {}

    LPVOID STDMETHODCALLTYPE GetBufferPointer() override {
        return data;
    }

    SIZE_T STDMETHODCALLTYPE GetBufferSize() override {
        return size;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override {
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
        return 1;
    }

    ULONG STDMETHODCALLTYPE Release() override {
        return 0;
    }

private:
    void* data;
    size_t size;
};

void DxilLogCallback(void* priv, const char* msg) {
    auto* error = static_cast<std::string*>(priv);
    if (error) {
        *error += msg;
    }
}

} // Anonymous namespace

ShaderCompiler::ShaderCompiler() {
    LoadLibraries();
    if (!IsValid()) {
        LOG_ERROR(Render_D3D12, "Shader compiler unavailable: {}", last_error);
    } else {
        LOG_INFO(Render_D3D12,
                 "Shader compiler ready (spirv_to_dxil, DXIL validator {}.{})",
                 validator_version >> 16, validator_version & 0xffff);
    }
}

ShaderCompiler::~ShaderCompiler() {
    if (validator) {
        static_cast<IDxcValidator*>(validator)->Release();
        validator = nullptr;
    }
}

void ShaderCompiler::LoadLibraries() {
    if (!spirv_to_dxil_library.Open("spirv_to_dxil.dll")) {
        last_error = "failed to load spirv_to_dxil.dll";
        return;
    }
    if (!spirv_to_dxil_library.GetSymbol("spirv_to_dxil", &spirv_to_dxil_fn) ||
        !spirv_to_dxil_library.GetSymbol("spirv_to_dxil_free", &spirv_to_dxil_free_fn)) {
        last_error = "spirv_to_dxil.dll is missing expected exports";
        return;
    }

    if (!dxil_library.Open("dxil.dll")) {
        last_error = "failed to load dxil.dll (required to sign DXIL)";
        return;
    }
    if (!dxil_library.GetSymbol("DxcCreateInstance", &dxil_create_instance)) {
        last_error = "dxil.dll is missing DxcCreateInstance";
        return;
    }
    const auto create_instance = reinterpret_cast<DxcCreateInstanceProc>(dxil_create_instance);

    ComPtr<IDxcValidator> new_validator;
    HRESULT hr = create_instance(CLSID_DxcValidator, IID_PPV_ARGS(&new_validator));
    if (FAILED(hr) || !new_validator) {
        last_error = "failed to create the DXIL validator";
        return;
    }
    // Keep the validator alive for the life of the compiler; re-creating it per compile
    // retains DXC per-instance state.
    validator = new_validator.Detach();

    ComPtr<IDxcVersionInfo> version_info;
    if (SUCCEEDED(static_cast<IDxcValidator*>(validator)->QueryInterface(
            IID_PPV_ARGS(&version_info)))) {
        UINT32 major = 0;
        UINT32 minor = 0;
        if (SUCCEEDED(version_info->GetVersion(&major, &minor)) && major == 1) {
            validator_version = DXIL_VALIDATOR_1_0 + std::min(minor, 8u);
        }
    }
    if (validator_version == 0) {
        validator_version = DXIL_VALIDATOR_1_4;
    }
}

std::vector<u8> ShaderCompiler::Compile(std::span<const u32> spirv, ShaderStage stage,
                                        ShaderMetadata& metadata) {
    std::scoped_lock lock{mutex};
    last_error.clear();

    if (!IsValid()) {
        last_error = "shader compiler is not initialised";
        return {};
    }

    std::vector<u8> dxil = Translate(spirv, stage, metadata);
    if (dxil.empty()) {
        return {};
    }
    if (!Sign(dxil)) {
        return {};
    }
    return dxil;
}

std::vector<u8> ShaderCompiler::Translate(std::span<const u32> spirv, ShaderStage stage,
                                          ShaderMetadata& metadata) {
    dxil_spirv_runtime_conf conf{};
    conf.runtime_data_cbv = {31, 0};
    conf.push_constant_cbv = {30, 0};
    conf.first_vertex_and_base_instance_mode = DXIL_SPIRV_SYSVAL_TYPE_ZERO;
    conf.workgroup_id_mode = DXIL_SPIRV_SYSVAL_TYPE_NATIVE;
    conf.yz_flip.mode = DXIL_SPIRV_YZ_FLIP_NONE;
    conf.declared_read_only_images_as_srvs = true;
    conf.inferred_read_only_images_as_srvs = true;
    conf.force_sample_rate_shading = false;
    conf.shader_model_max = SHADER_MODEL_6_2;

    dxil_spirv_debug_options debug{};
    debug.dump_nir = false;

    std::string logger_output;
    dxil_spirv_logger logger{};
    logger.priv = &logger_output;
    logger.log = &DxilLogCallback;

    dxil_spirv_object object{};
    const bool success =
        spirv_to_dxil_fn(spirv.data(), spirv.size(), nullptr, 0, stage, "main",
                         static_cast<dxil_validator_version>(validator_version), &debug, &conf,
                         &logger, &object);
    if (!success || object.binary.buffer == nullptr || object.binary.size == 0) {
        last_error = logger_output.empty() ? "spirv_to_dxil failed" : std::move(logger_output);
        return {};
    }

    metadata.requires_runtime_data = object.metadata.requires_runtime_data;
    metadata.needs_draw_sysvals = object.metadata.needs_draw_sysvals;

    std::vector<u8> dxil(object.binary.size);
    std::memcpy(dxil.data(), object.binary.buffer, object.binary.size);
    spirv_to_dxil_free_fn(&object);
    return dxil;
}

bool ShaderCompiler::Sign(std::vector<u8>& dxil) {
    ComPtr<IDxcValidator> dxc_validator;
    if (this->validator != nullptr) {
        dxc_validator = static_cast<IDxcValidator*>(this->validator);
    } else {
        const auto create_instance = reinterpret_cast<DxcCreateInstanceProc>(dxil_create_instance);
        HRESULT hr = create_instance(CLSID_DxcValidator, IID_PPV_ARGS(&dxc_validator));
        if (FAILED(hr) || !dxc_validator) {
            last_error = "failed to create the DXIL validator";
            return false;
        }
    }

    // The validator signs the container in place; the DXIL container is never resized.
    ShaderBlob blob(dxil.data(), dxil.size());
    ComPtr<IDxcOperationResult> result;
    dxc_validator->Validate(&blob, DxcValidatorFlags_InPlaceEdit, &result);

    HRESULT status = E_FAIL;
    if (result) {
        result->GetStatus(&status);
    }
    if (FAILED(status)) {
        last_error = "DXIL signing failed";
        ComPtr<IDxcBlobEncoding> errors;
        if (result && SUCCEEDED(result->GetErrorBuffer(&errors)) && errors) {
            last_error += ": ";
            last_error.append(static_cast<const char*>(errors->GetBufferPointer()),
                              errors->GetBufferSize());
        }
        return false;
    }
    return true;
}

} // namespace D3D12
