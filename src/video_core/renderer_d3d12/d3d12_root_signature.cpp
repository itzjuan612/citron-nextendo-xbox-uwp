// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/d3d12_root_signature.h"

#include <vector>

#include "common/logging.h"

namespace D3D12 {

RootSignature::RootSignature(ID3D12Device* device, const RootSignatureParams& params) {
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    parameters.reserve(6);
    ranges.reserve(4);

    const auto add_table = [&](D3D12_DESCRIPTOR_RANGE_TYPE type, u32 register_space, u32 count) {
        const u32 range_index = static_cast<u32>(ranges.size());
        D3D12_DESCRIPTOR_RANGE range{};
        range.RangeType = type;
        range.NumDescriptors = count;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = register_space;
        range.OffsetInDescriptorsFromTableStart = 0;
        ranges.push_back(range);

        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable.NumDescriptorRanges = 1;
        parameter.DescriptorTable.pDescriptorRanges = ranges.data() + range_index;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters.push_back(parameter);
        return static_cast<u32>(parameters.size() - 1);
    };

    const auto add_root_cbv = [&](u32 register_space) {
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameter.Descriptor.RegisterSpace = register_space;
        parameter.Descriptor.ShaderRegister = 0;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters.push_back(parameter);
        return static_cast<u32>(parameters.size() - 1);
    };

    if (params.num_cbv > 0) {
        cbv_table_index = add_table(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 0, params.num_cbv);
    }
    if (params.num_resources > 0) {
        srv_table_index = add_table(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, params.num_resources);
        sampler_table_index = add_table(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1,
                                        params.num_resources);
        uav_table_index = add_table(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, params.num_resources);
    }
    if (params.needs_push_constants) {
        push_constant_index = add_root_cbv(PUSH_CONSTANT_REGISTER_SPACE);
    }
    if (params.needs_runtime_data) {
        runtime_data_index = add_root_cbv(RUNTIME_DATA_REGISTER_SPACE);
    }

    // `ranges` is reserve()d to its maximum size above, so the `pDescriptorRanges`
    // pointers stored in the parameters stay valid.
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = static_cast<UINT>(parameters.size());
    desc.pParameters = parameters.empty() ? nullptr : parameters.data();
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "D3D12SerializeRootSignature failed: {}",
                  error ? static_cast<const char*>(error->GetBufferPointer()) : "unknown error");
        return;
    }

    hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                     IID_PPV_ARGS(&root_signature));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateRootSignature failed: {:#x}", static_cast<u32>(hr));
        return;
    }

    LOG_DEBUG(Render_D3D12, "Root signature created (cbv={}, res={}, push_constants={}, rt={})",
              params.num_cbv, params.num_resources, params.needs_push_constants,
              params.needs_runtime_data);
}

RootSignature::~RootSignature() = default;

} // namespace D3D12
