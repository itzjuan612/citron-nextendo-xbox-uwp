// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/d3d12_descriptor_heap.h"

#include "common/logging.h"

namespace D3D12 {

DescriptorHeap::DescriptorHeap(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type_,
                               u32 capacity_, bool shader_visible)
    : type{type_}, capacity{capacity_} {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = type;
    desc.NumDescriptors = capacity;
    desc.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
                                : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    desc.NodeMask = 0;

    const HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateDescriptorHeap(type={}, count={}) failed: {:#x}",
                  static_cast<u32>(type), capacity, static_cast<u32>(hr));
        return;
    }
    descriptor_size = device->GetDescriptorHandleIncrementSize(type);
}

DescriptorHeap::~DescriptorHeap() = default;

} // namespace D3D12
