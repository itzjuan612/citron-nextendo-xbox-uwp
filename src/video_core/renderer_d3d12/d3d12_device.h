// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <string>

namespace D3D12 {

using Microsoft::WRL::ComPtr;

/// Owns the D3D12 device and the single direct command queue used by the renderer.
class Device {
public:
    explicit Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] bool IsValid() const {
        return device != nullptr && queue != nullptr;
    }

    [[nodiscard]] ID3D12Device* GetDevice() const {
        return device.Get();
    }

    [[nodiscard]] ID3D12CommandQueue* GetQueue() const {
        return queue.Get();
    }

    [[nodiscard]] IDXGIFactory4* GetFactory() const {
        return factory.Get();
    }

    [[nodiscard]] D3D_FEATURE_LEVEL GetFeatureLevel() const {
        return feature_level;
    }

    [[nodiscard]] const std::string& GetAdapterName() const {
        return adapter_name;
    }

    /// Blocks until the GPU has finished the currently submitted work.
    void WaitForIdle();

private:
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    UINT64 fence_value{};
    HANDLE fence_event{};
    D3D_FEATURE_LEVEL feature_level{D3D_FEATURE_LEVEL_11_0};
    std::string adapter_name{"unknown"};
};

} // namespace D3D12
