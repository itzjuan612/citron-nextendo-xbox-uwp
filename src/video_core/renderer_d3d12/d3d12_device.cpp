// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging.h"
#include "common/string_util.h"
#include "video_core/renderer_d3d12/d3d12_device.h"

namespace D3D12 {

Device::Device() {
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateDXGIFactory1 failed: {:#x}", static_cast<u32>(hr));
        return;
    }

    // Prefer a hardware adapter (skip the WARP/software adapter).
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter.Reset();
            continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&device)))) {
            adapter_name = Common::UTF16ToUTF8(desc.Description);
            break;
        }
        adapter.Reset();
    }

    // Fall back to the default adapter (this is what succeeds on Xbox).
    if (!device) {
        hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (FAILED(hr)) {
            LOG_ERROR(Render_D3D12, "D3D12CreateDevice failed: {:#x}", static_cast<u32>(hr));
            return;
        }
        ComPtr<IDXGIDevice> dxgi_device;
        ComPtr<IDXGIAdapter> default_adapter;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dxgi_device))) &&
            SUCCEEDED(dxgi_device->GetAdapter(&default_adapter))) {
            DXGI_ADAPTER_DESC desc{};
            if (SUCCEEDED(default_adapter->GetDesc(&desc))) {
                adapter_name = Common::UTF16ToUTF8(desc.Description);
            }
        }
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateCommandQueue failed: {:#x}", static_cast<u32>(hr));
        return;
    }

    const D3D_FEATURE_LEVEL requested_levels[] = {
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    D3D12_FEATURE_DATA_FEATURE_LEVELS levels{};
    levels.NumFeatureLevels = static_cast<UINT>(std::size(requested_levels));
    levels.pFeatureLevelsRequested = requested_levels;
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &levels,
                                              sizeof(levels)))) {
        feature_level = levels.MaxSupportedFeatureLevel;
    }

    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (SUCCEEDED(hr)) {
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }

    LOG_INFO(Render_D3D12, "Using D3D12 adapter '{}', feature level {:#x}", adapter_name,
             static_cast<u32>(feature_level));
}

Device::~Device() {
    WaitForIdle();
    if (fence_event) {
        CloseHandle(fence_event);
    }
}

void Device::WaitForIdle() {
    if (!queue || !fence) {
        return;
    }
    // Serialize: `queue->Signal` values must increase monotonically and the shared
    // `fence_event` may only have one SetEventOnCompletion registration at a time.
    // Without this, concurrent waiters (channel threads flushing + the GPU thread
    // compositing) can issue non-monotonic signals or strand a waiter forever.
    std::scoped_lock lock{idle_mutex};
    const UINT64 value = ++fence_value;
    if (FAILED(queue->Signal(fence.Get(), value))) {
        return;
    }
    if (fence->GetCompletedValue() < value) {
        if (fence_event && SUCCEEDED(fence->SetEventOnCompletion(value, fence_event))) {
            WaitForSingleObject(fence_event, INFINITE);
        } else {
            while (fence->GetCompletedValue() < value) {
                Sleep(0);
            }
        }
    }
}

} // namespace D3D12
