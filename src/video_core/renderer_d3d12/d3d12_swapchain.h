// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include "common/common_types.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_d3d12/d3d12_device.h"

namespace D3D12 {

/// DXGI swapchain bound to either a CoreWindow (UWP/Xbox) or an HWND (desktop).
class Swapchain {
public:
    Swapchain(Device& device, const Core::Frontend::EmuWindow::WindowSystemInfo& window_info,
              u32 width, u32 height);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    [[nodiscard]] bool IsValid() const {
        return swapchain != nullptr;
    }

    /// Index of the back buffer that is currently being rendered to.
    [[nodiscard]] u32 GetCurrentBackBufferIndex() const;

    /// Current back buffer resource (owned by the swapchain).
    [[nodiscard]] ID3D12Resource* GetBackBuffer() const;

    /// RTV of the current back buffer (CPU handle into the swapchain RTV heap).
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE GetBackBufferRtv() const;

    /// Clears the current back buffer and presents it.
    void Present();

    /// Presents the current back buffer with a solid clear colour (bring-up path).
    void ClearAndPresent(f32 r, f32 g, f32 b);

    [[nodiscard]] u32 Width() const {
        return width;
    }

    [[nodiscard]] u32 Height() const {
        return height;
    }

private:
    void CreateBackBuffers();

    Device& device;
    ComPtr<IDXGISwapChain3> swapchain;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    ComPtr<ID3D12Resource> back_buffers[2];
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    UINT rtv_descriptor_size{};
    u32 width{};
    u32 height{};
};

} // namespace D3D12
