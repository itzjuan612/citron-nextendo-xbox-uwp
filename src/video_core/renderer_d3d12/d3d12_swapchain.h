// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include "common/common_types.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_d3d12/d3d12_device.h"

namespace D3D12 {

/// DXGI swapchain bound to either a CoreWindow (UWP/Xbox) or an HWND (desktop).
///
/// IMPORTANT: on Xbox the swapchain MUST be created on the thread that owns the
/// CoreWindow (the UI thread). Creating it from a different thread (e.g. the boot
/// thread) is a COM apartment violation: `CreateSwapChainForCoreWindow` returns S_OK
/// and buffers render fine (verified via GPU readback) but nothing ever reaches the
/// display. `Init()` therefore runs from `PresentPending()` on the UI thread.
class Swapchain {
public:
    Swapchain(Device& device, const Core::Frontend::EmuWindow::WindowSystemInfo& window_info,
              u32 width, u32 height);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    /// Creates the DXGI swapchain and back buffers. Must be called on the UI thread.
    void Init();

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

    /// Request a Present on the UI thread (Xbox CoreWindow presentation requires it;
    /// presenting from the GPU thread leaves the screen black).
    void RequestPresent() {
        present_done.store(false, std::memory_order_release);
        present_requested.store(true, std::memory_order_release);
    }

    /// Called from the UI thread: creates the swapchain on first call, then presents
    /// if a request is pending. MUST run on the CoreWindow owner thread.
    void PresentIfRequested() {
        if (!swapchain) {
            Init();
            // Signal done even on the init call so the GPU thread does not stall.
            present_done.store(true, std::memory_order_release);
            return;
        }
        if (present_requested.exchange(false, std::memory_order_acq_rel)) {
            Present();
            present_done.store(true, std::memory_order_release);
        }
    }

    /// Waits for the UI thread to present. Times out after 2s to avoid deadlock if the
    /// UI thread is busy or has gone away.
    void WaitForPresent() {
        for (int i = 0; i < 2000 && !present_done.load(std::memory_order_acquire); ++i) {
            Sleep(1);
        }
    }

    [[nodiscard]] u32 Width() const {
        return width;
    }

    [[nodiscard]] u32 Height() const {
        return height;
    }

private:
    void CreateBackBuffers();

    Device& device;
    Core::Frontend::EmuWindow::WindowSystemInfo saved_window_info{};
    ComPtr<IDXGISwapChain3> swapchain;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    ComPtr<ID3D12Resource> back_buffers[2];
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    std::atomic<bool> present_requested{};
    std::atomic<bool> present_done{true};
    UINT rtv_descriptor_size{};
    u32 width{};
    u32 height{};
};

} // namespace D3D12
