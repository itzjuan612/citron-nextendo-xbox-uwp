// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_swapchain.h"

namespace D3D12 {

Swapchain::Swapchain(Device& device_,
                     const Core::Frontend::EmuWindow::WindowSystemInfo& window_info, u32 width_,
                     u32 height_)
    : device{device_}, width{width_ != 0 ? width_ : 1280}, height{height_ != 0 ? height_ : 720} {
    // Store the window info; the actual DXGI swapchain is created in Init() on the UI thread.
    saved_window_info = window_info;
}

Swapchain::~Swapchain() = default;

void Swapchain::Init() {
    if (!device.IsValid()) {
        return;
    }

    const auto& window_info = saved_window_info;
    IDXGIFactory4* factory = device.GetFactory();
    ID3D12CommandQueue* queue = device.GetQueue();

    DXGI_SWAP_CHAIN_DESC1 desc{};
    // Use the requested framebuffer size rather than 0 ("use window size"): at swapchain
    // creation time the CoreWindow may not be laid out yet and DXGI would give us a tiny
    // (8x8) buffer.
    desc.Width = width;
    desc.Height = height;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.Stereo = FALSE;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    desc.Flags = 0;

    ComPtr<IDXGISwapChain1> swapchain1;
    HRESULT hr = E_FAIL;

    if (window_info.core_window) {
        // UWP/Xbox: this is the only presentation path that works on console.
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        hr = factory->CreateSwapChainForCoreWindow(
            queue, static_cast<IUnknown*>(window_info.core_window), &desc, nullptr,
            swapchain1.ReleaseAndGetAddressOf());
    } else if (window_info.render_surface) {
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.Width = width;
        desc.Height = height;
        hr = factory->CreateSwapChainForHwnd(queue, static_cast<HWND>(window_info.render_surface),
                                             &desc, nullptr, nullptr,
                                             swapchain1.ReleaseAndGetAddressOf());
    } else {
        LOG_ERROR(Render_D3D12, "No swapchain surface provided (headless)");
        return;
    }

    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "Failed to create swapchain: {:#x}", static_cast<u32>(hr));
        return;
    }

    hr = swapchain1->QueryInterface(IID_PPV_ARGS(&swapchain));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "Failed to query IDXGISwapChain3: {:#x}", static_cast<u32>(hr));
        return;
    }

    CreateBackBuffers();
    LOG_INFO(Render_D3D12, "Swapchain created ({}x{}, {})", width, height,
             window_info.core_window ? "CoreWindow" : "HWND");
}

void Swapchain::CreateBackBuffers() {
    ID3D12Device* d3d_device = device.GetDevice();

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.NumDescriptors = 2;
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    HRESULT hr = d3d_device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateDescriptorHeap(RTV) failed: {:#x}", static_cast<u32>(hr));
        return;
    }

    rtv_descriptor_size =
        d3d_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_CPU_DESCRIPTOR_HANDLE handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 2; ++i) {
        hr = swapchain->GetBuffer(i, IID_PPV_ARGS(&back_buffers[i]));
        if (FAILED(hr)) {
            LOG_ERROR(Render_D3D12, "GetBuffer({}) failed: {:#x}", i, static_cast<u32>(hr));
            return;
        }
        d3d_device->CreateRenderTargetView(back_buffers[i].Get(), nullptr, handle);
        handle.ptr += rtv_descriptor_size;
    }

    hr = d3d_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            IID_PPV_ARGS(&allocator));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateCommandAllocator failed: {:#x}", static_cast<u32>(hr));
        return;
    }
    hr = d3d_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                       IID_PPV_ARGS(&command_list));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateCommandList failed: {:#x}", static_cast<u32>(hr));
        return;
    }
    command_list->Close();
}

void Swapchain::ClearAndPresent(f32 r, f32 g, f32 b) {
    if (!swapchain || !command_list) {
        return;
    }

    allocator->Reset();
    command_list->Reset(allocator.Get(), nullptr);

    const UINT index = swapchain->GetCurrentBackBufferIndex();

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = back_buffers[index].Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    command_list->ResourceBarrier(1, &barrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += static_cast<SIZE_T>(index) * rtv_descriptor_size;
    const float clear_colour[4] = {r, g, b, 1.0f};
    command_list->ClearRenderTargetView(rtv, clear_colour, 0, nullptr);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    command_list->ResourceBarrier(1, &barrier);

    command_list->Close();
    ID3D12CommandList* lists[] = {command_list.Get()};
    device.GetQueue()->ExecuteCommandLists(1, lists);

    const HRESULT hr = swapchain->Present(1, 0);
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "Present failed: {:#x}, removed reason: {:#x}",
                  static_cast<u32>(hr),
                  static_cast<u32>(device.GetDevice()->GetDeviceRemovedReason()));
    }
}

void Swapchain::Present() {
    if (!swapchain) {
        return;
    }
    const HRESULT hr = swapchain->Present(1, 0);
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "Present failed: {:#x}, removed reason: {:#x}",
                  static_cast<u32>(hr),
                  static_cast<u32>(device.GetDevice()->GetDeviceRemovedReason()));
    }
}

u32 Swapchain::GetCurrentBackBufferIndex() const {
    if (!swapchain) {
        return 0;
    }
    return static_cast<u32>(swapchain->GetCurrentBackBufferIndex());
}

ID3D12Resource* Swapchain::GetBackBuffer() const {
    if (!swapchain) {
        return nullptr;
    }
    return back_buffers[swapchain->GetCurrentBackBufferIndex()].Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE Swapchain::GetBackBufferRtv() const {
    D3D12_CPU_DESCRIPTOR_HANDLE handle{};
    if (!swapchain || !rtv_heap) {
        return handle;
    }
    handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(swapchain->GetCurrentBackBufferIndex()) * rtv_descriptor_size;
    return handle;
}

} // namespace D3D12
