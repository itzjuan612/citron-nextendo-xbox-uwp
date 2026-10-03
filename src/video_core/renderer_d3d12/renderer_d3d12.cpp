// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/renderer_d3d12.h"

#include <array>
#include <cstring>
#include <stdexcept>

#include "common/logging.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_d3d12/d3d12_bringup_shaders.h"

namespace D3D12 {

RendererD3D12::RendererD3D12(Core::Frontend::EmuWindow& emu_window,
                             Tegra::MaxwellDeviceMemoryManager& device_memory_, Tegra::GPU& gpu_,
                             std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), device_memory{device_memory_}, gpu{gpu_},
      device{}, swapchain{device, render_window.GetWindowInfo(),
                          render_window.GetFramebufferLayout().width,
                          render_window.GetFramebufferLayout().height},
      command_list{device.GetDevice()}, shader_compiler{}, rasterizer{gpu} {
    if (!device.IsValid()) {
        LOG_ERROR(Render_D3D12, "D3D12 device initialization failed");
        throw std::runtime_error{"D3D12 device initialization failed"};
    }
    // The swapchain is created on the UI thread (CoreWindow owner) via PresentPending();
    // it is not available yet here.
    LOG_INFO(Render_D3D12, "RendererD3D12 initialised ({} {:#x})", device.GetAdapterName(),
             static_cast<u32>(device.GetFeatureLevel()));
}

RendererD3D12::~RendererD3D12() {
    device.WaitForIdle();
}

void RendererD3D12::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    // The swapchain is created on the UI thread (CoreWindow owner). Until that happens,
    // skip drawing and just nudge the UI thread to do its init.
    if (!swapchain.IsValid()) {
        swapchain.RequestPresent();
        swapchain.WaitForPresent();
        return;
    }

    if (!bringup_attempted) {
        PrepareBringUpTriangle();
    }

    if (bringup_ready) {
        RenderBringUpTriangle();
    } else {
        // Diagnostic clear: unmissable orange so a black screen can be told apart from a
        // present that works but a draw that does not run.
        swapchain.ClearAndPresent(1.0f, 0.25f, 0.05f);
    }

    // Present on the UI thread. Xbox CoreWindow presentation only works from the thread
    // that owns the CoreWindow; presenting from the GPU thread leaves the screen black.
    swapchain.RequestPresent();
    swapchain.WaitForPresent();

    if (m_current_frame < 8 || (m_current_frame % 120) == 0) {
        LOG_INFO(Render_D3D12, "Composite frame {} (bringup_ready={}, fb={}, tid={})",
                 m_current_frame, bringup_ready, framebuffers.size(),
                 GetCurrentThreadId());
    }
    m_current_frame++;
    gpu.RendererFrameEndNotify();
    render_window.OnFrameDisplayed();
}

void RendererD3D12::PresentPending() {
    swapchain.PresentIfRequested();
}

void RendererD3D12::PrepareBringUpTriangle() {
    bringup_attempted = true;

    if (!shader_compiler.IsValid()) {
        LOG_ERROR(Render_D3D12, "Bring-up triangle disabled: {}", shader_compiler.GetLastError());
        return;
    }

    ShaderMetadata vertex_metadata{};
    ShaderMetadata fragment_metadata{};
    std::vector<u8> vertex_dxil = shader_compiler.Compile(BRINGUP_VERTEX_SPIRV,
                                                          ShaderStage::DXIL_SPIRV_SHADER_VERTEX,
                                                          vertex_metadata);
    if (vertex_dxil.empty()) {
        LOG_ERROR(Render_D3D12, "Bring-up vertex shader failed: {}",
                  shader_compiler.GetLastError());
        return;
    }
    std::vector<u8> fragment_dxil = shader_compiler.Compile(
        BRINGUP_FRAGMENT_SPIRV, ShaderStage::DXIL_SPIRV_SHADER_FRAGMENT, fragment_metadata);
    if (fragment_dxil.empty()) {
        LOG_ERROR(Render_D3D12, "Bring-up fragment shader failed: {}",
                  shader_compiler.GetLastError());
        return;
    }

    GraphicsPipelineCacheKey key{};
    key.state.topology.Assign(Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Triangles);
    key.state.color_formats[0] = static_cast<u8>(Tegra::RenderTargetFormat::A8R8G8B8_UNORM);
    // Enable all four colour write masks on attachment 0 (the union's raw layout is one bit
    // per channel).
    key.state.attachments[0].raw = 0xF;

    std::array<std::vector<u8>, MAX_SHADER_STAGES> stages{};
    stages[0] = std::move(vertex_dxil);
    stages[4] = std::move(fragment_dxil);

    std::array<const Shader::Info*, MAX_SHADER_STAGES> infos{};
    std::array<bool, MAX_SHADER_STAGES> runtime_data{};

    bringup_pipeline.emplace(device.GetDevice(), key, stages, infos, runtime_data);
    if (!bringup_pipeline->IsValid()) {
        LOG_ERROR(Render_D3D12, "Bring-up graphics pipeline creation failed");
        bringup_pipeline.reset();
        return;
    }

    bringup_ready = true;
    LOG_INFO(Render_D3D12, "Bring-up triangle ready (SPIR-V -> DXIL -> signed PSO)");
}

void RendererD3D12::RenderBringUpTriangle() {
    if (!command_list.IsValid()) {
        return;
    }

    // Bring-up: keep the CPU and GPU in lockstep so the command allocators and the swapchain
    // back buffers can be reused without a full frame/fence ring.
    device.WaitForIdle();

    command_list.Reset();

    ID3D12Resource* back_buffer = swapchain.GetBackBuffer();
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = swapchain.GetBackBufferRtv();

    if (!logged_back_buffer) {
        logged_back_buffer = true;
        D3D12_RESOURCE_DESC desc{};
        if (back_buffer) {
            desc = back_buffer->GetDesc();
        }
        LOG_INFO(Render_D3D12,
                 "Bring-up draw: back_buffer={} fmt={} {}x{} rtv={:#x} (swapchain {}x{})",
                 back_buffer != nullptr, static_cast<u32>(desc.Format), desc.Width, desc.Height,
                 rtv.ptr, swapchain.Width(), swapchain.Height());
    }

    command_list.Transition(back_buffer, D3D12_RESOURCE_STATE_PRESENT,
                            D3D12_RESOURCE_STATE_RENDER_TARGET);
    command_list.SetRootSignature(bringup_pipeline->GetRootSignature().Get());
    command_list.SetPipelineState(bringup_pipeline->Get());
    command_list.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    command_list.OMSetRenderTargets(1, &rtv);

    D3D12_VIEWPORT viewport{};
    viewport.Width = static_cast<f32>(swapchain.Width());
    viewport.Height = static_cast<f32>(swapchain.Height());
    viewport.MaxDepth = 1.0f;
    command_list.SetViewport(viewport);

    D3D12_RECT scissor{};
    scissor.right = static_cast<LONG>(swapchain.Width());
    scissor.bottom = static_cast<LONG>(swapchain.Height());
    command_list.SetScissorRect(scissor);

    // Diagnostic clear: bright red so "clear+present works but the triangle does not
    // rasterize" can be told apart from "nothing is presented at all".
    const f32 clear_colour[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    command_list.ClearRenderTargetView(rtv, clear_colour);
    command_list.Draw(3, 1, 0, 0);
    command_list.Transition(back_buffer, D3D12_RESOURCE_STATE_RENDER_TARGET,
                            D3D12_RESOURCE_STATE_PRESENT);
    command_list.Execute(device);

    if (!probe_done) {
        probe_done = true;
        RunD3D12IsolationProbe(back_buffer, rtv);
    }
}

void RendererD3D12::RunD3D12IsolationProbe(ID3D12Resource* back_buffer,
                                           D3D12_CPU_DESCRIPTOR_HANDLE rtv) {
    ID3D12Device* d3d = device.GetDevice();
    device.WaitForIdle();

    // Helpers
    auto make_buffer = [&](u64 size, D3D12_RESOURCE_STATES state,
                           D3D12_HEAP_TYPE heap) -> ComPtr<ID3D12Resource> {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = heap;
        hp.CreationNodeMask = 1;
        hp.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = size;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> res;
        if (FAILED(d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                IID_PPV_ARGS(&res)))) {
            return {};
        }
        return res;
    };
    auto make_texture = [&](u32 w, u32 h) -> ComPtr<ID3D12Resource> {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        hp.CreationNodeMask = 1;
        hp.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w;
        rd.Height = h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        ComPtr<ID3D12Resource> res;
        D3D12_CLEAR_VALUE cv{};
        cv.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        cv.Color[0] = 1.0f;
        cv.Color[3] = 1.0f;
        if (FAILED(d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                D3D12_RESOURCE_STATE_RENDER_TARGET, &cv,
                                                IID_PPV_ARGS(&res)))) {
            return {};
        }
        return res;
    };
    auto make_rtv = [&](ID3D12Resource* res) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.NumDescriptors = 1;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ComPtr<ID3D12DescriptorHeap> heap;
        if (FAILED(d3d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)))) {
            return {};
        }
        const D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        d3d->CreateRenderTargetView(res, nullptr, h);
        // Leak the heap deliberately so the handle stays valid.
        heap.Detach();
        return h;
    };
    auto submit = [&](const auto& record) -> bool {
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> list;
        if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               IID_PPV_ARGS(&alloc))) ||
            FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                                          IID_PPV_ARGS(&list)))) {
            LOG_ERROR(Render_D3D12, "probe: allocator/list creation failed");
            return false;
        }
        list->Close();
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        record(list.Get());
        const HRESULT hr = list->Close();
        if (FAILED(hr)) {
            LOG_ERROR(Render_D3D12, "probe: Close failed {:#x}", static_cast<u32>(hr));
            return false;
        }
        ID3D12CommandList* lists[] = {list.Get()};
        device.GetQueue()->ExecuteCommandLists(1, lists);
        device.WaitForIdle();
        return true;
    };
    auto read_u32 = [&](ID3D12Resource* rb, const char* label) {
        if (!rb) {
            LOG_ERROR(Render_D3D12, "probe {}: no readback", label);
            return;
        }
        D3D12_RANGE range{0, 4};
        void* mapped = nullptr;
        if (FAILED(rb->Map(0, &range, &mapped)) || !mapped) {
            LOG_ERROR(Render_D3D12, "probe {}: Map failed", label);
            return;
        }
        u32 v = 0;
        std::memcpy(&v, mapped, 4);
        D3D12_RANGE written{0, 0};
        rb->Unmap(0, &written);
        LOG_INFO(Render_D3D12, "probe {} = {:#010x}", label, v);
    };

    // ---- Test 0: CPU canary. Proves the Map/readback path works with no GPU involved. ----
    {
        auto rb = make_buffer(256, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_READBACK);
        if (rb) {
            void* mapped = nullptr;
            if (SUCCEEDED(rb->Map(0, nullptr, &mapped)) && mapped) {
                u32 canary = 0xDEADBEEF;
                std::memcpy(mapped, &canary, 4);
                rb->Unmap(0, nullptr);
            }
            read_u32(rb.Get(), "0 CPU-canary (expect 0xdeadbeef)");
        } else {
            LOG_ERROR(Render_D3D12, "probe 0: readback buffer creation failed");
        }
    }

    // ---- Test 0b: GPU copy canary. Upload buffer -> CopyBufferRegion -> readback. ----
    {
        auto upload = make_buffer(256, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
        auto rb = make_buffer(256, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);
        if (upload && rb) {
            void* mapped = nullptr;
            if (SUCCEEDED(upload->Map(0, nullptr, &mapped)) && mapped) {
                u32 canary = 0xCAFEBABE;
                std::memcpy(mapped, &canary, 4);
                upload->Unmap(0, nullptr);
            }
            submit([&](ID3D12GraphicsCommandList* l) {
                l->CopyBufferRegion(rb.Get(), 0, upload.Get(), 0, 256);
            });
            read_u32(rb.Get(), "0b GPU-copy-canary (expect 0xcafebabe)");
        } else {
            LOG_ERROR(Render_D3D12, "probe 0b: setup failed upload={} rb={}", upload != nullptr,
                      rb != nullptr);
        }
    }

    // ---- Test A: GPU writes to its OWN texture ----
    {
        auto tex = make_texture(64, 64);
        auto rtv_h = make_rtv(tex.Get());
        auto rb = make_buffer(256, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);
        if (tex && rb && rtv_h.ptr != 0) {
            submit([&](ID3D12GraphicsCommandList* l) {
                const f32 green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
                l->ClearRenderTargetView(rtv_h, green, 0, nullptr);
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = tex.Get();
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                l->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION dst{};
                dst.pResource = rb.Get();
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                dst.PlacedFootprint.Footprint.Width = 4;
                dst.PlacedFootprint.Footprint.Height = 1;
                dst.PlacedFootprint.Footprint.Depth = 1;
                dst.PlacedFootprint.Footprint.RowPitch = 256;
                D3D12_TEXTURE_COPY_LOCATION src{};
                src.pResource = tex.Get();
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.SubresourceIndex = 0;
                D3D12_BOX box{};
                box.left = 8;
                box.top = 8;
                box.front = 0;
                box.right = 12;
                box.bottom = 9;
                box.back = 1;
                l->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
            });
            read_u32(rb.Get(), "A own-tex-clear (expect 0xff00ff00 green)");
        } else {
            LOG_ERROR(Render_D3D12, "probe A: setup failed tex={} rb={} rtv={}", tex != nullptr,
                      rb != nullptr, rtv_h.ptr != 0);
        }
    }

    // ---- Test B: copy a known colour INTO the swapchain back buffer, then read it back ----
    {
        auto src_tex = make_texture(4, 4);
        auto rb = make_buffer(256, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);
        if (src_tex && rb && back_buffer) {
            // Fill src_tex with blue via clear
            auto src_rtv = make_rtv(src_tex.Get());
            submit([&](ID3D12GraphicsCommandList* l) {
                if (src_rtv.ptr != 0) {
                    const f32 blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
                    l->ClearRenderTargetView(src_rtv, blue, 0, nullptr);
                }
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = src_tex.Get();
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                l->ResourceBarrier(1, &b);
                // back buffer -> COPY_DEST
                b.Transition.pResource = back_buffer;
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                l->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION dst{};
                dst.pResource = back_buffer;
                dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dst.SubresourceIndex = 0;
                D3D12_TEXTURE_COPY_LOCATION src{};
                src.pResource = src_tex.Get();
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.SubresourceIndex = 0;
                l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                // back buffer -> COPY_SOURCE -> readback
                b.Transition.pResource = back_buffer;
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                l->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION rdst{};
                rdst.pResource = rb.Get();
                rdst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                rdst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                rdst.PlacedFootprint.Footprint.Width = 4;
                rdst.PlacedFootprint.Footprint.Height = 1;
                rdst.PlacedFootprint.Footprint.Depth = 1;
                rdst.PlacedFootprint.Footprint.RowPitch = 256;
                D3D12_BOX box{};
                box.left = 0;
                box.top = 0;
                box.front = 0;
                box.right = 4;
                box.bottom = 1;
                box.back = 1;
                l->CopyTextureRegion(&rdst, 0, 0, 0, &dst, &box);
                // restore
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
                l->ResourceBarrier(1, &b);
            });
            read_u32(rb.Get(), "B copy-into-backbuffer (expect 0xff0000ff blue)");
        } else {
            LOG_ERROR(Render_D3D12, "probe B: setup failed src={} rb={} bb={}", src_tex != nullptr,
                      rb != nullptr, back_buffer != nullptr);
        }
    }

    // ---- Test C: clear the back buffer via RTV and read it back ----
    {
        auto rb = make_buffer(256, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);
        if (rb && back_buffer && rtv.ptr != 0) {
            submit([&](ID3D12GraphicsCommandList* l) {
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = back_buffer;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
                l->ResourceBarrier(1, &b);
                const f32 red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
                l->ClearRenderTargetView(rtv, red, 0, nullptr);
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                l->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION rdst{};
                rdst.pResource = rb.Get();
                rdst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                rdst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                rdst.PlacedFootprint.Footprint.Width = 4;
                rdst.PlacedFootprint.Footprint.Height = 1;
                rdst.PlacedFootprint.Footprint.Depth = 1;
                rdst.PlacedFootprint.Footprint.RowPitch = 256;
                D3D12_TEXTURE_COPY_LOCATION src{};
                src.pResource = back_buffer;
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.SubresourceIndex = 0;
                D3D12_BOX box{};
                box.left = 100;
                box.top = 100;
                box.front = 0;
                box.right = 104;
                box.bottom = 101;
                box.back = 1;
                l->CopyTextureRegion(&rdst, 0, 0, 0, &src, &box);
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
                l->ResourceBarrier(1, &b);
            });
            read_u32(rb.Get(), "C backbuffer-clear (expect 0xff0000ff red)");
        } else {
            LOG_ERROR(Render_D3D12, "probe C: setup failed rb={} bb={} rtv={}", rb != nullptr,
                      back_buffer != nullptr, rtv.ptr != 0);
        }
    }
}

std::vector<u8> RendererD3D12::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

std::string RendererD3D12::GetDeviceVendor() const {
    return device.GetAdapterName();
}

} // namespace D3D12
