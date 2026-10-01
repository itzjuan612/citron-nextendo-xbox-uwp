// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/renderer_d3d12.h"

#include <array>
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
    if (!swapchain.IsValid()) {
        LOG_ERROR(Render_D3D12, "D3D12 swapchain initialization failed");
        throw std::runtime_error{"D3D12 swapchain initialization failed"};
    }
    LOG_INFO(Render_D3D12, "RendererD3D12 initialised ({} {:#x})", device.GetAdapterName(),
             static_cast<u32>(device.GetFeatureLevel()));
}

RendererD3D12::~RendererD3D12() {
    device.WaitForIdle();
}

void RendererD3D12::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    if (!bringup_attempted) {
        PrepareBringUpTriangle();
    }

    if (bringup_ready) {
        RenderBringUpTriangle();
        swapchain.Present();
    } else {
        // Fall back to the cleared-frame bring-up path when the shader path failed.
        swapchain.ClearAndPresent(0.05f, 0.05f, 0.08f);
    }

    m_current_frame++;
    gpu.RendererFrameEndNotify();
    render_window.OnFrameDisplayed();
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

    const f32 clear_colour[4] = {0.02f, 0.02f, 0.05f, 1.0f};
    command_list.ClearRenderTargetView(rtv, clear_colour);
    command_list.Draw(3, 1, 0, 0);

    command_list.Transition(back_buffer, D3D12_RESOURCE_STATE_RENDER_TARGET,
                            D3D12_RESOURCE_STATE_PRESENT);
    command_list.Execute(device);
}

std::vector<u8> RendererD3D12::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

std::string RendererD3D12::GetDeviceVendor() const {
    return device.GetAdapterName();
}

} // namespace D3D12
