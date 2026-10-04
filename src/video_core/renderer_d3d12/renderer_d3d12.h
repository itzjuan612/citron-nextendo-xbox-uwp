// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_base.h"
#include <wrl/client.h>

#include "video_core/renderer_d3d12/d3d12_command_list.h"
#include "video_core/renderer_d3d12/d3d12_descriptor_heap.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_graphics_pipeline.h"
#include "video_core/renderer_d3d12/d3d12_rasterizer.h"
#include "video_core/renderer_d3d12/d3d12_shader_compiler.h"
#include "video_core/renderer_d3d12/d3d12_swapchain.h"

namespace Tegra {
class GPU;
} // namespace Tegra

namespace D3D12 {

/// Native Direct3D 12 renderer backend (Xbox / UWP target).
///
/// Bring-up status: device + CoreWindow/HWND swapchain; the SPIR-V -> DXIL shader path and
/// the pipeline/descriptor foundation (command lists, root signatures, PSOs) are in place and
/// exercised by a bring-up triangle. The Maxwell command translation (texture/buffer caches,
/// real draw translation) is built out incrementally on top of this foundation.
class RendererD3D12 final : public VideoCore::RendererBase {
public:
    explicit RendererD3D12(Core::Frontend::EmuWindow& emu_window,
                           Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu,
                           std::unique_ptr<Core::Frontend::GraphicsContext> context);
    ~RendererD3D12() override;

    void Composite(std::span<const Tegra::FramebufferConfig> framebuffer) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override {
        return &rasterizer;
    }

    void PresentPending() override;

    [[nodiscard]] std::string GetDeviceVendor() const override;

private:
    /// Compiles the embedded blit shaders and creates the blit PSO.
    void PrepareBlit();

    /// Blits the guest framebuffer that `info` points at into the current
    /// swapchain back buffer (aspect-correct full-viewport blit).
    void RenderBlit(const Tegra::FramebufferConfig& framebuffer,
                        const AccelerateDisplayInfo& info);

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Tegra::GPU& gpu;
    Device device;
    Swapchain swapchain;
    CommandList command_list;
    ShaderCompiler shader_compiler;
    /// Shader-visible SRV heap holding the guest framebuffer view for the blit.
    DescriptorHeap blit_srv_heap;
    /// Shader-visible sampler heap holding the blit sampler (recreated per frame).
    DescriptorHeap blit_sampler_heap;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> blit_root_signature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> blit_pipeline;
    bool blit_attempted{};
    bool blit_ready{};
    RasterizerD3D12 rasterizer;
};

} // namespace D3D12
