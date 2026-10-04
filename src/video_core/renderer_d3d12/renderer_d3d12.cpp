// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/renderer_d3d12.h"

#include <array>
#include <stdexcept>

#include <array>
#include <stdexcept>

#include "common/logging.h"
#include "common/settings.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_d3d12/d3d12_blit_shaders.h"

namespace D3D12 {

RendererD3D12::RendererD3D12(Core::Frontend::EmuWindow& emu_window,
                             Tegra::MaxwellDeviceMemoryManager& device_memory_, Tegra::GPU& gpu_,
                             std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), device_memory{device_memory_}, gpu{gpu_},
      device{}, swapchain{device, render_window.GetWindowInfo(),
                          render_window.GetFramebufferLayout().width,
                          render_window.GetFramebufferLayout().height},
      command_list{device.GetDevice()}, shader_compiler{},
      blit_srv_heap{device.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, true},
      blit_sampler_heap{device.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true},
      rasterizer{gpu, device_memory, device, shader_compiler} {
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

    if (!blit_attempted) {
        PrepareBlit();
    }

    // The present list is z-ordered; the last framebuffer is the topmost (game) frame.
    bool presented = false;
    u32 blit_width = 0;
    u32 blit_height = 0;
    Common::Rectangle<f32> blit_crop{};
    if (blit_ready && !framebuffers.empty()) {
        const Tegra::FramebufferConfig& framebuffer = framebuffers.back();
        const auto info = rasterizer.AccelerateDisplay(
            framebuffer, framebuffer.address + framebuffer.offset, framebuffer.stride);
        if (info && info->view && info->view->Sampled().ptr != 0) {
            RenderBlit(framebuffer, *info);
            presented = true;
            blit_width = info->width;
            blit_height = info->height;
            blit_crop = Tegra::NormalizeCrop(framebuffer, info->width, info->height);
        }
    }
    if (!presented) {
        // Diagnostic clear: unmissable orange so a black screen can be told apart from a
        // present that works but a draw that does not run.
        swapchain.ClearAndPresent(1.0f, 0.25f, 0.05f);
    }

    // Present-path tracing: one DEBUG line every 20 Composite calls.
    static u32 composite_frames = 0;
    if (++composite_frames % 20 == 0) {
        if (presented) {
            LOG_DEBUG(Render_D3D12,
                      "Composite frame {}: blit path (framebuffer present, "
                      "{}x{}, uv/crop=({}, {})-({}, {}))",
                      composite_frames, blit_width, blit_height, blit_crop.left,
                      blit_crop.top, blit_crop.right, blit_crop.bottom);
        } else {
            LOG_DEBUG(Render_D3D12, "Composite frame {}: clear path ({})",
                      composite_frames,
                      framebuffers.empty()
                          ? "no framebuffer"
                          : (!blit_ready ? "blit not ready"
                                         : "accelerate display failed"));
        }
    }

    // Present on the UI thread. Xbox CoreWindow presentation only works from the thread
    // that owns the CoreWindow; presenting from the GPU thread leaves the screen black.
    swapchain.RequestPresent();
    swapchain.WaitForPresent();

    m_current_frame++;
    gpu.RendererFrameEndNotify();
    render_window.OnFrameDisplayed();
}

void RendererD3D12::PresentPending() {
    swapchain.PresentIfRequested();
}

void RendererD3D12::PrepareBlit() {
    blit_attempted = true;

    if (!shader_compiler.IsValid()) {
        LOG_ERROR(Render_D3D12, "Present blit disabled: {}", shader_compiler.GetLastError());
        return;
    }

    ShaderMetadata vertex_metadata{};
    ShaderMetadata fragment_metadata{};
    std::vector<u8> vertex_dxil = shader_compiler.Compile(BLIT_VERTEX_SPIRV,
                                                          ShaderStage::DXIL_SPIRV_SHADER_VERTEX,
                                                          vertex_metadata);
    if (vertex_dxil.empty()) {
        LOG_ERROR(Render_D3D12, "Blit vertex shader failed: {}",
                  shader_compiler.GetLastError());
        return;
    }
    std::vector<u8> fragment_dxil = shader_compiler.Compile(
        BLIT_FRAGMENT_SPIRV, ShaderStage::DXIL_SPIRV_SHADER_FRAGMENT, fragment_metadata);
    if (fragment_dxil.empty()) {
        LOG_ERROR(Render_D3D12, "Blit fragment shader failed: {}",
                  shader_compiler.GetLastError());
        return;
    }

    // Blit root signature: an SRV table at t0 (PIXEL) fed from the shader-visible
    // SRV heap, a sampler table at s0 (PIXEL) fed from the shader-visible sampler
    // heap (recreated per frame from the scaling-filter setting), and 32-bit
    // constants at b0 (VERTEX) carrying the UV scale/offset.
    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 1;
    srv_range.BaseShaderRegister = 0;
    srv_range.RegisterSpace = 0;
    srv_range.OffsetInDescriptorsFromTableStart = 0;

    D3D12_DESCRIPTOR_RANGE sampler_range{};
    sampler_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    sampler_range.NumDescriptors = 1;
    sampler_range.BaseShaderRegister = 0;
    sampler_range.RegisterSpace = 0;
    sampler_range.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv_range;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &sampler_range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.Num32BitValues = 4;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.RegisterSpace = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    D3D12_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.NumParameters = 3;
    rs_desc.pParameters = params;
    rs_desc.NumStaticSamplers = 0;
    rs_desc.pStaticSamplers = nullptr;
    rs_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    Microsoft::WRL::ComPtr<ID3DBlob> rs_blob;
    Microsoft::WRL::ComPtr<ID3DBlob> rs_error;
    if (FAILED(D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                            &rs_blob, &rs_error))) {
        LOG_ERROR(Render_D3D12, "Blit root signature serialization failed");
        return;
    }
    ID3D12Device* const d3d = device.GetDevice();
    const HRESULT blit_rs_hr = d3d->CreateRootSignature(
        0, rs_blob->GetBufferPointer(), rs_blob->GetBufferSize(), IID_PPV_ARGS(&blit_root_signature));
    if (FAILED(blit_rs_hr)) {
        LOG_ERROR(Render_D3D12, "Blit root signature creation failed: {:#x}, removed reason: {:#x}",
                  static_cast<u32>(blit_rs_hr), static_cast<u32>(d3d->GetDeviceRemovedReason()));
        return;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = blit_root_signature.Get();
    desc.VS = {vertex_dxil.data(), vertex_dxil.size()};
    desc.PS = {fragment_dxil.data(), fragment_dxil.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.DepthStencilState.StencilEnable = FALSE;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    if (FAILED(d3d->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&blit_pipeline)))) {
        LOG_ERROR(Render_D3D12, "Blit pipeline creation failed");
        return;
    }

    blit_ready = true;
    LOG_INFO(Render_D3D12, "Present blit ready (SPIR-V -> DXIL -> signed PSO)");
}

void RendererD3D12::RenderBlit(const Tegra::FramebufferConfig& framebuffer,
                                const AccelerateDisplayInfo& info) {
    if (!command_list.IsValid() || !blit_pipeline) {
        return;
    }

    // Keep the CPU and GPU in lockstep so the command allocator and the swapchain
    // back buffers can be reused without a full frame/fence ring.
    device.WaitForIdle();

    command_list.Reset();

    ID3D12Resource* back_buffer = swapchain.GetBackBuffer();
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = swapchain.GetBackBufferRtv();

    command_list.Transition(back_buffer, D3D12_RESOURCE_STATE_PRESENT,
                            D3D12_RESOURCE_STATE_RENDER_TARGET);

    // Clear to black first so the letterbox bars stay black.
    const f32 black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    command_list.ClearRenderTargetView(rtv, black);

    // Blit sampler, recreated per frame from the scaling-filter setting: point
    // for nearest-neighbour, linear for everything else.
    D3D12_SAMPLER_DESC sampler_desc{};
    sampler_desc.Filter = Settings::values.scaling_filter.GetValue() ==
                                  Settings::ScalingFilter::NearestNeighbor
                              ? D3D12_FILTER_MIN_MAG_MIP_POINT
                              : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
    device.GetDevice()->CreateSampler(&sampler_desc, blit_sampler_heap.CpuHandle(0));

    // Source SRV: the texture-cache view lives in a CPU-visible heap; copy it
    // into the shader-visible blit heap for the descriptor table.
    device.GetDevice()->CopyDescriptorsSimple(1, blit_srv_heap.CpuHandle(0),
                                               info.view->Sampled(),
                                               D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ID3D12DescriptorHeap* const heaps[] = {blit_srv_heap.Get(), blit_sampler_heap.Get()};
    command_list.Get()->SetDescriptorHeaps(2, heaps);

    command_list.SetRootSignature(blit_root_signature.Get());
    command_list.SetPipelineState(blit_pipeline.Get());
    command_list.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    command_list.OMSetRenderTargets(1, &rtv);
    command_list.SetGraphicsRootDescriptorTable(0, blit_srv_heap.GpuHandle(0));
    command_list.SetGraphicsRootDescriptorTable(1, blit_sampler_heap.GpuHandle(0));

    // UV scale/offset from the framebuffer crop (the full frame when no crop
    // is set), so the guest's visible region maps onto the triangle's [0,1] UVs.
    const Common::Rectangle<f32> crop =
        Tegra::NormalizeCrop(framebuffer, info.width, info.height);
    const std::array<f32, 4> uv_scale_offset = {
        crop.GetWidth(), crop.GetHeight(), crop.left, crop.top,
    };
    command_list.Get()->SetGraphicsRoot32BitConstants(2, static_cast<u32>(uv_scale_offset.size()),
                                                       uv_scale_offset.data(), 0);

    // Aspect-correct letterboxed viewport: fit the guest framebuffer into the
    // window layout, centered. The scissor stays full-back-buffer.
    const Layout::FramebufferLayout layout = render_window.GetFramebufferLayout();
    const f32 guest_aspect = info.height != 0
                                 ? static_cast<f32>(info.width) / static_cast<f32>(info.height)
                                 : 1.0f;
    const f32 target_aspect = layout.height != 0
                                  ? static_cast<f32>(layout.width) / static_cast<f32>(layout.height)
                                  : 1.0f;
    u32 dst_width = layout.width;
    u32 dst_height = layout.height;
    if (guest_aspect > target_aspect) {
        dst_height = static_cast<u32>(static_cast<f32>(layout.width) / guest_aspect);
    } else {
        dst_width = static_cast<u32>(static_cast<f32>(layout.height) * guest_aspect);
    }
    const u32 dst_x = (layout.width - dst_width) / 2;
    const u32 dst_y = (layout.height - dst_height) / 2;

    D3D12_VIEWPORT viewport{};
    viewport.TopLeftX = static_cast<f32>(dst_x);
    viewport.TopLeftY = static_cast<f32>(dst_y);
    viewport.Width = static_cast<f32>(dst_width);
    viewport.Height = static_cast<f32>(dst_height);
    viewport.MaxDepth = 1.0f;
    command_list.SetViewport(viewport);

    D3D12_RECT scissor{};
    scissor.right = static_cast<LONG>(swapchain.Width());
    scissor.bottom = static_cast<LONG>(swapchain.Height());
    command_list.SetScissorRect(scissor);

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
