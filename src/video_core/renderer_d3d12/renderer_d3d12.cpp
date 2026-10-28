// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/renderer_d3d12.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_capture.h"
#include "common/settings.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_d3d12/d3d12_blit_shaders.h"

namespace D3D12 {

namespace {

// S21 cycle 9: the present counter RenderBlit uses for the frame-capture cadence. The scene-RT
// dump reads it during the frame (before RenderBlit increments it), so the dump fires on the
// exact frames that produce a capture (a true same-frame pair).
u32 g_capture_present_counter = 0;

// S21 cycle 9: set when a capture is written; consumed by the next composite draw so the scene
// RT is dumped for a near-same-frame pair even when the cadences cannot be aligned.
std::atomic<bool> g_capture_align_dump{false};

// TEMP DIAGNOSTIC (session 12): read back a 64x64 corner of the sampled display image,
// to separate "the guest never wrote the presented buffer" from "the blit/present path
// loses the content". 32-bit color formats only; capped after a few probes.
struct ResourceProbe {
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    Microsoft::WRL::ComPtr<ID3D12Resource> calibration;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> calibration_rtv_heap;
    std::array<bool, 7> armed{};
    u32 logged = 0;
    u32 frames = 0;
};

ResourceProbe g_resource_probe;

bool ProbeFrameWanted() {
    const u32 frame = g_resource_probe.frames++;
    return g_resource_probe.logged < 24 && (frame < 4 || frame % 240 == 0);
}

bool RecordResourceProbe(ID3D12Device* d3d, CommandList& command_list, u32 slot,
                         ID3D12Resource* src) {
    if (!src || slot >= g_resource_probe.armed.size()) {
        return false;
    }
    const D3D12_RESOURCE_DESC desc = src->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width == 0 ||
        desc.Height == 0) {
        return false;
    }
    switch (desc.Format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_TYPELESS:
        break;
    default: {
        static u32 reject_logs = 0;
        if (reject_logs < 8) {
            ++reject_logs;
            LOG_WARNING(Render_D3D12, "Content probe slot {}: unsupported fmt={:#x} dim={}", slot,
                        static_cast<u32>(desc.Format), static_cast<u32>(desc.Dimension));
        }
        return false;
    }
    }
    constexpr u32 kMaxExtent = 16;
    constexpr u64 kRowPitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    constexpr u64 kSlotBytes = kRowPitch * kMaxExtent;
    constexpr u32 kSlots = 7;
    const u32 width = static_cast<u32>(std::min<u64>(kMaxExtent, desc.Width));
    const u32 height = static_cast<u32>(std::min<u64>(kMaxExtent, desc.Height));
    if (width == 0 || height == 0) {
        return false;
    }
    if (!g_resource_probe.readback) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer_desc{};
        buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer_desc.Width = kSlotBytes * kSlots;
        buffer_desc.Height = 1;
        buffer_desc.DepthOrArraySize = 1;
        buffer_desc.MipLevels = 1;
        buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
        buffer_desc.SampleDesc.Count = 1;
        buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&g_resource_probe.readback)))) {
            g_resource_probe.logged = 24;
            return false;
        }
    }
    command_list.Transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = g_resource_probe.readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = static_cast<u64>(slot) * kSlotBytes;
    dst.PlacedFootprint.Footprint.Format = desc.Format;
    dst.PlacedFootprint.Footprint.Width = width;
    dst.PlacedFootprint.Footprint.Height = height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = static_cast<u32>(kRowPitch);
    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = src;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;
    D3D12_BOX box{};
    box.left = 0;
    box.top = 0;
    box.front = 0;
    box.right = width;
    box.bottom = height;
    box.back = 1;
    command_list.Get()->CopyTextureRegion(&dst, 0, 0, 0, &src_loc, &box);
    command_list.Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    g_resource_probe.armed[slot] = true;
    return true;
}

bool RecordCalibrationProbe(ID3D12Device* d3d, CommandList& command_list) {
    constexpr u32 kExtent = 16;
    if (!g_resource_probe.calibration) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = kExtent;
        texture_desc.Height = kExtent;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        texture_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texture_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&g_resource_probe.calibration)))) {
            return false;
        }
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heap_desc.NumDescriptors = 1;
        if (FAILED(d3d->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(
                                                       &g_resource_probe.calibration_rtv_heap)))) {
            return false;
        }
        d3d->CreateRenderTargetView(
            g_resource_probe.calibration.Get(), nullptr,
            g_resource_probe.calibration_rtv_heap->GetCPUDescriptorHandleForHeapStart());
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv =
        g_resource_probe.calibration_rtv_heap->GetCPUDescriptorHandleForHeapStart();
    command_list.Transition(g_resource_probe.calibration.Get(), D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_RENDER_TARGET);
    const f32 color[4] = {0.1f, 0.6f, 0.9f, 1.0f};
    command_list.ClearRenderTargetView(rtv, color);
    command_list.Transition(g_resource_probe.calibration.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                            D3D12_RESOURCE_STATE_COMMON);
    return RecordResourceProbe(d3d, command_list, 3, g_resource_probe.calibration.Get());
}

void LogResourceProbes() {
    if (!g_resource_probe.readback) {
        return;
    }
    constexpr u64 kRowPitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    constexpr u64 kSlotBytes = kRowPitch * 16;
    constexpr u64 kAllBytes = kSlotBytes * 7;
    static constexpr const char* kSlotNames[] = {"display", "scene", "any_sample", "calib",
                                                 "blit_src", "blit_dst", "scene3d"};
    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, kAllBytes};
    if (FAILED(g_resource_probe.readback->Map(0, &read_range, &mapped)) || mapped == nullptr) {
        return;
    }
    const auto* const base = static_cast<const u8*>(mapped);
    for (u32 slot = 0; slot < g_resource_probe.armed.size(); ++slot) {
        if (!g_resource_probe.armed[slot]) {
            continue;
        }
        g_resource_probe.armed[slot] = false;
        const u8* const bytes = base + static_cast<u64>(slot) * kSlotBytes;
        u32 min_value = 255;
        u32 max_value = 0;
        u64 sum = 0;
        u32 nonzero = 0;
        for (u32 y = 0; y < 16; ++y) {
            const u8* row = bytes + y * kRowPitch;
            for (u32 x = 0; x < 16 * 4; ++x) {
                const u32 value = row[x];
                min_value = value < min_value ? value : min_value;
                max_value = value > max_value ? value : max_value;
                sum += value;
                if (value != 0) {
                    ++nonzero;
                }
            }
        }
        LOG_WARNING(Render_D3D12, "Content probe #{} ({}): min={} max={} sum={} nonzero={}/{}",
                    g_resource_probe.logged, kSlotNames[slot], min_value, max_value, sum, nonzero,
                    16 * 16 * 4);
    }
    g_resource_probe.readback->Unmap(0, nullptr);
    ++g_resource_probe.logged;
}

// TEMP DIAGNOSTIC (session 12): periodic full-frame BMP capture of the presented image so the
// current frame can be fetched via Device Portal and inspected without a TV.
struct FrameCapture {
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    u32 width = 0;
    u32 height = 0;
    u32 index = 0;
    u32 bytes_per_pixel = 4;
    u32 dxgi_format = 0; // resource format, for packed-format decoding in the writer
    const void* resource_ptr = nullptr;
    bool is_bgra = false;
    bool armed = false;
};

FrameCapture g_frame_capture;
// TEMP DIAGNOSTIC (session 13): same machinery, but for the visible 3D scene render target
// (g_probe_scene3d): a 16x16 content probe cannot show whether the scene holds geometry, so
// dump the whole target as a BMP. Up to sixteen files are written, then the probe stops.
FrameCapture g_scene3d_capture;
std::string g_capture_directory;

// TEMP DIAGNOSTIC (session 15): the 3D scene renders near-black, so dump the sampled textures the
// scene draws read (most of them BC-compressed) to a BMP to see whether the guest source data
// holds anything at all. One dump per image, capped, and written once the list that carried the
// readback copy retires.
struct TextureContentDump {
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    u32 width = 0;
    u32 height = 0;
    u32 dxgi_format = 0;
    u32 guest_format = 0;
    u64 guest_addr = 0;
    u32 image_index = 0;
    // S24: array/cube layer this entry came from. The 13-cube env probe is 78 layers and the
    // scene's sky renders black, so cube 0 / face 0 alone cannot tell a wrong cube index from a
    // black face; the dumper writes the layer into the file name.
    u32 layer = 0;
    bool armed = false;
};

std::vector<TextureContentDump> g_texture_dumps;

// S21: guest addresses whose newest witness readback (dump id 995xxx) was all-zero on the GPU.
// See S21IsZeroGpuTexture in d3d12_capture.h.
std::vector<u64> g_s21_zero_gpu;

bool CaptureWanted() {
    if (g_capture_directory.empty() || !Settings::values.uwp_frame_capture.GetValue()) {
        return false;
    }
    std::error_code ec;
    return !std::filesystem::exists(g_capture_directory + "\\capture_off", ec);
}

void PruneCaptures() {
    constexpr size_t kKeep = 5;
    std::error_code ec;
    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
    for (const auto& entry : std::filesystem::directory_iterator{g_capture_directory, ec}) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("capture_", 0) != 0 || entry.path().extension() != ".bmp" ||
            name.size() <= 12) {
            continue;
        }
        u32 index = 0;
        const std::string digits = name.substr(8, name.size() - 12);
        const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), index);
        if (result.ec != std::errc{}) {
            continue;
        }
        const std::filesystem::file_time_type write_time =
            std::filesystem::last_write_time(entry.path(), ec);
        if (ec) {
            continue;
        }
        files.emplace_back(write_time, entry.path());
    }
    std::sort(files.begin(), files.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    while (files.size() > kKeep) {
        std::filesystem::remove(files.front().second, ec);
        files.erase(files.begin());
    }
}

bool RecordFrameCapture(ID3D12Device* d3d, CommandList& command_list, ID3D12Resource* src) {
    if (!src || g_frame_capture.armed) {
        return false;
    }
    const D3D12_RESOURCE_DESC desc = src->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width == 0 ||
        desc.Height == 0) {
        return false;
    }
    bool is_bgra = false;
    switch (desc.Format) {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        is_bgra = true;
        break;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        is_bgra = false;
        break;
    default:
        return false;
    }
    g_frame_capture.readback.Reset();
    UINT64 total_bytes = 0;
    UINT rows = 0;
    UINT64 row_size = 0;
    d3d->GetCopyableFootprints(&desc, 0, 1, 0, &g_frame_capture.footprint, &rows, &row_size,
                               &total_bytes);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = total_bytes;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                            IID_PPV_ARGS(&g_frame_capture.readback)))) {
        return false;
    }
    command_list.Transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = g_frame_capture.readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = g_frame_capture.footprint;
    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = src;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;
    command_list.Get()->CopyTextureRegion(&dst, 0, 0, 0, &src_loc, nullptr);
    command_list.Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    g_frame_capture.width = static_cast<u32>(desc.Width);
    g_frame_capture.height = desc.Height;
    g_frame_capture.is_bgra = is_bgra;
    g_frame_capture.armed = true;
    return true;
}

// TEMP DIAGNOSTIC (session 13): RecordFrameCapture for the scene3d probe RT. Same readback,
// footprint and COMMON/COPY_SOURCE handling; the format table additionally accepts the wider
// 4-bytes-per-pixel colour formats a scene render target can use, which take the RGBA path
// (the BMP is colour-swizzled for those, but still shows whether the target holds pixels).
bool RecordScene3dCapture(ID3D12Device* d3d, CommandList& command_list, ID3D12Resource* src) {
    if (!src || g_scene3d_capture.armed) {
        return false;
    }
    const D3D12_RESOURCE_DESC desc = src->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width == 0 ||
        desc.Height == 0) {
        return false;
    }
    bool is_bgra = false;
    switch (desc.Format) {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        is_bgra = true;
        g_scene3d_capture.bytes_per_pixel = 4;
        g_scene3d_capture.dxgi_format = static_cast<u32>(desc.Format);
        break;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        is_bgra = false;
        g_scene3d_capture.bytes_per_pixel = 4;
        break;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
        // 4 bytes per pixel: copied through and written unswizzled.
        is_bgra = false;
        g_scene3d_capture.bytes_per_pixel = 4;
        break;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
        // 8 bytes per pixel: the writer truncates each 16-bit channel to its low byte.
        is_bgra = false;
        g_scene3d_capture.bytes_per_pixel = 8;
        break;
    default:
        static bool scene3d_format_logged = false;
        if (!scene3d_format_logged) {
            scene3d_format_logged = true;
            LOG_WARNING(Render_D3D12, "Scene3D capture: unsupported fmt={:#x} {}x{}",
                        static_cast<u32>(desc.Format), desc.Width, desc.Height);
        }
        return false;
    }
    g_scene3d_capture.readback.Reset();
    UINT64 total_bytes = 0;
    UINT rows = 0;
    UINT64 row_size = 0;
    d3d->GetCopyableFootprints(&desc, 0, 1, 0, &g_scene3d_capture.footprint, &rows, &row_size,
                               &total_bytes);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = total_bytes;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                            IID_PPV_ARGS(&g_scene3d_capture.readback)))) {
        return false;
    }
    command_list.Transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = g_scene3d_capture.readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = g_scene3d_capture.footprint;
    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = src;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;
    command_list.Get()->CopyTextureRegion(&dst, 0, 0, 0, &src_loc, nullptr);
    command_list.Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    g_scene3d_capture.width = static_cast<u32>(desc.Width);
    g_scene3d_capture.height = desc.Height;
    g_scene3d_capture.dxgi_format = static_cast<u32>(desc.Format);
    g_scene3d_capture.is_bgra = is_bgra;
    g_scene3d_capture.resource_ptr = src;
    g_scene3d_capture.armed = true;
    return true;
}

void WriteFrameCapture() {
    if (!g_frame_capture.armed || !g_frame_capture.readback) {
        g_frame_capture.armed = false;
        return;
    }
    g_frame_capture.armed = false;
    const u32 width = g_frame_capture.width;
    const u32 height = g_frame_capture.height;
    const u32 pitch = g_frame_capture.footprint.Footprint.RowPitch;
    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, static_cast<SIZE_T>(pitch) * height};
    if (FAILED(g_frame_capture.readback->Map(0, &read_range, &mapped)) || mapped == nullptr) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(g_capture_directory, ec);
    const std::string path =
        g_capture_directory + "\\capture_" + std::to_string(g_frame_capture.index) + ".bmp";
    const u32 row_bytes = width * 4;
    const u32 file_size = 54 + row_bytes * height;
    std::array<u8, 54> header{};
    header[0] = 'B';
    header[1] = 'M';
    std::memcpy(header.data() + 2, &file_size, 4);
    const u32 pixel_offset = 54;
    std::memcpy(header.data() + 10, &pixel_offset, 4);
    const u32 dib_size = 40;
    std::memcpy(header.data() + 14, &dib_size, 4);
    std::memcpy(header.data() + 18, &width, 4);
    const s32 bmp_height = static_cast<s32>(height);
    std::memcpy(header.data() + 22, &bmp_height, 4);
    const u16 planes = 1;
    std::memcpy(header.data() + 26, &planes, 2);
    const u16 bpp = 32;
    std::memcpy(header.data() + 28, &bpp, 2);
    const u32 image_size = row_bytes * height;
    std::memcpy(header.data() + 34, &image_size, 4);
    std::ofstream out{path, std::ios::binary};
    if (out) {
        out.write(reinterpret_cast<const char*>(header.data()), header.size());
        const u8* const base = static_cast<const u8*>(mapped);
        std::vector<u8> row(row_bytes);
        for (s32 y = static_cast<s32>(height) - 1; y >= 0; --y) {
            const u8* const src_row = base + static_cast<size_t>(y) * pitch;
            if (g_frame_capture.is_bgra) {
                out.write(reinterpret_cast<const char*>(src_row), row_bytes);
            } else {
                for (u32 x = 0; x < row_bytes; x += 4) {
                    row[x] = src_row[x + 2];
                    row[x + 1] = src_row[x + 1];
                    row[x + 2] = src_row[x];
                    row[x + 3] = src_row[x + 3];
                }
                out.write(reinterpret_cast<const char*>(row.data()), row_bytes);
            }
        }
        // TEMP DIAGNOSTIC (S18, Phase 0.2.1): the guest-frame counter and the newest exposure
        // readback are the phase fingerprint - two runs' captures can only be compared when
        // they sit at the same guest phase (the style-screen dolly advances on guest time).
        LOG_WARNING(Render_D3D12,
                    "Frame capture written: {} ({}x{}) [S18 guest_frame={} expo={}]", path, width,
                    height, g_s18_guest_frames,
                    g_s18_last_exposure.empty() ? "none" : g_s18_last_exposure);
        // S21 cycle 9: ask the next composite draw to dump the scene RT so this capture and its
        // scene source can be compared (<=1 frame apart; the cadence ordering is not knowable).
        g_capture_align_dump.store(true, std::memory_order_relaxed);
        ++g_frame_capture.index;
    }
    g_frame_capture.readback->Unmap(0, nullptr);
    // Keep console storage bounded: only the newest captures are retained.
    PruneCaptures();
}

// TEMP DIAGNOSTIC (session 13): WriteFrameCapture for the scene3d probe RT, writing
// scene3d_<index>.bmp next to the presented-frame captures. Up to eight files are produced
// (the trigger stops arming once index == 8), and PruneCaptures never matches them, so they
// survive across runs until the next eight overwrite the same names.
void WriteScene3dCapture() {
    if (!g_scene3d_capture.armed || !g_scene3d_capture.readback) {
        g_scene3d_capture.armed = false;
        return;
    }
    g_scene3d_capture.armed = false;
    const u32 width = g_scene3d_capture.width;
    const u32 height = g_scene3d_capture.height;
    const u32 pitch = g_scene3d_capture.footprint.Footprint.RowPitch;
    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, static_cast<SIZE_T>(pitch) * height};
    if (FAILED(g_scene3d_capture.readback->Map(0, &read_range, &mapped)) || mapped == nullptr) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(g_capture_directory, ec);
    const std::string path =
        g_capture_directory + "\\scene3d_" + std::to_string(g_scene3d_capture.index) + ".bmp";
    const u32 row_bytes = width * 4;
    const u32 file_size = 54 + row_bytes * height;
    std::array<u8, 54> header{};
    header[0] = 'B';
    header[1] = 'M';
    std::memcpy(header.data() + 2, &file_size, 4);
    const u32 pixel_offset = 54;
    std::memcpy(header.data() + 10, &pixel_offset, 4);
    const u32 dib_size = 40;
    std::memcpy(header.data() + 14, &dib_size, 4);
    std::memcpy(header.data() + 18, &width, 4);
    const s32 bmp_height = static_cast<s32>(height);
    std::memcpy(header.data() + 22, &bmp_height, 4);
    const u16 planes = 1;
    std::memcpy(header.data() + 26, &planes, 2);
    const u16 bpp = 32;
    std::memcpy(header.data() + 28, &bpp, 2);
    const u32 image_size = row_bytes * height;
    std::memcpy(header.data() + 34, &image_size, 4);
    std::ofstream out{path, std::ios::binary};
    if (out) {
        out.write(reinterpret_cast<const char*>(header.data()), header.size());
        const u8* const base = static_cast<const u8*>(mapped);
        std::vector<u8> row(row_bytes);
        const u32 pixel_bytes = g_scene3d_capture.bytes_per_pixel;
        // Packed 4-byte colour formats (R11G11B10_FLOAT, R10G10B10A2, ...) cannot be byte-swizzled
        // the way R8G8B8A8 can, so decode each pixel per the stored format into BMP-order B,G,R,A.
        const u32 capture_format = g_scene3d_capture.dxgi_format;
        // value_bits packs a 5-bit exponent above mantissa_bits; bias is 15 for both 11- and
        // 10-bit unsuffixed floats. Subnormals are flushed to zero.
        const auto decode_unsigned_float = [](u32 value_bits, u32 mantissa_bits) -> float {
            const u32 exponent = value_bits >> mantissa_bits;
            if (exponent == 0) {
                return 0.0f;
            }
            const u32 mantissa = value_bits & ((1u << mantissa_bits) - 1u);
            float value =
                1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissa_bits);
            s32 power = static_cast<s32>(exponent) - 15;
            while (power > 0) {
                value *= 2.0f;
                --power;
            }
            while (power < 0) {
                value *= 0.5f;
                ++power;
            }
            return value;
        };
        const auto to_byte = [](float value) -> u8 {
            if (value < 0.0f) {
                value = 0.0f;
            } else if (value > 1.0f) {
                value = 1.0f;
            }
            return static_cast<u8>(value * 255.0f + 0.5f);
        };
        const auto convert_4bpp = [&](const u8* src, u8* dst) {
            const u32 raw = static_cast<u32>(src[0]) | (static_cast<u32>(src[1]) << 8) |
                            (static_cast<u32>(src[2]) << 16) |
                            (static_cast<u32>(src[3]) << 24);
            switch (capture_format) {
            case DXGI_FORMAT_R11G11B10_FLOAT: {
                const u32 r11 = raw & 0x7FFu;
                const u32 g11 = (raw >> 11) & 0x7FFu;
                const u32 b10 = (raw >> 22) & 0x3FFu;
                dst[0] = to_byte(decode_unsigned_float(b10, 5u));
                dst[1] = to_byte(decode_unsigned_float(g11, 6u));
                dst[2] = to_byte(decode_unsigned_float(r11, 6u));
                dst[3] = static_cast<u8>(255);
                break;
            }
            case DXGI_FORMAT_R10G10B10A2_UNORM:
            case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            case DXGI_FORMAT_R10G10B10A2_UINT: {
                const u32 r10 = raw & 0x3FFu;
                const u32 g10 = (raw >> 10) & 0x3FFu;
                const u32 b10 = (raw >> 20) & 0x3FFu;
                const u32 a2 = (raw >> 30) & 0x3u;
                dst[0] = static_cast<u8>(b10 * 255u / 1023u);
                dst[1] = static_cast<u8>(g10 * 255u / 1023u);
                dst[2] = static_cast<u8>(r10 * 255u / 1023u);
                dst[3] = static_cast<u8>(a2 * 255u / 3u);
                break;
            }
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            case DXGI_FORMAT_R8G8B8A8_TYPELESS:
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
                dst[3] = src[3];
                break;
            default:
                std::memcpy(dst, src, 4);
                break;
            }
        };
        for (s32 y = static_cast<s32>(height) - 1; y >= 0; --y) {
            const u8* const src_row = base + static_cast<size_t>(y) * pitch;
            if (pixel_bytes == 8) {
                for (u32 x = 0; x < width; ++x) {
                    const u8* const src_px = src_row + static_cast<size_t>(x) * 8;
                    // R16G16B16A16 -> BGRA, keeping each channel's low byte.
                    row[x * 4 + 0] = src_px[4];
                    row[x * 4 + 1] = src_px[2];
                    row[x * 4 + 2] = src_px[0];
                    row[x * 4 + 3] = src_px[6];
                }
                out.write(reinterpret_cast<const char*>(row.data()), row_bytes);
            } else if (g_scene3d_capture.is_bgra) {
                out.write(reinterpret_cast<const char*>(src_row), row_bytes);
            } else {
                for (u32 x = 0; x < width; ++x) {
                    convert_4bpp(src_row + static_cast<size_t>(x) * 4, row.data() + x * 4);
                }
                out.write(reinterpret_cast<const char*>(row.data()), row_bytes);
            }
        }
        LOG_WARNING(Render_D3D12, "Scene3D capture written: {} ({}x{} bpp={} fmt={:#x} res={})",
                    path, width, height, pixel_bytes, g_scene3d_capture.dxgi_format,
                    g_scene3d_capture.resource_ptr);
        ++g_scene3d_capture.index;
    }
    g_scene3d_capture.readback->Unmap(0, nullptr);
}

// TEMP DIAGNOSTIC (session 15): block-compressed formats are laid out as 4x4 blocks, so a
// full-mip readback covers block rows instead of pixel rows, and a row of blocks is not padded
// like a row of pixels. Only the row count matters to bound the mapped readback.
bool TextureFormatIsBlockCompressed(u32 dxgi_format) {
    switch (dxgi_format) {
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
    case DXGI_FORMAT_BC7_TYPELESS:
        return true;
    default:
        return false;
    }
}

void DecodeTextureToBgra(u32 dxgi_format, u32 width, u32 height, u32 row_pitch, size_t src_bytes,
                         const u8* src, std::vector<u8>& dst) {
    dst.assign(static_cast<size_t>(width) * height * 4, 0);
    const auto write_pixel = [&](u32 x, u32 y, u32 b, u32 g, u32 r, u32 a) {
        if (x >= width || y >= height) {
            return;
        }
        u8* const pixel = dst.data() + (static_cast<size_t>(y) * width + x) * 4;
        pixel[0] = static_cast<u8>(b);
        pixel[1] = static_cast<u8>(g);
        pixel[2] = static_cast<u8>(r);
        pixel[3] = static_cast<u8>(a);
    };
    const auto blend = [](u32 a, u32 b, u32 weight_a, u32 weight_b) -> u32 {
        const u32 total_weight = weight_a + weight_b;
        const auto mix_channel = [&](u32 shift) -> u32 {
            return (((a >> shift) & 0xFFu) * weight_a + ((b >> shift) & 0xFFu) * weight_b) /
                   total_weight;
        };
        return mix_channel(0) | (mix_channel(8) << 8) | (mix_channel(16) << 16) |
               (mix_channel(24) << 24);
    };
    const auto expand565 = [](u16 value) -> u32 {
        const u32 b = ((value & 0x1Fu) * 255u) / 31u;
        const u32 g = (((value >> 5) & 0x3Fu) * 255u) / 63u;
        const u32 r = (((value >> 11) & 0x1Fu) * 255u) / 31u;
        return b | (g << 8) | (r << 16) | 0xFF000000u;
    };
    const auto read_u16 = [](const u8* bytes) -> u16 {
        return static_cast<u16>(static_cast<u16>(bytes[0]) | (static_cast<u16>(bytes[1]) << 8));
    };
    const auto read_u32 = [](const u8* bytes) -> u32 {
        return static_cast<u32>(bytes[0]) | (static_cast<u32>(bytes[1]) << 8) |
               (static_cast<u32>(bytes[2]) << 16) | (static_cast<u32>(bytes[3]) << 24);
    };
    const auto read_u48 = [](const u8* bytes) -> u64 {
        u64 value = 0;
        for (u32 i = 0; i < 6; ++i) {
            value |= static_cast<u64>(bytes[i]) << (8 * i);
        }
        return value;
    };
    const auto read_f32 = [](const u8* bytes) -> float {
        float value = 0.0f;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    };
    const auto to_byte = [](float value) -> u8 {
        if (value < 0.0f) {
            value = 0.0f;
        } else if (value > 1.0f) {
            value = 1.0f;
        }
        return static_cast<u8>(value * 255.0f + 0.5f);
    };
    const auto decode_unsigned_float = [](u32 value_bits, u32 mantissa_bits) -> float {
        const u32 exponent = value_bits >> mantissa_bits;
        if (exponent == 0) {
            return 0.0f;
        }
        const u32 mantissa = value_bits & ((1u << mantissa_bits) - 1u);
        float value =
            1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissa_bits);
        s32 power = static_cast<s32>(exponent) - 15;
        while (power > 0) {
            value *= 2.0f;
            --power;
        }
        while (power < 0) {
            value *= 0.5f;
            ++power;
        }
        return value;
    };
    const auto decode_half = [](u16 bits) -> float {
        const u32 sign = static_cast<u32>(bits) >> 15;
        const u32 exponent = (static_cast<u32>(bits) >> 10) & 0x1Fu;
        const u32 mantissa = static_cast<u32>(bits) & 0x3FFu;
        float value = 0.0f;
        if (exponent == 0) {
            value = static_cast<float>(mantissa) / 1024.0f * 0.5f;
        } else if (exponent == 31) {
            value = mantissa != 0 ? 1.0f : 65504.0f;
        } else {
            value = 1.0f + static_cast<float>(mantissa) / 1024.0f;
            s32 power = static_cast<s32>(exponent) - 15;
            while (power > 0) {
                value *= 2.0f;
                --power;
            }
            while (power < 0) {
                value *= 0.5f;
                ++power;
            }
        }
        return sign != 0 ? -value : value;
    };
    // BC4/BC5 red block and the alpha block of BC3 share the 3-bit indexed interpolation table.
    const auto greyscale_palette = [](const u8* bytes) -> std::array<u32, 8> {
        const u32 low = bytes[0];
        const u32 high = bytes[1];
        std::array<u32, 8> palette{};
        palette[0] = low;
        palette[1] = high;
        if (low > high) {
            for (u32 i = 2; i < 8; ++i) {
                palette[i] = (low * (8 - i) + high * (i - 1)) / 7u;
            }
        } else {
            palette[2] = (low * 4u + high) / 5u;
            palette[3] = (low * 3u + high * 2u) / 5u;
            palette[4] = (low * 2u + high * 3u) / 5u;
            palette[5] = (low + high * 4u) / 5u;
            palette[6] = 0;
            palette[7] = 255;
        }
        return palette;
    };
    const auto decode_block_rows = [&](u32 block_bytes, const auto& decode_block) {
        const u32 blocks_per_row = (width + 3) / 4;
        const u32 block_rows = (height + 3) / 4;
        for (u32 block_y = 0; block_y < block_rows; ++block_y) {
            const size_t row_begin = static_cast<size_t>(block_y) * row_pitch;
            if (row_begin + static_cast<size_t>(blocks_per_row) * block_bytes > src_bytes) {
                break;
            }
            const u8* const src_row = src + row_begin;
            for (u32 block_x = 0; block_x < blocks_per_row; ++block_x) {
                decode_block(src_row + static_cast<size_t>(block_x) * block_bytes, block_x,
                             block_y);
            }
        }
    };
    const auto decode_unpacked_rows = [&](u32 pixel_bytes, const auto& decode_pixel) {
        for (u32 y = 0; y < height; ++y) {
            const size_t row_begin = static_cast<size_t>(y) * row_pitch;
            if (row_begin + static_cast<size_t>(width) * pixel_bytes > src_bytes) {
                break;
            }
            const u8* const src_row = src + row_begin;
            for (u32 x = 0; x < width; ++x) {
                const size_t offset = static_cast<size_t>(x) * pixel_bytes;
                if (offset + pixel_bytes > row_pitch) {
                    break;
                }
                const u32 color = decode_pixel(src_row + offset);
                write_pixel(x, y, color & 0xFFu, (color >> 8) & 0xFFu, (color >> 16) & 0xFFu,
                            (color >> 24) & 0xFFu);
            }
        }
    };
    const auto decode_bc1_block = [&](const u8* block, bool has_alpha_block, u32 block_x,
                                      u32 block_y) {
        const u8* const color_block = has_alpha_block ? block + 8 : block;
        const u16 color0 = read_u16(color_block);
        const u16 color1 = read_u16(color_block + 2);
        std::array<u32, 8> alphas{};
        u64 alpha_indices = 0;
        if (has_alpha_block) {
            alphas = greyscale_palette(block);
            alpha_indices = read_u48(block + 2);
        }
        std::array<u32, 4> palette{};
        palette[0] = expand565(color0);
        palette[1] = expand565(color1);
        if (color0 > color1) {
            palette[2] = blend(palette[0], palette[1], 2u, 1u);
            palette[3] = blend(palette[0], palette[1], 1u, 2u);
        } else {
            palette[2] = blend(palette[0], palette[1], 1u, 1u);
            palette[3] = 0u;
        }
        u32 indices = read_u32(color_block + 4);
        for (u32 y = 0; y < 4; ++y) {
            for (u32 x = 0; x < 4; ++x) {
                const u32 index = indices & 3u;
                indices >>= 2;
                u32 alpha = 0xFFu;
                if (has_alpha_block) {
                    alpha = alphas[alpha_indices & 7u];
                    alpha_indices >>= 3;
                } else if (color0 <= color1 && index == 3) {
                    alpha = 0u;
                }
                const u32 color = (palette[index] & 0x00FFFFFFu) | (alpha << 24);
                write_pixel(block_x * 4 + x, block_y * 4 + y, color & 0xFFu, (color >> 8) & 0xFFu,
                            (color >> 16) & 0xFFu, (color >> 24) & 0xFFu);
            }
        }
    };
    switch (dxgi_format) {
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC1_TYPELESS:
        decode_block_rows(8, [&](const u8* block, u32 block_x, u32 block_y) {
            decode_bc1_block(block, false, block_x, block_y);
        });
        return;
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
        decode_block_rows(16, [&](const u8* block, u32 block_x, u32 block_y) {
            decode_bc1_block(block, true, block_x, block_y);
        });
        return;
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC4_TYPELESS:
        decode_block_rows(8, [&](const u8* block, u32 block_x, u32 block_y) {
            const std::array<u32, 8> values = greyscale_palette(block);
            u64 indices = read_u48(block + 2);
            for (u32 y = 0; y < 4; ++y) {
                for (u32 x = 0; x < 4; ++x) {
                    const u32 value = values[indices & 7u];
                    indices >>= 3;
                    write_pixel(block_x * 4 + x, block_y * 4 + y, value, value, value, 0xFFu);
                }
            }
        });
        return;
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC5_TYPELESS:
        decode_block_rows(16, [&](const u8* block, u32 block_x, u32 block_y) {
            const std::array<u32, 8> red = greyscale_palette(block);
            const std::array<u32, 8> green = greyscale_palette(block + 8);
            u64 indices = read_u48(block + 2);
            u64 green_indices = read_u48(block + 10);
            for (u32 y = 0; y < 4; ++y) {
                for (u32 x = 0; x < 4; ++x) {
                    const u32 r = red[indices & 7u];
                    const u32 g = green[green_indices & 7u];
                    indices >>= 3;
                    green_indices >>= 3;
                    write_pixel(block_x * 4 + x, block_y * 4 + y, r, g, r, 0xFFu);
                }
            }
        });
        return;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        decode_unpacked_rows(4, [](const u8* pixel) -> u32 {
            return static_cast<u32>(pixel[2]) | (static_cast<u32>(pixel[1]) << 8) |
                   (static_cast<u32>(pixel[0]) << 16) | (static_cast<u32>(pixel[3]) << 24);
        });
        return;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        decode_unpacked_rows(4, [](const u8* pixel) -> u32 {
            return static_cast<u32>(pixel[0]) | (static_cast<u32>(pixel[1]) << 8) |
                   (static_cast<u32>(pixel[2]) << 16) | (static_cast<u32>(pixel[3]) << 24);
        });
        return;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        decode_unpacked_rows(4, [&](const u8* pixel) -> u32 {
            const u32 raw = read_u32(pixel);
            const u32 b = to_byte(decode_unsigned_float((raw >> 22) & 0x3FFu, 5u));
            const u32 g = to_byte(decode_unsigned_float((raw >> 11) & 0x7FFu, 6u));
            const u32 r = to_byte(decode_unsigned_float(raw & 0x7FFu, 6u));
            return b | (g << 8) | (r << 16) | 0xFF000000u;
        });
        return;
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
        decode_unpacked_rows(4, [&](const u8* pixel) -> u32 {
            const u32 r = to_byte(decode_half(read_u16(pixel)));
            const u32 g = to_byte(decode_half(read_u16(pixel + 2)));
            return (g << 8) | (r << 16) | 0xFF000000u;
        });
        return;
    case DXGI_FORMAT_R16_FLOAT:
        decode_unpacked_rows(2, [&](const u8* pixel) -> u32 {
            const u32 value = to_byte(decode_half(read_u16(pixel)));
            return value | (value << 8) | (value << 16) | 0xFF000000u;
        });
        return;
    case DXGI_FORMAT_R32_FLOAT:
        decode_unpacked_rows(4, [&](const u8* pixel) -> u32 {
            const u32 value = to_byte(read_f32(pixel));
            return value | (value << 8) | (value << 16) | 0xFF000000u;
        });
        return;
    case DXGI_FORMAT_R8_UNORM:
        decode_unpacked_rows(1, [](const u8* pixel) -> u32 {
            const u32 value = pixel[0];
            return value | (value << 8) | (value << 16) | 0xFF000000u;
        });
        return;
    default: {
        static bool no_decoder_logged = false;
        if (!no_decoder_logged) {
            no_decoder_logged = true;
            LOG_WARNING(Render_D3D12, "Texture dump: no decoder for dxgi fmt {:#x}", dxgi_format);
        }
        decode_unpacked_rows(4, [](const u8* pixel) -> u32 {
            const u32 value = pixel[0];
            return value | (value << 8) | (value << 16) | 0xFF000000u;
        });
        return;
    }
    }
}

// TEMP DIAGNOSTIC (session 15): WriteFrameCapture for sampled textures: every armed entry is
// logged, decoded and written as texdump_<entry>_img<image>.bmp next to the frame captures.
void WriteTextureContentDumps() {
    if (g_texture_dumps.empty()) {
        return;
    }
    // S24: the wipe in SetCaptureDirectory runs before the logger exists, so report the volume
    // here (first dump of the process) - a full app-data volume killed two runs mid-dump.
    {
        static bool s24_space_logged = false;
        if (!s24_space_logged) {
            s24_space_logged = true;
            std::error_code space_ec;
            const auto space = std::filesystem::space(g_capture_directory, space_ec);
            LOG_WARNING(Render_D3D12, "S24 capture dir: free={} MB of {} MB",
                        space.available / (1024 * 1024), space.capacity / (1024 * 1024));
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(g_capture_directory, ec);
    for (size_t i = 0; i < g_texture_dumps.size(); ++i) {
        TextureContentDump& entry = g_texture_dumps[i];
        if (!entry.armed || !entry.readback) {
            entry.armed = false;
            continue;
        }
        entry.armed = false;
        const u32 width = entry.width;
        const u32 height = entry.height;
        const u32 row_pitch = entry.footprint.Footprint.RowPitch;
        const u32 row_count =
            TextureFormatIsBlockCompressed(entry.dxgi_format) ? (height + 3) / 4 : height;
        const size_t total_bytes = std::min(static_cast<size_t>(row_pitch) * row_count,
                                            static_cast<size_t>(entry.readback->GetDesc().Width));
        void* mapped = nullptr;
        const D3D12_RANGE read_range{0, static_cast<SIZE_T>(total_bytes)};
        const HRESULT s15_map_hr = entry.readback->Map(0, &read_range, &mapped);
        if (FAILED(s15_map_hr) || mapped == nullptr) {
            LOG_WARNING(Render_D3D12,
                        "Texture dump #{}: Map failed hr={:#x} (img={}) res={} x{} x{} "
                        "dxgi={:#x} pitch={} range_end={}",
                        i, static_cast<u32>(s15_map_hr), entry.image_index,
                        entry.readback->GetDesc().Width, entry.width, entry.height,
                        entry.dxgi_format, entry.footprint.Footprint.RowPitch,
                        static_cast<u64>(read_range.End));
            continue;
        }
        const u8* const base = static_cast<const u8*>(mapped);
        u32 min_value = 255;
        u32 max_value = 0;
        u64 nonzero = 0;
        for (size_t offset = 0; offset < total_bytes; ++offset) {
            const u32 value = base[offset];
            min_value = value < min_value ? value : min_value;
            max_value = value > max_value ? value : max_value;
            if (value != 0) {
                ++nonzero;
            }
        }
        LOG_WARNING(Render_D3D12,
                    "Texture dump #{} img={} layer={} guest_fmt=0x{:x} dxgi=0x{:x} {}x{} bytes={} "
                    "nonzero={} min={} max={}",
                    i, entry.image_index, entry.layer, entry.guest_format, entry.dxgi_format,
                    width, height, total_bytes, nonzero, min_value, max_value);
        // S21 cycle 17: byte min/max cannot tell a lighting-table ramp from a NaN/Inf texel.
        // For R11G11B10 dumps (the 256x256 LUT and the 960x540 scene RT) decode the actual
        // 11/11/10 floats and log the non-finite count plus the finite per-channel maxima.
        if (entry.dxgi_format == 26 /* DXGI_FORMAT_R11G11B10_FLOAT */ && row_pitch >= width * 4) {
            u32 s21_bad = 0;
            u32 s21_finite = 0;
            u32 s21_over1 = 0;
            f32 s21_rmax = 0.0f;
            f32 s21_gmax = 0.0f;
            f32 s21_bmax = 0.0f;
            // S23: the maxima alone cannot tell an over-gained scene from a well-exposed one;
            // the composite's auto-exposure loop keys on the frame's AVERAGE luminance, so log
            // the Rec601 mean and how much of the surface is above the tonemap's knee.
            double s21_lsum = 0.0;
            const auto s21_dec = [](u32 bits, u32 mant_bits) -> f32 {
                const u32 exp = bits >> mant_bits;
                const u32 mant = bits & ((1u << mant_bits) - 1);
                if (exp == 31) {
                    return mant == 0 ? std::numeric_limits<f32>::infinity()
                                     : std::numeric_limits<f32>::quiet_NaN();
                }
                const f32 frac = static_cast<f32>(mant) / static_cast<f32>(1u << mant_bits);
                if (exp == 0) {
                    return frac * 6.103515625e-05f;
                }
                return (1.0f + frac) * std::exp2(static_cast<f32>(exp) - 15.0f);
            };
            for (u32 y = 0; y < height; ++y) {
                const u8* const row = base + static_cast<size_t>(y) * row_pitch;
                for (u32 x = 0; x < width; ++x) {
                    u32 word = 0;
                    std::memcpy(&word, row + static_cast<size_t>(x) * 4, sizeof(word));
                    const f32 r = s21_dec(word & 0x7FF, 6);
                    const f32 g = s21_dec((word >> 11) & 0x7FF, 6);
                    const f32 b = s21_dec((word >> 22) & 0x3FF, 5);
                    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
                        ++s21_bad;
                        continue;
                    }
                    ++s21_finite;
                    const f32 s21_lum = 0.2989f * r + 0.587f * g + 0.114f * b;
                    s21_lsum += s21_lum;
                    if (s21_lum > 1.0f) {
                        ++s21_over1;
                    }
                    s21_rmax = std::max(s21_rmax, r);
                    s21_gmax = std::max(s21_gmax, g);
                    s21_bmax = std::max(s21_bmax, b);
                }
            }
            LOG_WARNING(Render_D3D12,
                        "S21 float dump #{} img={}: {}x{} nonfinite={} finite={} rmax={} gmax={} "
                        "bmax={} lmean={:.5f} over1={:.3f}",
                        i, entry.image_index, width, height, s21_bad, s21_finite, s21_rmax,
                        s21_gmax, s21_bmax,
                        s21_finite != 0 ? s21_lsum / static_cast<double>(s21_finite) : 0.0,
                        s21_finite != 0 ? static_cast<double>(s21_over1) /
                                              static_cast<double>(s21_finite)
                                        : 0.0);
        }
        // S21: remember which witness-dumped textures came back empty on the GPU (and forget
        // them once they hold content again) so the witness can force a guest re-upload.
        if (entry.image_index >= 995000u && entry.image_index < 997000u) {
            const auto it = std::find(g_s21_zero_gpu.begin(), g_s21_zero_gpu.end(),
                                      entry.guest_addr);
            if (nonzero == 0 && total_bytes != 0) {
                if (it == g_s21_zero_gpu.end() && g_s21_zero_gpu.size() < 64) {
                    g_s21_zero_gpu.push_back(entry.guest_addr);
                }
            } else if (it != g_s21_zero_gpu.end()) {
                g_s21_zero_gpu.erase(it);
            }
        }
        std::vector<u8> pixels;
        DecodeTextureToBgra(entry.dxgi_format, width, height, row_pitch, total_bytes, base,
                            pixels);
        const std::string path = g_capture_directory + "\\texdump_" + std::to_string(i) + "_img" +
                                 std::to_string(entry.image_index) + "_l" +
                                 std::to_string(entry.layer) + "_gf" +
                                 std::to_string(g_s18_guest_frames) + ".bmp";
        const u32 row_bytes = width * 4;
        const u32 file_size = 54 + row_bytes * height;
        std::array<u8, 54> header{};
        header[0] = 'B';
        header[1] = 'M';
        std::memcpy(header.data() + 2, &file_size, 4);
        const u32 pixel_offset = 54;
        std::memcpy(header.data() + 10, &pixel_offset, 4);
        const u32 dib_size = 40;
        std::memcpy(header.data() + 14, &dib_size, 4);
        std::memcpy(header.data() + 18, &width, 4);
        const s32 bmp_height = static_cast<s32>(height);
        std::memcpy(header.data() + 22, &bmp_height, 4);
        const u16 planes = 1;
        std::memcpy(header.data() + 26, &planes, 2);
        const u16 bpp = 32;
        std::memcpy(header.data() + 28, &bpp, 2);
        const u32 image_size = row_bytes * height;
        std::memcpy(header.data() + 34, &image_size, 4);
        std::ofstream out{path, std::ios::binary};
        if (out) {
            out.write(reinterpret_cast<const char*>(header.data()), header.size());
            for (s32 y = static_cast<s32>(height) - 1; y >= 0; --y) {
                out.write(reinterpret_cast<const char*>(pixels.data() +
                                                        static_cast<size_t>(y) * row_bytes),
                          row_bytes);
            }
            LOG_WARNING(Render_D3D12, "Texture dump written: {} ({}x{})", path, width, height);
        }
        // TEMP DIAGNOSTIC (S17): alongside the decoded BMP, write the RAW readback bytes for
        // B10G11R11 dumps (the lighting LUT) and for R32G32B32A32_FLOAT (the 1x1 exposure input;
        // 16 bytes, decoded offline as four floats - Phase 0.2). In the 1.0.266 harvest the
        // decoded BMP of a populated LUT rendered all-white, so either the R11G11B10 decode or
        // the content is suspect - the .bin is a byte-for-byte copy of the readback.
        // NOTE: the 1x1 exposure texture reports DXGI_FORMAT_R32G32B32A32_TYPELESS (1), not
        // _FLOAT (2) - cover both or the exposure dump silently misses its .bin.
        if (static_cast<DXGI_FORMAT>(entry.dxgi_format) == DXGI_FORMAT_R11G11B10_FLOAT ||
            static_cast<DXGI_FORMAT>(entry.dxgi_format) == DXGI_FORMAT_R32G32B32A32_TYPELESS ||
            static_cast<DXGI_FORMAT>(entry.dxgi_format) == DXGI_FORMAT_R32G32B32A32_FLOAT) {
            const std::string bin_path = g_capture_directory + "\\texdump_" + std::to_string(i) +
                                         "_img" + std::to_string(entry.image_index) + "_l" +
                                         std::to_string(entry.layer) + "_gf" +
                                         std::to_string(g_s18_guest_frames) + ".bin";
            std::ofstream bin_out{bin_path, std::ios::binary};
            if (bin_out) {
                bin_out.write(reinterpret_cast<const char*>(base),
                              static_cast<std::streamsize>(total_bytes));
                LOG_WARNING(Render_D3D12, "Texture dump written: {} ({}x{}, raw {})", bin_path,
                            width, height, total_bytes);
            }
            // TEMP DIAGNOSTIC (S18, Phase 0.2.1): a 1x1 R32G32B32A32 readback IS the exposure
            // value (four floats, 16 bytes). Decode it from the raw bytes and keep it as the
            // phase fingerprint for subsequent captures; the guest-memory read of this texture
            // always returns zeros (GPU-authored), so this readback is the ground truth.
            if (width == 1 && height == 1 && total_bytes >= 16 &&
                (static_cast<DXGI_FORMAT>(entry.dxgi_format) ==
                     DXGI_FORMAT_R32G32B32A32_TYPELESS ||
                 static_cast<DXGI_FORMAT>(entry.dxgi_format) ==
                     DXGI_FORMAT_R32G32B32A32_FLOAT)) {
                f32 s18_expo[4]{};
                std::memcpy(s18_expo, base, sizeof(s18_expo));
                g_s18_last_exposure = fmt::format("{:.6f},{:.6f},{:.6f},{:.6f}", s18_expo[0],
                                                  s18_expo[1], s18_expo[2], s18_expo[3]);
                LOG_WARNING(Render_D3D12,
                            "S18 EXPOSURE readback: img={} f=({:.6f},{:.6f},{:.6f},{:.6f})",
                            entry.image_index, s18_expo[0], s18_expo[1], s18_expo[2],
                            s18_expo[3]);
            }
        }
        entry.readback->Unmap(0, nullptr);
    }
    // Keep the dedupe list in RecordTextureContentDump so an image is never dumped twice.
    g_texture_dumps.clear();
}

} // Anonymous namespace

bool S21IsZeroGpuTexture(u64 gpu_addr) {
    return std::find(g_s21_zero_gpu.begin(), g_s21_zero_gpu.end(), gpu_addr) !=
           g_s21_zero_gpu.end();
}

u32 GetCapturePresentIndex() {
    return g_capture_present_counter;
}

bool ConsumeCaptureAlignDumpRequest() {
    return g_capture_align_dump.exchange(false, std::memory_order_relaxed);
}

void RequestSceneAlignDump() {
    g_capture_align_dump.store(true, std::memory_order_relaxed);
}

void SetCaptureDirectory(std::string directory) {
    g_capture_directory = std::move(directory);
    // S24: the 60 s probes had accumulated 14 038 texdumps / 20.5 GB in LocalState\captures
    // (the dump ids repeat every run, so nothing there survives a process anyway, and the
    // unique _gf names added in 1.0.353 make a run add ~0.5 GB instead of recycling names).
    // Two back-to-back boot deaths on 1.0.353 (0xC00000FD and a silent kill during a dump
    // burst) make a full app-data volume the prime suspect, so wipe the previous process's
    // dumps once, before the first capture of this run can be written.
    std::error_code ec;
    size_t wiped = 0;
    for (const auto& entry : std::filesystem::directory_iterator{g_capture_directory, ec}) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("texdump_", 0) == 0 || name.rfind("scene3d_", 0) == 0) {
            std::filesystem::remove(entry.path(), ec);
            ec.clear();
            ++wiped;
        }
    }
    const auto space = std::filesystem::space(g_capture_directory, ec);
    LOG_WARNING(Render_D3D12, "S24 capture dir: wiped {} previous dumps, free={} MB of {} MB",
                wiped, space.available / (1024 * 1024), space.capacity / (1024 * 1024));
}

std::string GetCaptureDirectory() {
    return g_capture_directory;
}

// TEMP DIAGNOSTIC (session 14): render target of the latest fullscreen 6-vertex 1920x1080
// draw (the compositor/UI target), set in ConfigureDraw and dumped by the mid-frame scene3d
// capture (declared in d3d12_texture_cache.h).
Microsoft::WRL::ComPtr<ID3D12Resource> g_probe_ui;

// TEMP DIAGNOSTIC (S18, Phase 0.2.1): phase fingerprints for captures (declared in
// d3d12_capture.h). Both are only touched from the GPU thread (TickFrame / RenderBlit /
// WriteTextureContentDumps all run inside RendererD3D12::Composite), so no locking is needed.
u64 g_s18_guest_frames = 0;
std::string g_s18_last_exposure;

void RecordScene3dCaptureMidFrame(ID3D12Device* d3d, CommandList& command_list,
                                  ID3D12Resource* src) {
    if (g_scene3d_capture.index < 16) {
        RecordScene3dCapture(d3d, command_list, src);
    }
}

bool RecordTextureContentDump(ID3D12Device* d3d, CommandList& command_list, ID3D12Resource* src,
                              u32 guest_format, u64 guest_addr, u32 image_index, bool force) {
    static std::vector<std::pair<u64, u32>> seen;
    // S21 cycle 7: the per-frame dump queue was capped at 32; a witness window arms up to
    // 4 draws (14+8+14+7 sampled slots) so the 4th draw's readbacks (the character) were
    // silently dropped. 64 covers the whole window.
    if (!src || g_texture_dumps.size() >= 64) {
        return false;
    }
    if (!force &&
        std::find(seen.begin(), seen.end(), std::make_pair(guest_addr, image_index)) !=
            seen.end()) {
        return false;
    }
    const D3D12_RESOURCE_DESC desc = src->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width == 0 ||
        desc.Height == 0 || desc.Width > 4096 || desc.Height > 4096) {
        return false;
    }
    // S24: one layer/mip of the image into a readback of its own. Extracted from the old body so a
    // cubemap array can be dumped face by face: the 13-cube environment probe is 78 layers and
    // the scene's sky comes out black, so layer 0 (cube 0 / face 0) alone cannot tell a wrong
    // cube index from a black face.
    // S26: `mip` selects the mip level inside the layer (0 = the surface the fills paint).
    // The bank's coarse mips have never been observed and are structurally black (the members
    // are 1-mip images, the join copies level 0 only, the alias tail syncs are skipped), so
    // layer 0 mips 1-2 are armed once for the 946xxx timeline band.
    const auto arm_layer = [&](u32 layer, u32 mip = 0, u32 id_override = 0) -> bool {
        if (g_texture_dumps.size() >= 64) {
            return false;
        }
        TextureContentDump entry;
        UINT64 total_bytes = 0;
        UINT rows = 0;
        UINT64 row_size = 0;
        const UINT sub = layer * static_cast<u32>(desc.MipLevels) + mip;
        d3d->GetCopyableFootprints(&desc, sub, 1, 0, &entry.footprint, &rows, &row_size,
                                   &total_bytes);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer_desc{};
        buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer_desc.Width = total_bytes;
        buffer_desc.Height = 1;
        buffer_desc.DepthOrArraySize = 1;
        buffer_desc.MipLevels = 1;
        buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
        buffer_desc.SampleDesc.Count = 1;
        buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&entry.readback)))) {
            return false;
        }
        command_list.Transition(src, D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = entry.readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = entry.footprint;
        D3D12_TEXTURE_COPY_LOCATION src_loc{};
        src_loc.pResource = src;
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        // Subresources are mip-major: mip 0 of array slice N is subresource N * MipLevels. Using
        // the raw layer index would dump mips 1..MipLevels-1 of slice 0 as if they were faces
        // (the 13-cube probe has 6 mips, so layers 1-5 are mips, not faces).
        src_loc.SubresourceIndex = sub;
        command_list.Get()->CopyTextureRegion(&dst, 0, 0, 0, &src_loc, nullptr);
        command_list.Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                D3D12_RESOURCE_STATE_COMMON);
        entry.width = static_cast<u32>(desc.Width) >> mip;
        entry.height = static_cast<u32>(desc.Height) >> mip;
        entry.dxgi_format = static_cast<u32>(desc.Format);
        entry.guest_format = guest_format;
        entry.guest_addr = guest_addr;
        entry.image_index = id_override != 0 ? id_override : image_index;
        entry.layer = layer;
        entry.armed = true;
        g_texture_dumps.push_back(std::move(entry));
        return true;
    };
    if (!arm_layer(0)) {
        return false;
    }
    seen.emplace_back(guest_addr, image_index);
    // Small uncompressed array/cube images also get cube 0's remaining faces and the first face
    // of every later cube (all 13 cubes of the env probe = 18 entries including layer 0, capped
    // so one call cannot eat the 64-entry queue). Only the S24 probes (id bands 943xxx sky probe
    // and 946xxx env timeline) ask for layers: the boot-time LUT probes also hit a 256x256 cube
    // array and must not gain a pile of readbacks while the guest is loading. The desc cannot
    // distinguish a cube array from a plain 2D array, so a plain array in those bands gets the
    // same every-6th-slice sample - harmless, the probes only aim this at the env probe.
    // S25: the 947xxx band (the post-join bank dump) arms ONLY layers 0-5 — the six
    // 256x256 members — so one join answers "did the copies land" without eating the
    // 64-entry queue.
    const bool s25_members_only = image_index >= 946000u && image_index < 948000u;
    if (((image_index >= 943000u && image_index < 944000u) || s25_members_only) &&
        desc.DepthOrArraySize > 1 && desc.Width <= 512 && desc.Height <= 512 &&
        !TextureFormatIsBlockCompressed(static_cast<u32>(desc.Format))) {
        // S25: the 946xxx timeline rotates through layers 0-5 only (the six member faces
        // the fills paint); 18/24 were far tiles the game never writes in this scene.
        u32 added = 0;
        const u32 s25_layer_limit = s25_members_only ? 6u : desc.DepthOrArraySize;
        for (u32 layer = 1; layer < s25_layer_limit && added < 19; ++layer) {
            if (!s25_members_only && layer >= 6 && (layer % 6) != 0) {
                continue;
            }
            if (!arm_layer(layer)) {
                break;
            }
            ++added;
        }
        // S26: layer 0's coarse mips (never observed; structurally black in our pipeline -
        // the members are 1-mip images, the join copies level 0 only, the alias tail syncs
        // are skipped by the sub-4x4 guard). Armed twice, ids 948xxx.
        static u32 s26_mip_dumps = 0;
        if (s25_members_only && s26_mip_dumps < 2) {
            ++s26_mip_dumps;
            const u32 s26_mip_base = 948000u + s26_mip_dumps * 10u;
            (void)arm_layer(0, 1, s26_mip_base + 1);
            (void)arm_layer(0, 2, s26_mip_base + 2);
        }
    }
    return true;
}

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

    // Drive texture/buffer GC + staging pool ticks once per present (mirrors Vulkan
    // SwapBuffers, which calls rasterizer.TickFrame() per Composite). TickFrame takes its
    // own cache locks; no m_buffer_cache/m_texture_cache mutex is held at this point
    // (AccelerateDisplay's lock is scope-local and released on return).
    rasterizer.TickFrame();

    // Present-path tracing: one DEBUG line every 20 Composite calls, with memory stats
    // piggybacked every 120th to catch the OOM leak (~1.3 MB/s observed on console).
    static u32 composite_frames = 0;
    if (++composite_frames % 20 == 0) {
        if (composite_frames % 120 == 0) {
            rasterizer.LogMemoryStats();
        }
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
    // TEMP INSTRUMENTATION (S17): frame-rate probe. The user reports 1-2 FPS on console and we have
    // no number of our own, so log presents per 30 s window (bounded). Uses GetTickCount64 to avoid
    // pulling in <chrono> here. Read this with a WARM shader cache before drawing conclusions: the
    // first run after an install recompiles every pipeline on first use.
    static u32 s17_frames = 0;
    static u32 s17_logs = 0;
    static u64 s17_window_ms = GetTickCount64();
    swapchain.PresentIfRequested();
    ++s17_frames;
    const u64 s17_now_ms = GetTickCount64();
    if (s17_now_ms - s17_window_ms >= 30000 && s17_logs < 120) {
        ++s17_logs;
        const f64 s17_secs = static_cast<f64>(s17_now_ms - s17_window_ms) / 1000.0;
        LOG_WARNING(Render_D3D12, "S17 perf: {} presents in {:.1f}s = {:.2f} FPS", s17_frames,
                    s17_secs, static_cast<f64>(s17_frames) / s17_secs);
        s17_frames = 0;
        s17_window_ms = s17_now_ms;
    }
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

    const bool probe_frame = ProbeFrameWanted();
    if (probe_frame) {
        RecordResourceProbe(device.GetDevice(), command_list, 0, info.view->Resource());
        RecordResourceProbe(device.GetDevice(), command_list, 1, g_probe_scene.Get());
        RecordResourceProbe(device.GetDevice(), command_list, 2, g_probe_any_sampled.Get());
        // TEMP DIAGNOSTIC (session 13): F2D blit source/destination content, to see whether
        // the repeated 640x360 blit fills the texture the missing splash layer samples.
        RecordResourceProbe(device.GetDevice(), command_list, 4, g_probe_blit_src.Get());
        RecordResourceProbe(device.GetDevice(), command_list, 5, g_probe_blit_dst.Get());
        // TEMP DIAGNOSTIC (session 13): the render target of the first depth-tested 3D draw
        // (the scene behind the UI), to show whether the 3D scene RT actually receives content.
        RecordResourceProbe(device.GetDevice(), command_list, 6, g_probe_scene3d.Get());
        // TEMP DIAGNOSTIC (session 13): the full BMP of the same target used to be taken here,
        // but the probe frames run out long before the 3D scene is drawn, so the snapshots
        // always came from before it. It now follows the presented-frame cadence below.
        RecordCalibrationProbe(device.GetDevice(), command_list);
    }

    const u32 capture_this = g_capture_present_counter++;
    // TEMP DIAGNOSTIC (session 13): spread captures over long runs, so boot progress can be
    // watched over time (newest kept by PruneCaptures).
    // S21: a dense window across the style phase (guest frames 16200-22500) - the 900-frame
    // cadence aliases the ~3-frame "UI-only black" dips and the camera dolly, so the
    // settled-phase framing and the dip frequency were never measurable.
    const bool capture_dense = capture_this >= 16200 && capture_this <= 22500 &&
                               (capture_this % 150) == 0;
    // S21 cycle 6: dense captures across the style phase (gf 11700-15000) - the character is
    // the fragment cloud there, so ~4 frames per 900-frame cadence show the animation state.
    const bool capture_iso = capture_this >= 11700 && capture_this <= 15000 &&
                             (capture_this % 150) == 0;
    const bool capture_frame = CaptureWanted() && g_frame_capture.index < 200 &&
                               ((capture_this >= 600 && (capture_this % 900) == 0) || capture_dense ||
                                capture_iso);
    if (capture_frame) {
        RecordFrameCapture(device.GetDevice(), command_list, info.view->Resource());
    }

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
    if (probe_frame) {
        device.WaitForIdle();
        LogResourceProbes();
    }
    if (capture_frame) {
        device.WaitForIdle();
        WriteFrameCapture();
    }
    // TEMP DIAGNOSTIC (session 13): the scene3d copy was recorded on the same list that just
    // executed, so its readback is complete once the list (and the idle wait above) is done.
    if (g_scene3d_capture.armed) {
        if (g_scene3d_capture.index < 16) {
            device.WaitForIdle();
            WriteScene3dCapture();
        } else {
            g_scene3d_capture.armed = false;
        }
    }
    // TEMP DIAGNOSTIC (session 15): sampled-texture dumps are recorded mid-frame, so their
    // readbacks are complete once the list that carried the copy has executed.
    if (!g_texture_dumps.empty()) {
        device.WaitForIdle();
        WriteTextureContentDumps();
    }
}

std::vector<u8> RendererD3D12::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

std::string RendererD3D12::GetDeviceVendor() const {
    return device.GetAdapterName();
}

} // namespace D3D12
