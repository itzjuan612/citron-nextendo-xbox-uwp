// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>

#include "common/alignment.h"
#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_command_list.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_texture_cache.h"

namespace D3D12 {

namespace {

using Tegra::Texture::DepthCompareFunc;
using Tegra::Texture::SamplerReduction;
using Tegra::Texture::SwizzleSource;
using Tegra::Texture::TextureFilter;
using Tegra::Texture::TextureMipmapFilter;
using Tegra::Texture::TSCEntry;
using Tegra::Texture::WrapMode;
using VideoCommon::BufferImageCopy;
using VideoCommon::Extent3D;
using VideoCommon::ImageCopy;
using VideoCommon::ImageType;
using VideoCommon::ImageViewType;
using VideoCommon::Offset3D;
using VideoCommon::SubresourceLayers;
using VideoCore::Surface::PixelFormat;

// TEMP DIAGNOSTIC: ring buffer of the most recent texture uploads, dumped when the
// rasterizer's per-draw flush detects a device removal.
constexpr u32 TXUP_RING_SIZE = 16;
std::array<std::string, TXUP_RING_SIZE> g_txup_ring;
std::atomic<u32> g_txup_ring_index{0};
std::atomic<bool> g_txup_ring_dumped{false};

u32 CalcSubresource(u32 mip, u32 slice, u32 plane, u32 mips, u32 array_size) {
    return mip + slice * mips + plane * mips * array_size;
}

DXGI_FORMAT ResourceFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::R8_UNORM:
    case PixelFormat::R8_SNORM:
    case PixelFormat::R8_UINT:
    case PixelFormat::R8_SINT:
        return DXGI_FORMAT_R8_TYPELESS;
    case PixelFormat::R8G8_UNORM:
    case PixelFormat::R8G8_SNORM:
    case PixelFormat::R8G8_UINT:
    case PixelFormat::R8G8_SINT:
        return DXGI_FORMAT_R8G8_TYPELESS;
    case PixelFormat::R16_UNORM:
    case PixelFormat::R16_SNORM:
    case PixelFormat::R16_UINT:
    case PixelFormat::R16_SINT:
    case PixelFormat::R16_FLOAT:
        return DXGI_FORMAT_R16_TYPELESS;
    case PixelFormat::R32_UINT:
    case PixelFormat::R32_SINT:
    case PixelFormat::R32_FLOAT:
        return DXGI_FORMAT_R32_TYPELESS;
    case PixelFormat::A8B8G8R8_UNORM:
    case PixelFormat::A8B8G8R8_SNORM:
    case PixelFormat::A8B8G8R8_SINT:
    case PixelFormat::A8B8G8R8_UINT:
    case PixelFormat::A8B8G8R8_SRGB:
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case PixelFormat::B8G8R8A8_UNORM:
    case PixelFormat::B8G8R8A8_SRGB:
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case PixelFormat::D32_FLOAT:
        return DXGI_FORMAT_R32_TYPELESS;
    case PixelFormat::D16_UNORM:
        return DXGI_FORMAT_R16_TYPELESS;
    case PixelFormat::X8_D24_UNORM:
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
        return DXGI_FORMAT_R24G8_TYPELESS;
    case PixelFormat::D32_FLOAT_S8_UINT:
        return DXGI_FORMAT_R32G8X24_TYPELESS;
    case PixelFormat::BC1_RGBA_UNORM:
        return DXGI_FORMAT_BC1_TYPELESS;
    case PixelFormat::BC2_UNORM:
        return DXGI_FORMAT_BC2_TYPELESS;
    case PixelFormat::BC3_UNORM:
        return DXGI_FORMAT_BC3_TYPELESS;
    case PixelFormat::BC4_UNORM:
    case PixelFormat::BC4_SNORM:
        return DXGI_FORMAT_BC4_TYPELESS;
    case PixelFormat::BC5_UNORM:
    case PixelFormat::BC5_SNORM:
        return DXGI_FORMAT_BC5_TYPELESS;
    case PixelFormat::BC6H_UFLOAT:
    case PixelFormat::BC6H_SFLOAT:
        return DXGI_FORMAT_BC6H_TYPELESS;
    case PixelFormat::A2B10G10R10_UNORM:
    case PixelFormat::A2B10G10R10_UINT:
        return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    case PixelFormat::B10G11R11_FLOAT:
        // The resource is allocated typed as R11G11B10_FLOAT so the typed
        // SRV/RTV views created from it match. This format is not UAV-capable,
        // so the UAV flag is cleared by the whitelist below, which keys on the
        // resolved resource format.
        return DXGI_FORMAT_R11G11B10_FLOAT;
    case PixelFormat::BC7_UNORM:
    case PixelFormat::BC7_SRGB:
        return DXGI_FORMAT_BC7_TYPELESS;
    case PixelFormat::R16G16B16A16_FLOAT:
    case PixelFormat::R16G16B16A16_UNORM:
    case PixelFormat::R16G16B16A16_SNORM:
    case PixelFormat::R16G16B16A16_SINT:
    case PixelFormat::R16G16B16A16_UINT:
    case PixelFormat::R16G16B16X16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    case PixelFormat::R32G32B32A32_FLOAT:
    case PixelFormat::R32G32B32A32_UINT:
    case PixelFormat::R32G32B32A32_SINT:
        return DXGI_FORMAT_R32G32B32A32_TYPELESS;
    case PixelFormat::R32G32B32_FLOAT:
        return DXGI_FORMAT_R32G32B32_TYPELESS;
    case PixelFormat::R32G32_FLOAT:
    case PixelFormat::R32G32_SINT:
    case PixelFormat::R32G32_UINT:
        return DXGI_FORMAT_R32G32_TYPELESS;
    case PixelFormat::R16G16_FLOAT:
    case PixelFormat::R16G16_UNORM:
    case PixelFormat::R16G16_SNORM:
    case PixelFormat::R16G16_UINT:
    case PixelFormat::R16G16_SINT:
        return DXGI_FORMAT_R16G16_TYPELESS;
    case PixelFormat::BC1_RGBA_SRGB:
        return DXGI_FORMAT_BC1_TYPELESS;
    case PixelFormat::BC2_SRGB:
        return DXGI_FORMAT_BC2_TYPELESS;
    case PixelFormat::BC3_SRGB:
        return DXGI_FORMAT_BC3_TYPELESS;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

DXGI_FORMAT SampledFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::R8_UNORM:
        return DXGI_FORMAT_R8_UNORM;
    case PixelFormat::R8_SNORM:
        return DXGI_FORMAT_R8_SNORM;
    case PixelFormat::R8_UINT:
        return DXGI_FORMAT_R8_UINT;
    case PixelFormat::R8_SINT:
        return DXGI_FORMAT_R8_SINT;
    case PixelFormat::R8G8_UNORM:
        return DXGI_FORMAT_R8G8_UNORM;
    case PixelFormat::R8G8_SNORM:
        return DXGI_FORMAT_R8G8_SNORM;
    case PixelFormat::R8G8_UINT:
        return DXGI_FORMAT_R8G8_UINT;
    case PixelFormat::R8G8_SINT:
        return DXGI_FORMAT_R8G8_SINT;
    case PixelFormat::R16_UNORM:
        return DXGI_FORMAT_R16_UNORM;
    case PixelFormat::R16_SNORM:
        return DXGI_FORMAT_R16_SNORM;
    case PixelFormat::R16_UINT:
        return DXGI_FORMAT_R16_UINT;
    case PixelFormat::R16_SINT:
        return DXGI_FORMAT_R16_SINT;
    case PixelFormat::R16_FLOAT:
        return DXGI_FORMAT_R16_FLOAT;
    case PixelFormat::R32_UINT:
        return DXGI_FORMAT_R32_UINT;
    case PixelFormat::R32_SINT:
        return DXGI_FORMAT_R32_SINT;
    case PixelFormat::R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case PixelFormat::A8B8G8R8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case PixelFormat::A8B8G8R8_SNORM:
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    case PixelFormat::A8B8G8R8_UINT:
        return DXGI_FORMAT_R8G8B8A8_UINT;
    case PixelFormat::A8B8G8R8_SINT:
        return DXGI_FORMAT_R8G8B8A8_SINT;
    case PixelFormat::A8B8G8R8_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case PixelFormat::B8G8R8A8_UNORM:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case PixelFormat::B8G8R8A8_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case PixelFormat::D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case PixelFormat::D16_UNORM:
        return DXGI_FORMAT_R16_UNORM;
    case PixelFormat::X8_D24_UNORM:
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case PixelFormat::D32_FLOAT_S8_UINT:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case PixelFormat::BC1_RGBA_UNORM:
        return DXGI_FORMAT_BC1_UNORM;
    case PixelFormat::BC2_UNORM:
        return DXGI_FORMAT_BC2_UNORM;
    case PixelFormat::BC3_UNORM:
        return DXGI_FORMAT_BC3_UNORM;
    case PixelFormat::BC4_UNORM:
        return DXGI_FORMAT_BC4_UNORM;
    case PixelFormat::BC4_SNORM:
        return DXGI_FORMAT_BC4_SNORM;
    case PixelFormat::BC5_UNORM:
        return DXGI_FORMAT_BC5_UNORM;
    case PixelFormat::BC5_SNORM:
        return DXGI_FORMAT_BC5_SNORM;
    case PixelFormat::BC6H_UFLOAT:
        return DXGI_FORMAT_BC6H_UF16;
    case PixelFormat::BC6H_SFLOAT:
        return DXGI_FORMAT_BC6H_SF16;
    case PixelFormat::A2B10G10R10_UNORM:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case PixelFormat::A2B10G10R10_UINT:
        return DXGI_FORMAT_R10G10B10A2_UINT;
    case PixelFormat::B10G11R11_FLOAT:
        return DXGI_FORMAT_R11G11B10_FLOAT;
    case PixelFormat::BC7_UNORM:
        return DXGI_FORMAT_BC7_UNORM;
    case PixelFormat::BC7_SRGB:
        return DXGI_FORMAT_BC7_UNORM_SRGB;
    case PixelFormat::R16G16B16A16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case PixelFormat::R16G16B16A16_UNORM:
        return DXGI_FORMAT_R16G16B16A16_UNORM;
    case PixelFormat::R16G16B16A16_SNORM:
        return DXGI_FORMAT_R16G16B16A16_SNORM;
    case PixelFormat::R16G16B16A16_SINT:
        return DXGI_FORMAT_R16G16B16A16_SINT;
    case PixelFormat::R16G16B16A16_UINT:
        return DXGI_FORMAT_R16G16B16A16_UINT;
    case PixelFormat::R16G16B16X16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case PixelFormat::R32G32B32A32_FLOAT:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case PixelFormat::R32G32B32A32_UINT:
        return DXGI_FORMAT_R32G32B32A32_UINT;
    case PixelFormat::R32G32B32A32_SINT:
        return DXGI_FORMAT_R32G32B32A32_SINT;
    case PixelFormat::R32G32B32_FLOAT:
        return DXGI_FORMAT_R32G32B32_FLOAT;
    case PixelFormat::R32G32_FLOAT:
        return DXGI_FORMAT_R32G32_FLOAT;
    case PixelFormat::R32G32_SINT:
        return DXGI_FORMAT_R32G32_SINT;
    case PixelFormat::R32G32_UINT:
        return DXGI_FORMAT_R32G32_UINT;
    case PixelFormat::R16G16_FLOAT:
        return DXGI_FORMAT_R16G16_FLOAT;
    case PixelFormat::R16G16_UNORM:
        return DXGI_FORMAT_R16G16_UNORM;
    case PixelFormat::R16G16_SNORM:
        return DXGI_FORMAT_R16G16_SNORM;
    case PixelFormat::R16G16_UINT:
        return DXGI_FORMAT_R16G16_UINT;
    case PixelFormat::R16G16_SINT:
        return DXGI_FORMAT_R16G16_SINT;
    case PixelFormat::BC1_RGBA_SRGB:
        return DXGI_FORMAT_BC1_UNORM_SRGB;
    case PixelFormat::BC2_SRGB:
        return DXGI_FORMAT_BC2_UNORM_SRGB;
    case PixelFormat::BC3_SRGB:
        return DXGI_FORMAT_BC3_UNORM_SRGB;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

DXGI_FORMAT RenderTargetFormat(PixelFormat format) {
    const DXGI_FORMAT sampled = SampledFormat(format);
    switch (sampled) {
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return DXGI_FORMAT_UNKNOWN;
    default:
        return sampled;
    }
}

DXGI_FORMAT DepthStencilFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::D32_FLOAT:
        return DXGI_FORMAT_D32_FLOAT;
    case PixelFormat::D16_UNORM:
        return DXGI_FORMAT_D16_UNORM;
    case PixelFormat::X8_D24_UNORM:
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case PixelFormat::D32_FLOAT_S8_UINT:
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

bool IsCompressedFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::BC1_RGBA_UNORM:
    case PixelFormat::BC1_RGBA_SRGB:
    case PixelFormat::BC2_UNORM:
    case PixelFormat::BC2_SRGB:
    case PixelFormat::BC3_UNORM:
    case PixelFormat::BC3_SRGB:
    case PixelFormat::BC4_UNORM:
    case PixelFormat::BC4_SNORM:
    case PixelFormat::BC5_UNORM:
    case PixelFormat::BC5_SNORM:
    case PixelFormat::BC6H_UFLOAT:
    case PixelFormat::BC6H_SFLOAT:
    case PixelFormat::BC7_UNORM:
    case PixelFormat::BC7_SRGB:
        return true;
    default:
        return false;
    }
}

u32 ElementBytes(PixelFormat format) {
    switch (ResourceFormat(format)) {
    case DXGI_FORMAT_R8_TYPELESS:
        return 1;
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R16_TYPELESS:
        return 2;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_R24G8_TYPELESS:
        return 4;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC4_TYPELESS:
        return 8;
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC7_TYPELESS:
        return 16;
    default:
        return 4;
    }
}

u32 ComponentMapping(SwizzleSource source) {
    switch (source) {
    case SwizzleSource::R:
        return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0;
    case SwizzleSource::G:
        return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1;
    case SwizzleSource::B:
        return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2;
    case SwizzleSource::A:
        return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3;
    case SwizzleSource::Zero:
        return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0;
    case SwizzleSource::OneInt:
    case SwizzleSource::OneFloat:
        return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1;
    default:
        return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0;
    }
}

D3D12_TEXTURE_ADDRESS_MODE AddressMode(WrapMode mode) {
    switch (mode) {
    case WrapMode::Wrap:
        return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case WrapMode::Mirror:
        return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case WrapMode::ClampToEdge:
        return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case WrapMode::Border:
    case WrapMode::MirrorOnceBorder:
        return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    case WrapMode::Clamp:
        return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case WrapMode::MirrorOnceClampToEdge:
    case WrapMode::MirrorOnceClampOGL:
        return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    default:
        return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
}

D3D12_COMPARISON_FUNC ComparisonFunc(DepthCompareFunc func) {
    switch (func) {
    case DepthCompareFunc::Never:
        return D3D12_COMPARISON_FUNC_NEVER;
    case DepthCompareFunc::Less:
        return D3D12_COMPARISON_FUNC_LESS;
    case DepthCompareFunc::Equal:
        return D3D12_COMPARISON_FUNC_EQUAL;
    case DepthCompareFunc::LessEqual:
        return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    case DepthCompareFunc::Greater:
        return D3D12_COMPARISON_FUNC_GREATER;
    case DepthCompareFunc::NotEqual:
        return D3D12_COMPARISON_FUNC_NOT_EQUAL;
    case DepthCompareFunc::GreaterEqual:
        return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    case DepthCompareFunc::Always:
        return D3D12_COMPARISON_FUNC_ALWAYS;
    default:
        return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    }
}

} // Anonymous namespace

// TEMP FIX (session 11): D3D12 only allows UAV creation for formats with
// D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW support. Creating a UAV for a
// format without support (SRGB variants, R11G11B10_FLOAT, block-compressed formats,
// ...) is an invalid call that removes the device on Xbox. The resource flag alone is
// not sufficient: SRGB resources get ALLOW_UNORDERED_ACCESS because their typeless
// format is UAV-capable, but the typed SRGB view is not.
bool IsUavCompatibleFormat(ID3D12Device* device, DXGI_FORMAT format) {
    if (device == nullptr || format == DXGI_FORMAT_UNKNOWN) {
        return false;
    }
    const u32 index = static_cast<u32>(format);
    static std::array<std::atomic<s32>, 256> cache{};
    if (index >= cache.size()) {
        return false;
    }
    const s32 cached = cache[index].load(std::memory_order_relaxed);
    if (cached >= 0) {
        return cached != 0;
    }
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{};
    support.Format = format;
    const HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support,
                                                   sizeof(support));
    const bool supported =
        SUCCEEDED(hr) &&
        (support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0;
    cache[index].store(supported ? 1 : 0, std::memory_order_relaxed);
    return supported;
}

Image::Image(TextureCacheRuntime& runtime_, const VideoCommon::ImageInfo& info, GPUVAddr gpu_addr,
             VAddr cpu_addr)
    : VideoCommon::ImageBase{info, gpu_addr, cpu_addr}, device{&runtime_.device},
      runtime{&runtime_} {
    if (info.type == ImageType::Buffer || info.format == PixelFormat::Invalid) {
        return;
    }
    resource = runtime->CreateImageResource(info);
    if (!resource) {
        return;
    }
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    array_size = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                     ? 1
                     : static_cast<u32>(desc.DepthOrArraySize);
    is_depth_stencil = (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
}

Image::Image(const VideoCommon::NullImageParams&)
    : VideoCommon::ImageBase{VideoCommon::NullImageParams{}} {}

Image::~Image() = default;

D3D12_CPU_DESCRIPTOR_HANDLE Image::StorageImageView(s32 level) {
    for (const StorageView& view : storage_views) {
        if (view.level == level) {
            return view.handle;
        }
    }
    if (!resource || static_cast<size_t>(storage_views.size()) >= MAX_STORAGE_VIEWS) {
        return D3D12_CPU_DESCRIPTOR_HANDLE{0};
    }
    const DXGI_FORMAT format = SampledFormat(info.format);
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0 ||
        !IsUavCompatibleFormat(device->GetDevice(), format)) {
        // Xbox removes the device when a UAV is created on a resource without
        // D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS or for a format without UAV support
        // (SRGB, R11G11B10, compressed formats, ...).
        return D3D12_CPU_DESCRIPTOR_HANDLE{0};
    }
    const u32 index = runtime->view_heap.Allocate();
    if (index == DescriptorHeap::INVALID_INDEX) {
        return D3D12_CPU_DESCRIPTOR_HANDLE{0};
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE handle = runtime->view_heap.CpuHandle(index);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = format;
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uav.Texture3D.MipSlice = static_cast<UINT>(level);
        uav.Texture3D.FirstWSlice = 0;
        uav.Texture3D.WSize = static_cast<UINT>(-1);
    } else if (desc.DepthOrArraySize > 1) {
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav.Texture2DArray.MipSlice = static_cast<UINT>(level);
        uav.Texture2DArray.FirstArraySlice = 0;
        uav.Texture2DArray.ArraySize = desc.DepthOrArraySize;
        uav.Texture2DArray.PlaneSlice = 0;
    } else {
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Texture2D.MipSlice = static_cast<UINT>(level);
        uav.Texture2D.PlaneSlice = 0;
    }
    device->GetDevice()->CreateUnorderedAccessView(resource.Get(), nullptr, &uav, handle);
    storage_views.push_back(StorageView{.level = level, .handle = handle});
    return handle;
}

void Image::UploadMemory(ID3D12Resource* buffer, u64 offset,
                         std::span<const BufferImageCopy> copies) {
    if (!resource || info.num_samples > 1 || !runtime->command_list.IsValid()) {
        return;
    }
    if (::D3D12::ResourceFormat(info.format) == DXGI_FORMAT_UNKNOWN) {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            LOG_DEBUG(Render_D3D12, "Skipping Image::UploadMemory (buffer) GPU copy for "
                                    "unmapped guest format {}",
                      static_cast<u32>(info.format));
        }
        initialized = true;
        return;
    }
    runtime->Transition(resource.Get(), D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST);
    const std::span<u8> staging_span = runtime->staging_pool.MappedSpan(buffer);
    for (const BufferImageCopy& copy : copies) {
        UploadSubresource(buffer, staging_span, offset + copy.buffer_offset, copy);
    }
    runtime->command_list.Transition(resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                     D3D12_RESOURCE_STATE_COMMON);
    initialized = true;
}

void Image::UploadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    {
        // Diagnostic (session 12): is the CPU-side staging (guest texel data) actually
        // non-zero? Black rendering with zero staging means the guest read/upload path.
        static u32 upload_probe_logs = 0;
        if (upload_probe_logs < 24 && map.mapped_span.size() >= 64) {
            ++upload_probe_logs;
            const std::span<u8> span = map.mapped_span;
            u32 min_value = 255;
            u32 max_value = 0;
            u64 sum = 0;
            u32 nonzero = 0;
            const size_t count = std::min<size_t>(4096, span.size());
            for (size_t i = 0; i < count; ++i) {
                const u32 value = span[i];
                min_value = value < min_value ? value : min_value;
                max_value = value > max_value ? value : max_value;
                sum += value;
                if (value != 0) {
                    ++nonzero;
                }
            }
            LOG_WARNING(Render_D3D12,
                        "Tex upload data #{}: fmt={:#x} bytes={} min={} max={} sum={} nonzero={}",
                        upload_probe_logs, static_cast<u32>(info.format), span.size(), min_value,
                        max_value, sum, nonzero);
        }
    }
    if (!map.buffer) {
        // Staging allocation failed; skip the upload (the caller's
        // CPU-side state is still updated).
        return;
    }
    if (!resource || info.num_samples > 1 || !runtime->command_list.IsValid()) {
        return;
    }
    if (::D3D12::ResourceFormat(info.format) == DXGI_FORMAT_UNKNOWN) {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            LOG_DEBUG(Render_D3D12, "Skipping Image::UploadMemory (staging) GPU copy for "
                                    "unmapped guest format {}",
                      static_cast<u32>(info.format));
        }
        initialized = true;
        return;
    }
    runtime->Transition(resource.Get(), D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST);
    for (const BufferImageCopy& copy : copies) {
        UploadSubresource(map.buffer, map.mapped_span, map.offset + copy.buffer_offset, copy);
    }
    runtime->command_list.Transition(resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                     D3D12_RESOURCE_STATE_COMMON);
    initialized = true;
}

void Image::DownloadMemory(ID3D12Resource* buffer, size_t offset,
                           std::span<const BufferImageCopy> copies) {
    if (!resource || info.num_samples > 1 || !runtime->command_list.IsValid()) {
        return;
    }
    if (::D3D12::ResourceFormat(info.format) == DXGI_FORMAT_UNKNOWN) {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            LOG_DEBUG(Render_D3D12, "Skipping Image::DownloadMemory (buffer) GPU copy for "
                                    "unmapped guest format {}",
                      static_cast<u32>(info.format));
        }
        return;
    }
    runtime->Transition(resource.Get(), D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE);
    const std::span<u8> staging_span = runtime->staging_pool.MappedSpan(buffer);
    for (const BufferImageCopy& copy : copies) {
        DownloadSubresource(buffer, staging_span, offset + copy.buffer_offset, copy);
    }
    runtime->command_list.Transition(resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_COMMON);
}

void Image::DownloadMemory(std::span<ID3D12Resource*> buffers, std::span<size_t> offsets,
                           std::span<const BufferImageCopy> copies) {
    const size_t count = std::min({buffers.size(), offsets.size(), copies.size()});
    for (size_t i = 0; i < count; ++i) {
        DownloadMemory(buffers[i], offsets[i], std::span<const BufferImageCopy>{&copies[i], 1});
    }
}

void Image::DownloadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    if (!map.buffer) {
        // Staging allocation failed; skip the readback (the caller's
        // CPU-side state is still updated).
        return;
    }
    if (!resource || info.num_samples > 1 || !runtime->command_list.IsValid()) {
        return;
    }
    if (::D3D12::ResourceFormat(info.format) == DXGI_FORMAT_UNKNOWN) {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            LOG_DEBUG(Render_D3D12, "Skipping Image::DownloadMemory (staging) GPU copy for "
                                    "unmapped guest format {}",
                      static_cast<u32>(info.format));
        }
        return;
    }
    runtime->Transition(resource.Get(), D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (const BufferImageCopy& copy : copies) {
        DownloadSubresource(map.buffer, map.mapped_span, map.offset + copy.buffer_offset, copy);
    }
    runtime->command_list.Transition(resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_COMMON);
}

void DumpRecentTextureUploads() {
    if (g_txup_ring_dumped.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    const u32 total = g_txup_ring_index.load(std::memory_order_relaxed);
    const u32 count = std::min<u32>(total, TXUP_RING_SIZE);
    LOG_CRITICAL(Render_D3D12, "Recent texture uploads (last {} of {}):", count, total);
    for (u32 i = 0; i < count; ++i) {
        const u32 idx = (total - count + i) % TXUP_RING_SIZE;
        LOG_CRITICAL(Render_D3D12, "  {}", g_txup_ring[idx]);
    }
}

void Image::UploadSubresource(ID3D12Resource* src_buffer, std::span<const u8> staging_span,
                              u64 src_offset, const BufferImageCopy& copy) {
    if (!src_buffer) {
        return;
    }
    if (::D3D12::ResourceFormat(info.format) == DXGI_FORMAT_UNKNOWN) {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            LOG_DEBUG(Render_D3D12, "Skipping Image::UploadSubresource GPU copy for "
                                    "unmapped guest format {}",
                      static_cast<u32>(info.format));
        }
        return;
    }
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    const u32 mip = static_cast<u32>(copy.image_subresource.base_level);
    // A copy whose mip level or layer range exceeds the host resource computes an
    // out-of-range subresource index; recording it makes CommandList::Close fail with
    // E_INVALIDARG on Xbox and removes the device. Skip those copies.
    const u32 base_layer = static_cast<u32>(copy.image_subresource.base_layer);
    const u32 copy_layers = std::max<u32>(copy.image_subresource.num_layers, 1);
    const bool is_3d = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    if (mip >= desc.MipLevels || (!is_3d && base_layer + copy_layers > array_size)) {
        static std::atomic<u32> g_subresource_oor{0};
        if (g_subresource_oor.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_ERROR(Render_D3D12,
                      "TXUPLOAD subresource out of range: mip={} mips={} layer={} layers={} "
                      "copy_layers={} img={}x{}: copy skipped",
                      mip, desc.MipLevels, base_layer, array_size, copy_layers,
                      static_cast<u64>(desc.Width), desc.Height);
        }
        return;
    }
    const u32 extent_layers =
        desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : array_size;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    const u32 sub = CalcSubresource(mip, 0, 0, desc.MipLevels, extent_layers);
    device->GetDevice()->GetCopyableFootprints(&desc, sub, 1, 0, &footprint, nullptr, nullptr,
                                               nullptr);

    const bool compressed = IsCompressedFormat(info.format);
    const u32 elem_bytes = ElementBytes(info.format);
    const u32 width_units =
        compressed ? (copy.image_extent.width + 3) / 4 : copy.image_extent.width;
    const u32 rows =
        compressed ? (copy.image_extent.height + 3) / 4 : copy.image_extent.height;
    const u64 tight_pitch = static_cast<u64>(width_units) * elem_bytes;
    // For block-compressed formats D3D12 requires the copy box coordinates (and the
    // placed-footprint dimensions) to be aligned to the 4x4 block, even for the last
    // mips that are smaller than one block. A 2x2/1x1 box makes CopyTextureRegion
    // invalid: CommandList::Close fails with E_INVALIDARG and the device is removed.
    const u32 box_width = compressed ? Common::AlignUp(copy.image_extent.width, 4u)
                                     : copy.image_extent.width;
    const u32 box_height = compressed ? Common::AlignUp(copy.image_extent.height, 4u)
                                      : copy.image_extent.height;
    // Xbox rejects sub-4x4 copies (Close fails E_INVALIDARG); skip them.
    if (copy.image_extent.width < 4 || copy.image_extent.height < 4) {
        static std::atomic<u32> g_sub4_skip{0};
        if (g_sub4_skip.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_ERROR(Render_D3D12,
                      "TXUPLOAD sub-4x4 copy skipped (fmt={} lvl={} copy={}x{} compressed={})",
                      static_cast<u32>(info.format), copy.image_subresource.base_level,
                      copy.image_extent.width, copy.image_extent.height, compressed ? 1 : 0);
        }
        return;
    }
    // `buffer_row_length` is in texels; for block-compressed formats the row is a row of
    // blocks, so convert to blocks before multiplying by the block byte size. Using texels
    // directly made the pitch 4x too large, which pushed every row read past the guest
    // data (repack overrun) / recorded copies whose rows exceeded the source buffer
    // (device removal on Xbox).
    const u64 src_pitch =
        copy.buffer_row_length != 0
            ? static_cast<u64>(compressed ? (copy.buffer_row_length + 3) / 4
                                          : copy.buffer_row_length) *
                  elem_bytes
            : tight_pitch;
    const u64 canonical_pitch = footprint.Footprint.RowPitch;
    if (rows == 0 || copy.image_extent.width == 0 || copy.image_extent.depth == 0) {
        return;
    }
    const u32 layers = std::max<u32>(copy.image_subresource.num_layers, 1);
    const u64 layer_src_stride = copy.buffer_size / layers;

    // The guest buffer holds only the copy region (Vulkan layout: `rows` rows of
    // `src_pitch` per layer). Model it as a placed footprint of exactly the copy
    // extent and map image_offset onto the destination coordinates — a footprint
    // spanning the whole mip would read past the guest rows for sub-region copies.
    u64 copy_src_pitch = src_pitch;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = src_buffer;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    StagingBufferRef repack{};
    if (src_pitch != canonical_pitch) {
        // Template data is not in the canonical padded layout; repack rows through a
        // private staging buffer so the copy sees a legal 256-aligned pitch.
        if (staging_span.empty()) {
            return;
        }
        const u64 padded_size = canonical_pitch * rows * layers;
        repack = runtime->staging_pool.Request(padded_size, StagingUsage::Upload);
        if (!repack.buffer || repack.mapped_span.size() < padded_size) {
            return;
        }
        for (u32 l = 0; l < layers; ++l) {
            for (u32 r = 0; r < rows; ++r) {
                const u8* src_row = staging_span.data() + src_offset +
                                    static_cast<u64>(l) * layer_src_stride +
                                    static_cast<u64>(r) * src_pitch;
                u8* dst_row = repack.mapped_span.data() +
                              (static_cast<u64>(l) * rows + r) * canonical_pitch;
                std::memcpy(dst_row, src_row, tight_pitch);
            }
        }
        src.pResource = repack.buffer;
        src.PlacedFootprint.Offset = repack.offset;
        copy_src_pitch = canonical_pitch;
    } else {
        src.PlacedFootprint.Offset = src_offset;
    }
    src.PlacedFootprint.Footprint = footprint.Footprint;
    src.PlacedFootprint.Footprint.Width = box_width;
    src.PlacedFootprint.Footprint.Height = box_height;
    src.PlacedFootprint.Footprint.Depth = copy.image_extent.depth;
    src.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(copy_src_pitch);

    // TEMP DIAGNOSTIC: remember the copy parameters in a ring so the uploads preceding a
    // device-removal can be dumped.
    {
        const u32 ring_idx =
            g_txup_ring_index.fetch_add(1, std::memory_order_relaxed) % TXUP_RING_SIZE;
        g_txup_ring[ring_idx] = fmt::format(
            "guest_fmt={} res_fmt={:#x} img={}x{} copy={}x{}x{} lvl={} layer={} layers={} "
            "src_off={:#x} pitch={} canon={} src_buf={:#x} repack={}",
            static_cast<u32>(info.format), static_cast<u32>(desc.Format),
            static_cast<u64>(desc.Width), desc.Height, copy.image_extent.width,
            copy.image_extent.height, copy.image_extent.depth,
            copy.image_subresource.base_level, copy.image_subresource.base_layer, layers,
            src_offset, src_pitch, canonical_pitch,
            static_cast<u64>(src_buffer->GetDesc().Width), repack.buffer ? 1 : 0);
    }

    // Guard: a placed-footprint copy whose rows extend past the source buffer removes the
    // device on Xbox. Refuse to record such a copy and log the offender.
    {
        const u64 last_layer = layers > 1 ? static_cast<u64>(layer_src_stride) * (layers - 1) : 0;
        const u64 gpu_end = src_offset + last_layer +
                            static_cast<u64>(copy_src_pitch) * (rows - 1) + tight_pitch;
        const u64 gpu_size = static_cast<u64>(src_buffer->GetDesc().Width);
        const u64 cpu_end = src_offset + last_layer +
                            static_cast<u64>(src_pitch) * (rows - 1) + tight_pitch;
        const bool gpu_overrun = gpu_end > gpu_size;
        const bool cpu_overrun = !staging_span.empty() && cpu_end > staging_span.size();
        if ((!repack.buffer && gpu_overrun) || (repack.buffer && cpu_overrun)) {
            static std::atomic<u32> g_txup_overrun{0};
            if (g_txup_overrun.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_ERROR(Render_D3D12,
                          "TXUPLOAD OVERRUN guest_fmt={} img={}x{} copy={}x{} lvl={} pitch={} "
                          "canon={} src_off={:#x} need_end={:#x} buf_size={:#x} repack={}: copy "
                          "skipped",
                          static_cast<u32>(info.format), static_cast<u64>(desc.Width), desc.Height,
                          copy.image_extent.width, copy.image_extent.height,
                          copy.image_subresource.base_level, copy_src_pitch, canonical_pitch,
                          src_offset, repack.buffer ? cpu_end : gpu_end,
                          repack.buffer ? static_cast<u64>(staging_span.size()) : gpu_size,
                          repack.buffer ? 1 : 0);
            }
            return;
        }
    }

    for (u32 l = 0; l < layers; ++l) {
        const u32 layer = static_cast<u32>(copy.image_subresource.base_layer) + l;
        const u32 layer_sub =
            CalcSubresource(mip, layer, 0, desc.MipLevels, extent_layers);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = resource.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = layer_sub;
        D3D12_BOX box{};
        box.left = 0;
        box.top = 0;
        box.front = 0;
        box.right = box_width;
        box.bottom = box_height;
        box.back = copy.image_extent.depth;
        D3D12_TEXTURE_COPY_LOCATION layer_src = src;
        if (!repack.buffer) {
            layer_src.PlacedFootprint.Offset = src.PlacedFootprint.Offset + l * layer_src_stride;
        } else {
            layer_src.PlacedFootprint.Offset =
                repack.offset + static_cast<u64>(l) * canonical_pitch * rows;
        }
        runtime->command_list.Get()->CopyTextureRegion(
            &dst, static_cast<UINT>(copy.image_offset.x), static_cast<UINT>(copy.image_offset.y),
            static_cast<UINT>(copy.image_offset.z), &layer_src, &box);
    }
}

void Image::DownloadSubresource(ID3D12Resource* dst_buffer, std::span<u8> staging_span,
                                 u64 dst_offset, const BufferImageCopy& copy) {
    if (!dst_buffer) {
        return;
    }
    if (::D3D12::ResourceFormat(info.format) == DXGI_FORMAT_UNKNOWN) {
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            LOG_DEBUG(Render_D3D12, "Skipping Image::DownloadSubresource GPU copy for "
                                    "unmapped guest format {}",
                      static_cast<u32>(info.format));
        }
        return;
    }
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    const u32 mip = static_cast<u32>(copy.image_subresource.base_level);
    // A copy whose mip level or layer range exceeds the host resource computes an
    // out-of-range subresource index; recording it makes CommandList::Close fail with
    // E_INVALIDARG on Xbox and removes the device. Skip those copies.
    const u32 base_layer = static_cast<u32>(copy.image_subresource.base_layer);
    const u32 copy_layers = std::max<u32>(copy.image_subresource.num_layers, 1);
    const bool is_3d = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    if (mip >= desc.MipLevels || (!is_3d && base_layer + copy_layers > array_size)) {
        static std::atomic<u32> g_subresource_oor{0};
        if (g_subresource_oor.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_ERROR(Render_D3D12,
                      "TXUPLOAD subresource out of range: mip={} mips={} layer={} layers={} "
                      "copy_layers={} img={}x{}: copy skipped",
                      mip, desc.MipLevels, base_layer, array_size, copy_layers,
                      static_cast<u64>(desc.Width), desc.Height);
        }
        return;
    }
    const u32 extent_layers =
        desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : array_size;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    device->GetDevice()->GetCopyableFootprints(&desc,
                                               CalcSubresource(mip, 0, 0, desc.MipLevels,
                                                                    extent_layers),
                                               1, 0, &footprint, nullptr, nullptr, nullptr);

    const bool compressed = IsCompressedFormat(info.format);
    const u32 elem_bytes = ElementBytes(info.format);
    const u32 width_units =
        compressed ? (copy.image_extent.width + 3) / 4 : copy.image_extent.width;
    const u32 rows =
        compressed ? (copy.image_extent.height + 3) / 4 : copy.image_extent.height;
    const u64 tight_pitch = static_cast<u64>(width_units) * elem_bytes;
    // For block-compressed formats D3D12 requires the copy box coordinates (and the
    // placed-footprint dimensions) to be aligned to the 4x4 block, even for the last
    // mips that are smaller than one block. A 2x2/1x1 box makes CopyTextureRegion
    // invalid: CommandList::Close fails with E_INVALIDARG and the device is removed.
    const u32 box_width = compressed ? Common::AlignUp(copy.image_extent.width, 4u)
                                     : copy.image_extent.width;
    const u32 box_height = compressed ? Common::AlignUp(copy.image_extent.height, 4u)
                                      : copy.image_extent.height;
    // Xbox rejects sub-4x4 copies (Close fails E_INVALIDARG); skip them.
    if (copy.image_extent.width < 4 || copy.image_extent.height < 4) {
        static std::atomic<u32> g_sub4_skip{0};
        if (g_sub4_skip.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_ERROR(Render_D3D12,
                      "TXUPLOAD sub-4x4 copy skipped (fmt={} lvl={} copy={}x{} compressed={})",
                      static_cast<u32>(info.format), copy.image_subresource.base_level,
                      copy.image_extent.width, copy.image_extent.height, compressed ? 1 : 0);
        }
        return;
    }
    const u64 src_pitch =
        copy.buffer_row_length != 0 ? static_cast<u64>(copy.buffer_row_length) * elem_bytes
                                    : tight_pitch;
    const u64 canonical_pitch = footprint.Footprint.RowPitch;
    const u32 layers = std::max<u32>(copy.image_subresource.num_layers, 1);
    if (rows == 0 || copy.image_extent.width == 0 || copy.image_extent.depth == 0) {
        return;
    }

    // The destination footprint describes only the copy region (the guest expects
    // `rows` rows of `src_pitch` per layer), matching the upload path.
    StagingBufferRef repack{};
    D3D12_TEXTURE_COPY_LOCATION gpu_dst{};
    u64 layer_dst_stride = 0;
    if (src_pitch != canonical_pitch) {
        // Copy into a padded private staging buffer, then unpack tight rows into the
        // template's staging buffer (which expects its own layout).
        if (staging_span.empty()) {
            return;
        }
        const u64 padded_size = canonical_pitch * rows * layers;
        repack = runtime->staging_pool.Request(padded_size, StagingUsage::Download);
        if (!repack.buffer || repack.mapped_span.size() < padded_size) {
            return;
        }
        gpu_dst.pResource = repack.buffer;
        gpu_dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        gpu_dst.PlacedFootprint.Offset = repack.offset;
        gpu_dst.PlacedFootprint.Footprint = footprint.Footprint;
        layer_dst_stride = canonical_pitch * rows;
    } else {
        gpu_dst.pResource = dst_buffer;
        gpu_dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        gpu_dst.PlacedFootprint.Offset = dst_offset;
        gpu_dst.PlacedFootprint.Footprint = footprint.Footprint;
        layer_dst_stride = copy.buffer_size / layers;
    }
    gpu_dst.PlacedFootprint.Footprint.Width = box_width;
    gpu_dst.PlacedFootprint.Footprint.Height = box_height;
    gpu_dst.PlacedFootprint.Footprint.Depth = copy.image_extent.depth;
    gpu_dst.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(
        src_pitch != canonical_pitch ? canonical_pitch : src_pitch);

    for (u32 l = 0; l < layers; ++l) {
        const u32 layer = static_cast<u32>(copy.image_subresource.base_layer) + l;
        const u32 layer_sub =
            CalcSubresource(mip, layer, 0, desc.MipLevels, extent_layers);
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = resource.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = layer_sub;
        D3D12_BOX box{};
        box.left = static_cast<UINT>(copy.image_offset.x);
        box.top = static_cast<UINT>(copy.image_offset.y);
        box.front = static_cast<UINT>(copy.image_offset.z);
        box.right = box.left + box_width;
        box.bottom = box.top + box_height;
        box.back = box.front + (copy.image_extent.depth == 0 ? 1 : copy.image_extent.depth);
        D3D12_TEXTURE_COPY_LOCATION layer_dst = gpu_dst;
        layer_dst.PlacedFootprint.Offset += static_cast<u64>(l) * layer_dst_stride;
        runtime->command_list.Get()->CopyTextureRegion(&layer_dst, 0, 0, 0, &src, &box);

        if (repack.buffer) {
            // Unpack the padded rows into the template's staging buffer.
            const u8* src_base = repack.mapped_span.data() + static_cast<u64>(l) *
                                                                       canonical_pitch * rows;
            u8* dst_base = staging_span.data() + dst_offset +
                           static_cast<u64>(l) * (copy.buffer_size / (layers == 0 ? 1 : layers));
            for (u32 r = 0; r < rows; ++r) {
                std::memcpy(dst_base + static_cast<u64>(r) * src_pitch,
                            src_base + static_cast<u64>(r) * canonical_pitch, tight_pitch);
            }
        }
    }
}

ImageView::ImageView(TextureCacheRuntime& runtime_, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id, Image& image)
    : VideoCommon::ImageViewBase{info, image.info, image_id, image.gpu_addr} {
    CreateViews(image, info);
}

ImageView::ImageView(TextureCacheRuntime& runtime_, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id, Image& image, const Common::SlotVector<Image>&)
    : VideoCommon::ImageViewBase{info, image.info, image_id, image.gpu_addr} {
    CreateViews(image, info);
}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo& info,
                     const VideoCommon::ImageViewInfo& view_info, GPUVAddr addr)
    : VideoCommon::ImageViewBase{info, view_info, addr} {
    // Texture-buffer (buffer-backed) view: the descriptor is created by the buffer cache
    // step when texel buffer bindings are translated.
}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::NullImageViewParams&)
    : VideoCommon::ImageViewBase{VideoCommon::NullImageViewParams{}} {}

ImageView::~ImageView() = default;

void ImageView::CreateViews(Image& image, const VideoCommon::ImageViewInfo& view_info) {
    ID3D12Resource* resource_ = image.Handle();
    if (!resource_) {
        return;
    }
    resource = resource_;
    const DXGI_FORMAT sampled_format = SampledFormat(format);
    if (sampled_format == DXGI_FORMAT_UNKNOWN) {
        return;
    }
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();

    const std::array<SwizzleSource, 4> mapping_sources = view_info.Swizzle();
    const UINT mapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
        ComponentMapping(mapping_sources[0]), ComponentMapping(mapping_sources[1]),
        ComponentMapping(mapping_sources[2]), ComponentMapping(mapping_sources[3]));

    const UINT most_detailed = static_cast<UINT>(range.base.level);
    const UINT num_mips = static_cast<UINT>(std::max<s32>(range.extent.levels, 1));
    const UINT first_layer = static_cast<UINT>(range.base.layer);
    const UINT num_layers = static_cast<UINT>(std::max<s32>(range.extent.layers, 1));

    const u32 srv_index = image.runtime->view_heap.Allocate();
    if (srv_index != DescriptorHeap::INVALID_INDEX) {
        sampled_view = image.runtime->view_heap.CpuHandle(srv_index);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = sampled_format;
        srv.Shader4ComponentMapping = mapping;
        if (desc.SampleDesc.Count > 1) {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        } else if (type == ImageViewType::e1D) {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            srv.Texture1D.MostDetailedMip = most_detailed;
            srv.Texture1D.MipLevels = num_mips;
        } else if (type == ImageViewType::e3D) {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srv.Texture3D.MostDetailedMip = most_detailed;
            srv.Texture3D.MipLevels = num_mips;
        } else if (type == ImageViewType::Cube) {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srv.TextureCube.MostDetailedMip = most_detailed;
            srv.TextureCube.MipLevels = num_mips;
        } else if (num_layers > 1 || desc.DepthOrArraySize > 1) {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srv.Texture2DArray.MostDetailedMip = most_detailed;
            srv.Texture2DArray.MipLevels = num_mips;
            srv.Texture2DArray.FirstArraySlice = first_layer;
            srv.Texture2DArray.ArraySize = num_layers;
        } else {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MostDetailedMip = most_detailed;
            srv.Texture2D.MipLevels = num_mips;
        }
        image.device->GetDevice()->CreateShaderResourceView(resource, &srv, sampled_view);
    }

    if (image.IsDepthStencil()) {
        const DXGI_FORMAT dsv_format = DepthStencilFormat(image.info.format);
        if (dsv_format != DXGI_FORMAT_UNKNOWN) {
            const u32 dsv_index = image.runtime->dsv_heap.Allocate();
            if (dsv_index != DescriptorHeap::INVALID_INDEX) {
                rt_view = image.runtime->dsv_heap.CpuHandle(dsv_index);
                D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
                dsv.Format = dsv_format;
                dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                dsv.Texture2D.MipSlice = most_detailed;
                dsv.Flags = D3D12_DSV_FLAG_NONE;
                image.device->GetDevice()->CreateDepthStencilView(resource, &dsv, rt_view);
            }
        }
        return;
    }

    const DXGI_FORMAT rtv_format = RenderTargetFormat(image.info.format);
    if (rtv_format != DXGI_FORMAT_UNKNOWN && desc.SampleDesc.Count <= 1) {
        const u32 rtv_index = image.runtime->rtv_heap.Allocate();
        if (rtv_index != DescriptorHeap::INVALID_INDEX) {
            rt_view = image.runtime->rtv_heap.CpuHandle(rtv_index);
            D3D12_RENDER_TARGET_VIEW_DESC rtv{};
            rtv.Format = rtv_format;
            if (type == ImageViewType::e3D) {
                rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
                rtv.Texture3D.MipSlice = most_detailed;
                rtv.Texture3D.FirstWSlice = first_layer;
                rtv.Texture3D.WSize = num_layers;
            } else if (num_layers > 1 || desc.DepthOrArraySize > 1) {
                rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                rtv.Texture2DArray.MipSlice = most_detailed;
                rtv.Texture2DArray.FirstArraySlice = first_layer;
                rtv.Texture2DArray.ArraySize = num_layers;
            } else {
                rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                rtv.Texture2D.MipSlice = most_detailed;
            }
            image.device->GetDevice()->CreateRenderTargetView(resource, &rtv, rt_view);
        }
    }

    if (desc.SampleDesc.Count == 1 && !image.IsDepthStencil() &&
        (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0 &&
        IsUavCompatibleFormat(image.device->GetDevice(),
                              rtv_format != DXGI_FORMAT_UNKNOWN ? rtv_format : sampled_format)) {
        const u32 uav_index = image.runtime->view_heap.Allocate();
        if (uav_index != DescriptorHeap::INVALID_INDEX) {
            storage_view = image.runtime->view_heap.CpuHandle(uav_index);
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format = rtv_format != DXGI_FORMAT_UNKNOWN ? rtv_format : sampled_format;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav.Texture2D.MipSlice = most_detailed;
            image.device->GetDevice()->CreateUnorderedAccessView(resource, nullptr, &uav,
                                                                 storage_view);
        }
    }
}

Sampler::Sampler(TextureCacheRuntime&, const TSCEntry& config) {
    const bool mag_linear = config.mag_filter == TextureFilter::Linear;
    const bool min_linear = config.min_filter == TextureFilter::Linear;
    const bool mip_linear = config.mipmap_filter == TextureMipmapFilter::Linear;
    const f32 max_anisotropy = std::clamp(config.MaxAnisotropy(), 1.0f, 16.0f);
    const bool use_aniso = max_anisotropy > 1.0f && mag_linear && min_linear;
    const bool compare = config.depth_compare_enabled != 0;

    const D3D12_FILTER_TYPE min_type =
        min_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mag_type =
        mag_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mip_type =
        mip_linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    D3D12_FILTER filter{};
    if (use_aniso && !compare) {
        filter = D3D12_FILTER_ANISOTROPIC;
    } else if (use_aniso) {
        filter = D3D12_FILTER_COMPARISON_ANISOTROPIC;
    } else if (compare) {
        filter = D3D12_ENCODE_BASIC_FILTER(min_type, mag_type, mip_type,
                                           D3D12_FILTER_REDUCTION_TYPE_COMPARISON);
    } else {
        filter = D3D12_ENCODE_BASIC_FILTER(min_type, mag_type, mip_type,
                                           D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    }
    desc.Filter = filter;
    desc.AddressU = AddressMode(config.wrap_u);
    desc.AddressV = AddressMode(config.wrap_v);
    desc.AddressW = AddressMode(config.wrap_p);
    desc.MipLODBias = config.LodBias();
    desc.MaxAnisotropy = use_aniso ? static_cast<UINT>(max_anisotropy) : 1;
    desc.ComparisonFunc =
        config.depth_compare_enabled ? ComparisonFunc(config.depth_compare_func)
                                      : D3D12_COMPARISON_FUNC_NEVER;
    const bool has_mips = config.mipmap_filter != TextureMipmapFilter::None;
    desc.MinLOD = has_mips ? config.MinLod() : 0.0f;
    desc.MaxLOD = has_mips ? config.MaxLod() : 0.25f;
    // D3D12 only supports preset border colors; approximate the guest color.
    const std::array<f32, 4> border = config.BorderColor();
    const bool white = border[0] >= 0.99f && border[1] >= 0.99f && border[2] >= 0.99f;
    const bool opaque = border[3] >= 0.99f;
    desc.BorderColor[0] = white ? 1.0f : 0.0f;
    desc.BorderColor[1] = white ? 1.0f : 0.0f;
    desc.BorderColor[2] = white ? 1.0f : 0.0f;
    desc.BorderColor[3] = opaque ? 1.0f : 0.0f;
}

Framebuffer::Framebuffer(TextureCacheRuntime&,
                         std::array<ImageView*, VideoCommon::NUM_RT> color_buffers_,
                         ImageView* depth_buffer_, const VideoCommon::RenderTargets& key)
    : color_buffers{color_buffers_}, depth_buffer{depth_buffer_} {
    // Render area: the render-targets key size clamped to the smallest bound color
    // buffer (the draw translation also clamps its viewport/scissor to this).
    VideoCommon::Extent2D size{key.size};
    for (ImageView* view : color_buffers) {
        if (view) {
            size.width = std::min(size.width, view->size.width);
            size.height = std::min(size.height, view->size.height);
        }
    }
    if (depth_buffer) {
        size.width = std::min(size.width, depth_buffer->size.width);
        size.height = std::min(size.height, depth_buffer->size.height);
    }
    render_area = size;
}

TextureCacheRuntime::TextureCacheRuntime(Device& device_, CommandList& command_list_,
                                         StagingBufferPool& staging_pool_)
    : device{device_}, command_list{command_list_}, staging_pool{staging_pool_},
      view_heap{device_.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8192, false},
      rtv_heap{device_.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2048, false},
      dsv_heap{device_.GetDevice(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 512, false} {}

void TextureCacheRuntime::Finish() {
    device.WaitForIdle();
}

StagingBufferRef TextureCacheRuntime::UploadStagingBuffer(size_t size) {
    return staging_pool.Request(size, StagingUsage::Upload);
}

StagingBufferRef TextureCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, StagingUsage::Download, deferred);
}

void TextureCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

void TextureCacheRuntime::TickFrame() {
    ++tick;
    staging_pool.TickFrame();
}

/// Queries the LOCAL video-memory segment for the adapter backing the D3D12 device.
/// Least-invasive path: Device stores no adapter getter, so this goes through the
/// ID3D12Device -> IDXGIDevice -> IDXGIAdapter -> IDXGIAdapter3 chain (the same chain
/// Device::Device uses for its default-adapter fallback). Queried once per call;
/// callers sample this at most once per frame, so nothing is cached here.
static bool QueryLocalVideoMemoryInfo(ID3D12Device* d3d_device,
                                      DXGI_QUERY_VIDEO_MEMORY_INFO* out) {
    if (!d3d_device || !out) {
        return false;
    }
    ComPtr<IDXGIDevice> dxgi_device;
    if (FAILED(d3d_device->QueryInterface(IID_PPV_ARGS(&dxgi_device)))) {
        return false;
    }
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgi_device->GetAdapter(&adapter))) {
        return false;
    }
    ComPtr<IDXGIAdapter3> adapter3;
    if (FAILED(adapter->QueryInterface(IID_PPV_ARGS(&adapter3)))) {
        return false;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        return false;
    }
    *out = info;
    return true;
}

u64 TextureCacheRuntime::GetDeviceLocalMemory() const {
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (QueryLocalVideoMemoryInfo(device.GetDevice(), &info)) {
        return info.Budget;
    }
    // Fallback budget when the DXGI memory query is unavailable (Xbox). The shared texture
    // cache derives its GC watermarks from this value: 384 MiB makes the cache evict under
    // pressure (normal GC above 0, aggressive at >= 384 MiB) so host commit stays inside
    // the UWP ceiling instead of growing without bound.
    return 384ull * 1024ull * 1024ull;
}

u64 TextureCacheRuntime::GetDeviceMemoryUsage() const {
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (QueryLocalVideoMemoryInfo(device.GetDevice(), &info)) {
        return info.CurrentUsage;
    }
    return 0;
}

bool TextureCacheRuntime::CanReportMemoryUsage() const {
    // Xbox does not implement IDXGIAdapter3::QueryVideoMemoryInfo (it always fails), so the
    // shared texture-cache GC cannot use the DXGI budget. Report false so the GC falls back
    // to its own accounted usage (total_used_memory) for memory pressure.
    return false;
}

bool TextureCacheRuntime::IsDepthStencilFormat(VideoCore::Surface::PixelFormat format) {
    return DepthStencilFormat(format) != DXGI_FORMAT_UNKNOWN;
}

bool TextureCacheRuntime::IsRepresentableRenderTarget(VideoCore::Surface::PixelFormat format) {
    if (DepthStencilFormat(format) != DXGI_FORMAT_UNKNOWN) {
        return false;
    }
    return RenderTargetFormat(format) != DXGI_FORMAT_UNKNOWN;
}

void TextureCacheRuntime::TransitionImageLayout(Image& image) {
    // Images live in COMMON between operations (same excursion model as buffers), so
    // sampled/read use from COMMON is legal and there is nothing to transition here.
    (void)image;
}

bool TextureCacheRuntime::CanImageBeCopied(Image& dst, Image& src) {
    if (!dst.Handle() || !src.Handle()) {
        return false;
    }
    const D3D12_RESOURCE_DESC dst_desc = dst.Handle()->GetDesc();
    const D3D12_RESOURCE_DESC src_desc = src.Handle()->GetDesc();
    return dst_desc.Format == src_desc.Format &&
           dst_desc.SampleDesc.Count == src_desc.SampleDesc.Count &&
           dst_desc.Dimension == src_desc.Dimension;
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src,
                                    std::span<const ImageCopy> copies) {
    if (!command_list.IsValid() || !CanImageBeCopied(dst, src)) {
        return;
    }
    Transition(dst.Handle(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    Transition(src.Handle(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const D3D12_RESOURCE_DESC src_desc = src.Handle()->GetDesc();
    for (const ImageCopy& copy : copies) {
        const u32 src_sub = CalcSubresource(
            static_cast<u32>(copy.src_subresource.base_level),
            static_cast<u32>(copy.src_subresource.base_layer), 0, src_desc.MipLevels,
            src_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                ? 1
                : static_cast<u32>(src_desc.DepthOrArraySize));
        D3D12_TEXTURE_COPY_LOCATION src_loc{};
        src_loc.pResource = src.Handle();
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src_loc.SubresourceIndex = src_sub;
        const D3D12_RESOURCE_DESC dst_desc = dst.Handle()->GetDesc();
        const u32 dst_sub = CalcSubresource(
            static_cast<u32>(copy.dst_subresource.base_level),
            static_cast<u32>(copy.dst_subresource.base_layer), 0, dst_desc.MipLevels,
            dst_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                ? 1
                : static_cast<u32>(dst_desc.DepthOrArraySize));
        D3D12_TEXTURE_COPY_LOCATION dst_loc{};
        dst_loc.pResource = dst.Handle();
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst_loc.SubresourceIndex = dst_sub;
        D3D12_BOX box{};
        box.left = static_cast<UINT>(copy.src_offset.x);
        box.top = static_cast<UINT>(copy.src_offset.y);
        box.front = static_cast<UINT>(copy.src_offset.z);
        box.right = box.left + copy.extent.width;
        box.bottom = box.top + copy.extent.height;
        box.back = box.front + (copy.extent.depth == 0 ? 1 : copy.extent.depth);
        command_list.Get()->CopyTextureRegion(&dst_loc, copy.dst_offset.x, copy.dst_offset.y,
                                              copy.dst_offset.z, &src_loc, &box);
    }
    Transition(dst.Handle(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    Transition(src.Handle(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

void TextureCacheRuntime::EmulateCopyImage(Image&, Image&,
                                            std::span<const ImageCopy>) {
    if (!logged_emulated_copy) {
        logged_emulated_copy = true;
        LOG_ERROR(Render_D3D12, "Emulated image copies not implemented yet");
    }
}

bool TextureCacheRuntime::ShouldReinterpret(Image& dst, Image& src) {
    return dst.Handle() && src.Handle() && dst.guest_size_bytes == src.guest_size_bytes &&
           dst.info.format != src.info.format;
}

void TextureCacheRuntime::ReinterpretImage(Image& dst, Image& src,
                                           std::span<const ImageCopy> copies) {
    if (!command_list.IsValid() || !dst.Handle() || !src.Handle()) {
        return;
    }
    ID3D12Resource* const dst_resource = dst.Handle();
    ID3D12Resource* const src_resource = src.Handle();
    if (dst_resource == src_resource) {
        return;
    }
    const D3D12_RESOURCE_DESC src_desc = src_resource->GetDesc();
    const D3D12_RESOURCE_DESC dst_desc = dst_resource->GetDesc();
    // D3D12's CopyTextureRegion cannot convert between different resource formats; the
    // cross-family raw copy is an invalid call on Xbox (device removal). The guest
    // reinterprets the same memory with a different format, which our typeless resources
    // cannot express directly, so skip the copy until a buffer-roundtrip path exists.
    if (src_desc.Format != dst_desc.Format) {
        static std::atomic<u32> g_reinterpret_skip{0};
        if (g_reinterpret_skip.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_ERROR(Render_D3D12,
                      "ReinterpretImage fmt mismatch src={:#x} dst={:#x} copies={}: copy "
                      "skipped",
                      static_cast<u32>(src_desc.Format), static_cast<u32>(dst_desc.Format),
                      copies.size());
        }
        return;
    }
    Transition(src_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(dst_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    const u32 src_array_size =
        src_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1
            : static_cast<u32>(src_desc.DepthOrArraySize);
    const u32 dst_array_size =
        dst_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1
            : static_cast<u32>(dst_desc.DepthOrArraySize);
    for (const ImageCopy& copy : copies) {
        if (copy.extent.width == 0 || copy.extent.height == 0 || copy.extent.depth == 0) {
            continue;
        }
        D3D12_TEXTURE_COPY_LOCATION src_loc{};
        src_loc.pResource = src_resource;
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src_loc.SubresourceIndex = CalcSubresource(
            static_cast<u32>(copy.src_subresource.base_level),
            static_cast<u32>(copy.src_subresource.base_layer), 0, src_desc.MipLevels,
            src_array_size);
        D3D12_TEXTURE_COPY_LOCATION dst_loc{};
        dst_loc.pResource = dst_resource;
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst_loc.SubresourceIndex = CalcSubresource(
            static_cast<u32>(copy.dst_subresource.base_level),
            static_cast<u32>(copy.dst_subresource.base_layer), 0, dst_desc.MipLevels,
            dst_array_size);
        D3D12_BOX box{};
        box.left = static_cast<UINT>(copy.src_offset.x);
        box.top = static_cast<UINT>(copy.src_offset.y);
        box.front = 0;
        box.right = box.left + copy.extent.width;
        box.bottom = box.top + copy.extent.height;
        box.back = 1;
        // 2D-only path per the template (UNIMPLEMENTED_IF on non-e2D types).
        command_list.Get()->CopyTextureRegion(&dst_loc, static_cast<UINT>(copy.dst_offset.x),
                                              static_cast<UINT>(copy.dst_offset.y), 0, &src_loc,
                                              &box);
    }
    Transition(dst_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    Transition(src_resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

void TextureCacheRuntime::CopyImageMSAA(Image&, Image&, std::span<const ImageCopy>) {
    if (!logged_msaa) {
        logged_msaa = true;
        LOG_ERROR(Render_D3D12, "MSAA image copies not implemented yet");
    }
}

Microsoft::WRL::ComPtr<ID3D12Resource> g_probe_blit_src;
Microsoft::WRL::ComPtr<ID3D12Resource> g_probe_blit_dst;
Microsoft::WRL::ComPtr<ID3D12Resource> g_probe_scene;
Microsoft::WRL::ComPtr<ID3D12Resource> g_probe_sampled;
Microsoft::WRL::ComPtr<ID3D12Resource> g_probe_any_sampled;

bool TextureCacheRuntime::BlitImage(Framebuffer* dst_framebuffer, ImageView& dst_view,
                                    ImageView& src_view,
                                    const VideoCommon::Region2D& dst_region,
                                    const VideoCommon::Region2D& src_region,
                                    Tegra::Engines::Fermi2D::Filter filter,
                                    Tegra::Engines::Fermi2D::Operation operation) {
    (void)dst_framebuffer;
    (void)filter;
    ID3D12Resource* const dst_resource = dst_view.Resource();
    ID3D12Resource* const src_resource = src_view.Resource();
    if (!command_list.IsValid() || !dst_resource || !src_resource ||
        dst_resource == src_resource) {
        return false;
    }
    // Mirror Vulkan's dispatch: only SrcCopy is accelerated, ROP/blend ops need the CPU path.
    if (operation != Tegra::Engines::Fermi2D::Operation::SrcCopy) {
        if (!logged_blit) {
            logged_blit = true;
            LOG_DEBUG(Render_D3D12, "Skipping non-SrcCopy image blit (op={})",
                      static_cast<u32>(operation));
        }
        return false;
    }
    const D3D12_RESOURCE_DESC src_desc = src_resource->GetDesc();
    const D3D12_RESOURCE_DESC dst_desc = dst_resource->GetDesc();
    // Raw byte copies need identical typeless resource formats. This is also the
    // aspect-compatibility check: depth/stencil resources never match color ones, and true
    // format conversions need a shader draw (see ConvertImage), so never attempt them here.
    if (src_desc.Format != dst_desc.Format ||
        src_desc.Dimension != dst_desc.Dimension ||
        src_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
        if (!logged_blit) {
            logged_blit = true;
            LOG_DEBUG(Render_D3D12, "Skipping incompatible image blit (format {} -> {})",
                      static_cast<u32>(src_desc.Format), static_cast<u32>(dst_desc.Format));
        }
        return false;
    }
    const s32 src_width = src_region.end.x - src_region.start.x;
    const s32 src_height = src_region.end.y - src_region.start.y;
    const s32 dst_width = dst_region.end.x - dst_region.start.x;
    const s32 dst_height = dst_region.end.y - dst_region.start.y;
    // CopyTextureRegion cannot stretch: scaled blits stay on the CPU fallback.
    // (A 1:1 bilinear blit samples texel centers, so it is an exact copy.)
    if (src_width <= 0 || src_height <= 0 || src_width != dst_width || src_height != dst_height) {
        if (!logged_blit) {
            logged_blit = true;
            LOG_DEBUG(Render_D3D12, "Skipping scaled image blit ({}x{} -> {}x{})", src_width,
                      src_height, dst_width, dst_height);
        }
        return false;
    }
    // Never record an out-of-bounds copy.
    if (src_region.start.x < 0 || src_region.start.y < 0 ||
        src_region.end.x > static_cast<s32>(src_view.size.width) ||
        src_region.end.y > static_cast<s32>(src_view.size.height) ||
        dst_region.start.x < 0 || dst_region.start.y < 0 ||
        dst_region.end.x > static_cast<s32>(dst_view.size.width) ||
        dst_region.end.y > static_cast<s32>(dst_view.size.height)) {
        if (!logged_blit) {
            logged_blit = true;
            LOG_DEBUG(Render_D3D12, "Skipping out-of-bounds image blit");
        }
        return false;
    }
    const u32 src_samples = src_desc.SampleDesc.Count;
    const u32 dst_samples = dst_desc.SampleDesc.Count;
    const bool is_resolve = src_samples > 1 && dst_samples == 1;
    if (src_samples != dst_samples && !is_resolve) {
        if (!logged_blit) {
            logged_blit = true;
            LOG_DEBUG(Render_D3D12, "Skipping MSAA image blit ({}x -> {}x)", src_samples,
                      dst_samples);
        }
        return false;
    }
    if (is_resolve) {
        // ResolveSubresource covers whole subresources, so it is only valid for full-view blits.
        const bool is_full_view =
            src_region.start.x == 0 && src_region.start.y == 0 &&
            src_region.end.x == static_cast<s32>(src_view.size.width) &&
            src_region.end.y == static_cast<s32>(src_view.size.height) &&
            dst_region.start.x == 0 && dst_region.start.y == 0 &&
            dst_region.end.x == static_cast<s32>(dst_view.size.width) &&
            dst_region.end.y == static_cast<s32>(dst_view.size.height);
        if (!is_full_view) {
            if (!logged_blit) {
                logged_blit = true;
                LOG_DEBUG(Render_D3D12, "Skipping sub-rectangle MSAA resolve blit");
            }
            return false;
        }
    }
    const u32 src_array_size =
        src_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1
            : static_cast<u32>(src_desc.DepthOrArraySize);
    const u32 dst_array_size =
        dst_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1
            : static_cast<u32>(dst_desc.DepthOrArraySize);
    Transition(src_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(dst_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    const s32 levels = std::min(src_view.range.extent.levels, dst_view.range.extent.levels);
    const s32 layers = std::min(src_view.range.extent.layers, dst_view.range.extent.layers);
    for (s32 level = 0; level < levels; ++level) {
        for (s32 layer = 0; layer < layers; ++layer) {
            const u32 src_sub = CalcSubresource(
                static_cast<u32>(src_view.range.base.level + level),
                static_cast<u32>(src_view.range.base.layer + layer), 0, src_desc.MipLevels,
                src_array_size);
            const u32 dst_sub = CalcSubresource(
                static_cast<u32>(dst_view.range.base.level + level),
                static_cast<u32>(dst_view.range.base.layer + layer), 0, dst_desc.MipLevels,
                dst_array_size);
            if (is_resolve) {
                command_list.Get()->ResolveSubresource(dst_resource, dst_sub, src_resource,
                                                       src_sub, dst_desc.Format);
                continue;
            }
            D3D12_TEXTURE_COPY_LOCATION src_loc{};
            src_loc.pResource = src_resource;
            src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src_loc.SubresourceIndex = src_sub;
            D3D12_TEXTURE_COPY_LOCATION dst_loc{};
            dst_loc.pResource = dst_resource;
            dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst_loc.SubresourceIndex = dst_sub;
            D3D12_BOX box{};
            box.left = static_cast<UINT>(src_region.start.x);
            box.top = static_cast<UINT>(src_region.start.y);
            box.front = 0;
            box.right = static_cast<UINT>(src_region.end.x);
            box.bottom = static_cast<UINT>(src_region.end.y);
            box.back = 1;
            command_list.Get()->CopyTextureRegion(&dst_loc,
                                                  static_cast<UINT>(dst_region.start.x),
                                                  static_cast<UINT>(dst_region.start.y), 0,
                                                  &src_loc, &box);
        }
    }
    Transition(dst_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    Transition(src_resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    g_probe_blit_src = src_resource;
    g_probe_blit_dst = dst_resource;
    return true;
}

void TextureCacheRuntime::ConvertImage(Framebuffer*, ImageView& dst_view,
                                       ImageView& src_view) {
    ID3D12Resource* const dst_resource = dst_view.Resource();
    ID3D12Resource* const src_resource = src_view.Resource();
    if (!command_list.IsValid() || !dst_resource || !src_resource ||
        dst_resource == src_resource) {
        return;
    }
    if (src_resource->GetDesc().Format != dst_resource->GetDesc().Format) {
        if (!logged_convert) {
            logged_convert = true;
            LOG_ERROR(Render_D3D12, "Image format conversion not implemented yet");
        }
        return;
    }
    const u32 width = std::min(dst_view.size.width, src_view.size.width);
    const u32 height = std::min(dst_view.size.height, src_view.size.height);
    if (width == 0 || height == 0) {
        return;
    }
    Transition(src_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(dst_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    const D3D12_RESOURCE_DESC src_desc = src_resource->GetDesc();
    const D3D12_RESOURCE_DESC dst_desc = dst_resource->GetDesc();
    const u32 src_array_size =
        src_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1
            : static_cast<u32>(src_desc.DepthOrArraySize);
    const u32 dst_array_size =
        dst_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1
            : static_cast<u32>(dst_desc.DepthOrArraySize);
    D3D12_BOX box{};
    box.left = 0;
    box.top = 0;
    box.front = 0;
    box.right = width;
    box.bottom = height;
    box.back = 1;
    for (s32 level = 0; level < src_view.range.extent.levels; ++level) {
        for (s32 layer = 0; layer < src_view.range.extent.layers; ++layer) {
            D3D12_TEXTURE_COPY_LOCATION src_loc{};
            src_loc.pResource = src_resource;
            src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src_loc.SubresourceIndex = CalcSubresource(
                static_cast<u32>(src_view.range.base.level + level),
                static_cast<u32>(src_view.range.base.layer + layer), 0, src_desc.MipLevels,
                src_array_size);
            D3D12_TEXTURE_COPY_LOCATION dst_loc{};
            dst_loc.pResource = dst_resource;
            dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst_loc.SubresourceIndex = CalcSubresource(
                static_cast<u32>(dst_view.range.base.level + level),
                static_cast<u32>(dst_view.range.base.layer + layer), 0, dst_desc.MipLevels,
                dst_array_size);
            command_list.Get()->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, &box);
        }
    }
    Transition(dst_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    Transition(src_resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

std::span<const DXGI_FORMAT> TextureCacheRuntime::ViewFormats(PixelFormat format) {
    const size_t index = static_cast<size_t>(format);
    if (index < view_formats.size() && view_formats[index].empty()) {
        const DXGI_FORMAT view = SampledFormat(format);
        if (view != DXGI_FORMAT_UNKNOWN) {
            view_formats[index].push_back(view);
        }
    }
    if (index < view_formats.size()) {
        return view_formats[index];
    }
    return {};
}

DXGI_FORMAT TextureCacheRuntime::ResourceFormat(PixelFormat format) const {
    const DXGI_FORMAT resource = ::D3D12::ResourceFormat(format);
    if (resource != DXGI_FORMAT_UNKNOWN) {
        return resource;
    }
    LOG_CRITICAL(Render_D3D12, "Unsupported guest pixel format {}", static_cast<u32>(format));
    return DXGI_FORMAT_R8G8B8A8_TYPELESS;
}

ComPtr<ID3D12Resource> TextureCacheRuntime::CreateImageResource(
    const VideoCommon::ImageInfo& info) {
    const DXGI_FORMAT format = ResourceFormat(info.format);
    D3D12_RESOURCE_DIMENSION dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    u32 array_size = 1;
    u32 depth = info.size.depth == 0 ? 1 : info.size.depth;
    switch (info.type) {
    case ImageType::e1D:
        dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        array_size = static_cast<u32>(info.resources.layers);
        break;
    case ImageType::e2D:
    case ImageType::Linear:
        dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        array_size = static_cast<u32>(info.resources.layers);
        break;
    case ImageType::e3D:
        dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        array_size = 1;
        break;
    default:
        break;
    }
    if (array_size == 0) {
        array_size = 1;
    }

    // Block-compressed formats cannot be bound as render targets or UAVs;
    // they stay plain sample-only resources (no RTV/UAV flags possible).
    const bool is_compressed =
        info.format == PixelFormat::BC1_RGBA_UNORM ||
        info.format == PixelFormat::BC1_RGBA_SRGB ||
        info.format == PixelFormat::BC2_UNORM ||
        info.format == PixelFormat::BC2_SRGB ||
        info.format == PixelFormat::BC3_UNORM ||
        info.format == PixelFormat::BC3_SRGB ||
        info.format == PixelFormat::BC4_UNORM ||
        info.format == PixelFormat::BC4_SNORM ||
        info.format == PixelFormat::BC5_UNORM ||
        info.format == PixelFormat::BC5_SNORM ||
        info.format == PixelFormat::BC6H_UFLOAT ||
        info.format == PixelFormat::BC6H_SFLOAT ||
        info.format == PixelFormat::BC7_UNORM ||
        info.format == PixelFormat::BC7_SRGB;

    const bool depth_stencil = IsDepthStencilFormat(info.format);
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
    if (depth_stencil) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    } else if (!is_compressed) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        // R11G11B10_FLOAT does NOT support UAV views: a UAV flag on such a
        // resource makes the (void, unchecked) CreateUnorderedAccessView call
        // fail and leaves a garbage descriptor that removes the device once
        // bound. Allow UAV only on formats with known UAV support, keyed on
        // the resolved typeless resource format.
        if (info.num_samples <= 1) {
            switch (format) {
            case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            case DXGI_FORMAT_R16G16_TYPELESS:
            case DXGI_FORMAT_R32G32_TYPELESS:
            case DXGI_FORMAT_R32_TYPELESS:
            case DXGI_FORMAT_R16_TYPELESS:
            case DXGI_FORMAT_R8G8_TYPELESS:
            case DXGI_FORMAT_R8_TYPELESS:
                flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                break;
            default:
                break;
            }
        }
    }

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dimension;
    desc.Width = std::max<u64>(info.size.width, 1);
    desc.Height = std::max<u32>(info.size.height, 1);
    desc.DepthOrArraySize = dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                ? static_cast<UINT16>(depth)
                                : static_cast<UINT16>(array_size);
    desc.MipLevels = static_cast<UINT16>(std::max<s32>(info.resources.levels, 1));
    desc.Format = format;
    desc.SampleDesc.Count =
        info.num_samples > 1 ? info.num_samples : 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;

    D3D12_CLEAR_VALUE clear{};
    const D3D12_CLEAR_VALUE* clear_value = nullptr;
    if (depth_stencil) {
        clear.Format = DepthStencilFormat(info.format);
        clear.DepthStencil.Depth = 1.0f;
        clear.DepthStencil.Stencil = 0;
        clear_value = &clear;
    } else if ((flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0 &&
               SampledFormat(info.format) != DXGI_FORMAT_UNKNOWN) {
        // CreateCommittedResource rejects typeless clear formats
        // (E_INVALIDARG), so only clear when the sampled format is
        // fully typed; otherwise leave clear_value nullptr.
        clear.Format = SampledFormat(info.format);
        clear_value = &clear;
    }

    ComPtr<ID3D12Resource> resource;
    const HRESULT create_hr = device.GetDevice()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, clear_value,
            IID_PPV_ARGS(&resource));
    if (FAILED(create_hr)) {
        LOG_CRITICAL(Render_D3D12,
                     "Image creation failed ({}x{} fmt={} mips={} array={} "
                     "samples={} flags={:#x} hr={:#x})",
                     desc.Width, desc.Height, static_cast<u32>(format), desc.MipLevels,
                     desc.DepthOrArraySize, desc.SampleDesc.Count,
                     static_cast<u32>(desc.Flags), static_cast<u32>(create_hr));
        return nullptr;
    }
    return resource;
}

void TextureCacheRuntime::Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                     D3D12_RESOURCE_STATES after) {
    if (command_list.IsValid() && resource) {
        command_list.Transition(resource, before, after);
    }
}

} // namespace D3D12