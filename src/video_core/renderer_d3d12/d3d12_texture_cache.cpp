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
    case PixelFormat::BC7_UNORM:
    case PixelFormat::BC7_SRGB:
        return DXGI_FORMAT_BC7_TYPELESS;
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
    case PixelFormat::BC7_UNORM:
        return DXGI_FORMAT_BC7_UNORM;
    case PixelFormat::BC7_SRGB:
        return DXGI_FORMAT_BC7_UNORM_SRGB;
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

bool IsDepthStencilFormat(PixelFormat format) {
    return DepthStencilFormat(format) != DXGI_FORMAT_UNKNOWN;
}

bool IsCompressedFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::BC1_RGBA_UNORM:
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
        return 8;
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
    const u32 index = runtime->view_heap.Allocate();
    if (index == DescriptorHeap::INVALID_INDEX) {
        return D3D12_CPU_DESCRIPTOR_HANDLE{0};
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE handle = runtime->view_heap.CpuHandle(index);
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
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
    if (!resource || info.num_samples > 1 || !runtime->command_list.IsValid()) {
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
    if (!resource || info.num_samples > 1 || !runtime->command_list.IsValid()) {
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

void Image::UploadSubresource(ID3D12Resource* src_buffer, std::span<const u8> staging_span,
                              u64 src_offset, const BufferImageCopy& copy) {
    if (!src_buffer) {
        return;
    }
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    const u32 mip = static_cast<u32>(copy.image_subresource.base_level);
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
    const u64 src_pitch =
        copy.buffer_row_length != 0
            ? static_cast<u64>(copy.buffer_row_length) * (compressed ? elem_bytes : elem_bytes)
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
    src.PlacedFootprint.Footprint.Width = copy.image_extent.width;
    src.PlacedFootprint.Footprint.Height = copy.image_extent.height;
    src.PlacedFootprint.Footprint.Depth = copy.image_extent.depth;
    src.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(copy_src_pitch);

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
        box.right = copy.image_extent.width;
        box.bottom = copy.image_extent.height;
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
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    const u32 mip = static_cast<u32>(copy.image_subresource.base_level);
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
    gpu_dst.PlacedFootprint.Footprint.Width = copy.image_extent.width;
    gpu_dst.PlacedFootprint.Footprint.Height = copy.image_extent.height;
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
        box.right = box.left + copy.image_extent.width;
        box.bottom = box.top + copy.image_extent.height;
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

    if (desc.SampleDesc.Count == 1 && !image.IsDepthStencil()) {
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

Framebuffer::Framebuffer(TextureCacheRuntime&, std::array<ImageView*, VideoCommon::NUM_RT>,
                         ImageView*, const VideoCommon::RenderTargets&) {}

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

u64 TextureCacheRuntime::GetDeviceLocalMemory() const {
    return 0;
}

u64 TextureCacheRuntime::GetDeviceMemoryUsage() const {
    return 0;
}

bool TextureCacheRuntime::CanReportMemoryUsage() const {
    return false;
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

void TextureCacheRuntime::ReinterpretImage(Image&, Image&,
                                           std::span<const ImageCopy>) {
    if (!logged_reinterpret) {
        logged_reinterpret = true;
        LOG_ERROR(Render_D3D12, "Reinterpret image copies not implemented yet");
    }
}

void TextureCacheRuntime::CopyImageMSAA(Image&, Image&, std::span<const ImageCopy>) {
    if (!logged_msaa) {
        logged_msaa = true;
        LOG_ERROR(Render_D3D12, "MSAA image copies not implemented yet");
    }
}

void TextureCacheRuntime::BlitImage(Framebuffer*, ImageView& dst_view, ImageView& src_view,
                                    const VideoCommon::Region2D& dst_region,
                                    const VideoCommon::Region2D& src_region,
                                    Tegra::Engines::Fermi2D::Filter filter,
                                    Tegra::Engines::Fermi2D::Operation operation) {
    // Only plain 1:1 same-format SrcCopy blits are handled for now; scaled/filtered and
    // format-converting blits belong to the framebuffer blit step.
    if (!logged_blit) {
        logged_blit = true;
        LOG_ERROR(Render_D3D12, "General image blits not implemented yet (filter={}, op={})",
                  static_cast<u32>(filter), static_cast<u32>(operation));
    }
    (void)dst_view;
    (void)src_view;
    (void)dst_region;
    (void)src_region;
}

void TextureCacheRuntime::ConvertImage(Framebuffer*, ImageView&, ImageView&) {
    if (!logged_convert) {
        logged_convert = true;
        LOG_ERROR(Render_D3D12, "Image format conversion not implemented yet");
    }
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

    const bool depth_stencil = IsDepthStencilFormat(info.format);
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
    if (depth_stencil) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    } else {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (info.num_samples <= 1) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
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
    } else if ((flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0) {
        clear.Format = SampledFormat(info.format) != DXGI_FORMAT_UNKNOWN
                           ? SampledFormat(info.format)
                           : format;
        clear_value = &clear;
    }

    ComPtr<ID3D12Resource> resource;
    if (FAILED(device.GetDevice()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, clear_value,
            IID_PPV_ARGS(&resource)))) {
        LOG_CRITICAL(Render_D3D12, "Image creation failed ({}x{} fmt={})", desc.Width, desc.Height,
                     static_cast<u32>(format));
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