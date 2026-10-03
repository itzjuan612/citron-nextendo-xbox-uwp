// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <array>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/renderer_d3d12/d3d12_descriptor_heap.h"
#include "video_core/renderer_d3d12/d3d12_staging_buffer_pool.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"

namespace D3D12 {

class CommandList;
class Device;
class Framebuffer;
class Image;
class ImageView;
class Sampler;
class TextureCacheRuntime;

using Microsoft::WRL::ComPtr;
using VideoCommon::ImageId;

// Guest image backed by a committed D3D12 texture resource.
//
// Resources are created in the typeless variant of the guest format so subresource views
// can reinterpret the format when the template asks for it. The home state is COMMON;
// copies transition in and out explicitly. Sampled/rt/depth views can read from COMMON
// through D3D12's read-state promotion rules.
class Image : public VideoCommon::ImageBase {
    friend ImageView;

public:    explicit Image(TextureCacheRuntime& runtime, const VideoCommon::ImageInfo& info,
                   GPUVAddr gpu_addr, VAddr cpu_addr);
    explicit Image(const VideoCommon::NullImageParams&);

    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    void UploadMemory(ID3D12Resource* buffer, u64 offset,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void UploadMemory(const StagingBufferRef& map,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(ID3D12Resource* buffer, size_t offset,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(std::span<ID3D12Resource*> buffers, std::span<size_t> offsets,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(const StagingBufferRef& map,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    [[nodiscard]] ID3D12Resource* Handle() const noexcept {
        return resource.Get();
    }

    [[nodiscard]] bool ExchangeInitialization() noexcept {
        return std::exchange(initialized, true);
    }

    /// UAV for image-store access to a mip level (created on first use).
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE StorageImageView(s32 level);

    [[nodiscard]] bool IsRescaled() const noexcept {
        return false;
    }

    [[nodiscard]] bool ScaleUp(bool ignore = false) {
        return false;
    }

    [[nodiscard]] bool ScaleDown(bool ignore = false) {
        return false;
    }

    [[nodiscard]] bool IsDepthStencil() const noexcept {
        return is_depth_stencil;
    }

    [[nodiscard]] u32 ArraySize() const noexcept {
        return array_size;
    }

private:
    struct StorageView {
        s32 level;
        D3D12_CPU_DESCRIPTOR_HANDLE handle;
    };

    void UploadSubresource(ID3D12Resource* src_buffer, std::span<const u8> staging_span,
                           u64 src_offset, const VideoCommon::BufferImageCopy& copy);
    void DownloadSubresource(ID3D12Resource* dst_buffer, std::span<u8> staging_span,
                             u64 dst_offset, const VideoCommon::BufferImageCopy& copy);

    static constexpr s32 MAX_STORAGE_VIEWS = 16;

    Device* device{};
    TextureCacheRuntime* runtime{};
    ComPtr<ID3D12Resource> resource;
    std::vector<StorageView> storage_views;
    u32 array_size{1};
    bool initialized{};
    bool is_depth_stencil{};
};

/// View over a cached image, materialized as CPU descriptors in the runtime's heaps.
/// The draw translation copies these into shader-visible descriptor tables.
class ImageView : public VideoCommon::ImageViewBase {
public:
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&,
                       const Common::SlotVector<Image>&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo&,
                       const VideoCommon::ImageViewInfo&, GPUVAddr);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::NullImageViewParams&);

    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    /// Sampled (SRV) view.
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE Sampled() const noexcept {
        return sampled_view;
    }

    /// Render-target (RTV) or depth (DSV) view.
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE RenderTarget() const noexcept {
        return rt_view;
    }

    /// Storage (UAV) view for image stores.
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE Storage() const noexcept {
        return storage_view;
    }

    [[nodiscard]] bool IsRescaled() const noexcept {
        return false;
    }

    [[nodiscard]] u32 BufferSize() const noexcept {
        return 0;
    }

    [[nodiscard]] ID3D12Resource* Resource() const noexcept {
        return resource;
    }

private:
    void CreateViews(Image& image, const VideoCommon::ImageViewInfo& view_info);

    ID3D12Resource* resource{};
    D3D12_CPU_DESCRIPTOR_HANDLE sampled_view{};
    D3D12_CPU_DESCRIPTOR_HANDLE storage_view{};
    D3D12_CPU_DESCRIPTOR_HANDLE rt_view{};
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

/// Guest sampler state translated to a D3D12 sampler description. The draw translation
/// materializes this into a sampler-heap descriptor when flushing descriptor tables.
class Sampler {
public:
    explicit Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry& config);

    [[nodiscard]] const D3D12_SAMPLER_DESC& Desc() const noexcept {
        return desc;
    }

private:
    D3D12_SAMPLER_DESC desc{};
};

/// D3D12 has no framebuffer objects. This class exists because the template owns a
/// framebuffer slot vector and threads it through the blit entry points; the actual
/// render-target state is consumed by the draw translation (framebuffer blit step).
class Framebuffer {
public:
    explicit Framebuffer(TextureCacheRuntime& runtime,
                         std::array<ImageView*, VideoCommon::NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key);

    [[nodiscard]] const std::array<ImageView*, VideoCommon::NUM_RT>& ColorBuffers() const noexcept {
        return color_buffers;
    }

    [[nodiscard]] ImageView* DepthBuffer() const noexcept {
        return depth_buffer;
    }

private:
    std::array<ImageView*, VideoCommon::NUM_RT> color_buffers{};
    ImageView* depth_buffer{};
};

class TextureCacheRuntime {
    friend Image;
    friend ImageView;

public:
    explicit TextureCacheRuntime(Device& device, CommandList& command_list,
                                 StagingBufferPool& staging_pool);

    void Finish();

    StagingBufferRef UploadStagingBuffer(size_t size);

    StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);

    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void TickFrame();

    u64 GetDeviceLocalMemory() const;

    u64 GetDeviceMemoryUsage() const;

    bool CanReportMemoryUsage() const;

    bool HasBrokenTextureViewFormats() const noexcept {
        return false;
    }

    bool HasNativeBgr() const noexcept {
        return true;
    }

    bool CanUploadMSAA() const noexcept {
        return false;
    }

    bool CanAccelerateImageUpload(Image&) const noexcept {
        return false;
    }

    void AccelerateImageUpload(Image&, const StagingBufferRef&,
                               std::span<const VideoCommon::SwizzleParameters>) {}

    void InsertUploadMemoryBarrier() {}

    void TransitionImageLayout(Image& image);

    void BarrierFeedbackLoop() {}

    bool CanImageBeCopied(Image& dst, Image& src);

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void EmulateCopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    bool ShouldReinterpret(Image& dst, Image& src);

    void ReinterpretImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void CopyImageMSAA(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void BlitImage(Framebuffer* dst_framebuffer, ImageView& dst_view, ImageView& src_view,
                   const VideoCommon::Region2D& dst_region, const VideoCommon::Region2D& src_region,
                   Tegra::Engines::Fermi2D::Filter filter,
                   Tegra::Engines::Fermi2D::Operation operation);

    void ConvertImage(Framebuffer* dst_framebuffer, ImageView& dst_view, ImageView& src_view);

    std::span<const DXGI_FORMAT> ViewFormats(VideoCore::Surface::PixelFormat format);

private:
    DXGI_FORMAT ResourceFormat(VideoCore::Surface::PixelFormat format) const;
    ComPtr<ID3D12Resource> CreateImageResource(const VideoCommon::ImageInfo& info);
    void Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after);

    Device& device;
    CommandList& command_list;
    StagingBufferPool& staging_pool;
    DescriptorHeap view_heap;
    DescriptorHeap rtv_heap;
    DescriptorHeap dsv_heap;

    std::array<std::vector<DXGI_FORMAT>, VideoCore::Surface::MaxPixelFormat> view_formats;

    u64 tick{};
    bool logged_blit{};
    bool logged_convert{};
    bool logged_msaa{};
    bool logged_reinterpret{};
    bool logged_emulated_copy{};
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;
    static constexpr bool HAS_EMULATED_COPIES = false;
    static constexpr bool HAS_DEVICE_MEMORY_INFO = true;
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = true;

    using Runtime = D3D12::TextureCacheRuntime;
    using Image = D3D12::Image;
    using ImageAlloc = D3D12::ImageAlloc;
    using ImageView = D3D12::ImageView;
    using Sampler = D3D12::Sampler;
    using Framebuffer = D3D12::Framebuffer;
    using AsyncBuffer = D3D12::StagingBufferRef;
    using BufferType = ID3D12Resource*;
};

using TextureCache = VideoCommon::TextureCache<TextureCacheParams>;

} // namespace D3D12
