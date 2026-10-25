// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <unordered_set>
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/settings.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::GetFormatType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceType;
using namespace Common::Literals;

template <class P>
TextureCache<P>::TextureCache(Runtime& runtime_, Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : runtime{runtime_}, device_memory{device_memory_} {
    // Configure null sampler
    TSCEntry sampler_descriptor{};
    sampler_descriptor.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    sampler_descriptor.cubemap_anisotropy.Assign(1);

    // These values were chosen based on typical peak swizzle data sizes seen in some titles
    static constexpr size_t SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 8_MiB;
    static constexpr size_t UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 1_MiB;
    swizzle_data_buffer.resize_destructive(SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);
    unswizzle_data_buffer.resize_destructive(UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);

    // Make sure the first index is reserved for the null resources
    // This way the null resource becomes a compile time constant
    void(slot_images.insert(NullImageParams{}));
    void(slot_image_views.insert(runtime, NullImageViewParams{}));
    void(slot_samplers.insert(runtime, sampler_descriptor));

    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
        const s64 min_spacing_expected = device_local_memory - 1_GiB;
        const s64 min_spacing_critical = device_local_memory - 512_MiB;
        const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
        const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
        const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
        expected_memory = static_cast<u64>(
            (std::min)(device_local_memory,
                       (std::max)((std::min)(device_local_memory - min_vacancy_expected,
                                             min_spacing_expected),
                                  DEFAULT_EXPECTED_MEMORY)));
        critical_memory = static_cast<u64>(
            (std::min)(device_local_memory,
                       (std::max)((std::min)(device_local_memory - min_vacancy_critical,
                                             min_spacing_critical),
                                  DEFAULT_CRITICAL_MEMORY)));
        minimum_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    } else {
        expected_memory = DEFAULT_EXPECTED_MEMORY + 512_MiB;
        critical_memory = DEFAULT_CRITICAL_MEMORY + 1_GiB;
        minimum_memory = 0;
    }
}

template <class P>
void TextureCache<P>::RunGarbageCollector() {
    bool high_priority_mode = false;
    bool aggressive_mode = false;
    u64 ticks_to_destroy = 0;
    size_t num_iterations = 0;
    size_t num_inspected = 0;

    const auto Configure = [&](bool allow_aggressive) {
        num_inspected = 0;
        high_priority_mode = total_used_memory >= expected_memory;
        aggressive_mode = allow_aggressive && total_used_memory >= critical_memory;
        ticks_to_destroy = aggressive_mode ? 10ULL : high_priority_mode ? 25ULL : 50ULL;
        num_iterations = aggressive_mode ? 40 : (high_priority_mode ? 20 : 10);
        // Many small images never reach the memory thresholds, so bound the count too.
        if (gc_stale_only) {
            num_iterations = (std::min<size_t>)(texture_count - IMAGE_COUNT_SOFT_LIMIT + 10, 256);
        }
    };

    std::vector<ImageViewId> removed_views;
    bool deleted_any = false;
    const auto Cleanup = [this, &num_iterations, &num_inspected, &high_priority_mode,
                          &aggressive_mode, &removed_views, &deleted_any](ImageId image_id) {
        // Only deletions spend the budget, so protected old images cannot starve the rest.
        if (num_iterations == 0 || num_inspected++ >= GC_MAX_INSPECTED) {
            return true;
        }
        auto& image = slot_images[image_id];

        if (True(image.flags & ImageFlagBits::IsDecoding)) {
            return false;
        }

        // Without memory pressure, only drop images whose guest memory was rewritten.
        if (gc_stale_only && (False(image.flags & ImageFlagBits::CpuModified) ||
                              True(image.flags & ImageFlagBits::GpuModified))) {
            return false;
        }

        if (!aggressive_mode && True(image.flags & ImageFlagBits::CostlyLoad)) {
            return false;
        }

        const bool must_download =
            image.IsSafeDownload() && False(image.flags & ImageFlagBits::BadOverlap);

        if (!high_priority_mode && must_download) {
            return false;
        }

        if (must_download) {
            auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
            const auto copies = FullDownloadCopies(image.info);
            image.DownloadMemory(map, copies);
            runtime.Finish();
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span,
                         swizzle_data_buffer);
        }

        --num_iterations;
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        UnregisterImage(image_id);
        DeleteImage(image_id, image.scale_tick > frame_tick + 5, &removed_views);
        deleted_any = true;
        if (aggressive_mode && total_used_memory < critical_memory) {
            num_iterations >>= 2;
            aggressive_mode = false;
        } else if (high_priority_mode && total_used_memory < expected_memory) {
            num_iterations >>= 1;
            high_priority_mode = false;
        }
        return false;
    };

    Configure(false);
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, Cleanup);
    if (total_used_memory >= critical_memory) {
        Configure(true);
        lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, Cleanup);
    }
    if (deleted_any) {
        FinishDeferredDeletes(removed_views);
    }
}

template <class P>
void TextureCache<P>::TickFrame() {
    const u64 gc_memory_usage =
        runtime.CanReportMemoryUsage() ? runtime.GetDeviceMemoryUsage() : total_used_memory;
    const bool memory_pressure = gc_memory_usage > minimum_memory;
    if (memory_pressure || texture_count > IMAGE_COUNT_SOFT_LIMIT) {
        gc_stale_only = !memory_pressure;
        RunGarbageCollector();
    }
    sentenced_images.Tick();
    sentenced_framebuffers.Tick();
    sentenced_image_view.Tick();
    TickAsyncDecode();

    runtime.TickFrame();
    ++frame_tick;

    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        for (auto& buffer : async_buffers_death_ring) {
            runtime.FreeDeferredStagingBuffer(buffer);
        }
        async_buffers_death_ring.clear();
    }
}

template <class P>
typename TextureCache<P>::VRAMStats TextureCache<P>::GetVRAMStats() const noexcept {
    return VRAMStats{
        .total_used_bytes = total_used_memory,
        .texture_bytes = total_used_memory - sparse_texture_memory,
        .sparse_texture_bytes = sparse_texture_memory,
        .evicted_this_frame = evicted_this_frame,
        .evicted_total = evicted_total,
        .texture_count = texture_count,
        .sparse_texture_count = sparse_texture_count,
    };
}

template <class P>
const typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) const noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(u32 index) noexcept {
    const auto image_view_id = VisitImageView(channel_state->graphics_image_table,
                                              channel_state->graphics_image_view_ids, index);
    return slot_image_views[image_view_id];
}

template <class P>
void TextureCache<P>::MarkModification(ImageId id) noexcept {
    MarkModification(slot_images[id]);
}

template <class P>
template <bool has_blacklists>
void TextureCache<P>::FillGraphicsImageViews(std::span<ImageViewInOut> views) {
    FillImageViews<has_blacklists>(channel_state->graphics_image_table,
                                   channel_state->graphics_image_view_ids, views);
}

template <class P>
void TextureCache<P>::FillComputeImageViews(std::span<ImageViewInOut> views) {
    FillImageViews<true>(channel_state->compute_image_table, channel_state->compute_image_view_ids,
                         views);
}

template <class P>
u64 TextureCache<P>::GraphicsImageTableGeneration() const noexcept {
    return channel_state->graphics_image_table.Generation();
}

template <class P>
u64 TextureCache<P>::ComputeImageTableGeneration() const noexcept {
    return channel_state->compute_image_table.Generation();
}

template <class P>
void TextureCache<P>::CheckFeedbackLoop(std::span<const ImageViewInOut> views) {
    if (!Settings::values.barrier_feedback_loops.GetValue()) {
        return;
    }

    const bool requires_barrier = [&] {
        for (const auto& view : views) {
            if (!view.id) {
                continue;
            }
            auto& image_view = slot_image_views[view.id];

            // Check color targets
            for (const auto& ct_view_id : render_targets.color_buffer_ids) {
                if (ct_view_id) {
                    auto& ct_view = slot_image_views[ct_view_id];
                    if (image_view.image_id == ct_view.image_id) {
                        return true;
                    }
                }
            }

            // Check zeta target
            if (render_targets.depth_buffer_id) {
                auto& zt_view = slot_image_views[render_targets.depth_buffer_id];
                if (image_view.image_id == zt_view.image_id) {
                    return true;
                }
            }
        }

        return false;
    }();

    if (requires_barrier) {
        runtime.BarrierFeedbackLoop();
    }
}

template <class P>
typename P::Sampler* TextureCache<P>::GetGraphicsSampler(u32 index) {
    return &slot_samplers[GetGraphicsSamplerId(index)];
}

template <class P>
typename P::Sampler* TextureCache<P>::GetComputeSampler(u32 index) {
    return &slot_samplers[GetComputeSamplerId(index)];
}

template <class P>
SamplerId TextureCache<P>::GetGraphicsSamplerId(u32 index) {
    if (index > channel_state->graphics_sampler_table.Limit()) {
        LOG_DEBUG(HW_GPU, "Invalid sampler index={}", index);
        return NULL_SAMPLER_ID;
    }
    const auto [descriptor, is_new] = channel_state->graphics_sampler_table.Read(index);
    SamplerId& id = channel_state->graphics_sampler_ids[index];
    if (is_new) {
        id = FindSampler(descriptor);
    }
    return id;
}

template <class P>
SamplerId TextureCache<P>::GetComputeSamplerId(u32 index) {
    if (index > channel_state->compute_sampler_table.Limit()) {
        LOG_DEBUG(HW_GPU, "Invalid sampler index={}", index);
        return NULL_SAMPLER_ID;
    }
    const auto [descriptor, is_new] = channel_state->compute_sampler_table.Read(index);
    SamplerId& id = channel_state->compute_sampler_ids[index];
    if (is_new) {
        id = FindSampler(descriptor);
    }
    return id;
}

template <class P>
const typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) const noexcept {
    return slot_samplers[id];
}

template <class P>
typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) noexcept {
    return slot_samplers[id];
}

template <class P>
void TextureCache<P>::SynchronizeGraphicsDescriptors() {
    using SamplerBinding = Tegra::Engines::Maxwell3D::Regs::SamplerBinding;
    const bool linked_tsc = maxwell3d->regs.sampler_binding == SamplerBinding::ViaHeaderBinding;
    const u32 tic_limit = maxwell3d->regs.tex_header.limit;
    const u32 tsc_limit = linked_tsc ? tic_limit : maxwell3d->regs.tex_sampler.limit;
    if (channel_state->graphics_sampler_table.Synchronize(maxwell3d->regs.tex_sampler.Address(),
                                                          tsc_limit)) {
        channel_state->graphics_sampler_ids.resize(tsc_limit + 1, CORRUPT_ID);
    }
    if (channel_state->graphics_image_table.Synchronize(maxwell3d->regs.tex_header.Address(),
                                                        tic_limit)) {
        channel_state->graphics_image_view_ids.resize(tic_limit + 1, CORRUPT_ID);
    }
}

template <class P>
void TextureCache<P>::SynchronizeComputeDescriptors() {
    const bool linked_tsc = kepler_compute->launch_description.linked_tsc;
    const u32 tic_limit = kepler_compute->regs.tic.limit;
    const u32 tsc_limit = linked_tsc ? tic_limit : kepler_compute->regs.tsc.limit;
    const GPUVAddr tsc_gpu_addr = kepler_compute->regs.tsc.Address();
    if (channel_state->compute_sampler_table.Synchronize(tsc_gpu_addr, tsc_limit)) {
        channel_state->compute_sampler_ids.resize(tsc_limit + 1, CORRUPT_ID);
    }
    if (channel_state->compute_image_table.Synchronize(kepler_compute->regs.tic.Address(),
                                                       tic_limit)) {
        channel_state->compute_image_view_ids.resize(tic_limit + 1, CORRUPT_ID);
    }
}

template <class P>
bool TextureCache<P>::RescaleRenderTargets() {
    auto& flags = maxwell3d->dirty.flags;
    u32 scale_rating = 0;
    bool rescaled = false;
    std::array<ImageId, NUM_RT> tmp_color_images{};
    ImageId tmp_depth_image{};
    do {
        flags[Dirty::RenderTargets] = false;

        has_deleted_images = false;
        // Render target control is used on all render targets, so force look ups when this one is
        // up
        const bool force = flags[Dirty::RenderTargetControl];
        flags[Dirty::RenderTargetControl] = false;

        scale_rating = 0;
        bool any_rescaled = false;
        bool can_rescale = true;
        const auto check_rescale = [&](ImageViewId view_id, ImageId& id_save) {
            if (view_id != NULL_IMAGE_VIEW_ID && view_id != ImageViewId{}) {
                const auto& view = slot_image_views[view_id];
                const auto image_id = view.image_id;
                id_save = image_id;
                auto& image = slot_images[image_id];
                can_rescale &= ImageCanRescale(image);
                any_rescaled |= True(image.flags & ImageFlagBits::Rescaled) ||
                                GetFormatType(image.info.format) != SurfaceType::ColorTexture;
                scale_rating = std::max<u32>(scale_rating, image.scale_tick <= frame_tick
                                                               ? image.scale_rating + 1U
                                                               : image.scale_rating);
            } else {
                id_save = CORRUPT_ID;
            }
        };
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            if (flags[Dirty::ColorBuffer0 + index] || force) {
                flags[Dirty::ColorBuffer0 + index] = false;
                BindRenderTarget(&color_buffer_id, FindColorBuffer(index));
            }
            check_rescale(color_buffer_id, tmp_color_images[index]);
        }
        if (flags[Dirty::ZetaBuffer] || force) {
            flags[Dirty::ZetaBuffer] = false;
            BindRenderTarget(&render_targets.depth_buffer_id, FindDepthBuffer());
        }
        check_rescale(render_targets.depth_buffer_id, tmp_depth_image);

        if (can_rescale) {
            rescaled = any_rescaled || scale_rating >= 2;
            const auto scale_up = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleUp(image);
                }
            };
            if (rescaled) {
                for (size_t index = 0; index < NUM_RT; ++index) {
                    scale_up(tmp_color_images[index]);
                }
                scale_up(tmp_depth_image);
                scale_rating = 2;
            }
        } else {
            rescaled = false;
            const auto scale_down = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleDown(image);
                }
            };
            for (size_t index = 0; index < NUM_RT; ++index) {
                scale_down(tmp_color_images[index]);
            }
            scale_down(tmp_depth_image);
            scale_rating = 1;
        }
    } while (has_deleted_images);
    const auto set_rating = [this, scale_rating](ImageId image_id) {
        if (image_id != CORRUPT_ID) {
            Image& image = slot_images[image_id];
            image.scale_rating = scale_rating;
            if (image.scale_tick <= frame_tick) {
                image.scale_tick = frame_tick + 1;
            }
        }
    };
    for (size_t index = 0; index < NUM_RT; ++index) {
        set_rating(tmp_color_images[index]);
    }
    set_rating(tmp_depth_image);

    return rescaled;
}

template <class P>
void TextureCache<P>::UpdateRenderTargets(bool is_clear) {
    using namespace VideoCommon::Dirty;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::RenderTargets]) {
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
        }
        const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;
        PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));
        return;
    }

    const bool rescaled = RescaleRenderTargets();
    if (is_rescaling != rescaled) {
        flags[Dirty::RescaleViewports] = true;
        flags[Dirty::RescaleScissors] = true;
        is_rescaling = rescaled;
    }

    for (size_t index = 0; index < NUM_RT; ++index) {
        ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
        PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
    }
    const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;

    PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));

    for (size_t index = 0; index < NUM_RT; ++index) {
        render_targets.draw_buffers[index] = static_cast<u8>(maxwell3d->regs.rt_control.Map(index));
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (is_rescaling) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    render_targets.size = Extent2D{
        (maxwell3d->regs.surface_clip.width * up_scale) >> down_shift,
        (maxwell3d->regs.surface_clip.height * up_scale) >> down_shift,
    };
    render_targets.is_rescaled = is_rescaling;

    flags[Dirty::DepthBiasGlobal] = true;
}

template <class P>
typename P::Framebuffer* TextureCache<P>::GetFramebuffer() {
    return &slot_framebuffers[GetFramebufferId(render_targets)];
}

template <class P>
template <bool has_blacklists>
void TextureCache<P>::FillImageViews(DescriptorTable<TICEntry>& table,
                                     std::span<ImageViewId> cached_image_view_ids,
                                     std::span<ImageViewInOut> views) {
    bool has_blacklisted = false;
    do {
        has_deleted_images = false;
        if constexpr (has_blacklists) {
            has_blacklisted = false;
        }
        for (ImageViewInOut& view : views) {
            if (view.id_cached) {
                // Already resolved and prepared during cache population.
                // Batch-prepare unique images below instead of per-entry PrepareImageView.
            } else {
                view.id = VisitImageView(table, cached_image_view_ids, view.index);
            }
            if constexpr (has_blacklists) {
                if (view.blacklist && view.id != NULL_IMAGE_VIEW_ID) {
                    const ImageViewBase& image_view{slot_image_views[view.id]};
                    auto& image = slot_images[image_view.image_id];
                    has_blacklisted |= ScaleDown(image);
                    image.scale_rating = 0;
                }
            }
        }
    } while (has_deleted_images || (has_blacklists && has_blacklisted));

    // Batch-prepare unique images from id_cached views.  The cache-miss path
    // already called VisitImageView (which does PrepareImageView) for each
    // entry; here we only need to touch/refresh each *unique* underlying
    // image once, rather than 1024 individual PrepareImageView calls.
    boost::container::small_vector<ImageId, 64> cached_image_ids;
    for (const ImageViewInOut& view : views) {
        if (view.id_cached && view.id != NULL_IMAGE_VIEW_ID) {
            const ImageViewBase& iv = slot_image_views[view.id];
            if (!iv.IsBuffer()) {
                cached_image_ids.push_back(iv.image_id);
            }
        }
    }
    std::sort(cached_image_ids.begin(), cached_image_ids.end());
    cached_image_ids.erase(
        std::unique(cached_image_ids.begin(), cached_image_ids.end()),
        cached_image_ids.end());
    for (const ImageId id : cached_image_ids) {
        PrepareImage(id, false, false);
    }
}

template <class P>
ImageViewId TextureCache<P>::VisitImageView(DescriptorTable<TICEntry>& table,
                                            std::span<ImageViewId> cached_image_view_ids,
                                            u32 index) {
    if (index > table.Limit()) {
        LOG_DEBUG(HW_GPU, "Invalid image view index={}", index);
        return NULL_IMAGE_VIEW_ID;
    }
    const auto [descriptor, is_new] = table.Read(index);
    ImageViewId& image_view_id = cached_image_view_ids[index];
    if (is_new) {
        image_view_id = FindImageView(descriptor);
    }
    if (image_view_id != NULL_IMAGE_VIEW_ID) {
        PrepareImageView(image_view_id, false, false);
    }
    return image_view_id;
}

template <class P>
FramebufferId TextureCache<P>::GetFramebufferId(const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer_id = pair->second;
    if (!is_new) {
        return framebuffer_id;
    }
    std::array<ImageView*, NUM_RT> color_buffers;
    std::ranges::transform(key.color_buffer_ids, color_buffers.begin(),
                           [this](ImageViewId id) { return id ? &slot_image_views[id] : nullptr; });
    ImageView* const depth_buffer =
        key.depth_buffer_id ? &slot_image_views[key.depth_buffer_id] : nullptr;
    framebuffer_id = slot_framebuffers.insert(runtime, color_buffers, depth_buffer, key);
    return framebuffer_id;
}

template <class P>
void TextureCache<P>::WriteMemory(DAddr cpu_addr, size_t size) {
    // TEMP DIAGNOSTIC (S17): notification side of the LUT/mask investigation. Log raw write
    // notifications whose range covers the bank's cpu range (0x29426000 in every harvested
    // run) before any per-image guard, so a missing line means "nothing was ever notified"
    // rather than "a guard ate it". Bounded to 24 raw lines.
    {
        static u32 s17_notify_raw_logs = 0;
        const DAddr s17_bank_cpu = 0x29426000u;
        // S19: the LUT-fill input windows (probe heap dev 0x9418a000, sample-table heap dev
        // 0xbaec8000) - a raw write notification touching them means the guest (or a download)
        // wrote the scratch heap the GPU-authored probe/table lives in.
        const bool s19_input_cpu =
            (cpu_addr < 0x9418a000u + 0x408000u && cpu_addr + size > 0x9418a000u) ||
            (cpu_addr < 0xbaec8000u + 0x10000u && cpu_addr + size > 0xbaec8000u);
        if (s17_notify_raw_logs < 64 &&
            ((cpu_addr < s17_bank_cpu + 0x1a40000u && cpu_addr + size > s17_bank_cpu) ||
             s19_input_cpu)) {
            ++s17_notify_raw_logs;
            LOG_WARNING(Render_D3D12,
                        "S17 write notify: raw cpu={:#x} size={} (bank={} input={})",
                        static_cast<u64>(cpu_addr), size,
                        (cpu_addr < s17_bank_cpu + 0x1a40000u && cpu_addr + size > s17_bank_cpu) ? 1
                                                                                                : 0,
                        s19_input_cpu ? 1 : 0);
        }
    }
    // Release the whole write-tracked page; GPU-written images keep exact bounds.
    const DAddr page_start = Common::AlignDown(cpu_addr, Core::DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(cpu_addr + size, Core::DEVICE_PAGESIZE);
    ForEachImageInRegion(page_start, page_end - page_start,
                          [this, cpu_addr, size](ImageId image_id, Image& image) {
        // TEMP DIAGNOSTIC (S17): per-image decision for the bank/mask images, matched by GPU
        // VA so the mask family is covered even though its cpu range is unknown. 'mark' => the
        // next sample re-uploads this image from guest memory; 'swallow-gpu' => the 1.0.233
        // guard refuses it; 'pending' => already marked for upload. Bounded to 24 lines.
        {
            static u32 s17_notify_image_logs = 0;
            const u64 s17_gpu = static_cast<u64>(image.gpu_addr);
            // Bank window corrected (see S17ProducerRange): 0x5ed00000 was a 1.6 GB address.
            const bool s17_bank = s17_gpu >= 0x5e9000000ull && s17_gpu < 0x5ee7b4000ull;
            const bool s17_mask =
                (s17_gpu >= 0x510fa0000ull && s17_gpu < 0x510fa0000ull + 0x100000ull) ||
                (s17_gpu >= 0x510f30000ull && s17_gpu < 0x510f30000ull + 0x100000ull);
            // S19: the two LUT-fill input windows (the probe image and the sample table).
            const bool s19_input_img =
                (s17_gpu >= 0x516170000ull && s17_gpu < 0x516170000ull + 0x408000ull) ||
                (s17_gpu >= 0x5d0630000ull && s17_gpu < 0x5d0631000ull);
            if (s17_notify_image_logs < 64 && (s17_bank || s17_mask || s19_input_img)) {
                ++s17_notify_image_logs;
                const char* const s17_action =
                    True(image.flags & ImageFlagBits::CpuModified)  ? "pending"
                    : True(image.flags & ImageFlagBits::GpuModified) ? "swallow-gpu"
                                                                     : "mark";
                LOG_WARNING(Render_D3D12,
                            "S17 write notify: img={} gpu={:#x} flags={:#x} size={} action={} "
                            "write_cpu={:#x} write_size={}",
                            image_id.index, s17_gpu, static_cast<u32>(image.flags),
                            image.guest_size_bytes, s17_action, static_cast<u64>(cpu_addr), size);
            }
        }
        // S23 cycle 3: dedicated notify probe for the 1024x1024 BC5 at 0x5e9430000 and the
        // 960x540 mask at 0x510f30000 - the two images whose GPU content separates the lit run
        // from the black one. Own budget so the boot-time S17/S18 caps cannot swallow it.
        {
            static u32 s23_notify_logs = 0;
            const u64 s23_gpu = static_cast<u64>(image.gpu_addr);
            const bool s23_track = (s23_gpu >= 0x5e9430000ull && s23_gpu < 0x5e9580000ull) ||
                                   (s23_gpu >= 0x510f30000ull && s23_gpu < 0x510f40000ull);
            if (s23_notify_logs < 48 && s23_track) {
                ++s23_notify_logs;
                const char* const s23_action =
                    True(image.flags & ImageFlagBits::CpuModified)  ? "pending"
                    : True(image.flags & ImageFlagBits::GpuModified) ? "swallow-gpu"
                                                                     : "mark";
                LOG_WARNING(Render_D3D12,
                            "S23 target notify: img={} gpu={:#x} flags={:#x} size={} action={} "
                            "write_cpu={:#x} write_size={}",
                            image_id.index, s23_gpu, static_cast<u32>(image.flags),
                            image.guest_size_bytes, s23_action, static_cast<u64>(cpu_addr), size);
            }
        }
        // TEMP DIAGNOSTIC (S18, Phase 0.2.1): 1x1 images (the exposure candidates) fall outside
        // the bank/mask filter above; log a write notification touching one with the action the
        // guards below take (mark => the next sample re-uploads it; swallow-gpu => the GPU-owned
        // guard refuses the write; pending => already queued). Bounded to 12 lines.
        {
            static u32 s18_tiny_notify_logs = 0;
            // S18 cycle 3: widened from <=2x2 so the exposure chain's other members are covered
            // too - the 8x1 reduction RT and the 128x1 R32G32B32A32 luminance array (the
            // reduction draw's real input, sampled by it in the 1.0.272 run).
            const u32 s18_nw = image.info.size.width;
            const u32 s18_nh = image.info.size.height;
            const bool s18_notify_tiny =
                (s18_nw <= 2 && s18_nh <= 2) || (s18_nw <= 128 && s18_nh <= 1) ||
                (s18_nw <= 1 && s18_nh <= 128);
            if (s18_tiny_notify_logs < 12 && image.info.type == ImageType::e2D &&
                s18_notify_tiny) {
                ++s18_tiny_notify_logs;
                const char* const s18_action =
                    True(image.flags & ImageFlagBits::CpuModified)  ? "pending"
                    : True(image.flags & ImageFlagBits::GpuModified) ? "swallow-gpu"
                                                                     : "mark";
                LOG_WARNING(Render_D3D12,
                            "S18 tiny notify: img={} gpu={:#x} flags={:#x} {}x{} fmt={:#x} "
                            "write_cpu={:#x} write_size={} action={}",
                            image_id.index, static_cast<u64>(image.gpu_addr),
                            static_cast<u32>(image.flags), image.info.size.width,
                            image.info.size.height, static_cast<u32>(image.info.format),
                            static_cast<u64>(cpu_addr), size, s18_action);
            }
        }
        if (True(image.flags & ImageFlagBits::CpuModified)) {
            return;
        }
        // FIX (S15/S16): never let a guest CPU write mark a GPU-authored image as CPU-modified.
        // Such an image's content is owned by the GPU (the scene render target was being
        // re-uploaded from zero guest memory and wiped between the scene draw and the
        // composite). The previous guard kept the CPU mark whenever the write overlapped the
        // image, which page-aligned writes routinely do. Leaving GpuModified images untouched
        // keeps the GPU content authoritative (a real CPU overwrite of such an image must go
        // through an explicit download/invalidate, as everywhere else).
        if (True(image.flags & ImageFlagBits::GpuModified)) {
            // TEMP DIAGNOSTIC (S15): H-cpu probe - this early return swallows guest CPU
            // writes whose page range overlaps a GPU-authored image (candidate writer of
            // the wiper input at cpu 0x9501a000 and of the atlas images); bounded log of
            // exactly what is being skipped.
            static u32 s15_write_swallow_logs = 0;
            if (s15_write_swallow_logs < 16) {
                ++s15_write_swallow_logs;
                LOG_WARNING(Render_D3D12,
                            "S15 WriteMemory swallowed: img_gpu={:#x} img_flags={:#x} "
                            "write_cpu={:#x} size={} gpu_modified=1",
                            static_cast<u64>(image.gpu_addr),
                            static_cast<u32>(image.flags), static_cast<u64>(cpu_addr), size);
            }
            return;
        }
        image.flags |= ImageFlagBits::CpuModified;
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
    });
}

template <class P>
void TextureCache<P>::DownloadMemory(DAddr cpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> images;
    ForEachImageInRegion(cpu_addr, size, [&images](ImageId image_id, ImageBase& image) {
        if (!image.IsSafeDownload()) {
            return;
        }
        image.flags &= ~ImageFlagBits::GpuModified;
        images.push_back(image_id);
    });
    if (images.empty()) {
        return;
    }
    std::ranges::sort(images, [this](ImageId lhs, ImageId rhs) {
        return slot_images[lhs].modification_tick < slot_images[rhs].modification_tick;
    });
    for (const ImageId image_id : images) {
        Image& image = slot_images[image_id];
        auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
        const auto copies = FullDownloadCopies(image.info);
        image.DownloadMemory(map, copies);
        runtime.Finish();
        SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span,
                     swizzle_data_buffer);
    }
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> TextureCache<P>::GetFlushArea(DAddr cpu_addr,
                                                                               u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    ForEachImageInRegion(cpu_addr, size, [&](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return;
        }
        if (!area) {
            area.emplace();
            area->start_address = cpu_addr;
            area->end_address = cpu_addr + size;
            area->preemtive = true;
        }
        area->start_address = std::min(area->start_address, image.cpu_addr);
        area->end_address = std::max(area->end_address, image.cpu_addr_end);
        for (auto image_view_id : image.image_view_ids) {
            auto& image_view = slot_image_views[image_view_id];
            image_view.flags |= ImageViewFlagBits::PreemtiveDownload;
        }
        area->preemtive &= image.info.forced_flushed;
        image.info.forced_flushed = true;
    });
    return area;
}

template <class P>
void TextureCache<P>::UnmapMemory(DAddr cpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    std::vector<ImageViewId> removed_views;
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, id);
        }
        UnregisterImage(id);
        DeleteImage(id, false, &removed_views);
    }
    if (!deleted_images.empty()) {
        FinishDeferredDeletes(removed_views);
    }
}

template <class P>
void TextureCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegionGPU(as_id, gpu_addr, size,
                            [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        if (False(image.flags & ImageFlagBits::CpuModified)) {
            image.flags |= ImageFlagBits::CpuModified;
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, id);
            }
        }

        if (True(image.flags & ImageFlagBits::Remapped)) {
            continue;
        }
        image.flags |= ImageFlagBits::Remapped;
    }
}

template <class P>
bool TextureCache<P>::BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                                const Tegra::Engines::Fermi2D::Surface& src,
                                const Tegra::Engines::Fermi2D::Config& copy) {
    const auto result = GetBlitImages(dst, src, copy);
    if (!result) {
        return false;
    }
    const BlitImages images = *result;
    const ImageId dst_id = images.dst_id;
    const ImageId src_id = images.src_id;

    PrepareImage(src_id, false, false);
    PrepareImage(dst_id, true, false);

    Image& dst_image = slot_images[dst_id];
    Image& src_image = slot_images[src_id];
    // TEMP DIAGNOSTIC (S18, Phase 0.2.1): Fermi2D blits with a 1x1 side - the other candidate
    // writer of the exposure texel (a downsample chain ending in a 1x1 blit). Cap 12.
    {
        static u32 s18_tiny_blit_logs = 0;
        // S18 cycle 3: any member of the exposure family (1x1 // 8x1 // 128x1) on either side.
        const auto s18_blit_tiny = [](const ImageBase& s18_img) {
            const u32 s18_w = s18_img.info.size.width;
            const u32 s18_h = s18_img.info.size.height;
            return (s18_w <= 2 && s18_h <= 2) || (s18_w <= 128 && s18_h <= 1) ||
                   (s18_w <= 1 && s18_h <= 128);
        };
        const bool s18_dst_tiny = s18_blit_tiny(dst_image);
        const bool s18_src_tiny = s18_blit_tiny(src_image);
        if (s18_tiny_blit_logs < 12 && (s18_dst_tiny || s18_src_tiny)) {
            ++s18_tiny_blit_logs;
            LOG_WARNING(Render_D3D12,
                        "S18 tiny blit: dst=({} gpu={:#x} {}x{} fmt={:#x}) src=({} gpu={:#x} "
                        "{}x{} fmt={:#x}) src_region=({},{})-({},{}) dst_region=({},{})-({},{}) "
                        "filter={} op={}",
                        dst_id.index, static_cast<u64>(dst_image.gpu_addr),
                        dst_image.info.size.width, dst_image.info.size.height,
                        static_cast<u32>(dst_image.info.format), src_id.index,
                        static_cast<u64>(src_image.gpu_addr), src_image.info.size.width,
                        src_image.info.size.height, static_cast<u32>(src_image.info.format),
                        copy.src_x0, copy.src_y0, copy.src_x1, copy.src_y1, copy.dst_x0,
                        copy.dst_y0, copy.dst_x1, copy.dst_y1, static_cast<u32>(copy.filter),
                        static_cast<u32>(copy.operation));
        }
    }
    bool is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
    bool is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);

    const bool is_resolve = src_image.info.num_samples != 1 && dst_image.info.num_samples == 1;
    if (is_src_rescaled != is_dst_rescaled) {
        if (ImageCanRescale(src_image)) {
            ScaleUp(src_image);
            is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
            if (is_resolve) {
                dst_image.info.rescaleable = true;
                for (const auto& alias : dst_image.aliased_images) {
                    Image& other_image = slot_images[alias.id];
                    other_image.info.rescaleable = true;
                }
            }
        }
        if (ImageCanRescale(dst_image)) {
            ScaleUp(dst_image);
            is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
        }
    }
    if (is_resolve && (is_src_rescaled != is_dst_rescaled)) {
        // A resolve requires both images to be the same dimensions. Resize down if needed.
        ScaleDown(src_image);
        ScaleDown(dst_image);
        is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
        is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
    }
    const auto& resolution = Settings::values.resolution_info;
    const auto scale_region = [&](Region2D& region) {
        region.start.x = resolution.ScaleUp(region.start.x);
        region.start.y = resolution.ScaleUp(region.start.y);
        region.end.x = resolution.ScaleUp(region.end.x);
        region.end.y = resolution.ScaleUp(region.end.y);
    };

    // TODO: Deduplicate
    const std::optional src_base = src_image.TryFindBase(src.Address());
    const SubresourceRange src_range{.base = src_base.value(), .extent = {1, 1}};
    const ImageViewInfo src_view_info(ImageViewType::e2D, images.src_format, src_range);
    const auto [src_framebuffer_id, src_view_id] = RenderTargetFromImage(src_id, src_view_info);
    const auto [src_samples_x, src_samples_y] = SamplesLog2(src_image.info.num_samples);
    Region2D src_region{
        Offset2D{.x = copy.src_x0 >> src_samples_x, .y = copy.src_y0 >> src_samples_y},
        Offset2D{.x = copy.src_x1 >> src_samples_x, .y = copy.src_y1 >> src_samples_y},
    };
    if (is_src_rescaled) {
        scale_region(src_region);
    }

    const std::optional dst_base = dst_image.TryFindBase(dst.Address());
    const SubresourceRange dst_range{.base = dst_base.value(), .extent = {1, 1}};
    const ImageViewInfo dst_view_info(ImageViewType::e2D, images.dst_format, dst_range);
    const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
    const auto [dst_samples_x, dst_samples_y] = SamplesLog2(dst_image.info.num_samples);
    Region2D dst_region{
        Offset2D{.x = copy.dst_x0 >> dst_samples_x, .y = copy.dst_y0 >> dst_samples_y},
        Offset2D{.x = copy.dst_x1 >> dst_samples_x, .y = copy.dst_y1 >> dst_samples_y},
    };
    if (is_dst_rescaled) {
        scale_region(dst_region);
    }

    // Always call this after src_framebuffer_id was queried, as the address might be invalidated.
    Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
    if constexpr (FRAMEBUFFER_BLITS) {
        // OpenGL blits from framebuffers, not images
        Framebuffer* const src_framebuffer = &slot_framebuffers[src_framebuffer_id];
        runtime.BlitFramebuffer(dst_framebuffer, src_framebuffer, dst_region, src_region,
                                copy.filter, copy.operation);
    } else {
        // Vulkan can blit images, but it lacks format reinterpretations
        // Provide a framebuffer in case it's necessary
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        if (!runtime.BlitImage(dst_framebuffer, dst_view, src_view, dst_region, src_region,
                               copy.filter, copy.operation)) {
            return false;
        }
    }
    return true;
}

template <class P>
std::pair<typename P::ImageView*, bool> TextureCache<P>::TryFindFramebufferImageView(
    const Tegra::FramebufferConfig& config, DAddr cpu_addr) {
    // TODO: Properly implement this
    const auto it = page_table.find(cpu_addr >> CITRON_PAGEBITS);
    if (it == page_table.end()) {
        return {};
    }
    const auto& image_map_ids = it->second;
    boost::container::small_vector<ImageId, 4> valid_image_ids;
    for (const ImageMapId map_id : image_map_ids) {
        const ImageMapView& map = slot_map_views[map_id];
        const ImageBase& image = slot_images[map.image_id];
        if (image.cpu_addr != cpu_addr) {
            continue;
        }
        if (image.image_view_ids.empty()) {
            continue;
        }
        valid_image_ids.push_back(map.image_id);
    }

    {
        // Diagnostic: list every cache image registered at the displayed CPU page, so the
        // guest display hand-off (flinger buffer device address vs render GPU VA) is traceable.
        static u64 fb_lookup_logs = 0;
        const u64 log_index = fb_lookup_logs++;
        if (log_index < 8 || log_index % 400 == 0) {
            LOG_WARNING(Render_D3D12, "FB lookup cpu_addr={:#x}: {} image(s), frame_tick={}",
                        cpu_addr, valid_image_ids.size(), frame_tick);
            for (const ImageId id : valid_image_ids) {
                const ImageBase& image = slot_images[id];
                LOG_WARNING(Render_D3D12,
                            "  image id={} gpu={:#x} cpu={:#x} tick={} fmt={:#x} {}x{} "
                            "pitch={} bytes={} flags={:#x}",
                            id.index, image.gpu_addr, image.cpu_addr, image.modification_tick,
                            static_cast<u32>(image.info.format), image.info.size.width,
                            image.info.size.height, image.info.pitch, image.guest_size_bytes,
                            static_cast<u32>(image.flags));
            }
        }
    }

    const auto view_format = [&]() {
        switch (config.pixel_format) {
        case Service::android::PixelFormat::Rgb565:
            return PixelFormat::R5G6B5_UNORM;
        case Service::android::PixelFormat::Bgra8888:
            return PixelFormat::B8G8R8A8_UNORM;
        default:
            return PixelFormat::A8B8G8R8_UNORM;
        }
    }();

    const auto GetImageViewForFramebuffer = [&](ImageId image_id) {
        // The framebuffer-accelerated display path historically skipped the same
        // "ensure GPU contents reflect any pending CPU writes" step every other
        // image-view accessor goes through. Titles that compose their displayed
        // framebuffer via CPU writes (not GPU draws) could therefore have their
        // CpuModified flag never get flushed before being sampled for display.
        PrepareImage(image_id, false, false);
        ImageViewInfo info{ImageViewType::e2D, view_format};
        if (config.blending == Tegra::BlendMode::Opaque) {
            info.x_source = static_cast<u8>(SwizzleSource::R);
            info.y_source = static_cast<u8>(SwizzleSource::G);
            info.z_source = static_cast<u8>(SwizzleSource::B);
            info.w_source = static_cast<u8>(SwizzleSource::OneFloat);
        }
        return std::make_pair(&slot_image_views[FindOrEmplaceImageView(image_id, info)],
                              slot_images[image_id].IsRescaled());
    };

    if (valid_image_ids.size() == 1) [[likely]] {
        return GetImageViewForFramebuffer(valid_image_ids.front());
    }

    if (valid_image_ids.size() > 0) [[unlikely]] {
        auto most_recent = std::ranges::max_element(valid_image_ids, [&](auto a, auto b) {
            return slot_images[a].modification_tick < slot_images[b].modification_tick;
        });
        return GetImageViewForFramebuffer(*most_recent);
    }

    return {};
}

template <class P>
bool TextureCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_downloads.empty();
}

template <class P>
bool TextureCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return !committed_downloads.empty() && !committed_downloads.front().empty();
}

template <class P>
void TextureCache<P>::CommitAsyncFlushes() {
    // This is intentionally passing the value by copy
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        auto& download_ids = uncommitted_downloads;
        if (download_ids.empty()) {
            committed_downloads.emplace_back(std::move(uncommitted_downloads));
            uncommitted_downloads.clear();
            async_buffers.emplace_back(std::move(uncommitted_async_buffers));
            uncommitted_async_buffers.clear();
            return;
        }
        size_t total_size_bytes = 0;
        size_t last_async_buffer_id = uncommitted_async_buffers.size();
        bool any_none_dma = false;
        for (PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes +=
                    Common::AlignUp(slot_images[download_info.object_id].unswizzled_size_bytes, 64);
                any_none_dma = true;
                download_info.async_buffer_id = last_async_buffer_id;
            }
        }

        if (any_none_dma) {
            auto download_map = runtime.DownloadStagingBuffer(total_size_bytes, true);
            for (const PendingDownload& download_info : download_ids) {
                if (download_info.is_swizzle) {
                    Image& image = slot_images[download_info.object_id];
                    const auto copies = FullDownloadCopies(image.info);
                    image.DownloadMemory(download_map, copies);
                    download_map.offset += Common::AlignUp(image.unswizzled_size_bytes, 64);
                }
            }
            uncommitted_async_buffers.emplace_back(download_map);
        }

        async_buffers.emplace_back(std::move(uncommitted_async_buffers));
        uncommitted_async_buffers.clear();
    }
    committed_downloads.emplace_back(std::move(uncommitted_downloads));
    uncommitted_downloads.clear();
}

template <class P>
void TextureCache<P>::PopAsyncFlushes() {
    if (committed_downloads.empty()) {
        return;
    }
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            async_buffers.pop_front();
            return;
        }
        auto download_map = std::move(async_buffers.front());
        for (size_t i = download_ids.size(); i > 0; i--) {
            auto& download_info = download_ids[i - 1];
            auto& download_buffer = download_map[download_info.async_buffer_id];
            if (download_info.is_swizzle) {
                const ImageBase& image = slot_images[download_info.object_id];
                const auto copies = FullDownloadCopies(image.info);
                download_buffer.offset -= Common::AlignUp(image.unswizzled_size_bytes, 64);
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span,
                             swizzle_data_buffer);
            } else {
                const BufferDownload& buffer_info = slot_buffer_downloads[download_info.object_id];
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                gpu_memory->WriteBlockUnsafe(buffer_info.address, download_span.data(),
                                             buffer_info.size);
                slot_buffer_downloads.erase(download_info.object_id);
            }
        }
        for (auto& download_buffer : download_map) {
            async_buffers_death_ring.emplace_back(download_buffer);
        }
        committed_downloads.pop_front();
        async_buffers.pop_front();
    } else {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            return;
        }
        size_t total_size_bytes = 0;
        for (const PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes += slot_images[download_info.object_id].unswizzled_size_bytes;
            }
        }
        auto download_map = runtime.DownloadStagingBuffer(total_size_bytes);
        const size_t original_offset = download_map.offset;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            Image& image = slot_images[download_info.object_id];
            const auto copies = FullDownloadCopies(image.info);
            image.DownloadMemory(download_map, copies);
            download_map.offset += image.unswizzled_size_bytes;
        }
        // Wait for downloads to finish
        runtime.Finish();
        download_map.offset = original_offset;
        std::span<u8> download_span = download_map.mapped_span;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            const ImageBase& image = slot_images[download_info.object_id];
            const auto copies = FullDownloadCopies(image.info);
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span,
                         swizzle_data_buffer);
            download_map.offset += image.unswizzled_size_bytes;
            download_span = download_span.subspan(image.unswizzled_size_bytes);
        }
        committed_downloads.pop_front();
    }
}

template <class P>
ImageId TextureCache<P>::DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload) {
    const ImageInfo dst_info(operand);
    const ImageId dst_id = FindDMAImage(dst_info, operand.address);
    if (!dst_id) {
        return NULL_IMAGE_ID;
    }
    auto& image = slot_images[dst_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        // No need to waste time on an image that's synced with guest
        return NULL_IMAGE_ID;
    }
    if (image.info.type == ImageType::e3D) {
        // Don't accelerate 3D images.
        return NULL_IMAGE_ID;
    }
    if (!is_upload && !image.info.dma_downloaded) {
        // Force a full sync.
        image.info.dma_downloaded = true;
        return NULL_IMAGE_ID;
    }
    const auto base = image.TryFindBase(operand.address);
    if (!base) {
        return NULL_IMAGE_ID;
    }
    return dst_id;
}

template <class P>
bool TextureCache<P>::IsRescaling() const noexcept {
    return is_rescaling;
}

template <class P>
bool TextureCache<P>::IsRescaling(const ImageViewBase& image_view) const noexcept {
    if (image_view.type == ImageViewType::Buffer) {
        return false;
    }
    const ImageBase& image = slot_images[image_view.image_id];
    return True(image.flags & ImageFlagBits::Rescaled);
}

template <class P>
bool TextureCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    bool is_modified = false;
    ForEachImageInRegion(addr, size, [&is_modified](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return false;
        }
        is_modified = true;
        return true;
    });
    return is_modified;
}

template <class P>
std::pair<typename TextureCache<P>::Image*, BufferImageCopy> TextureCache<P>::DmaBufferImageCopy(
    const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
    const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image) {
    const auto [level, base] = PrepareDmaImage(image_id, image_operand.address, modifies_image);
    auto* image = &slot_images[image_id];
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    const u32 bpp = VideoCore::Surface::BytesPerBlock(image->info.format);
    const auto convert = [old_bpp = image_operand.bytes_per_pixel, bpp](u32 value) {
        return (old_bpp * value) / bpp;
    };
    const u32 base_x = convert(image_operand.params.origin.x.Value());
    const u32 base_y = image_operand.params.origin.y.Value();
    const u32 length_x = convert(copy_info.length_x);
    const u32 length_y = copy_info.length_y;

    const BufferImageCopy copy{
        .buffer_offset = 0,
        .buffer_size = buffer_size,
        .buffer_row_length = convert(buffer_operand.pitch),
        .buffer_image_height = buffer_operand.height,
        .image_subresource =
            {
                .base_level = static_cast<s32>(level),
                .base_layer = static_cast<s32>(base),
                .num_layers = 1,
            },
        .image_offset =
            {
                .x = static_cast<s32>(base_x),
                .y = static_cast<s32>(base_y),
                .z = 0,
            },
        .image_extent =
            {
                .width = length_x,
                .height = length_y,
                .depth = 1,
            },
    };
    return {image, copy};
}

template <class P>
void TextureCache<P>::DownloadImageIntoBuffer(typename TextureCache<P>::Image* image,
                                              typename TextureCache<P>::BufferType buffer,
                                              size_t buffer_offset,
                                              std::span<const VideoCommon::BufferImageCopy> copies,
                                              GPUVAddr address, size_t size) {
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const BufferDownload new_buffer_download{address, size};
        auto slot = slot_buffer_downloads.insert(new_buffer_download);
        const PendingDownload new_download{false, uncommitted_async_buffers.size(), slot};
        uncommitted_downloads.emplace_back(new_download);
        auto download_map = runtime.DownloadStagingBuffer(size, true);
        uncommitted_async_buffers.emplace_back(download_map);
        std::array buffers{
            buffer,
            download_map.buffer,
        };
        std::array<size_t, 2> buffer_offsets{
            buffer_offset,
            download_map.offset,
        };
        image->DownloadMemory(buffers, buffer_offsets, copies);
    } else {
        image->DownloadMemory(buffer, buffer_offset, copies);
    }
}

template <class P>
void TextureCache<P>::RefreshContents(Image& image, ImageId image_id) {
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        // Only upload modified images
        return;
    }
    image.flags &= ~ImageFlagBits::CpuModified;
    TrackImage(image, image_id);

    // FIX (S15/S16): never upload guest bytes over a GPU-authored image. The scene's lighting
    // LUT bank, mask and dynamic-res render targets are written by the GPU while the guest
    // memory beneath them stays zero; any stale CpuModified mark (UnmapGPUMemory, the
    // full-clear invalidate window, or an alias sibling) would otherwise let the upload path
    // wipe the rendered content with zeros. This is the single choke point every mark path
    // shares. A legitimate CPU overwrite must go through an explicit download/invalidate.
    if (True(image.flags & ImageFlagBits::GpuModified)) {
        static u32 refresh_gpu_wins_logs = 0;
        if (refresh_gpu_wins_logs < 24) {
            ++refresh_gpu_wins_logs;
            LOG_WARNING(Render_D3D12,
                        "RefreshContents: skipping upload over GpuModified image gpu={:#x} "
                        "{}x{} fmt={:#x}",
                        static_cast<u64>(image.gpu_addr), image.info.size.width,
                        image.info.size.height, static_cast<u32>(image.info.format));
        }
        return;
    }

    if (image.info.num_samples > 1 && !runtime.CanUploadMSAA()) {
        LOG_WARNING(HW_GPU, "MSAA image uploads are not implemented");
        runtime.TransitionImageLayout(image);
        return;
    }
    if (True(image.flags & ImageFlagBits::AsynchronousDecode)) {
        QueueAsyncDecode(image, image_id);
        return;
    }
    auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
    // TEMP DIAGNOSTIC (S15): log uploads that could clobber the 960x540 scene RT
    // (gpu 0x5103e0000, ~2073600 bytes). If a stray CPU write marks this image CpuModified,
    // RefreshContents re-uploads guest memory over the freshly GPU-rendered content; log a
    // small guest-memory prefix to tell whether the uploaded bytes are zero. Log-only.
    {
        static u32 refresh_diag_logs = 0;
        const u64 diag_size = image.guest_size_bytes;
        const u64 diag_gpu = static_cast<u64>(image.gpu_addr);
        // Scene RT, the 256x256 lighting-LUT bank (JoinImages layer stack), the mask, and
        // the wiper-input images (480x270 R16G16 window at 0x517000000/0x517180000).
        // S19: the two LUT-fill input windows joined the list - the probe image (4086 @
        // 0x516170000) held 87% content right after its join copy and read zero 555 s later
        // (timeline 925001/925002). An upload recorded here means an upload happened despite
        // the GpuModified guard OR the image lost that flag; prefix_nz says whether the guest
        // bytes behind it are zero (a zero re-upload = the wipe).
        const bool diag_addr = diag_gpu == 0x5103e0000ULL || diag_gpu == 0x5eb340000ULL ||
                               diag_gpu == 0x510f30000ULL || diag_gpu == 0x517000000ULL ||
                               diag_gpu == 0x517180000ULL ||
                               (diag_gpu >= 0x516170000ull &&
                                diag_gpu < 0x516170000ull + 0x408000ull) ||
                               (diag_gpu >= 0x5d0630000ull && diag_gpu < 0x5d0631000ull);
        const bool diag_size_match = (diag_size >= 2000000ULL && diag_size <= 2100000ULL);
        if (refresh_diag_logs < 64 && (diag_addr || diag_size_match)) {
            ++refresh_diag_logs;
            std::array<u8, 16> diag_prefix{};
            u32 diag_nz = 0;
            const std::size_t diag_read =
                std::min<std::size_t>(diag_prefix.size(), staging.mapped_span.size());
            gpu_memory->ReadBlock(static_cast<GPUVAddr>(image.gpu_addr), diag_prefix.data(),
                                  diag_read, VideoCommon::CacheType::NoTextureCache);
            for (std::size_t i = 0; i < diag_read; ++i) {
                if (diag_prefix[i] != 0) {
                    ++diag_nz;
                }
            }
            LOG_WARNING(Render_D3D12,
                        "S15 RefreshContents upload: img={} gpu={:#x} size={} {}x{} fmt={:#x} "
                        "flags={:#x} prefix_nz={}/{} "
                        "prefix={:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                        image_id.index, static_cast<u64>(image.gpu_addr), diag_size,
                        image.info.size.width, image.info.size.height,
                        static_cast<u32>(image.info.format), static_cast<u32>(image.flags),
                        diag_nz, static_cast<u32>(diag_read), diag_prefix[0], diag_prefix[1],
                        diag_prefix[2], diag_prefix[3], diag_prefix[4], diag_prefix[5],
                        diag_prefix[6], diag_prefix[7]);
        }
    }
    // TEMP DIAGNOSTIC (S17): per-upload record for the 27 MB LUT bank. The lit/black alternation
    // observed WITHIN one run (capture_8 lit at t=1203, capture_9 black at t=1307, capture_10 lit
    // at t=1408) is either the sampled image being re-uploaded from zero guest memory on
    // alternating frames, or something else clobbering its resource. This logs every upload of a
    // bank-overlapping image with its flags/tick so the two are distinguishable. Bounded to 64.
    {
        // Cap raised 64 -> 1024: the 64-entry cap was EXACTLY saturated by the creation burst
        // (64 events in 0.27 s at t=750.9-751.2), which made 'no uploads after creation' an
        // artefact of the cap rather than an observation.
        static u32 s17_bank_upload_logs = 0;
        // S19: the LUT-fill input windows are recorded here too. This block sits BELOW the
        // GpuModified early-return, so any line for the probe image (0x516170000) or the sample
        // table (0x5d0630000) IS a real upload over their content - the wipe, if it is an upload.
        const bool s19_input_upload =
            (static_cast<u64>(image.cpu_addr) < 0x9418a000u + 0x408000u &&
             static_cast<u64>(image.cpu_addr) + image.guest_size_bytes > 0x9418a000u) ||
            (static_cast<u64>(image.cpu_addr) < 0xbaec8000u + 0x10000u &&
             static_cast<u64>(image.cpu_addr) + image.guest_size_bytes > 0xbaec8000u);
        if (s17_bank_upload_logs < 1024 &&
            ((image.cpu_addr < (0x29426000u + 0x1a40000u) &&
              static_cast<u64>(image.cpu_addr) + image.guest_size_bytes > 0x29426000u) ||
             s19_input_upload)) {
            ++s17_bank_upload_logs;
            LOG_WARNING(Render_D3D12,
                        "S17 bank upload #{}: img={} gpu={:#x} size={} flags={:#x} tick={} "
                        "input={}",
                        s17_bank_upload_logs, image_id.index, static_cast<u64>(image.gpu_addr),
                        image.guest_size_bytes, static_cast<u32>(image.flags),
                        image.modification_tick, s19_input_upload ? 1 : 0);
        }
    }
    UploadImageContents(image, staging);
    runtime.InsertUploadMemoryBarrier();
}

template <class P>
template <typename StagingBuffer>
void TextureCache<P>::UploadImageContents(Image& image, StagingBuffer& staging) {
    const std::span<u8> mapped_span = staging.mapped_span;
    const GPUVAddr gpu_addr = image.gpu_addr;

    if (True(image.flags & ImageFlagBits::AcceleratedUpload)) {
        gpu_memory->ReadBlock(gpu_addr, mapped_span.data(), mapped_span.size_bytes(),
                              VideoCommon::CacheType::NoTextureCache);
        const auto uploads = FullUploadSwizzles(image.info);
        runtime.AccelerateImageUpload(image, staging, uploads);
        return;
    }

    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(
        *gpu_memory, gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);

    if (True(image.flags & ImageFlagBits::Converted)) {
        unswizzle_data_buffer.resize_destructive(image.unswizzled_size_bytes);
        auto copies =
            UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data, unswizzle_data_buffer);
        ConvertImage(unswizzle_data_buffer, image.info, mapped_span, copies);
        image.UploadMemory(staging, copies);
    } else {
        const auto copies =
            UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data, mapped_span);
        // S23 cycle 3: WHICH upload decides whether the 3D scene is lit. The 1024x1024 BC5 at
        // 0x5e9430000 reads 37.8 % nonzero with a lit scene (1.0.328) and 0 % with a black scene
        // (1.0.329), while the image's flags are identical (0xf0, clean) in both runs - so the
        // difference is this upload. Log every upload of that window (and of the 960x540 mask at
        // 0x510f30000) with the source's nonzero fraction: a zero source means the image was
        // created/uploaded BEFORE the guest wrote the texture, and nothing re-uploaded it after.
        {
            static u32 s23_upload_logs = 0;
            const u64 s23_ug = static_cast<u64>(gpu_addr);
            const bool s23_target = (s23_ug >= 0x5e9430000ull && s23_ug < 0x5e9580000ull) ||
                                    (s23_ug >= 0x510f30000ull && s23_ug < 0x510f40000ull);
            if (s23_upload_logs < 48 && s23_target) {
                ++s23_upload_logs;
                const size_t s23_n =
                    mapped_span.size_bytes() < 4096 ? mapped_span.size_bytes() : 4096;
                u32 s23_nz = 0;
                for (size_t s23_i = 0; s23_i < s23_n; ++s23_i) {
                    s23_nz += mapped_span[s23_i] != 0 ? 1u : 0u;
                }
                LOG_WARNING(Render_D3D12,
                            "S23 target upload: gpu={:#x} {}x{} fmt={:#x} size={} src_nz={}/{} "
                            "flags={:#x}",
                            s23_ug, image.info.size.width, image.info.size.height,
                            static_cast<u32>(image.info.format), image.guest_size_bytes, s23_nz,
                            static_cast<u32>(s23_n), static_cast<u32>(image.flags));
            }
        }
        // TEMP DIAGNOSTIC (S18, Phase 0.2.1): an upload INTO a 1x1 R32G32B32A32 image is a
        // candidate writer of the exposure texel. Log the exact floats being pushed into the
        // resource (read after the unswizzle, so the bytes are what the upload uses); all-zero
        // means the guest bytes behind the image are zero - the GPU-authored case. Cap 12.
        {
            static u32 s18_tiny_upload_logs = 0;
            // S18 cycle 3: widened to the whole exposure family (1x1 // 8x1 // 128x1) - the
            // uploaded bytes are the guest-side source of every member; only the first texel is
            // decoded (the rest would need per-row strides the staging buffer may not keep).
            const u32 s18_uw = image.info.size.width;
            const u32 s18_uh = image.info.size.height;
            const bool s18_upload_tiny = (s18_uw <= 2 && s18_uh <= 2) ||
                                         (s18_uw <= 128 && s18_uh <= 1) ||
                                         (s18_uw <= 1 && s18_uh <= 128);
            if (s18_tiny_upload_logs < 12 && s18_upload_tiny &&
                image.info.format == PixelFormat::R32G32B32A32_FLOAT &&
                mapped_span.size_bytes() >= 16) {
                ++s18_tiny_upload_logs;
                f32 s18_vals[4]{};
                std::memcpy(s18_vals, mapped_span.data(), sizeof(s18_vals));
                LOG_WARNING(Render_D3D12,
                            "S18 tiny upload: gpu={:#x} size={} f=({:.6f},{:.6f},{:.6f},{:.6f}) "
                            "bytes={:02x}{:02x}{:02x}{:02x}",
                            static_cast<u64>(gpu_addr), image.guest_size_bytes, s18_vals[0],
                            s18_vals[1], s18_vals[2], s18_vals[3], mapped_span[0],
                            mapped_span[1], mapped_span[2], mapped_span[3]);
            }
        }
        image.UploadMemory(staging, copies);
    }
}

template <class P>
ImageViewId TextureCache<P>::FindImageView(const TICEntry& config) {
    if (!IsValidEntry(*gpu_memory, config)) {
        return NULL_IMAGE_VIEW_ID;
    }
    using Tegra::Texture::TextureType;
    TICEntry cache_key = config;
    const auto tex_type = config.texture_type.Value();
    if (tex_type == TextureType::Texture2D || tex_type == TextureType::Texture2DNoMipmap) {
        cache_key.depth_minus_1.Assign(0);
    }
    const auto [pair, is_new] = channel_state->image_views.try_emplace(cache_key);
    ImageViewId& image_view_id = pair->second;
    if (is_new) {
        image_view_id = CreateImageView(config);
    }
    return image_view_id;
}

template <class P>
ImageViewId TextureCache<P>::CreateImageView(const TICEntry& config) {
    const ImageInfo info(config);
    if (info.type == ImageType::Buffer) {
        const ImageViewInfo view_info(config, 0);
        return slot_image_views.insert(runtime, info, view_info, config.Address());
    }
    const u32 layer_offset = config.BaseLayer() * info.layer_stride;
    const GPUVAddr image_gpu_addr = config.Address() - layer_offset;
    // S23 FIX: a sampled view may be a few texels narrower/shorter than the surface it lives on
    // (the game declares its dynamic-resolution render target at 992x556 and samples it through a
    // 990x556 view). With the default strict size comparison that created a SECOND image at the
    // same guest address, and the composite then sampled an image nothing had rendered
    // (measured: `same_res=0`, the resized scene dumps read all-zero while the 960x540 member had
    // content). Relaxing the size check lets FindImage match the existing surface through the
    // block-linear (GOB) alignment, which is what the hardware does.
    const ImageId image_id = FindOrInsertImage(info, image_gpu_addr, RelaxedOptions::Size);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    ImageBase& image = slot_images[image_id];
    const SubresourceBase base = image.TryFindBase(config.Address()).value();
    ASSERT(base.level == 0);
    const ImageViewInfo view_info(config, base.layer);
    const ImageViewId image_view_id = FindOrEmplaceImageView(image_id, view_info);
    ImageViewBase& image_view = slot_image_views[image_view_id];
    image_view.flags |= ImageViewFlagBits::Strong;
    image.flags |= ImageFlagBits::Strong;
    return image_view_id;
}

template <class P>
ImageId TextureCache<P>::FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                           RelaxedOptions options) {
    if (const ImageId image_id = FindImage(info, gpu_addr, options); image_id) {
        return image_id;
    }
    return InsertImage(info, gpu_addr, options);
}

template <class P>
ImageId TextureCache<P>::FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                   RelaxedOptions options) {
    std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!cpu_addr) {
        cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
        if (!cpu_addr) {
            return ImageId{};
        }
    }
    const bool broken_views =
        runtime.HasBrokenTextureViewFormats() || True(options & RelaxedOptions::ForceBrokenViews);
    const bool native_bgr = runtime.HasNativeBgr();
    const bool flexible_formats = True(options & RelaxedOptions::Format);
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    boost::container::small_vector<ImageId, 8> exact_image_ids;
    // S20 fix: the scan must not stop at the first exact-format match. Guest scratch heaps
    // (the LUT bank VA, the env-probe VA) host INCARNATION CHAINS - several images over the
    // same guest range, the stale older ones still registered as bad overlaps. The old
    // early-out returned the FIRST registered (oldest) compatible image, so a sampled view
    // could resolve to a stale incarnation whose resource no longer holds the guest's data
    // (the 27 MB LUT image read real content at t=764 and all-zero at t=828 while the guest
    // kept writing the newest incarnation). Collect every compatible image and let the
    // selection below pick the one the guest last wrote.
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            return false;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = False(options & RelaxedOptions::Size) &&
                                     True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, broken_views, native_bgr)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                if (existing.format == info.format) {
                    exact_image_ids.push_back(existing_image_id);
                }
            }
        } else if (IsSubresource(info, existing_image, gpu_addr, options, broken_views,
                                 native_bgr)) {
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            if (existing_image.info.format == info.format) {
                exact_image_ids.push_back(existing_image_id);
            }
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.empty()) {
        return image_id;
    }
    // Exact-format matches win when the caller did not relax the format; among the pool, a
    // pending guest upload (CpuModified) is the freshest guest intent, then the newest GPU
    // modification tick.
    const auto& pool = (!flexible_formats && !exact_image_ids.empty()) ? exact_image_ids
                                                                       : image_ids;
    if (pool.size() <= 1) [[likely]] {
        return pool.front();
    }
    boost::container::small_vector<ImageId, 8> cpu_ids;
    for (const ImageId id : pool) {
        if (True(slot_images[id].flags & ImageFlagBits::CpuModified)) {
            cpu_ids.push_back(id);
        }
    }
    const auto& pick = cpu_ids.empty() ? pool : cpu_ids;
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(pick, image_ids_compare);
}

template <class P>
bool TextureCache<P>::ImageCanRescale(ImageBase& image) {
    if (!image.info.rescaleable) {
        return false;
    }
    if (Settings::values.resolution_info.downscale && !image.info.downscaleable) {
        return false;
    }
    if (True(image.flags & (ImageFlagBits::Rescaled | ImageFlagBits::CheckingRescalable))) {
        return true;
    }
    if (True(image.flags & ImageFlagBits::IsRescalable)) {
        return true;
    }
    image.flags |= ImageFlagBits::CheckingRescalable;
    for (const auto& alias : image.aliased_images) {
        Image& other_image = slot_images[alias.id];
        if (!ImageCanRescale(other_image)) {
            image.flags &= ~ImageFlagBits::CheckingRescalable;
            return false;
        }
    }
    image.flags &= ~ImageFlagBits::CheckingRescalable;
    image.flags |= ImageFlagBits::IsRescalable;
    return true;
}

template <class P>
void TextureCache<P>::InvalidateScale(Image& image) {
    if (image.scale_tick <= frame_tick) {
        image.scale_tick = frame_tick + 1;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);
    for (const ImageViewId image_view_id : image_view_ids) {
        sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        slot_image_views.erase(image_view_id);
    }
    image.image_view_ids.clear();
    image.image_view_infos.clear();
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
        if constexpr (ENABLE_VALIDATION) {
            std::ranges::fill(channel_info.graphics_image_view_ids, CORRUPT_ID);
            std::ranges::fill(channel_info.compute_image_view_ids, CORRUPT_ID);
        }
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
u64 TextureCache<P>::GetScaledImageSizeBytes(const ImageBase& image) {
    const u64 scale_up = static_cast<u64>(Settings::values.resolution_info.up_scale *
                                          Settings::values.resolution_info.up_scale);
    const u64 down_shift = static_cast<u64>(Settings::values.resolution_info.down_shift +
                                            Settings::values.resolution_info.down_shift);
    const u64 image_size_bytes =
        static_cast<u64>(std::max(image.guest_size_bytes, image.unswizzled_size_bytes));
    const u64 tentative_size = (image_size_bytes * scale_up) >> down_shift;
    const u64 fitted_size = Common::AlignUp(tentative_size, 1024);
    return fitted_size;
}

template <class P>
void TextureCache<P>::QueueAsyncDecode(Image& image, ImageId image_id) {
    UNIMPLEMENTED_IF(False(image.flags & ImageFlagBits::Converted));
    LOG_INFO(HW_GPU, "Queuing async texture decode");

    image.flags |= ImageFlagBits::IsDecoding;
    auto decode = std::make_unique<AsyncDecodeContext>();
    auto* decode_ptr = decode.get();
    decode->image_id = image_id;
    async_decodes.push_back(std::move(decode));

    static Common::ScratchBuffer<u8> local_unswizzle_data_buffer;
    local_unswizzle_data_buffer.resize_destructive(image.unswizzled_size_bytes);
    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(
        *gpu_memory, image.gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);

    auto copies = UnswizzleImage(*gpu_memory, image.gpu_addr, image.info, swizzle_data,
                                 local_unswizzle_data_buffer);
    const size_t out_size = MapSizeBytes(image);

    auto func = [out_size, copies, info = image.info,
                 input = std::move(local_unswizzle_data_buffer),
                 async_decode = decode_ptr]() mutable {
        async_decode->decoded_data.resize_destructive(out_size);
        std::span copies_span{copies.data(), copies.size()};
        ConvertImage(input, info, async_decode->decoded_data, copies_span);

        // TODO: Do we need this lock?
        std::unique_lock lock{async_decode->mutex};
        async_decode->copies = std::move(copies);
        async_decode->complete = true;
    };
    texture_decode_worker.QueueWork(std::move(func));
}

template <class P>
void TextureCache<P>::TickAsyncDecode() {
    bool has_uploads{};
    auto i = async_decodes.begin();
    while (i != async_decodes.end()) {
        auto* async_decode = i->get();
        std::unique_lock lock{async_decode->mutex};
        if (!async_decode->complete) {
            ++i;
            continue;
        }
        Image& image = slot_images[async_decode->image_id];
        auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
        std::memcpy(staging.mapped_span.data(), async_decode->decoded_data.data(),
                    async_decode->decoded_data.size());
        image.UploadMemory(staging, async_decode->copies);
        image.flags &= ~ImageFlagBits::IsDecoding;
        has_uploads = true;
        i = async_decodes.erase(i);
    }
    if (has_uploads) {
        runtime.InsertUploadMemoryBarrier();
    }
}

template <class P>
bool TextureCache<P>::ScaleUp(Image& image) {
    const bool has_copy = image.HasScaled();
    const bool rescaled = image.ScaleUp();
    if (!rescaled) {
        return false;
    }
    if (!has_copy) {
        const u64 scaled_size = GetScaledImageSizeBytes(image);
        total_used_memory += scaled_size;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
bool TextureCache<P>::ScaleDown(Image& image) {
    const bool rescaled = image.ScaleDown();
    if (!rescaled) {
        return false;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
ImageId TextureCache<P>::InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                     RelaxedOptions options) {
    std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!cpu_addr) {
        const auto size = CalculateGuestSizeInBytes(info);
        cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr, size);
        if (!cpu_addr) {
            const DAddr fake_addr = ~(1ULL << 40ULL) + virtual_invalid_space;
            virtual_invalid_space += Common::AlignUp(size, 32);
            cpu_addr = std::optional<DAddr>(fake_addr);
        }
    }
    ASSERT_MSG(cpu_addr, "Tried to insert an image to an invalid gpu_addr=0x{:x}", gpu_addr);
    const ImageId image_id = JoinImages(info, gpu_addr, *cpu_addr);
    const Image& image = slot_images[image_id];
    // Using "image.gpu_addr" instead of "gpu_addr" is important because it might be different
    const auto [it, is_new] = image_allocs_table.try_emplace(image.gpu_addr);
    if (is_new) {
        it->second = slot_image_allocs.insert();
    }
    slot_image_allocs[it->second].images.push_back(image_id);
    return image_id;
}

template <class P>
ImageId TextureCache<P>::JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr) {
    // Diagnostic (session 12): trace creation of large (framebuffer-sized) images, to see
    // which GPU/CPU addresses the display buffers are born under.
    if (CalculateGuestSizeInBytes(info) >= 4 * 1024 * 1024) {
        static u32 big_image_logs = 0;
        if (big_image_logs < 40 || big_image_logs % 200 == 0) {
            LOG_WARNING(Render_D3D12,
                        "Big image create: gpu={:#x} cpu={:#x} {}x{} fmt={:#x} bytes={}",
                        static_cast<u64>(gpu_addr), static_cast<u64>(cpu_addr), info.size.width,
                        info.size.height, static_cast<u32>(info.format),
                        CalculateGuestSizeInBytes(info));
        }
        ++big_image_logs;
    }
    ImageInfo new_info = info;
    const size_t size_bytes = CalculateGuestSizeInBytes(new_info);
    const bool broken_views = runtime.HasBrokenTextureViewFormats();
    const bool native_bgr = runtime.HasNativeBgr();
    join_overlap_ids.clear();
    join_overlaps_found.clear();
    join_left_aliased_ids.clear();
    join_right_aliased_ids.clear();
    join_ignore_textures.clear();
    join_bad_overlap_ids.clear();
    join_copies_to_do.clear();
    join_alias_indices.clear();
    const bool this_is_linear = info.type == ImageType::Linear;
    const auto region_check = [&](ImageId overlap_id, ImageBase& overlap) {
        if (True(overlap.flags & ImageFlagBits::Remapped)) {
            join_ignore_textures.insert(overlap_id);
            return;
        }
        const bool overlap_is_linear = overlap.info.type == ImageType::Linear;
        if (this_is_linear != overlap_is_linear) {
            return;
        }
        if (this_is_linear && overlap_is_linear) {
            if (info.pitch == overlap.info.pitch && gpu_addr == overlap.gpu_addr) {
                // Alias linear images with the same pitch
                join_left_aliased_ids.push_back(overlap_id);
            }
            return;
        }
        join_overlaps_found.insert(overlap_id);
        static constexpr bool strict_size = true;
        const std::optional<OverlapResult> solution = ResolveOverlap(
            new_info, gpu_addr, cpu_addr, overlap, strict_size, broken_views, native_bgr);
        if (solution) {
            gpu_addr = solution->gpu_addr;
            cpu_addr = solution->cpu_addr;
            new_info.resources = solution->resources;
            if (new_info.resources.layers > 1) {
                // The union may have raised layers above 1 from an info whose stride was
                // never set (valid when layers == 1); CalculateGuestSizeInBytes needs a
                // real stride once layers > 1, or it computes a guest size of 0.
                new_info.layer_stride = CalculateLayerStride(new_info);
                new_info.maybe_unaligned_layer_stride = CalculateLayerSize(new_info);
            }
            join_overlap_ids.push_back(overlap_id);
            join_copies_to_do.emplace_back(JoinCopy{false, overlap_id});
            return;
        }
        static constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
        const ImageBase new_image_base(new_info, gpu_addr, cpu_addr);
        if (IsSubresource(new_info, overlap, gpu_addr, options, broken_views, native_bgr)) {
            join_left_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else if (IsSubresource(overlap.info, new_image_base, overlap.gpu_addr, options,
                                 broken_views, native_bgr)) {
            join_right_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else if (True(overlap.flags & ImageFlagBits::CpuModified) &&
                   False(overlap.flags & ImageFlagBits::GpuModified)) {
            // Its guest memory was rewritten and it holds no GPU data, so it is stale.
            join_ignore_textures.insert(overlap_id);
        } else {
            join_bad_overlap_ids.push_back(overlap_id);
        }
    };
    ForEachImageInRegion(cpu_addr, size_bytes, region_check);
    const auto region_check_gpu = [&](ImageId overlap_id, ImageBase& overlap) {
        if (!join_overlaps_found.contains(overlap_id)) {
            if (True(overlap.flags & ImageFlagBits::Remapped)) {
                join_ignore_textures.insert(overlap_id);
            }
            if (overlap.gpu_addr == gpu_addr && overlap.guest_size_bytes == size_bytes) {
                join_ignore_textures.insert(overlap_id);
            }
        }
    };
    ForEachSparseImageInRegion(channel_state->gpu_memory.GetID(), gpu_addr, size_bytes,
                               region_check_gpu);

    // TEMP DIAGNOSTIC (S19): which branch did the join take for the LUT fills' two input
    // windows? (0x516170000: the 256x256 probe - its producer draw writes 87% nonzero content
    // (dump 923001) yet the sampled view reads zero at t=760; 0x5d0630800: the 256x4 sample
    // table - uploaded with content (nonzero=2550 at t=221.1) yet zero at t=760.) The counts
    // say whether the member content was queued for copying (solution/ignore), aliased, or
    // dropped on the bad-overlap floor. Bounded 24.
    const auto s19_join_window = [](u64 gpu) {
        return (gpu >= 0x516170000ull && gpu < 0x516170000ull + 0x408000ull) ||
               (gpu >= 0x5d0630000ull && gpu < 0x5d0631000ull) ||
               // S22 cycle 21: the lighting-LUT bank. Its join creates the 27 MB image right
               // after the six 256x256 fills, then RefreshContents uploads ZERO guest bytes
               // over it (t=761.302 in 1.0.320). The branch counts say whether the filled
               // members were queued for copying, aliased, or dropped on the bad-overlap floor.
               (gpu >= 0x5e9000000ull && gpu < 0x5ee7b4000ull);
    };
    if (s19_join_window(static_cast<u64>(gpu_addr))) {
        static u32 s19_join_logs = 0;
        if (s19_join_logs < 48) {
            ++s19_join_logs;
            LOG_WARNING(Render_D3D12,
                        "S19 join branch: gpu={:#x} {}x{} fmt={:#x} overlaps={} leftalias={} "
                        "rightalias={} ignore={} bad={} copies={}",
                        static_cast<u64>(gpu_addr), info.size.width, info.size.height,
                        static_cast<u32>(info.format), join_overlap_ids.size(),
                        join_left_aliased_ids.size(), join_right_aliased_ids.size(),
                        join_ignore_textures.size(), join_bad_overlap_ids.size(),
                        join_copies_to_do.size());
        }
    }

    bool can_rescale = info.rescaleable;
    bool any_rescaled = false;
    for (const auto& copy : join_copies_to_do) {
        if (!can_rescale) {
            break;
        }
        Image& sibling = slot_images[copy.id];
        can_rescale &= ImageCanRescale(sibling);
        any_rescaled |= True(sibling.flags & ImageFlagBits::Rescaled);
    }

    can_rescale &= any_rescaled;

    if (can_rescale) {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleUp(sibling);
        }
    } else {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleDown(sibling);
        }
    }

    const ImageId new_image_id = slot_images.insert(runtime, new_info, gpu_addr, cpu_addr);
    Image& new_image = slot_images[new_image_id];

    if (!gpu_memory->IsContinuousRange(new_image.gpu_addr, new_image.guest_size_bytes) &&
        new_info.is_sparse) {
        new_image.flags |= ImageFlagBits::Sparse;
    }

    for (const ImageId overlap_id : join_ignore_textures) {
        Image& overlap = slot_images[overlap_id];
        if (s19_join_window(static_cast<u64>(overlap.gpu_addr)) ||
            s19_join_window(static_cast<u64>(new_image.gpu_addr))) {
            static u32 s19_ignore_logs = 0;
            if (s19_ignore_logs < 24) {
                ++s19_ignore_logs;
                LOG_WARNING(Render_D3D12,
                            "S19 join ignore: new={} gpu={:#x} overlap={} gpu={:#x} {}x{} "
                            "gpu_modified={} flags={:#x}",
                            new_image_id.index, static_cast<u64>(new_image.gpu_addr),
                            overlap_id.index, static_cast<u64>(overlap.gpu_addr),
                            overlap.info.size.width, overlap.info.size.height,
                            True(overlap.flags & ImageFlagBits::GpuModified) ? 1 : 0,
                            static_cast<u32>(overlap.flags));
            }
        }
        if (True(overlap.flags & ImageFlagBits::GpuModified)) {
            // For ignored textures that are GPU modified, try to preserve the data
            // by copying it to the new image if possible, otherwise mark as modified
            const auto base_opt = new_image.TryFindBase(overlap.gpu_addr);
            if (base_opt.has_value() && overlap.info.format == new_info.format &&
                overlap.info.num_samples == new_info.num_samples) {
                // Formats match, copy the data
                new_image.flags |= ImageFlagBits::GpuModified;
                const auto& resolution = Settings::values.resolution_info;
                const u32 up_scale = can_rescale ? resolution.up_scale : 1;
                const u32 down_shift = can_rescale ? resolution.down_shift : 0;
                auto copies = MakeShrinkImageCopies(new_info, overlap.info, base_opt.value(),
                                                    up_scale, down_shift);
                runtime.CopyImage(new_image, overlap, std::move(copies));
                new_image.modification_tick =
                    std::max(new_image.modification_tick, overlap.modification_tick);
            } else {
                // Formats don't match or can't find base, just mark as modified
                new_image.flags |= ImageFlagBits::GpuModified;
                new_image.modification_tick =
                    std::max(new_image.modification_tick, overlap.modification_tick);
            }
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, overlap_id);
        }
        UnregisterImage(overlap_id);
        DeleteImage(overlap_id);
    }

    // TODO: Only upload what we need
    RefreshContents(new_image, new_image_id);

    if (can_rescale) {
        ScaleUp(new_image);
    } else {
        ScaleDown(new_image);
    }

    std::ranges::sort(join_copies_to_do, [this](const JoinCopy& lhs, const JoinCopy& rhs) {
        const ImageBase& lhs_image = slot_images[lhs.id];
        const ImageBase& rhs_image = slot_images[rhs.id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });

    ImageBase& new_image_base = new_image;
    for (const ImageId aliased_id : join_right_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(new_image_base, aliased, new_image_id, aliased_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_left_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(aliased, new_image_base, aliased_id, new_image_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_bad_overlap_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        // TEMP FIX (S19, the lighting-LUT bank): an overlap the join could not classify as a
        // subresource in either direction, but which holds GPU-authored content, is the
        // producer of the new image's content. The six 256x256 LUT fill draws render into
        // per-slot images at the bank VAs (0x5eb340000 + k*0x56000); the joined 27 MB array
        // image is created afterwards, uploads ZERO guest bytes (the LUT is GPU-authored) and
        // previously never received that content - so every sampled 256x256 sub-view read zero
        // and the scene rendered unlit (black silhouettes). Inherit the content exactly like
        // the ignore-textures path above does, then retire the overlap so one image owns each
        // guest range again. When the base or the format does not line up, keep the old
        // BadOverlap linking (no copy, no delete) - the behaviour before this fix.
        // Deliberately bounded to the LUT bank window this cycle (the same range the S17
        // producer tracers use): the dynamic-res family has the same shape but its joins are
        // load-bearing for the whole scene chain, so generalising waits for its own A/B.
        const bool s19_lut_bank_overlap = aliased.gpu_addr >= 0x5e9000000ull &&
                                          aliased.gpu_addr < 0x5ee7b4000ull;
        if (s19_lut_bank_overlap && True(aliased.flags & ImageFlagBits::GpuModified) &&
            aliased.info.num_samples == 1 &&
            aliased.guest_size_bytes < new_image.guest_size_bytes) {
            const auto s19_inherit_base = new_image.TryFindBase(aliased.gpu_addr);
            if (s19_inherit_base.has_value() && aliased.info.format == new_info.format) {
                static u32 s19_inherit_logs = 0;
                const auto& resolution = Settings::values.resolution_info;
                const u32 up_scale = can_rescale ? resolution.up_scale : 1;
                const u32 down_shift = can_rescale ? resolution.down_shift : 0;
                auto copies = MakeShrinkImageCopies(new_info, aliased.info,
                                                    s19_inherit_base.value(), up_scale, down_shift);
                const size_t s19_num_copies = copies.size();
                new_image.flags |= ImageFlagBits::GpuModified;
                Image& s19_inherit_src = slot_images[aliased_id];
                runtime.CopyImage(new_image, s19_inherit_src, std::move(copies));
                new_image.modification_tick =
                    std::max(new_image.modification_tick, aliased.modification_tick);
                if (s19_inherit_logs < 24) {
                    ++s19_inherit_logs;
                    LOG_WARNING(Render_D3D12,
                                "S19 join inherit #{}: overlap={} gpu={:#x} {}x{} fmt={:#x} -> "
                                "new={} gpu={:#x} copies={}",
                                s19_inherit_logs, aliased_id.index,
                                static_cast<u64>(aliased.gpu_addr), aliased.info.size.width,
                                aliased.info.size.height, static_cast<u32>(aliased.info.format),
                                new_image_id.index, static_cast<u64>(new_image.gpu_addr),
                                s19_num_copies);
                }
                if (True(aliased.flags & ImageFlagBits::Tracked)) {
                    UntrackImage(aliased, aliased_id);
                }
                UnregisterImage(aliased_id);
                DeleteImage(aliased_id);
                continue;
            }
        }
        aliased.overlapping_images.push_back(new_image_id);
        new_image.overlapping_images.push_back(aliased_id);
        if (aliased.info.resources.levels == 1 && aliased.info.block.depth == 0 &&
            aliased.overlapping_images.size() > 1) {
            aliased.flags |= ImageFlagBits::BadOverlap;
        }
        if (new_image.info.resources.levels == 1 && new_image.info.block.depth == 0 &&
            new_image.overlapping_images.size() > 1) {
            new_image.flags |= ImageFlagBits::BadOverlap;
        }
    }

    for (const auto& copy_object : join_copies_to_do) {
        Image& overlap = slot_images[copy_object.id];
        // S22 cycle 22: the bank's own copy log. The shared S19 budget is exhausted long
        // before the bank join (t~790), so the one datum that explains the zero LUT - whether
        // the filled members were GpuModified when the copy was queued - has never been seen.
        // The zero RefreshContents upload happens BEFORE this loop (line ~1998), so the final
        // content is whatever these copies produce.
        {
            const u64 s22_new_gpu = static_cast<u64>(new_image.gpu_addr);
            const u64 s22_ov_gpu = static_cast<u64>(overlap.gpu_addr);
            const bool s22_bank =
                (s22_ov_gpu >= 0x5e9000000ull && s22_ov_gpu < 0x5ee7b4000ull) ||
                (s22_new_gpu >= 0x5e9000000ull && s22_new_gpu < 0x5ee7b4000ull);
            static u32 s22_bank_copy_logs = 0;
            if (s22_bank && s22_bank_copy_logs < 24) {
                ++s22_bank_copy_logs;
                LOG_WARNING(Render_D3D12,
                            "S22 bank join copy #{}: new={} gpu={:#x} overlap={} gpu={:#x} {}x{} "
                            "alias={} gpu_modified={} flags={:#x}",
                            s22_bank_copy_logs, new_image_id.index, s22_new_gpu,
                            copy_object.id.index, s22_ov_gpu, overlap.info.size.width,
                            overlap.info.size.height, copy_object.is_alias ? 1 : 0,
                            True(overlap.flags & ImageFlagBits::GpuModified) ? 1 : 0,
                            static_cast<u32>(overlap.flags));
            }
        }
        if (s19_join_window(static_cast<u64>(overlap.gpu_addr)) ||
            s19_join_window(static_cast<u64>(new_image.gpu_addr))) {
            static u32 s19_copy_logs = 0;
            if (s19_copy_logs < 24) {
                ++s19_copy_logs;
                LOG_WARNING(Render_D3D12,
                            "S19 join copy: new={} gpu={:#x} overlap={} gpu={:#x} {}x{} alias={} "
                            "gpu_modified={} flags={:#x}",
                            new_image_id.index, static_cast<u64>(new_image.gpu_addr),
                            copy_object.id.index, static_cast<u64>(overlap.gpu_addr),
                            overlap.info.size.width, overlap.info.size.height,
                            copy_object.is_alias ? 1 : 0,
                            True(overlap.flags & ImageFlagBits::GpuModified) ? 1 : 0,
                            static_cast<u32>(overlap.flags));
            }
        }
        if (copy_object.is_alias) {
            // FIX (S15/S16): preserve GPU-authored alias content. IsSafeDownload() is a
            // guest-download safety predicate; using it here skipped the device-local
            // GPU->GPU preservation copy for exactly the images that need it (GpuModified),
            // leaving the new joined image zero (the lighting-LUT bank was lost this way).
            // Copy iff the overlap holds GPU-authored content; CPU-authored content is
            // re-uploaded by RefreshContents below. MSAA stays on the old skip path
            // (CopyImage does not support it here).
            if (False(overlap.flags & ImageFlagBits::GpuModified) || overlap.info.num_samples > 1) {
                continue;
            }
            const auto alias_pointer = join_alias_indices.find(copy_object.id);
            if (alias_pointer == join_alias_indices.end()) {
                continue;
            }
            const AliasedImage& aliased = new_image.aliased_images[alias_pointer->second];
            CopyImage(new_image_id, aliased.id, aliased.copies);
            new_image.modification_tick = overlap.modification_tick;
            continue;
        }
        if (True(overlap.flags & ImageFlagBits::GpuModified)) {
            new_image.flags |= ImageFlagBits::GpuModified;
            const auto& resolution = Settings::values.resolution_info;
            const SubresourceBase base = new_image.TryFindBase(overlap.gpu_addr).value();
            const u32 up_scale = can_rescale ? resolution.up_scale : 1;
            const u32 down_shift = can_rescale ? resolution.down_shift : 0;
            auto copies = MakeShrinkImageCopies(new_info, overlap.info, base, up_scale, down_shift);
            // S19: the copy that was supposed to carry the probe member's 87%-nonzero content
            // (t=222.6) into the new image yields a ZERO sampled view at t=749 - log the exact
            // copy boxes so a bad offset/extent is distinguishable from a later overwrite.
            if (s19_join_window(static_cast<u64>(overlap.gpu_addr)) ||
                s19_join_window(static_cast<u64>(new_image.gpu_addr))) {
                static u32 s19_boxes_logs = 0;
                if (s19_boxes_logs < 12 && !copies.empty()) {
                    ++s19_boxes_logs;
                    const ImageCopy& c0 = copies.front();
                    LOG_WARNING(Render_D3D12,
                                "S19 join boxes: overlap={} copies={} c0=(src=({},{},{}) "
                                "dst=({},{},{}) ext={}x{}x{} src_lvl={} src_lay={} dst_lvl={} "
                                "dst_lay={} base_lvl={} base_lay={})",
                                copy_object.id.index, copies.size(), c0.src_offset.x,
                                c0.src_offset.y, c0.src_offset.z, c0.dst_offset.x, c0.dst_offset.y,
                                c0.dst_offset.z, c0.extent.width, c0.extent.height, c0.extent.depth,
                                c0.src_subresource.base_level, c0.src_subresource.base_layer,
                                c0.dst_subresource.base_level, c0.dst_subresource.base_layer,
                                base.level, base.layer);
                }
            }
            if (overlap.info.num_samples != new_image.info.num_samples) {
                runtime.CopyImageMSAA(new_image, overlap, std::move(copies));
            } else {
                runtime.CopyImage(new_image, overlap, std::move(copies));
            }
            new_image.modification_tick = overlap.modification_tick;
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, copy_object.id);
        }
        UnregisterImage(copy_object.id);
        DeleteImage(copy_object.id);
    }

    // S20 fix: a join that creates a new image over a guest range where other images already
    // live (subresources, bad overlaps, ignored stale siblings) makes every cached descriptor
    // -> view mapping of those images ambiguous: the newest image now holds the guest's
    // current content at that VA, while a sample that reuses the cached mapping keeps reading
    // the stale incarnation (the bank LUT image held real content at t=764 and read all-zero
    // at t=828 through one unresolved view). Purge the affected images' cached views and force
    // every descriptor to re-resolve on its next use, so a sample binds the image the guest
    // last wrote. Same cleanup DeleteImage performs for the views it retires.
    {
        boost::container::small_vector<ImageViewId, 32> s20_stale_views;
        const auto s20_collect_views = [this, &s20_stale_views](ImageId id) {
            const std::span<const ImageViewId> view_ids = slot_images[id].image_view_ids;
            s20_stale_views.insert(s20_stale_views.end(), view_ids.begin(), view_ids.end());
        };
        for (const ImageId id : join_overlap_ids) {
            s20_collect_views(id);
        }
        for (const ImageId id : join_left_aliased_ids) {
            s20_collect_views(id);
        }
        for (const ImageId id : join_right_aliased_ids) {
            s20_collect_views(id);
        }
        for (const ImageId id : join_ignore_textures) {
            s20_collect_views(id);
        }
        for (const ImageId id : join_bad_overlap_ids) {
            s20_collect_views(id);
        }
        for (const auto& join_copy : join_copies_to_do) {
            s20_collect_views(join_copy.id);
        }
        if (!s20_stale_views.empty()) {
            RemoveImageViewReferences(s20_stale_views);
            InvalidateImageTables();
        }
    }
    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
std::optional<typename TextureCache<P>::BlitImages> TextureCache<P>::GetBlitImages(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
    const Tegra::Engines::Fermi2D::Config& copy) {

    static constexpr auto FIND_OPTIONS = RelaxedOptions::Samples;
    const GPUVAddr dst_addr = dst.Address();
    const GPUVAddr src_addr = src.Address();
    ImageInfo dst_info(dst);
    ImageInfo src_info(src);
    const bool can_be_depth_blit =
        dst_info.format == src_info.format && copy.filter == Tegra::Engines::Fermi2D::Filter::Point;
    ImageId dst_id;
    ImageId src_id;
    RelaxedOptions try_options = FIND_OPTIONS;
    if (can_be_depth_blit) {
        try_options |= RelaxedOptions::Format;
    }
    do {
        has_deleted_images = false;
        src_id = FindImage(src_info, src_addr, try_options);
        dst_id = FindImage(dst_info, dst_addr, try_options);
        if (!copy.must_accelerate) {
            do {
                if (!src_id && !dst_id) {
                    return std::nullopt;
                }
                if (src_id && True(slot_images[src_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                if (dst_id && True(slot_images[dst_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                return std::nullopt;
            } while (false);
        }
        const ImageBase* const src_image = src_id ? &slot_images[src_id] : nullptr;
        if (src_image && src_image->info.num_samples > 1) {
            RelaxedOptions find_options{FIND_OPTIONS | RelaxedOptions::ForceBrokenViews};
            src_id = FindOrInsertImage(src_info, src_addr, find_options);
            dst_id = FindOrInsertImage(dst_info, dst_addr, find_options);
            if (has_deleted_images) {
                continue;
            }
            break;
        }
        if (can_be_depth_blit) {
            const ImageBase* const dst_image = dst_id ? &slot_images[dst_id] : nullptr;
            DeduceBlitImages(dst_info, src_info, dst_image, src_image);
            if (GetFormatType(dst_info.format) != GetFormatType(src_info.format)) {
                continue;
            }
        }
        if (!src_id) {
            src_id = InsertImage(src_info, src_addr, RelaxedOptions{});
        }
        if (!dst_id) {
            dst_id = InsertImage(dst_info, dst_addr, RelaxedOptions{});
        }
    } while (has_deleted_images);
    const ImageBase& src_image = slot_images[src_id];
    const ImageBase& dst_image = slot_images[dst_id];
    const bool native_bgr = runtime.HasNativeBgr();
    if (GetFormatType(dst_info.format) != GetFormatType(dst_image.info.format) ||
        GetFormatType(src_info.format) != GetFormatType(src_image.info.format) ||
        !VideoCore::Surface::IsViewCompatible(dst_info.format, dst_image.info.format, false,
                                              native_bgr) ||
        !VideoCore::Surface::IsViewCompatible(src_info.format, src_image.info.format, false,
                                              native_bgr)) {
        // Make sure the images match the expected format.
        do {
            has_deleted_images = false;
            src_id = FindOrInsertImage(src_info, src_addr, RelaxedOptions{});
            dst_id = FindOrInsertImage(dst_info, dst_addr, RelaxedOptions{});
        } while (has_deleted_images);
    }
    return {BlitImages{
        .dst_id = dst_id,
        .src_id = src_id,
        .dst_format = dst_info.format,
        .src_format = src_info.format,
    }};
}

template <class P>
ImageId TextureCache<P>::FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr) {
    std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!cpu_addr) {
        cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
        if (!cpu_addr) {
            return ImageId{};
        }
    }
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            return false;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, false, true)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return true;
            }
        } else if (IsSubCopy(info, existing_image, gpu_addr)) {
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            return true;
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.size() <= 1) [[likely]] {
        return image_id;
    }
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(image_ids, image_ids_compare);
}

template <class P>
std::pair<u32, u32> TextureCache<P>::PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                     bool mark_as_modified) {
    const auto& image = slot_images[dst_id];
    const auto base = image.TryFindBase(base_addr);
    PrepareImage(dst_id, mark_as_modified, false);
    const auto& new_image = slot_images[dst_id];
    lru_cache.Touch(new_image.lru_index, frame_tick);
    return std::make_pair(base->level, base->layer);
}

template <class P>
SamplerId TextureCache<P>::FindSampler(const TSCEntry& config) {
    if (std::ranges::all_of(config.raw, [](u64 value) { return value == 0; })) {
        return NULL_SAMPLER_ID;
    }
    const auto [pair, is_new] = channel_state->samplers.try_emplace(config);
    if (is_new) {
        pair->second = slot_samplers.insert(runtime, config);
    }
    return pair->second;
}

template <class P>
ImageViewId TextureCache<P>::FindColorBuffer(size_t index) {
    const auto& regs = maxwell3d->regs;
    if (index >= regs.rt_control.count) {
        return ImageViewId{};
    }
    const auto& rt = regs.rt[index];
    const GPUVAddr gpu_addr = rt.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    if (rt.format == Tegra::RenderTargetFormat::NONE) {
        return ImageViewId{};
    }
    const ImageInfo info(regs.rt[index], regs.anti_alias_samples_mode);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindDepthBuffer() {
    const auto& regs = maxwell3d->regs;
    if (!regs.zeta_enable) {
        return ImageViewId{};
    }
    const GPUVAddr gpu_addr = regs.zeta.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    const ImageInfo info(regs.zeta, regs.zeta_size, regs.anti_alias_samples_mode);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr) {
    ImageId image_id{};
    bool delete_state = has_deleted_images;
    do {
        has_deleted_images = false;
        // S23: the render target keeps the strict size match (it defines the surface). Only the
        // sampled view is relaxed (see CreateImageView) - letting a render target attach to a
        // surface created at a different extent let a recycled, never-written resource reach the
        // sampler (1.0.344 run: `rmax=65024 lmean=442` at t=2050, then the scene pass stopped).
        image_id = FindOrInsertImage(info, gpu_addr);
        delete_state |= has_deleted_images;
    } while (has_deleted_images);
    has_deleted_images = delete_state;
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    Image& image = slot_images[image_id];
    const ImageViewType view_type = RenderTargetImageViewType(info);
    SubresourceBase base;
    if (image.info.type == ImageType::Linear) {
        base = SubresourceBase{.level = 0, .layer = 0};
    } else {
        base = image.TryFindBase(gpu_addr).value();
    }
    const s32 layers = image.info.type == ImageType::e3D ? info.size.depth : info.resources.layers;
    const SubresourceRange range{
        .base = base,
        .extent = {.levels = 1, .layers = layers},
    };
    return FindOrEmplaceImageView(image_id, ImageViewInfo(view_type, info.format, range));
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 32> images;
    boost::container::small_vector<ImageMapId, 32> maps;
    ForEachCPUPage(cpu_addr, size, [this, &images, &maps, cpu_addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const ImageMapId map_id : it->second) {
            ImageMapView& map = slot_map_views[map_id];
            if (map.picked) {
                continue;
            }
            if (!map.Overlaps(cpu_addr, size)) {
                continue;
            }
            map.picked = true;
            maps.push_back(map_id);
            Image& image = slot_images[map.image_id];
            if (True(image.flags & ImageFlagBits::Picked)) {
                continue;
            }
            image.flags |= ImageFlagBits::Picked;
            images.push_back(map.image_id);
            if constexpr (BOOL_BREAK) {
                if (func(map.image_id, image)) {
                    return true;
                }
            } else {
                func(map.image_id, image);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
    for (const ImageMapId map_id : maps) {
        slot_map_views[map_id].picked = false;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                              Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& gpu_page_table = gpu_page_table_storage[*storage_id * 2];
    ForEachGPUPage(gpu_addr, size,
                   [this, &gpu_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = gpu_page_table.find(page);
                       if (it == gpu_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                                 Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& sparse_page_table = gpu_page_table_storage[*storage_id * 2 + 1];
    ForEachGPUPage(gpu_addr, size,
                   [this, &sparse_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = sparse_page_table.find(page);
                       if (it == sparse_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseSegment(ImageBase& image, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, GPUVAddr, DAddr, size_t>::type;
    static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
    const auto segments = gpu_memory->GetSubmappedRange(image.gpu_addr, image.guest_size_bytes);
    for (const auto& [gpu_addr, size] : segments) {
        std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        ASSERT(cpu_addr);
        if constexpr (RETURNS_BOOL) {
            if (func(gpu_addr, *cpu_addr, size)) {
                break;
            }
        } else {
            func(gpu_addr, *cpu_addr, size);
        }
    }
}

template <class P>
ImageViewId TextureCache<P>::FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info) {
    Image& image = slot_images[image_id];
    if (const ImageViewId image_view_id = image.FindView(info); image_view_id) {
        return image_view_id;
    }
    const ImageViewId image_view_id =
        slot_image_views.insert(runtime, info, image_id, image, slot_images);
    image.InsertView(info, image_view_id);
    return image_view_id;
}

template <class P>
void TextureCache<P>::RegisterImage(ImageId image_id) {
    ImageBase& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    const auto storage_id = getStorageID(channel_state->gpu_memory.GetID());
    if (storage_id) {
        image.registration_storage_id = *storage_id;
    }
    u64 tentative_size = std::max(image.guest_size_bytes, image.unswizzled_size_bytes);
    if ((IsPixelFormatASTC(image.info.format) &&
         True(image.flags & ImageFlagBits::AcceleratedUpload)) ||
        True(image.flags & ImageFlagBits::Converted)) {
        tentative_size = TranscodedAstcSize(tentative_size, image.info.format);
    }
    const u64 aligned_size = Common::AlignUp(tentative_size, 1024);
    total_used_memory += aligned_size;

    // FIXED: VRAM leak prevention - Track texture statistics
    ++texture_count;
    const bool is_sparse = True(image.flags & ImageFlagBits::Sparse);
    const bool is_large = aligned_size >= LARGE_TEXTURE_THRESHOLD;

    if (is_sparse) {
        sparse_texture_memory += aligned_size;
        ++sparse_texture_count;
    }
    if (is_large) {
        large_texture_memory += aligned_size;
    }

    image.lru_index = lru_cache.Insert(image_id, frame_tick);

    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->gpu_page_table)[page].push_back(image_id);
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        auto map_id =
            slot_map_views.insert(image.gpu_addr, image.cpu_addr, image.guest_size_bytes, image_id);
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes,
                       [this, map_id](u64 page) { page_table[page].push_back(map_id); });
        image.map_view_id = map_id;
        return;
    }
    boost::container::small_vector<ImageViewId, 16> sparse_maps;
    ForEachSparseSegment(
        image, [this, image_id, &sparse_maps](GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
            auto map_id = slot_map_views.insert(gpu_addr, cpu_addr, size, image_id);
            ForEachCPUPage(cpu_addr, size,
                           [this, map_id](u64 page) { page_table[page].push_back(map_id); });
            sparse_maps.push_back(map_id);
        });
    sparse_views.emplace(image_id, std::move(sparse_maps));
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->sparse_page_table)[page].push_back(image_id);
    });
}

template <class P>
void TextureCache<P>::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already registered image");
    image.flags &= ~ImageFlagBits::Registered;
    image.flags &= ~ImageFlagBits::BadOverlap;
    lru_cache.Free(image.lru_index);
    // Use the page tables this image was registered in (channel may have changed since then).
    TextureCacheGPUMap* gpu_table = channel_state->gpu_page_table;
    TextureCacheGPUMap* sparse_table = channel_state->sparse_page_table;
    if (image.registration_storage_id &&
        *image.registration_storage_id * 2 + 1 < gpu_page_table_storage.size()) {
        gpu_table = &gpu_page_table_storage[*image.registration_storage_id * 2];
        sparse_table = &gpu_page_table_storage[*image.registration_storage_id * 2 + 1];
    }
    image.registration_storage_id = std::nullopt;
    const auto& clear_page_table =
        [image_id](u64 page,
                   std::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>&
                       selected_page_table) {
            const auto page_it = selected_page_table.find(page);
            if (page_it == selected_page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page=0x{:x}", page << CITRON_PAGEBITS);
                return;
            }
            std::vector<ImageId>& image_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_ids, image_id);
            if (vector_it == image_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}",
                           page << CITRON_PAGEBITS);
                return;
            }
            image_ids.erase(vector_it);
        };
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [&clear_page_table, gpu_table](u64 page) {
        clear_page_table(page, *gpu_table);
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        const auto map_id = image.map_view_id;
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes, [this, map_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page=0x{:x}", page << CITRON_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_map_ids, map_id);
            if (vector_it == image_map_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}",
                           page << CITRON_PAGEBITS);
                return;
            }
            image_map_ids.erase(vector_it);
        });
        slot_map_views.erase(map_id);
        return;
    }
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [&clear_page_table, sparse_table](u64 page) {
        clear_page_table(page, *sparse_table);
    });
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map_range = slot_map_views[map_view_id];
        const DAddr cpu_addr = map_range.cpu_addr;
        const std::size_t size = map_range.size;
        ForEachCPUPage(cpu_addr, size, [this, image_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page=0x{:x}", page << CITRON_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            auto vector_it = image_map_ids.begin();
            while (vector_it != image_map_ids.end()) {
                ImageMapView& map = slot_map_views[*vector_it];
                if (map.image_id != image_id) {
                    vector_it++;
                    continue;
                }
                if (!map.picked) {
                    map.picked = true;
                }
                vector_it = image_map_ids.erase(vector_it);
            }
        });
        slot_map_views.erase(map_view_id);
    }
    sparse_views.erase(it);
}

template <class P>
void TextureCache<P>::TrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(False(image.flags & ImageFlagBits::Tracked));
    image.flags |= ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, 1);
        }
        return;
    }
    if (True(image.flags & ImageFlagBits::Registered)) {
        auto it = sparse_views.find(image_id);
        ASSERT(it != sparse_views.end());
        auto& sparse_maps = it->second;
        for (auto& map_view_id : sparse_maps) {
            const auto& map = slot_map_views[map_view_id];
            const DAddr cpu_addr = map.cpu_addr;
            const std::size_t size = map.size;
            device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
        }
        return;
    }
    ForEachSparseSegment(image,
                         [this]([[maybe_unused]] GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
                             device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
                         });
}

template <class P>
void TextureCache<P>::UntrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(True(image.flags & ImageFlagBits::Tracked));
    image.flags &= ~ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, -1);
        }
        return;
    }
    ASSERT(True(image.flags & ImageFlagBits::Registered));
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map = slot_map_views[map_view_id];
        const DAddr cpu_addr = map.cpu_addr;
        const std::size_t size = map.size;
        device_memory.UpdatePagesCachedCount(cpu_addr, size, -1);
    }
}

template <class P>
void TextureCache<P>::DeleteImage(ImageId image_id, bool immediate_delete,
                                  std::vector<ImageViewId>* deferred_view_refs) {
    ImageBase& image = slot_images[image_id];
    if (image.HasScaled()) {
        const u64 scaled_size = GetScaledImageSizeBytes(image);
        ASSERT(total_used_memory >= scaled_size);
        total_used_memory -= std::min(total_used_memory, scaled_size);
    }
    u64 tentative_size = std::max(image.guest_size_bytes, image.unswizzled_size_bytes);
    if ((IsPixelFormatASTC(image.info.format) &&
         True(image.flags & ImageFlagBits::AcceleratedUpload)) ||
        True(image.flags & ImageFlagBits::Converted)) {
        tentative_size = TranscodedAstcSize(tentative_size, image.info.format);
    }
    const u64 aligned_size = Common::AlignUp(tentative_size, 1024);
    ASSERT(total_used_memory >= aligned_size);
    total_used_memory -= std::min(total_used_memory, aligned_size);

    if (texture_count > 0) {
        --texture_count;
    }
    if (True(image.flags & ImageFlagBits::Sparse) && sparse_texture_count > 0) {
        --sparse_texture_count;
        sparse_texture_memory -= std::min(sparse_texture_memory, aligned_size);
    }
    if (aligned_size >= LARGE_TEXTURE_THRESHOLD) {
        large_texture_memory -= std::min(large_texture_memory, aligned_size);
    }
    const GPUVAddr gpu_addr = image.gpu_addr;
    const auto alloc_it = image_allocs_table.find(gpu_addr);
    if (alloc_it == image_allocs_table.end()) {
        ASSERT_MSG(false, "Trying to delete an image alloc that does not exist in address 0x{:x}",
                   gpu_addr);
        return;
    }
    const ImageAllocId alloc_id = alloc_it->second;
    std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
    const auto alloc_image_it = std::ranges::find(alloc_images, image_id);
    if (alloc_image_it == alloc_images.end()) {
        ASSERT_MSG(false, "Trying to delete an image that does not exist");
        return;
    }
    ASSERT_MSG(False(image.flags & ImageFlagBits::Tracked), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Mark render targets as dirty
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    if (deferred_view_refs) {
        deferred_view_refs->insert(deferred_view_refs->end(), image_view_ids.begin(),
                                   image_view_ids.end());
    } else {
        RemoveImageViewReferences(image_view_ids);
        RemoveFramebuffers(image_view_ids);
    }

    for (const AliasedImage& alias : image.aliased_images) {
        ImageBase& other_image = slot_images[alias.id];
        [[maybe_unused]] const size_t num_removed_aliases =
            std::erase_if(other_image.aliased_images, [image_id](const AliasedImage& other_alias) {
                return other_alias.id == image_id;
            });
        other_image.CheckAliasState();
        ASSERT_MSG(num_removed_aliases == 1, "Invalid number of removed aliases: {}",
                   num_removed_aliases);
    }
    for (const ImageId overlap_id : image.overlapping_images) {
        ImageBase& other_image = slot_images[overlap_id];
        [[maybe_unused]] const size_t num_removed_overlaps = std::erase_if(
            other_image.overlapping_images,
            [image_id](const ImageId other_overlap_id) { return other_overlap_id == image_id; });
        other_image.CheckBadOverlapState();
        ASSERT_MSG(num_removed_overlaps == 1, "Invalid number of removed overlapps: {}",
                   num_removed_overlaps);
    }
    for (const ImageViewId image_view_id : image_view_ids) {
        if (!immediate_delete) {
            sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        }
        slot_image_views.erase(image_view_id);
    }
    if (!immediate_delete) {
        sentenced_images.Push(std::move(slot_images[image_id]));
    }
    slot_images.erase(image_id);

    alloc_images.erase(alloc_image_it);
    if (alloc_images.empty()) {
        image_allocs_table.erase(alloc_it);
    }
    has_deleted_images = true;
    if (deferred_view_refs) {
        return;
    }
    InvalidateImageTables();
}

template <class P>
void TextureCache<P>::FinishDeferredDeletes(std::span<const ImageViewId> removed_views) {
    if (!removed_views.empty()) {
        RemoveImageViewReferences(removed_views);
        RemoveFramebuffers(removed_views);
    }
    InvalidateImageTables();
}

template <class P>
void TextureCache<P>::InvalidateImageTables() {
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
        if constexpr (ENABLE_VALIDATION) {
            std::ranges::fill(channel_info.graphics_image_view_ids, CORRUPT_ID);
            std::ranges::fill(channel_info.compute_image_view_ids, CORRUPT_ID);
        }
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
void TextureCache<P>::RemoveImageViewReferences(std::span<const ImageViewId> removed_views) {
    std::vector<ImageViewId> sorted(removed_views.begin(), removed_views.end());
    std::ranges::sort(sorted);
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
        auto it = channel_info.image_views.begin();
        while (it != channel_info.image_views.end()) {
            if (std::ranges::binary_search(sorted, it->second)) {
                it = channel_info.image_views.erase(it);
            } else {
                ++it;
            }
        }
    }
}

template <class P>
void TextureCache<P>::RemoveFramebuffers(std::span<const ImageViewId> removed_views) {
    std::vector<ImageViewId> sorted(removed_views.begin(), removed_views.end());
    std::ranges::sort(sorted);
    const auto removed = [&sorted](ImageViewId id) { return std::ranges::binary_search(sorted, id); };
    auto it = framebuffers.begin();
    while (it != framebuffers.end()) {
        if (std::ranges::any_of(it->first.color_buffer_ids, removed) ||
            removed(it->first.depth_buffer_id)) {
            auto framebuffer_id = it->second;
            ASSERT(framebuffer_id);
            sentenced_framebuffers.Push(std::move(slot_framebuffers[framebuffer_id]));
            it = framebuffers.erase(it);
        } else {
            ++it;
        }
    }
}

template <class P>
void TextureCache<P>::MarkModification(ImageBase& image) noexcept {
    image.flags |= ImageFlagBits::GpuModified;
    image.modification_tick = ++modification_tick;
}

template <class P>
void TextureCache<P>::SynchronizeAliases(ImageId image_id) {
    boost::container::small_vector<const AliasedImage*, 8> aliased_images;
    Image& image = slot_images[image_id];
    // TEMP DIAGNOSTIC (S17): the 27 MB LUT bank (cpu 0x29426000) is a family of small 512-byte
    // images flagged Alias|GpuModified sharing memory with the 27 MB image the shaders sample.
    // The sampled image only receives the LUT when this propagation runs, and the population is
    // intermittent run-to-run, so log exactly which of the three failure modes applies:
    //   aliases=0                     -> propagation was never registered (JoinImages missed it)
    //   aliases>0 but newer=0 for all -> registered, but the tick comparison never admits a copy
    //   newer=1 but copies=0          -> admitted, but the copy list is empty (nothing to copy)
    // Bounded to 24 lines; only fires for images overlapping the bank.
    {
        static u32 s17_alias_logs = 0;
        if (s17_alias_logs < 24 && image.cpu_addr < (0x29426000u + 0x1a40000u) &&
            static_cast<u64>(image.cpu_addr) + image.guest_size_bytes > 0x29426000u) {
            ++s17_alias_logs;
            std::string s17_aliases;
            for (const AliasedImage& s17_a : image.aliased_images) {
                ImageBase& s17_ai = slot_images[s17_a.id];
                s17_aliases +=
                    fmt::format("[id={} gpu={:#x} tick={} flags={:#x} newer={} copies={}]",
                                s17_a.id.index, static_cast<u64>(s17_ai.gpu_addr),
                                s17_ai.modification_tick, static_cast<u32>(s17_ai.flags),
                                image.modification_tick < s17_ai.modification_tick,
                                s17_a.copies.size());
            }
            LOG_WARNING(Render_D3D12,
                        "S17 alias: img={} gpu={:#x} tick={} flags={:#x} size={} aliases={} {}",
                        image_id.index, static_cast<u64>(image.gpu_addr),
                        image.modification_tick, static_cast<u32>(image.flags),
                        image.guest_size_bytes, image.aliased_images.size(), s17_aliases);
        }
    }
    bool any_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    bool any_modified = True(image.flags & ImageFlagBits::GpuModified);
    u64 most_recent_tick = image.modification_tick;
    for (const AliasedImage& aliased : image.aliased_images) {
        ImageBase& aliased_image = slot_images[aliased.id];
        if (image.modification_tick < aliased_image.modification_tick) {
            most_recent_tick = std::max(most_recent_tick, aliased_image.modification_tick);
            aliased_images.push_back(&aliased);
            any_rescaled |= True(aliased_image.flags & ImageFlagBits::Rescaled);
            any_modified |= True(aliased_image.flags & ImageFlagBits::GpuModified);
        }
    }
    if (aliased_images.empty()) {
        return;
    }
    const bool can_rescale = ImageCanRescale(image);
    if (any_rescaled) {
        if (can_rescale) {
            ScaleUp(image);
        } else {
            ScaleDown(image);
        }
    }
    image.modification_tick = most_recent_tick;
    if (any_modified) {
        image.flags |= ImageFlagBits::GpuModified;
    }
    std::ranges::sort(aliased_images, [this](const AliasedImage* lhs, const AliasedImage* rhs) {
        const ImageBase& lhs_image = slot_images[lhs->id];
        const ImageBase& rhs_image = slot_images[rhs->id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });
    const auto& resolution = Settings::values.resolution_info;
    for (const AliasedImage* const aliased : aliased_images) {
        if (!resolution.active || !any_rescaled) {
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        Image& aliased_image = slot_images[aliased->id];
        if (!can_rescale) {
            ScaleDown(aliased_image);
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        ScaleUp(aliased_image);
        CopyImage(image_id, aliased->id, aliased->copies);
    }
}

template <class P>
bool TextureCache<P>::ForceGuestReupload(ImageId image_id) {
    if (!image_id) {
        return false;
    }
    Image& image = slot_images[image_id];
    // Only images that hold NO content at all may be re-seeded from guest memory: no pending CPU
    // upload and never authored by the GPU. That cannot destroy rendered content, and the S21
    // harvest proved these images' guest bytes hold the real data while their resource stayed
    // empty (six scene-shading inputs read 0.0-0.1% nonzero on the GPU for a whole run).
    if (True(image.flags & (ImageFlagBits::CpuModified | ImageFlagBits::GpuModified))) {
        return false;
    }
    if (image.info.type != ImageType::e2D || image.info.num_samples > 1) {
        return false;
    }
    image.flags |= ImageFlagBits::CpuModified;
    if (True(image.flags & ImageFlagBits::Tracked)) {
        UntrackImage(image, image_id);
    }
    return true;
}

template <class P>
void TextureCache<P>::PrepareImage(ImageId image_id, bool is_modification, bool invalidate) {
    Image& image = slot_images[image_id];
    if (invalidate) {
        image.flags &= ~(ImageFlagBits::CpuModified | ImageFlagBits::GpuModified);
        if (False(image.flags & ImageFlagBits::Tracked)) {
            TrackImage(image, image_id);
        }
    } else {
        // TEMP DIAGNOSTIC (S17): forced re-upload probe for the 256x256 lighting-LUT sub-view
        // (gpu 0x5eb340000). Every producer forensic says the LUT is guest-authored, yet the
        // sampled resource has stayed zero in most runs while the guest bytes are (per upload
        // forensics) not reaching it. When the sub-view signature matches, the image carries no
        // pending content, and the guest bytes are genuinely nonzero, mark the image
        // CpuModified so the normal upload path restores the REAL data (swizzle-correct, unlike
        // an alias byte copy or the retired white clear). Bounded: sub-view signature only,
        // <=4 fires, <=16 state-change logs, <=8 GpuModified skips.
        if (static_cast<u64>(image.gpu_addr) == 0x5eb340000ull &&
            image.info.size.width == 256 && image.info.size.height == 256 &&
            image.info.format == PixelFormat::B10G11R11_FLOAT) {
            static u32 s17_subview_fires = 0;
            static u32 s17_subview_evals = 0;
            static u32 s17_subview_gpu_skips = 0;
            static bool s17_subview_last_nonzero = false;
            const bool s17_subview_clean =
                False(image.flags & (ImageFlagBits::CpuModified | ImageFlagBits::GpuModified));
            if (s17_subview_fires < 4 && s17_subview_clean && gpu_memory != nullptr) {
                std::array<u8, 4096> s17_prefix{};
                gpu_memory->ReadBlock(image.gpu_addr, s17_prefix.data(), s17_prefix.size(),
                                      VideoCommon::CacheType::NoTextureCache);
                u32 s17_nz = 0;
                for (const u8 s17_byte : s17_prefix) {
                    s17_nz += s17_byte != 0 ? 1u : 0u;
                }
                const bool s17_nonzero = s17_nz != 0;
                if (s17_nonzero != s17_subview_last_nonzero && s17_subview_evals < 16) {
                    s17_subview_last_nonzero = s17_nonzero;
                    ++s17_subview_evals;
                    LOG_WARNING(Render_D3D12,
                                "S17 sub-view probe: img={} guest_prefix_nz={}/{} flags={:#x}",
                                image_id.index, s17_nz, static_cast<u32>(s17_prefix.size()),
                                static_cast<u32>(image.flags));
                }
                if (s17_nonzero) {
                    ++s17_subview_fires;
                    image.flags |= ImageFlagBits::CpuModified;
                    if (True(image.flags & ImageFlagBits::Tracked)) {
                        UntrackImage(image, image_id);
                    }
                    LOG_WARNING(Render_D3D12,
                                "S17 sub-view refresh: img={} forcing guest re-upload "
                                "(guest_prefix_nz={}/{}, fire {}/4)",
                                image_id.index, s17_nz, static_cast<u32>(s17_prefix.size()),
                                s17_subview_fires);
                }
            } else if (!s17_subview_clean && True(image.flags & ImageFlagBits::GpuModified) &&
                       s17_subview_gpu_skips < 8) {
                ++s17_subview_gpu_skips;
                LOG_WARNING(Render_D3D12,
                            "S17 sub-view refresh skipped: img={} is GpuModified (flags={:#x}); "
                            "the RefreshContents guard would refuse the upload",
                            image_id.index, static_cast<u32>(image.flags));
            }
        }
        RefreshContents(image, image_id);
        SynchronizeAliases(image_id);
    }
    if (is_modification) {
        MarkModification(image);
    }
    lru_cache.Touch(image.lru_index, frame_tick);
}

template <class P>
void TextureCache<P>::PrepareImageView(ImageViewId image_view_id, bool is_modification,
                                       bool invalidate) {
    if (!image_view_id) {
        return;
    }
    const ImageViewBase& image_view = slot_image_views[image_view_id];
    if (image_view.IsBuffer()) {
        return;
    }
    PrepareImage(image_view.image_id, is_modification, invalidate);
}

template <class P>
void TextureCache<P>::CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies) {
    Image& dst = slot_images[dst_id];
    Image& src = slot_images[src_id];
    // S22 cycle 24: any cache-level copy touching the lighting-LUT bank. The LUT is finite at
    // t=771 and all-NaN at t=834 with no clear, no alias propagation and no later draw burst
    // logged, so a copy is the remaining writer class.
    {
        static u32 s22_bank_copy_logs = 0;
        static std::chrono::steady_clock::time_point s22_bank_copy_last{};
        const u64 s22_dst_gpu = static_cast<u64>(dst.gpu_addr);
        const u64 s22_src_gpu = static_cast<u64>(src.gpu_addr);
        const bool s22_dst_bank =
            s22_dst_gpu >= 0x5e9000000ull && s22_dst_gpu < 0x5ee7b4000ull;
        const bool s22_src_bank =
            s22_src_gpu >= 0x5e9000000ull && s22_src_gpu < 0x5ee7b4000ull;
        // Burst mode: the join emits a storm of tile copies in one millisecond, so a plain cap
        // hides every later copy. Log the first copy of each new burst (>=1 s gap) instead.
        const auto s22_copy_now = std::chrono::steady_clock::now();
        const bool s22_copy_new_burst =
            s22_copy_now - s22_bank_copy_last > std::chrono::seconds(1);
        if (s22_bank_copy_logs < 64 && s22_copy_new_burst && (s22_dst_bank || s22_src_bank)) {
            s22_bank_copy_last = s22_copy_now;
            ++s22_bank_copy_logs;
            // The box matters: SynchronizeAliases copies FROM the 16x8 alias tile INTO this
            // 256x256 image, so a full-extent box here overwrites the whole LUT with one tile.
            const ImageCopy* s22_first = copies.empty() ? nullptr : &copies.front();
            LOG_WARNING(Render_D3D12,
                        "S22 bank copy #{}: dst={} gpu={:#x} {}x{} fmt={:#x} src={} gpu={:#x} "
                        "{}x{} fmt={:#x} copies={} ext=({},{},{}) dst=({},{},{}) src=({},{},{})",
                        s22_bank_copy_logs, dst_id.index, s22_dst_gpu, dst.info.size.width,
                        dst.info.size.height, static_cast<u32>(dst.info.format), src_id.index,
                        s22_src_gpu, src.info.size.width, src.info.size.height,
                        static_cast<u32>(src.info.format), copies.size(),
                        s22_first ? s22_first->extent.width : 0,
                        s22_first ? s22_first->extent.height : 0,
                        s22_first ? s22_first->extent.depth : 0,
                        s22_first ? s22_first->dst_offset.x : 0,
                        s22_first ? s22_first->dst_offset.y : 0,
                        s22_first ? s22_first->dst_offset.z : 0,
                        s22_first ? s22_first->src_offset.x : 0,
                        s22_first ? s22_first->src_offset.y : 0,
                        s22_first ? s22_first->src_offset.z : 0);
        }
    }
    // TEMP DIAGNOSTIC (S18, Phase 0.2.1): a copy with a 1x1 side is a candidate writer (alias
    // propagation) of the exposure texel. The bank filter below requires both sides > 4 KiB, so
    // a 16-byte 1x1 copy would slip past it - this check is size-independent. Cap 12.
    {
        static u32 s18_tiny_copy_logs = 0;
        // S18 cycle 3: any member of the exposure family (1x1 // 8x1 // 128x1) on either side.
        const auto s18_copy_tiny = [](const ImageBase& s18_img) {
            const u32 s18_w = s18_img.info.size.width;
            const u32 s18_h = s18_img.info.size.height;
            return (s18_w <= 2 && s18_h <= 2) || (s18_w <= 128 && s18_h <= 1) ||
                   (s18_w <= 1 && s18_h <= 128);
        };
        const bool s18_dst_tiny = s18_copy_tiny(dst);
        const bool s18_src_tiny = s18_copy_tiny(src);
        if (s18_tiny_copy_logs < 12 && (s18_dst_tiny || s18_src_tiny)) {
            ++s18_tiny_copy_logs;
            LOG_WARNING(Render_D3D12,
                        "S18 tiny copyimg: dst=({} gpu={:#x} {}x{} fmt={:#x} flags={:#x} size={}) "
                        "src=({} gpu={:#x} {}x{} fmt={:#x} flags={:#x} size={}) copies={}",
                        dst_id.index, static_cast<u64>(dst.gpu_addr), dst.info.size.width,
                        dst.info.size.height, static_cast<u32>(dst.info.format),
                        static_cast<u32>(dst.flags), dst.guest_size_bytes, src_id.index,
                        static_cast<u64>(src.gpu_addr), src.info.size.width,
                        src.info.size.height, static_cast<u32>(src.info.format),
                        static_cast<u32>(src.flags), src.guest_size_bytes, copies.size());
        }
    }
    // TEMP DIAGNOSTIC (S19): any cache-level copy with a LUT-fill input window on either side.
    // The probe image (4086 @ 0x516170000) held 87% content at t=220.9 and zero at t=741.8 with
    // the SAME resource, no upload and no write notification - the alias path (which copies from
    // an aliased image with a newer modification_tick) is the prime remaining writer; a
    // zero-valued sibling would wipe it exactly like this. Bounded 24.
    {
        static u32 s19_input_copy_logs = 0;
        const auto s19_copy_window = [](u64 gpu) {
            return (gpu >= 0x516170000ull && gpu < 0x516170000ull + 0x408000ull) ||
                   (gpu >= 0x5d0630000ull && gpu < 0x5d0631000ull);
        };
        const u64 s19_copy_dst = static_cast<u64>(dst.gpu_addr);
        const u64 s19_copy_src = static_cast<u64>(src.gpu_addr);
        if (s19_input_copy_logs < 24 &&
            (s19_copy_window(s19_copy_dst) || s19_copy_window(s19_copy_src))) {
            ++s19_input_copy_logs;
            LOG_WARNING(Render_D3D12,
                        "S19 input copy: dst=({} gpu={:#x} {}x{} fmt={:#x} flags={:#x} size={}) "
                        "src=({} gpu={:#x} {}x{} fmt={:#x} flags={:#x} size={}) copies={} "
                        "tick_dst={} tick_src={}",
                        dst_id.index, s19_copy_dst, dst.info.size.width, dst.info.size.height,
                        static_cast<u32>(dst.info.format), static_cast<u32>(dst.flags),
                        dst.guest_size_bytes, src_id.index, s19_copy_src, src.info.size.width,
                        src.info.size.height, static_cast<u32>(src.info.format),
                        static_cast<u32>(src.flags), src.guest_size_bytes, copies.size(),
                        dst.modification_tick, src.modification_tick);
        }
    }
    // TEMP DIAGNOSTIC (S17): name every texture-cache-side copy touching the 27 MB LUT bank
    // (cpu 0x29426000). This catches the JoinImages copy paths and any alias synchronisation, so
    // together with the existing 'RefreshContents upload' line it accounts for every writer of the
    // sampled image's resource - the run-to-run intermittency needs that attribution. Bounded 24.
    {
        static u32 s17_copy_logs = 0;
        const auto s17_hits = [](const ImageBase& i) {
            return i.cpu_addr < (0x29426000u + 0x1a40000u) &&
                   static_cast<u64>(i.cpu_addr) + i.guest_size_bytes > 0x29426000u;
        };
        // S17 (1.0.266 harvest): the creation burst saturates this cap with bidirectional
        // 512-byte sibling pairs, hiding any later content-bearing copy. Require both sides
        // > 4 KiB so the burst pairs are excluded and a later large copy is always visible.
        if (s17_copy_logs < 24 && dst.guest_size_bytes > 4096 && src.guest_size_bytes > 4096 &&
            (s17_hits(dst) || s17_hits(src))) {
            ++s17_copy_logs;
            LOG_WARNING(Render_D3D12,
                        "S17 copyimg: dst=({} gpu={:#x} size={} flags={:#x}) src=({} gpu={:#x} "
                        "size={} flags={:#x}) copies={}",
                        dst_id.index, static_cast<u64>(dst.gpu_addr), dst.guest_size_bytes,
                        static_cast<u32>(dst.flags), src_id.index,
                        static_cast<u64>(src.gpu_addr), src.guest_size_bytes,
                        static_cast<u32>(src.flags), copies.size());
        }
    }
    // S23 cycle 4: dedicated copy tracer for the lit/black 1024x1024 BC5 at 0x5e9430000 (and its
    // 0x510f30000 mask sibling). Own budget; logs dst and src side by side, any size.
    {
        static u32 s23_bc5_copy_logs = 0;
        const auto s23_bc5_hits = [](const ImageBase& i) {
            const u64 g = static_cast<u64>(i.gpu_addr);
            return (g >= 0x5e9430000ull && g < 0x5e9580000ull) ||
                   (g >= 0x510f30000ull && g < 0x510f40000ull);
        };
        if (s23_bc5_copy_logs < 24 && (s23_bc5_hits(dst) || s23_bc5_hits(src))) {
            ++s23_bc5_copy_logs;
            LOG_WARNING(Render_D3D12,
                        "S23 BC5 copyimg: dst=({} gpu={:#x} {}x{} size={} flags={:#x}) src=({} "
                        "gpu={:#x} {}x{} size={} flags={:#x}) copies={} tick={}/{}",
                        dst_id.index, static_cast<u64>(dst.gpu_addr), dst.info.size.width,
                        dst.info.size.height, dst.guest_size_bytes, static_cast<u32>(dst.flags),
                        src_id.index, static_cast<u64>(src.gpu_addr), src.info.size.width,
                        src.info.size.height, src.guest_size_bytes, static_cast<u32>(src.flags),
                        copies.size(), dst.modification_tick, src.modification_tick);
        }
    }
    const bool is_rescaled = True(src.flags & ImageFlagBits::Rescaled);
    if (is_rescaled) {
        ASSERT(True(dst.flags & ImageFlagBits::Rescaled));
        const bool both_2d{src.info.type == ImageType::e2D && dst.info.type == ImageType::e2D};
        const auto& resolution = Settings::values.resolution_info;
        for (auto& copy : copies) {
            copy.src_offset.x = resolution.ScaleUp(copy.src_offset.x);
            copy.dst_offset.x = resolution.ScaleUp(copy.dst_offset.x);
            copy.extent.width = resolution.ScaleUp(copy.extent.width);
            if (both_2d) {
                copy.src_offset.y = resolution.ScaleUp(copy.src_offset.y);
                copy.dst_offset.y = resolution.ScaleUp(copy.dst_offset.y);
                copy.extent.height = resolution.ScaleUp(copy.extent.height);
            }
        }
    }
    const auto dst_format_type = GetFormatType(dst.info.format);
    const auto src_format_type = GetFormatType(src.info.format);
    if (src_format_type == dst_format_type) {
        if constexpr (HAS_EMULATED_COPIES) {
            if (!runtime.CanImageBeCopied(dst, src)) {
                return runtime.EmulateCopyImage(dst, src, copies);
            }
        }
        return runtime.CopyImage(dst, src, copies);
    }
    UNIMPLEMENTED_IF(dst.info.type != ImageType::e2D);
    UNIMPLEMENTED_IF(src.info.type != ImageType::e2D);
    if (runtime.ShouldReinterpret(dst, src)) {
        return runtime.ReinterpretImage(dst, src, copies);
    }
    for (const ImageCopy& copy : copies) {
        UNIMPLEMENTED_IF(copy.dst_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_offset != Offset3D{});
        UNIMPLEMENTED_IF(copy.dst_offset != Offset3D{});

        const SubresourceBase dst_base{
            .level = copy.dst_subresource.base_level,
            .layer = copy.dst_subresource.base_layer,
        };
        const SubresourceBase src_base{
            .level = copy.src_subresource.base_level,
            .layer = copy.src_subresource.base_layer,
        };
        const SubresourceExtent dst_extent{.levels = 1, .layers = 1};
        const SubresourceExtent src_extent{.levels = 1, .layers = 1};
        const SubresourceRange dst_range{.base = dst_base, .extent = dst_extent};
        const SubresourceRange src_range{.base = src_base, .extent = src_extent};
        PixelFormat dst_format = dst.info.format;
        if (GetFormatType(src.info.format) == SurfaceType::DepthStencil &&
            GetFormatType(dst_format) == SurfaceType::ColorTexture &&
            BytesPerBlock(dst_format) == 4) {
            dst_format = PixelFormat::A8B8G8R8_UNORM;
        }
        const ImageViewInfo dst_view_info(ImageViewType::e2D, dst_format, dst_range);
        const ImageViewInfo src_view_info(ImageViewType::e2D, src.info.format, src_range);
        const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
        const ImageViewId src_view_id = FindOrEmplaceImageView(src_id, src_view_info);
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        [[maybe_unused]] const Extent3D expected_size{
            .width = std::min(dst_view.size.width, src_view.size.width),
            .height = std::min(dst_view.size.height, src_view.size.height),
            .depth = std::min(dst_view.size.depth, src_view.size.depth),
        };
        const Extent3D scaled_extent = [is_rescaled, expected_size]() {
            if (!is_rescaled) {
                return expected_size;
            }
            const auto& resolution = Settings::values.resolution_info;
            return Extent3D{
                .width = resolution.ScaleUp(expected_size.width),
                .height = resolution.ScaleUp(expected_size.height),
                .depth = expected_size.depth,
            };
        }();
        UNIMPLEMENTED_IF(copy.extent != scaled_extent);

        runtime.ConvertImage(dst_framebuffer, dst_view, src_view);
    }
}

template <class P>
void TextureCache<P>::BindRenderTarget(ImageViewId* old_id, ImageViewId new_id) {
    if (*old_id == new_id) {
        return;
    }
    if (new_id) {
        const ImageViewBase& old_view = slot_image_views[new_id];
        if (True(old_view.flags & ImageViewFlagBits::PreemtiveDownload)) {
            const PendingDownload new_download{true, 0, old_view.image_id};
            uncommitted_downloads.emplace_back(new_download);
        }
    }
    *old_id = new_id;
}

template <class P>
std::pair<FramebufferId, ImageViewId> TextureCache<P>::RenderTargetFromImage(
    ImageId image_id, const ImageViewInfo& view_info) {
    const ImageViewId view_id = FindOrEmplaceImageView(image_id, view_info);
    const ImageBase& image = slot_images[image_id];
    const bool is_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    const bool is_color = GetFormatType(image.info.format) == SurfaceType::ColorTexture;
    const ImageViewId color_view_id = is_color ? view_id : ImageViewId{};
    const ImageViewId depth_view_id = is_color ? ImageViewId{} : view_id;
    Extent3D extent = MipSize(image.info.size, view_info.range.base.level);
    if (is_rescaled) {
        const auto& resolution = Settings::values.resolution_info;
        extent.width = resolution.ScaleUp(extent.width);
        if (image.info.type == ImageType::e2D) {
            extent.height = resolution.ScaleUp(extent.height);
        }
    }
    const u32 num_samples = image.info.num_samples;
    const auto [samples_x, samples_y] = SamplesLog2(num_samples);
    const FramebufferId framebuffer_id = GetFramebufferId(RenderTargets{
        .color_buffer_ids = {color_view_id},
        .depth_buffer_id = depth_view_id,
        .size = {extent.width >> samples_x, extent.height >> samples_y},
        .is_rescaled = is_rescaled,
    });
    return {framebuffer_id, view_id};
}

template <class P>
bool TextureCache<P>::IsFullClear(ImageViewId id) {
    if (!id) {
        return true;
    }
    const ImageViewBase& image_view = slot_image_views[id];
    const ImageBase& image = slot_images[image_view.image_id];
    const Extent3D size = image_view.size;
    const auto& regs = maxwell3d->regs;
    const auto& scissor = regs.scissor_test[0];
    if (image.info.resources.levels > 1 || image.info.resources.layers > 1) {
        // Images with multiple resources can't be cleared in a single call
        return false;
    }
    if (regs.clear_control.use_scissor == 0) {
        // If scissor testing is disabled, the clear is always full
        return true;
    }
    // Make sure the clear covers all texels in the subresource
    return scissor.min_x == 0 && scissor.min_y == 0 && scissor.max_x >= size.width &&
           scissor.max_y >= size.height;
}

template <class P>
void TextureCache<P>::CreateChannel(struct Tegra::Control::ChannelState& channel) {
    VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo>::CreateChannel(channel);
    const auto it = channel_map.find(channel.bind_id);
    auto* this_state = &channel_storage[it->second];
    const auto& this_as_ref = address_spaces[channel.memory_manager->GetID()];
    this_state->gpu_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2];
    this_state->sparse_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2 + 1];
}

/// Bind a channel for execution.
template <class P>
void TextureCache<P>::OnGPUASRegister([[maybe_unused]] size_t map_id) {
    gpu_page_table_storage.emplace_back();
    gpu_page_table_storage.emplace_back();
}

} // namespace VideoCommon
