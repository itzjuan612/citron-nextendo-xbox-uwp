// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/sw_blitter/blitter.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"

using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;

namespace Tegra::Engines {

using namespace Texture;

Fermi2D::Fermi2D(MemoryManager& memory_manager_) : memory_manager{memory_manager_} {
    sw_blitter = std::make_unique<Blitter::SoftwareBlitEngine>(memory_manager);
    // Nvidia's OpenGL driver seems to assume these values
    regs.src.depth = 1;
    regs.dst.depth = 1;

    execution_mask.reset();
    execution_mask[FERMI2D_REG_INDEX(pixels_from_memory.src_y0) + 1] = true;
}

Fermi2D::~Fermi2D() = default;

void Fermi2D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

void Fermi2D::CallMethod(u32 method, u32 method_argument, bool is_last_call) {
    ASSERT_MSG(method < Regs::NUM_REGS,
               "Invalid Fermi2D register, increase the size of the Regs structure");
    regs.reg_array[method] = method_argument;

    if (method == FERMI2D_REG_INDEX(pixels_from_memory.src_y0) + 1) {
        Blit();
    }
}

void Fermi2D::CallMultiMethod(u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    for (u32 i = 0; i < amount; ++i) {
        CallMethod(method, base_start[i], methods_pending - i <= 1);
    }
}

void Fermi2D::ConsumeSinkImpl() {
    for (auto [method, value] : method_sink) {
        regs.reg_array[method] = value;
    }
    method_sink.clear();
}

void Fermi2D::Blit() {
    // TEMP DIAGNOSTIC (session 13): the guest blits every frame; logging each call is a
    // large console cost. Keep the first few and sample periodically.
    static u64 blit_calls = 0;
    const bool log_blit_call = blit_calls < 8 || (blit_calls % 500) == 0;
    ++blit_calls;
    if (log_blit_call) {
        const auto src_cpu = memory_manager.GpuToCpuAddress(regs.src.Address());
        const auto dst_cpu = memory_manager.GpuToCpuAddress(regs.dst.Address());
        LOG_DEBUG(HW_GPU,
                  "called. source address=0x{:x} (cpu=0x{:x}), destination address=0x{:x} "
                  "(cpu=0x{:x})",
                  regs.src.Address(), src_cpu.value_or(0), regs.dst.Address(),
                  dst_cpu.value_or(0));
    }

    if (regs.operation != Operation::SrcCopy) {
        LOG_WARNING(HW_GPU, "Operation is not SrcCopy ({}), skipping blit", static_cast<u32>(regs.operation));
        return;
    }
    if (regs.src.layer != 0) {
        LOG_DEBUG(HW_GPU, "Source layer is {}, expected 0 - using layer 0", regs.src.layer);
    }
    if (regs.dst.layer != 0) {
        LOG_DEBUG(HW_GPU, "Destination layer is {}, expected 0 - using layer 0", regs.dst.layer);
    }
    if (regs.src.depth != 1) {
        LOG_DEBUG(HW_GPU, "Source depth is {}, expected 1 - using first layer", regs.src.depth);
    }
    if (regs.clip_enable != 0) {
        LOG_DEBUG(HW_GPU, "Clipped blit enabled - ignoring clip");
    }

    const auto& args = regs.pixels_from_memory;
    constexpr s64 null_derivative = 1ULL << 32;
    Surface src = regs.src;
    const auto bytes_per_pixel = BytesPerBlock(PixelFormatFromRenderTargetFormat(src.format));
    const bool delegate_to_gpu = src.width > 512 && src.height > 512 && bytes_per_pixel <= 8 &&
                                 src.format != regs.dst.format;

    auto srcX = args.src_x0;
    auto srcY = args.src_y0;
    if (args.sample_mode.origin == Origin::Corner) {
        srcX -= (args.du_dx >> 33) << 32;
        srcY -= (args.dv_dy >> 33) << 32;
    }

    Config config{
        .operation = regs.operation,
        .filter = args.sample_mode.filter,
        .must_accelerate =
            args.du_dx != null_derivative || args.dv_dy != null_derivative || delegate_to_gpu,
        .dst_x0 = args.dst_x0,
        .dst_y0 = args.dst_y0,
        .dst_x1 = args.dst_x0 + args.dst_width,
        .dst_y1 = args.dst_y0 + args.dst_height,
        .src_x0 = static_cast<s32>(srcX >> 32),
        .src_y0 = static_cast<s32>(srcY >> 32),
        .src_x1 = static_cast<s32>((srcX + args.du_dx * args.dst_width) >> 32),
        .src_y1 = static_cast<s32>((srcY + args.dv_dy * args.dst_height) >> 32),
    };

    const auto need_align_to_pitch =
        src.linear == Tegra::Engines::Fermi2D::MemoryLayout::Pitch &&
        static_cast<s32>(src.width) == config.src_x1 &&
        config.src_x1 > static_cast<s32>(src.pitch / bytes_per_pixel) && config.src_x0 > 0;
    if (need_align_to_pitch) {
        auto address = src.Address() + config.src_x0 * bytes_per_pixel;
        src.addr_upper = static_cast<u32>(address >> 32);
        src.addr_lower = static_cast<u32>(address);
        src.width -= config.src_x0;
        config.src_x1 -= config.src_x0;
        config.src_x0 = 0;
    }

    memory_manager.FlushCaching();
    const bool accelerated = rasterizer->AccelerateSurfaceCopy(src, regs.dst, config);
    if (!accelerated) {
        sw_blitter->Blit(src, regs.dst, config);
    }
    // Diagnostic: is the guest's composition blit running on the GPU or the CPU fallback?
    static u64 blit_total = 0;
    static u64 blit_accelerated = 0;
    ++blit_total;
    if (accelerated) {
        ++blit_accelerated;
    }
    if (blit_total <= 8 || blit_total % 500 == 0) {
        LOG_INFO(HW_GPU,
                 "Blit result #{}: accelerated={} total_accelerated={} "
                 "dst=({},{})-({},{}) src=({},{})-({},{}) filter={}",
                 blit_total, accelerated, blit_accelerated, config.dst_x0, config.dst_y0,
                 config.dst_x1, config.dst_y1, config.src_x0, config.src_y0, config.src_x1,
                 config.src_y1, static_cast<u32>(config.filter));
    }
    // TEMP DIAGNOSTIC (S15): the wiper-input window [0x517000000, 0x517200000) has no
    // logged draw/compute producer; catch a Fermi2D blit whose destination lands in it.
    static u32 s15_window_blit_logs = 0;
    if (s15_window_blit_logs < 8) {
        const GPUVAddr s15_dst_gpu = regs.dst.Address();
        if (s15_dst_gpu >= 0x517000000ull && s15_dst_gpu < 0x517200000ull) {
            ++s15_window_blit_logs;
            LOG_WARNING(Render_D3D12,
                        "S15 window blit: src={:#x} dst={:#x} src_box=({},{})-({},{}) "
                        "dst_box=({},{})-({},{})",
                        src.Address(), s15_dst_gpu, config.src_x0, config.src_y0,
                        config.src_x1, config.src_y1, config.dst_x0, config.dst_y0,
                        config.dst_x1, config.dst_y1);
        }
    }
    // TEMP DIAGNOSTIC (S17): the S15 block above only watches the 0x517 window. Also catch a
    // blit whose SOURCE or DESTINATION is in the 27 MB LUT bank VAs (0x5e9900000 / 0x5eb340000
    // / 0x5ecd80000, 0x1a34000 bytes each) or the mask VAs (0x510fa0000 / 0x510f30000) - a
    // producer neither the draw nor the compute tracers can see. Bounded to 16 lines.
    static u32 s17_blit_logs = 0;
    if (s17_blit_logs < 16) {
        const auto s17_in = [](GPUVAddr a, GPUVAddr base, u64 length) {
            return a >= base && a < base + length;
        };
        const auto s17_lut_addr = [&](GPUVAddr a) {
            return s17_in(a, 0x5e9900000ull, 0x1a34000ull) ||
                   s17_in(a, 0x5eb340000ull, 0x1a34000ull) ||
                   s17_in(a, 0x5ecd80000ull, 0x1a34000ull) ||
                   s17_in(a, 0x510fa0000ull, 0x100000ull) ||
                   s17_in(a, 0x510f30000ull, 0x100000ull);
        };
        const GPUVAddr s17_src = src.Address();
        const GPUVAddr s17_dst = regs.dst.Address();
        const bool s17_src_hit = s17_lut_addr(s17_src);
        const bool s17_dst_hit = s17_lut_addr(s17_dst);
        if (s17_src_hit || s17_dst_hit) {
            ++s17_blit_logs;
            LOG_WARNING(Render_D3D12,
                        "S17 LUT producer: blit src={:#x} src_hit={} dst={:#x} dst_hit={} "
                        "src_box=({},{})-({},{}) dst_box=({},{})-({},{})",
                        s17_src, static_cast<u32>(s17_src_hit), s17_dst,
                        static_cast<u32>(s17_dst_hit), config.src_x0, config.src_y0, config.src_x1,
                        config.src_y1, config.dst_x0, config.dst_y0, config.dst_x1,
                        config.dst_y1);
        }
    }
}

} // namespace Tegra::Engines
