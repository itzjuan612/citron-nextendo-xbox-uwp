// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "citron_uwp/emu_window.h"

#include <algorithm>

#include "common/logging.h"

namespace CitronUWP {

EmuWindowUWP::EmuWindowUWP(void* hwnd, void* core_window, u32 width, u32 height, float scale) {
    window_info.type = Core::Frontend::WindowSystemType::Windows;
    OnSurfaceChanged(hwnd, core_window, width, height, scale);
}

void EmuWindowUWP::OnSurfaceChanged(void* hwnd, void* core_window, u32 width, u32 height,
                                    float scale) {
    const u32 framebuffer_width = std::max(width, 1u);
    const u32 framebuffer_height = std::max(height, 1u);

    window_info.render_surface = hwnd;
    window_info.core_window = core_window;
    window_info.render_surface_scale = scale;

    NotifyClientAreaSizeChanged({framebuffer_width, framebuffer_height});
    UpdateCurrentFramebufferLayout(framebuffer_width, framebuffer_height);

    LOG_INFO(Frontend, "UWP surface: {}x{} scale {} (hwnd {:#x}, core_window {:#x})",
             framebuffer_width, framebuffer_height, scale,
             reinterpret_cast<uintptr_t>(hwnd), reinterpret_cast<uintptr_t>(core_window));
}

void EmuWindowUWP::OnFrameDisplayed() {
    if (!first_frame) {
        first_frame = true;
        LOG_INFO(Frontend, "UWP first frame displayed");
        if (on_first_frame) {
            on_first_frame();
        }
    }
}

} // namespace CitronUWP