// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <memory>

#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"

namespace CitronUWP {

/// The UWP frontend does not need a real context: the D3D12 device is owned by the renderer.
class GraphicsContextUWP final : public Core::Frontend::GraphicsContext {
public:
    ~GraphicsContextUWP() override = default;
};

class EmuWindowUWP final : public Core::Frontend::EmuWindow {
public:
    EmuWindowUWP(void* hwnd, void* core_window, u32 width, u32 height, float scale);
    ~EmuWindowUWP() override = default;

    void OnSurfaceChanged(void* hwnd, void* core_window, u32 width, u32 height, float scale);
    void OnFrameDisplayed() override;

    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<GraphicsContextUWP>();
    }

    bool IsShown() const override {
        return true;
    }

    /// Called once, when the first frame has been presented.
    void SetFirstFrameCallback(std::function<void()> callback) {
        on_first_frame = std::move(callback);
    }

private:
    bool first_frame{false};
    std::function<void()> on_first_frame;
};

} // namespace CitronUWP
