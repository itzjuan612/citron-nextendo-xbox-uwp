// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "common/detached_tasks.h"
#include "core/core.h"
#include "frontend_common/content_manager.h"
#include "input_common/main.h"
#include "citron_uwp/emu_window.h"
#include "video_core/rasterizer_interface.h"

namespace CitronUWP {

using LifecycleCallback = void (*)(int result);

/// Owns the emulated system and drives the same boot sequence as the iOS frontend.
class EmulationSession final {
public:
    EmulationSession();
    ~EmulationSession();

    static EmulationSession& GetInstance();

    Core::System& System();
    InputCommon::InputSubsystem& GetInputSubsystem();

    [[nodiscard]] bool IsInitialized() const;
    [[nodiscard]] bool IsRunning() const;

    void Initialize(const std::string& app_directory);
    void SetWindow(void* hwnd, void* core_window, u32 width, u32 height, float scale);
    Core::SystemResultStatus Launch(const std::string& filepath, std::size_t program_index);
    void Stop();
    void Shutdown();

    void SetButtonState(std::size_t player_index, int button_id, bool pressed);
    void SetStickPosition(std::size_t player_index, int stick_id, float x, float y);

    void SetCallbacks(LifecycleCallback started, LifecycleCallback stopped);

private:
    void ConfigureFilesystemProvider(const std::string& filepath);
    void InitializeSystem(bool reload);
    Core::SystemResultStatus InitializeEmulation(const std::string& filepath,
                                                 std::size_t program_index);
    void RunEmulation();
    void ShutdownEmulation(Core::SystemResultStatus result);

    static void LoadDiskCacheProgress(VideoCore::LoadCallbackStage stage, int progress, int max);

    std::unique_ptr<EmuWindowUWP> window;
    void* hwnd{};
    void* core_window{};
    u32 surface_width{1280};
    u32 surface_height{720};
    float surface_scale{1.0f};

    Core::System system;
    InputCommon::InputSubsystem input_subsystem;
    Common::DetachedTasks detached_tasks;
    std::shared_ptr<FileSys::VfsFilesystem> vfs;
    std::unique_ptr<FileSys::ManualContentProvider> manual_provider;
    Core::SystemResultStatus load_result{Core::SystemResultStatus::ErrorNotInitialized};

    std::atomic<bool> is_initialized = false;
    std::atomic<bool> is_running = false;
    std::condition_variable_any cv;
    mutable std::mutex mutex;
    std::thread emulation_thread;

    LifecycleCallback on_started{};
    LifecycleCallback on_stopped{};
};

} // namespace CitronUWP