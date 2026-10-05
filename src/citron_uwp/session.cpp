// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "citron_uwp/session.h"

#include <chrono>
#include <filesystem>
#include <memory>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/cpu_manager.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/card_image.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/fs_filesystem.h"
#include "core/file_sys/submission_package.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/frontend/applets/cabinet.h"
#include "core/frontend/applets/controller.h"
#include "core/frontend/applets/error.h"
#include "core/frontend/applets/general.h"
#include "core/frontend/applets/mii_edit.h"
#include "core/frontend/applets/profile_select.h"
#include "core/frontend/applets/software_keyboard.h"
#include "core/frontend/applets/web_browser.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/service/am/frontend/applets.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/loader.h"
#include "hid_core/hid_core.h"
#include "input_common/drivers/virtual_gamepad.h"
#include "video_core/renderer_base.h"
#include "video_core/rasterizer_interface.h"

namespace CitronUWP {

namespace {

/// Defaults tuned for the Xbox Series X|S D3D12 backend.
void ApplyUwpRuntimeSettings() {
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::D3D12);
    Settings::values.sink_id.SetValue(Settings::AudioEngine::XAudio2);
    Settings::values.audio_output_device_id.SetValue("auto");
    Settings::values.audio_input_device_id.SetValue("null");
    Settings::values.log_filter.SetValue("*:Info Render.D3D12:Debug HW.GPU:Debug");

    // No GPU NVDEC/ASTC on the D3D12 backend yet, and fastmem is disabled on UWP.
    Settings::values.nvdec_emulation.SetValue(Settings::NvdecEmulation::Cpu);
    Settings::values.accelerate_astc.SetValue(Settings::AstcDecodeMode::Cpu);
    Settings::values.async_presentation.SetValue(false);
    Settings::values.use_reactive_flushing.SetValue(false);
    Settings::values.use_fast_gpu_time.SetValue(false);
    Settings::values.use_vulkan_driver_pipeline_cache.SetValue(false);
    Settings::values.use_disk_shader_cache.SetValue(false);
}

} // Anonymous namespace

EmulationSession::EmulationSession() : vfs{std::make_shared<FileSys::RealVfsFilesystem>()} {}

EmulationSession::~EmulationSession() {
    Shutdown();
}

EmulationSession& EmulationSession::GetInstance() {
    // Function-local static: Core::System must not be constructed during static initialization
    // (it eagerly creates a RegisteredCache, which would touch Settings::values too early).
    static EmulationSession instance;
    return instance;
}

Core::System& EmulationSession::System() {
    return system;
}

InputCommon::InputSubsystem& EmulationSession::GetInputSubsystem() {
    return input_subsystem;
}

bool EmulationSession::IsInitialized() const {
    return is_initialized;
}

bool EmulationSession::IsRunning() const {
    return is_running;
}

void EmulationSession::SetCallbacks(LifecycleCallback started, LifecycleCallback stopped) {
    std::scoped_lock lock{mutex};
    on_started = started;
    on_stopped = stopped;
}

void EmulationSession::Initialize(const std::string& app_directory) {
    std::scoped_lock lock{mutex};
    if (is_initialized) {
        return;
    }

    // All Citron paths (keys/, nand/, config/, log/, ...) are derived from this directory.
    Common::FS::SetAppDirectory(app_directory);

    Common::Log::Initialize();
    Common::Log::SetColorConsoleBackendEnabled(false);
    Common::Log::Start();

    Core::Crypto::KeyManager::Instance().ReloadKeys();
    LOG_INFO(Frontend, "UWP app directory: {}", app_directory);
    LOG_INFO(Frontend, "UWP keys directory: {}",
             Common::FS::GetCitronPathString(Common::FS::CitronPath::KeysDir));
    LOG_INFO(Frontend, "UWP key files: prod.keys={}, title.keys={}",
             Core::Crypto::KeyManager::KeyFileExists(false),
             Core::Crypto::KeyManager::KeyFileExists(true));

    input_subsystem.Initialize();
    system.SetFilesystem(vfs);

    ApplyUwpRuntimeSettings();
    is_initialized = true;
}

void EmulationSession::SetWindow(void* hwnd_, void* core_window_, u32 width, u32 height,
                                 float scale) {
    std::scoped_lock lock{mutex};
    hwnd = hwnd_;
    core_window = core_window_;
    surface_width = width;
    surface_height = height;
    surface_scale = scale;

    if (window) {
        window->OnSurfaceChanged(hwnd, core_window, surface_width, surface_height, surface_scale);
    }
}

void EmulationSession::ConfigureFilesystemProvider(const std::string& filepath) {
    const auto file = system.GetFilesystem()->OpenFile(filepath, FileSys::OpenMode::Read);
    if (!file) {
        return;
    }

    auto loader = Loader::GetLoader(system, file);
    if (!loader) {
        return;
    }

    const auto file_type = loader->GetFileType();
    if (file_type == Loader::FileType::Unknown || file_type == Loader::FileType::Error) {
        return;
    }

    u64 program_id = 0;
    const auto result = loader->ReadProgramId(program_id);
    if (result == Loader::ResultStatus::Success && file_type == Loader::FileType::NCA) {
        manual_provider->AddEntry(FileSys::TitleType::Application,
                                  FileSys::GetCRTypeFromNCAType(FileSys::NCA{file}.GetType()),
                                  program_id, file);
    } else if (result == Loader::ResultStatus::Success &&
               (file_type == Loader::FileType::XCI || file_type == Loader::FileType::NSP)) {
        const auto nsp = file_type == Loader::FileType::NSP
                             ? std::make_shared<FileSys::NSP>(file)
                             : FileSys::XCI{file}.GetSecurePartitionNSP();
        for (const auto& title : nsp->GetNCAs()) {
            for (const auto& entry : title.second) {
                manual_provider->AddEntry(entry.first.first, entry.first.second, title.first,
                                          entry.second->GetBaseFile());
            }
        }
    }
}

void EmulationSession::InitializeSystem(bool reload) {
    if (!reload) {
        system.SetFilesystem(vfs);
    }

    system.GetUserChannel().clear();
    manual_provider = std::make_unique<FileSys::ManualContentProvider>();
    system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
    system.RegisterContentProvider(FileSys::ContentProviderUnionSlot::FrontendManual,
                                   manual_provider.get());
    system.GetFileSystemController().InitializeContentSystem(*vfs);
}

Core::SystemResultStatus EmulationSession::InitializeEmulation(const std::string& filepath,
                                                               std::size_t program_index) {
    std::scoped_lock lock{mutex};
    if (hwnd == nullptr && core_window == nullptr) {
        return Core::SystemResultStatus::ErrorVideoCore;
    }

    window = std::make_unique<EmuWindowUWP>(hwnd, core_window, surface_width, surface_height,
                                            surface_scale);
    window->SetFirstFrameCallback([this] {
        if (on_started) {
            on_started(static_cast<int>(Core::SystemResultStatus::Success));
        }
    });

    system.SetShuttingDown(false);
    system.ApplySettings();
    Settings::LogSettings();
    system.HIDCore().ReloadInputDevices();
    system.SetFrontendAppletSet({
        nullptr, // Amiibo Settings
        nullptr, // Controller Selector
        nullptr, // Error Display
        nullptr, // Mii Editor
        nullptr, // Parental Controls
        nullptr, // Photo Viewer
        nullptr, // Profile Selector
        nullptr, // Software Keyboard
        nullptr, // Web Browser
    });

    ConfigureFilesystemProvider(filepath);

    Service::AM::FrontendAppletParameters params{
        .applet_id = Service::AM::AppletId::Application,
        .launch_type = Service::AM::LaunchType::FrontendInitiated,
        .program_index = static_cast<s32>(program_index),
    };
    load_result = system.Load(*window, filepath, params);
    if (load_result != Core::SystemResultStatus::Success) {
        LOG_ERROR(Frontend, "Failed to load {} (result {})", filepath,
                  static_cast<int>(load_result));
        return load_result;
    }

    system.GPU().Start();
    system.GetCpuManager().OnGpuReady();
    system.RegisterExitCallback([this] {
        std::scoped_lock callback_lock{mutex};
        is_running = false;
        cv.notify_one();
    });
    return Core::SystemResultStatus::Success;
}

Core::SystemResultStatus EmulationSession::Launch(const std::string& filepath,
                                                  std::size_t program_index) {
    Stop();
    InitializeSystem(false);

    const auto result = InitializeEmulation(filepath, program_index);
    if (result != Core::SystemResultStatus::Success) {
        ShutdownEmulation(result);
        return result;
    }

    {
        std::scoped_lock lock{mutex};
        is_running = true;
    }
    emulation_thread = std::thread{[this] { RunEmulation(); }};
    return result;
}

void EmulationSession::Stop() {
    {
        std::scoped_lock lock{mutex};
        if (!is_running && !emulation_thread.joinable()) {
            return;
        }
        is_running = false;
        cv.notify_one();
    }

    if (emulation_thread.joinable()) {
        emulation_thread.join();
    }
}

void EmulationSession::Shutdown() {
    Stop();
    std::scoped_lock lock{mutex};
    if (is_initialized) {
        input_subsystem.Shutdown();
        Common::Log::Stop();
        is_initialized = false;
    }
}

void EmulationSession::PresentFrame() {
    // Called from the UI thread. The renderer's Present only works on the thread that
    // owns the CoreWindow (Xbox), so the GPU thread hands the present off here.
    if (!is_running) {
        return;
    }
    try {
        if (system.IsPoweredOn()) {
            system.Renderer().PresentPending();
        }
    } catch (...) {
    }
}

void EmulationSession::RunEmulation() {
    if (Settings::values.use_disk_shader_cache.GetValue()) {
        LoadDiskCacheProgress(VideoCore::LoadCallbackStage::Prepare, 0, 0);
        system.Renderer().ReadRasterizer()->LoadDiskResources(
            system.GetApplicationProcessProgramID(), std::stop_token{}, LoadDiskCacheProgress);
        LoadDiskCacheProgress(VideoCore::LoadCallbackStage::Complete, 0, 0);
    }

    void(system.Run());

    while (true) {
        std::unique_lock lock{mutex};
        if (cv.wait_for(lock, std::chrono::milliseconds(800), [this] { return !is_running; })) {
            break;
        }
    }

    ShutdownEmulation(Core::SystemResultStatus::Success);
}

void EmulationSession::ShutdownEmulation(Core::SystemResultStatus result) {
    std::scoped_lock lock{mutex};
    is_running = false;

    system.HIDCore().UnloadInputDevices();
    system.HIDCore().SetSupportedStyleTag({Core::HID::NpadStyleSet::All});

    if (load_result == Core::SystemResultStatus::Success) {
        system.DetachDebugger();
        system.ShutdownMainProcess();
        detached_tasks.WaitForAllTasks();
        load_result = Core::SystemResultStatus::ErrorNotInitialized;
    }

    window.reset();
    if (on_stopped) {
        on_stopped(static_cast<int>(result));
    }
}

void EmulationSession::SetButtonState(std::size_t player_index, int button_id, bool pressed) {
    if (auto* virtual_gamepad = input_subsystem.GetVirtualGamepad()) {
        virtual_gamepad->SetButtonState(player_index, button_id, pressed);
    }
}

void EmulationSession::SetStickPosition(std::size_t player_index, int stick_id, float x, float y) {
    if (auto* virtual_gamepad = input_subsystem.GetVirtualGamepad()) {
        virtual_gamepad->SetStickPosition(player_index, stick_id, x, y);
    }
}

void EmulationSession::LoadDiskCacheProgress(VideoCore::LoadCallbackStage, int, int) {}

} // namespace CitronUWP