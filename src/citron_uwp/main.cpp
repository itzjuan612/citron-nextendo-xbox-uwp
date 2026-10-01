// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// UWP (Xbox / Windows Store) frontend host.
//
// Owns the CoreWindow, hands the surface to the emulation session and pumps controller
// input through Windows.Gaming.Input into the input subsystem.

#include <unknwn.h>
#include <inspectable.h>
#include <winrt/base.h>
#include <winrt/Windows.ApplicationModel.Activation.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Gaming.Input.h>
#include <winrt/Windows.Graphics.Display.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.UI.Core.h>

#include <windows.h>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "citron_uwp/content_locator.h"
#include "citron_uwp/session.h"
#include "input_common/drivers/virtual_gamepad.h"
#include "common/logging.h"

using namespace winrt;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::UI::Core;

namespace {

// ICoreWindowInterop (declared manually to avoid windows.ui.core.h conflicting with C++/WinRT).
MIDL_INTERFACE("45d64a29-a63e-4cb6-b498-5781d298cb4f")
ICoreWindowInterop : public ::IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE get_WindowHandle(HWND* hwnd) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_MessageHandled(unsigned char value) = 0;
};

std::atomic<bool> g_boot_started{false};
std::thread g_boot_thread;

/// Logs unhandled exceptions (code, faulting module and offset) so crashes are symbolizable
/// with the shipped PDB; the console exposes no crash dumps through Device Portal.
LONG WINAPI CrashHandler(EXCEPTION_POINTERS* info) {
    const auto* record = info->ExceptionRecord;
    const auto address = reinterpret_cast<uintptr_t>(record->ExceptionAddress);

    HMODULE module = nullptr;
    wchar_t module_path[MAX_PATH]{};
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(address), &module)) {
        GetModuleFileNameW(module, module_path, MAX_PATH);
    }
    const auto module_base = reinterpret_cast<uintptr_t>(module);
    const auto module_name =
        module_path[0] != L'\0' ? winrt::to_string(winrt::hstring{module_path}) : "(unknown)";

    if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2) {
        LOG_CRITICAL(Frontend,
                     "Unhandled exception 0x{:08X} ({} address 0x{:016X}) at {} + 0x{:X} "
                     "(thread {})",
                     record->ExceptionCode,
                     record->ExceptionInformation[0] != 0 ? "writing" : "reading",
                     record->ExceptionInformation[1], module_name, address - module_base,
                     GetCurrentThreadId());
    } else {
        LOG_CRITICAL(Frontend, "Unhandled exception 0x{:08X} at {} + 0x{:X} (thread {})",
                     record->ExceptionCode, module_name, address - module_base,
                     GetCurrentThreadId());
    }

    void* frames[48]{};
    const USHORT frame_count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
    for (USHORT i = 0; i < frame_count; ++i) {
        const auto frame_address = reinterpret_cast<uintptr_t>(frames[i]);
        HMODULE frame_module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(frame_address), &frame_module)) {
            continue;
        }
        LOG_CRITICAL(Frontend, "  frame #{}: {} + 0x{:X}", i, "module",
                     frame_address - reinterpret_cast<uintptr_t>(frame_module));
    }

    Common::Log::Stop();
    return EXCEPTION_EXECUTE_HANDLER;
}

/// Polls Windows.Gaming.Input and forwards state to the emulated controller.
void PollGamepads() {
    using winrt::Windows::Gaming::Input::Gamepad;
    using VirtualButton = InputCommon::VirtualGamepad::VirtualButton;
    using VirtualStick = InputCommon::VirtualGamepad::VirtualStick;

    static Gamepad pad{nullptr};
    static uint64_t last_buttons = 0;

    auto pads = Gamepad::Gamepads();
    if (pads.Size() == 0) {
        pad = nullptr;
        return;
    }
    pad = pads.GetAt(0);
    if (!pad) {
        return;
    }

    const auto reading = pad.GetCurrentReading();
    const uint64_t buttons = static_cast<uint64_t>(reading.Buttons);
    if (buttons == last_buttons && pad == nullptr) {
        return;
    }

    auto& session = CitronUWP::EmulationSession::GetInstance();
    auto set = [&](VirtualButton button, bool pressed) {
        session.SetButtonState(0, static_cast<int>(button), pressed);
    };

    const auto button_state = [buttons](winrt::Windows::Gaming::Input::GamepadButtons mask) {
        return (buttons & static_cast<uint64_t>(mask)) != 0;
    };

    set(VirtualButton::ButtonA, button_state(winrt::Windows::Gaming::Input::GamepadButtons::A));
    set(VirtualButton::ButtonB, button_state(winrt::Windows::Gaming::Input::GamepadButtons::B));
    set(VirtualButton::ButtonX, button_state(winrt::Windows::Gaming::Input::GamepadButtons::X));
    set(VirtualButton::ButtonY, button_state(winrt::Windows::Gaming::Input::GamepadButtons::Y));
    set(VirtualButton::TriggerL, button_state(winrt::Windows::Gaming::Input::GamepadButtons::LeftShoulder));
    set(VirtualButton::TriggerR, button_state(winrt::Windows::Gaming::Input::GamepadButtons::RightShoulder));
    set(VirtualButton::StickL, button_state(winrt::Windows::Gaming::Input::GamepadButtons::LeftThumbstick));
    set(VirtualButton::StickR, button_state(winrt::Windows::Gaming::Input::GamepadButtons::RightThumbstick));
    set(VirtualButton::ButtonMinus, button_state(winrt::Windows::Gaming::Input::GamepadButtons::View));
    set(VirtualButton::ButtonPlus, button_state(winrt::Windows::Gaming::Input::GamepadButtons::Menu));
    set(VirtualButton::ButtonLeft, button_state(winrt::Windows::Gaming::Input::GamepadButtons::DPadLeft));
    set(VirtualButton::ButtonUp, button_state(winrt::Windows::Gaming::Input::GamepadButtons::DPadUp));
    set(VirtualButton::ButtonRight, button_state(winrt::Windows::Gaming::Input::GamepadButtons::DPadRight));
    set(VirtualButton::ButtonDown, button_state(winrt::Windows::Gaming::Input::GamepadButtons::DPadDown));

    session.SetStickPosition(0, static_cast<int>(VirtualStick::Left),
                             static_cast<float>(reading.LeftThumbstickX),
                             static_cast<float>(reading.LeftThumbstickY));
    session.SetStickPosition(0, static_cast<int>(VirtualStick::Right),
                             static_cast<float>(reading.RightThumbstickX),
                             static_cast<float>(reading.RightThumbstickY));

    last_buttons = buttons;
}

class CitronUwpView : public implements<CitronUwpView, IFrameworkView> {
public:
    void Initialize(CoreApplicationView const& view) {
        view.Activated([this](CoreApplicationView const&,
                              Windows::ApplicationModel::Activation::IActivatedEventArgs const&) {
            if (window_) {
                window_.Activate();
            }
        });
    }

    void SetWindow(CoreWindow const& window) {
        window_ = window;
        window_.Closed([this](CoreWindow const&, CoreWindowEventArgs const&) { closed_ = true; });

        float width = 1280.0f;
        float height = 720.0f;
        float scale = 1.0f;
        try {
            const auto bounds = window_.Bounds();
            scale = static_cast<float>(
                winrt::Windows::Graphics::Display::DisplayInformation::GetForCurrentView()
                    .LogicalDpi()) /
                    96.0f;
            width = static_cast<float>(bounds.Width) * scale;
            height = static_cast<float>(bounds.Height) * scale;
        } catch (...) {
        }

        void* hwnd = nullptr;
        void* core_window = reinterpret_cast<void*>(winrt::get_unknown(window_));
        winrt::com_ptr<ICoreWindowInterop> interop;
        if (SUCCEEDED(winrt::get_unknown(window_)->QueryInterface(
                __uuidof(ICoreWindowInterop), interop.put_void()))) {
            HWND native = nullptr;
            if (SUCCEEDED(interop->get_WindowHandle(&native))) {
                hwnd = native;
            }
        }

        CitronUWP::EmulationSession::GetInstance().SetWindow(hwnd, core_window,
                                                             static_cast<u32>(width),
                                                             static_cast<u32>(height), scale);
    }

    void Load(hstring const&) {}
    void Uninitialize() {}

    void Run() {
        auto& session = CitronUWP::EmulationSession::GetInstance();

        // Resolve the app data directory and boot content once the window exists.
        try {
            const auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
            const std::string app_dir = winrt::to_string(folder.Path());

            if (!g_boot_started.exchange(true)) {
                g_boot_thread = std::thread([app_dir] {
                    // The boot worker uses WinRT for removable-storage discovery.
                    try {
                        winrt::init_apartment(winrt::apartment_type::multi_threaded);
                    } catch (...) {
                    }

                    // Pull prod.keys / firmware from a USB drive before the session reads them.
                    try {
                        CitronUWP::ImportExternalContent(app_dir);
                    } catch (const std::exception& e) {
                        LOG_ERROR(Frontend, "External content import failed: {}", e.what());
                    }

                    auto& boot_session = CitronUWP::EmulationSession::GetInstance();
                    boot_session.Initialize(app_dir);

                    const std::string game = CitronUWP::FindBootableContent(app_dir);
                    if (game.empty()) {
                        LOG_ERROR(Frontend,
                                  "No bootable content found. Put an NSP/XCI/NCA in "
                                  "LocalFolder\\games, or on a USB drive (games\\ folder or "
                                  "Citron\\games\\), with prod.keys in keys\\ and firmware in "
                                  "firmware\\.");
                        return;
                    }
                    LOG_INFO(Frontend, "Booting {}", game);
                    const auto result = boot_session.Launch(game, 0);
                    if (result != Core::SystemResultStatus::Success) {
                        LOG_ERROR(Frontend, "Launch failed: {}", static_cast<int>(result));
                    }
                });
            }
        } catch (const hresult_error& e) {
            LOG_ERROR(Frontend, "Failed to resolve LocalFolder: {}", winrt::to_string(e.message()));
        }

        while (!closed_) {
            window_.Dispatcher().ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
            if (session.IsRunning()) {
                try {
                    PollGamepads();
                } catch (...) {
                }
            }
            Sleep(8);
        }

        session.Stop();
        session.Shutdown();
        if (g_boot_thread.joinable()) {
            g_boot_thread.join();
        }
    }

private:
    CoreWindow window_{nullptr};
    bool closed_{false};
};

class CitronUwpSource : public implements<CitronUwpSource, IFrameworkViewSource> {
public:
    IFrameworkView CreateView() { return make<CitronUwpView>(); }
};

class CitronUwpFactory : public implements<CitronUwpFactory, IActivationFactory> {
public:
    hstring GetRuntimeClassName() const { return L"CitronUwp.App"; }
    IInspectable ActivateInstance() const { return make<CitronUwpSource>().as<IInspectable>(); }
};

} // namespace

extern "C" __declspec(dllexport) HRESULT __stdcall DllGetActivationFactory(
    HSTRING, IActivationFactory** factory) {
    *factory = reinterpret_cast<IActivationFactory*>(detach_abi(make<CitronUwpFactory>()));
    return S_OK;
}

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    // UWP already initializes the apartment; do not call init_apartment here.
    SetUnhandledExceptionFilter(CrashHandler);
    std::set_terminate([] {
        LOG_CRITICAL(Frontend, "std::terminate called");
        Common::Log::Stop();
        std::abort();
    });

    CoreApplication::Run(make<CitronUwpSource>());
    return 0;
}