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
#include <winrt/Windows.System.Display.h>
#include <winrt/Windows.UI.Core.h>

#include <windows.h>

#include <algorithm>
#include <exception>
#include <csignal>
#include <filesystem>
#include <new.h>
#include <psapi.h>
#include <string>
#include <thread>
#include <vector>

#include "citron_uwp/content_locator.h"
#include "citron_uwp/session.h"
#include "input_common/drivers/virtual_gamepad.h"
#include "common/host_memory.h"
#include "common/logging.h"
#include "core/hle/kernel/k_process.h"

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

/// Number of poll cycles to hold an injected B press after the system Back button.
constexpr int kBackPulseFrames = 8;
std::atomic<int> g_back_pulse_frames{0};

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
    // B doubles as the Xbox system Back button. BackRequested is swallowed and injects a
    // short pulse here so the guest sees the press even if the system demotes B out of the
    // WGI reading while it handles the Back gesture.
    set(VirtualButton::ButtonB,
        button_state(winrt::Windows::Gaming::Input::GamepadButtons::B) ||
            g_back_pulse_frames.load() > 0);
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
    if (g_back_pulse_frames.load() > 0) {
        --g_back_pulse_frames;
    }
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

        // On Xbox the B button is the system Back button: with no BackRequested handler
        // the system terminates the app on every B press. Swallow the navigation and
        // inject a short B pulse so the emulated game still sees the button.
        try {
            SystemNavigationManager::GetForCurrentView().BackRequested(
                [](IInspectable const&, BackRequestedEventArgs const& e) {
                    e.Handled(true);
                    g_back_pulse_frames.store(kBackPulseFrames);
                });
        } catch (...) {
        }

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

        // Activate the CoreWindow before anything presents to it. Without this the
        // compositor never shows swapchain content: Present() returns S_OK and frames
        // flow, but the screen stays black (this cost a whole bring-up cycle).
        try {
            window_.Activate();
        } catch (...) {
        }

        // Prevent Xbox idle-suspend from killing unattended console runs (no input
        // for minutes returns the app to the dashboard with zero log output).
        try {
            if (!display_request_) {
                display_request_ = winrt::Windows::System::Display::DisplayRequest{};
            }
            display_request_.RequestActive();
        } catch (...) {
        }

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

        // Process-memory sentinel: silent fail-fast deaths (OOM/heap-corruption bypass all
        // exception filters) leave no trace; watching commit climb toward the 5 GB budget
        // every ~10 s identifies the killer before it strikes.
        static constexpr ULONGLONG memory_log_interval_ms = 2000;
        ULONGLONG last_memory_log_tick = GetTickCount64();
        u64 last_commit_mb = 0;
        auto log_process_memory = [&] {
            PROCESS_MEMORY_COUNTERS_EX pmc{};
            pmc.cb = sizeof(pmc);
            if (GetProcessMemoryInfo(GetCurrentProcess(),
                                     reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                                     sizeof(pmc))) {
                const u64 commit_mb = static_cast<u64>(pmc.PrivateUsage) / (1024ULL * 1024ULL);
                if (last_commit_mb != 0 && commit_mb > last_commit_mb + 48) {
                    LOG_CRITICAL(Frontend,
                                 "COMMIT JUMP: {} -> {} MB (+{} MB) within one sentinel interval",
                                 last_commit_mb, commit_mb, commit_mb - last_commit_mb);
                }
                last_commit_mb = commit_mb;
                // Classify committed regions: RWX private ~= JIT code cache (the top
                // suspect for the ~1.3 MB/s boot-time climb), other private, mapped.
                u64 rwx_private = 0;
                u64 other_private = 0;
                u64 mapped = 0;
                struct BigRegion {
                    const void* base;
                    u64 size;
                    u32 protect;
                };
                BigRegion big_regions[8]{};
                size_t big_region_count = 0;
                SYSTEM_INFO si{};
                GetSystemInfo(&si);
                for (LPCVOID addr = si.lpMinimumApplicationAddress; addr < si.lpMaximumApplicationAddress;) {
                    MEMORY_BASIC_INFORMATION mbi{};
                    if (VirtualQuery(addr, &mbi, sizeof(mbi)) == 0) {
                        break;
                    }
                    if (mbi.State == MEM_COMMIT) {
                        const u64 size = static_cast<u64>(mbi.RegionSize);
                        const bool private_mem = (mbi.Type == MEM_PRIVATE);
                        const bool executable =
                            (mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ |
                                            PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
                        if (private_mem && executable) {
                            rwx_private += size;
                        } else if (private_mem) {
                            other_private += size;
                            // Track the largest non-executable private regions so a growing
                            // consumer is identifiable by base address across samples.
                            if (size >= (32ULL * 1024 * 1024)) {
                                size_t pos = 0;
                                while (pos < big_region_count && big_regions[pos].size >= size) {
                                    ++pos;
                                }
                                if (pos < 8) {
                                    const size_t last = big_region_count < 8 ? big_region_count : 7;
                                    for (size_t move = last; move > pos; --move) {
                                        big_regions[move] = big_regions[move - 1];
                                    }
                                    big_regions[pos] = {mbi.BaseAddress, size,
                                                        static_cast<u32>(mbi.Protect)};
                                    if (big_region_count < 8) {
                                        ++big_region_count;
                                    }
                                }
                            }
                        } else {
                            mapped += size;
                        }
                    }
                    addr = reinterpret_cast<LPCVOID>(reinterpret_cast<uintptr_t>(mbi.BaseAddress) +
                                                     mbi.RegionSize);
                }
                LOG_INFO(Frontend,
                         "Memory sentinel: commit={} MB workingset={} MB rwx(JIT)={} MB "
                         "priv={} MB mapped={} MB",
                         static_cast<u64>(pmc.PrivateUsage) / (1024ULL * 1024ULL),
                         static_cast<u64>(pmc.WorkingSetSize) / (1024ULL * 1024ULL),
                         rwx_private / (1024ULL * 1024ULL), other_private / (1024ULL * 1024ULL),
                         mapped / (1024ULL * 1024ULL));
                for (size_t i = 0; i < big_region_count; ++i) {
                    LOG_INFO(Frontend, "  big priv region #{}: base={:#x} size={} MB prot={:#x}", i,
                             reinterpret_cast<uintptr_t>(big_regions[i].base),
                             big_regions[i].size / (1024ULL * 1024ULL), big_regions[i].protect);
                }
            }
            // Guest-side attribution for the commit climb: normal memory is the heap region
            // plus mapped physical memory in guest units. If it tracks the host climb, guest
            // heap growth is the culprit; if it stays flat, the growth is host-side (GPU page
            // tables, caches, kernel mapping structures).
            const Kernel::KProcess* const process = session.System().ApplicationProcess();
            if (process != nullptr) {
                LOG_INFO(Frontend,
                         "Guest memory sentinel: normal={} MB used={} MB total={} MB "
                         "lazy_commits={} sparse_committed={} MB",
                         static_cast<u64>(process->GetNormalMemorySize()) / (1024ULL * 1024ULL),
                         static_cast<u64>(process->GetUsedUserPhysicalMemorySize()) /
                             (1024ULL * 1024ULL),
                         static_cast<u64>(process->GetTotalUserPhysicalMemorySize()) /
                             (1024ULL * 1024ULL),
                         Common::GetLazyBackingCommitCount(),
                         Common::g_sparse_lazy_commit_total_bytes.load(std::memory_order_relaxed) /
                             (1024ULL * 1024ULL));
            }
        };

        while (!closed_) {
            window_.Dispatcher().ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
            // Presents the renderer's pending frame. Must run on this thread (CoreWindow owner).
            session.PresentFrame();
            if (session.IsRunning()) {
                try {
                    PollGamepads();
                } catch (...) {
                }
            }
            const ULONGLONG now_tick = GetTickCount64();
            if (now_tick - last_memory_log_tick >= memory_log_interval_ms) {
                last_memory_log_tick = now_tick;
                log_process_memory();
            }
            Sleep(8);
        }

        try {
            if (display_request_) {
                display_request_.RequestRelease();
                display_request_ = nullptr;
            }
        } catch (...) {
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
    // Lives as long as the view (app lifetime) so the active request is never released
    // during emulation. Xbox idle-suspend returns unattended apps to the dashboard
    // after minutes with no input and zero logs.
    winrt::Windows::System::Display::DisplayRequest display_request_{nullptr};
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
    // Fail-fast paths (OOM, invalid parameters, heap corruption) bypass all exception
    // filters; these hooks are the only way to leave a trace before the silent death.
    _set_new_mode(1);
    _set_new_handler([](size_t size) -> int {
        static std::atomic<u32> oom_count{0};
        const u32 count = oom_count.fetch_add(1, std::memory_order_relaxed) + 1;
        LOG_CRITICAL(Frontend,
                     "operator new failed (OOM): requested {} MB ({} bytes), commit exhausted "
                     "(count {})",
                     static_cast<u64>(size) / (1024ULL * 1024ULL), static_cast<u64>(size), count);
        // Frames 0-1 are the CRT's new machinery; frames 2+ identify the requesting code.
        // Only the first few failures pay for symbolization, and logging stays up because
        // callers can recover (pipeline translation catches bad_alloc).
        if (count <= 4) {
            void* frames[32]{};
            const USHORT frame_count = RtlCaptureStackBackTrace(0, 32, frames, nullptr);
            for (USHORT i = 0; i < frame_count; ++i) {
                const auto frame_address = reinterpret_cast<uintptr_t>(frames[i]);
                HMODULE frame_module = nullptr;
                if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                        reinterpret_cast<LPCWSTR>(frame_address), &frame_module)) {
                    continue;
                }
                LOG_CRITICAL(Frontend, "  new-fail frame #{}: module + 0x{:X}", i,
                             frame_address - reinterpret_cast<uintptr_t>(frame_module));
            }
        }
        throw std::bad_alloc{};
    });
    signal(SIGABRT, [](int) {
        LOG_CRITICAL(Frontend, "SIGABRT raised (abort()/assert path)");
        Common::Log::Stop();
    });
    signal(SIGFPE, [](int) {
        LOG_CRITICAL(Frontend, "SIGFPE raised");
        Common::Log::Stop();
    });
    signal(SIGILL, [](int) {
        LOG_CRITICAL(Frontend, "SIGILL raised");
        Common::Log::Stop();
    });

    CoreApplication::Run(make<CitronUwpSource>());
    return 0;
}