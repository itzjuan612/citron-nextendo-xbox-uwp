// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "citron_uwp/content_locator.h"

#include <algorithm>
#include <cctype>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Storage.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "core/file_sys/submission_package.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/loader/loader.h"

namespace CitronUWP {

namespace {

namespace fs = std::filesystem;

using winrt::Windows::Storage::KnownFolders;
using winrt::Windows::Storage::NameCollisionOption;
using winrt::Windows::Storage::StorageFile;
using winrt::Windows::Storage::StorageFolder;

std::wstring ToLower(winrt::hstring name) {
    std::wstring lower{name.c_str()};
    std::transform(lower.begin(), lower.end(), lower.begin(), [](wchar_t c) {
        return static_cast<wchar_t>(std::towlower(c));
    });
    return lower;
}

bool HasExtension(const winrt::hstring& name, std::wstring_view extension) {
    return ToLower(name).ends_with(extension);
}

bool IsContentFile(const winrt::hstring& name) {
    return HasExtension(name, L".nsp") || HasExtension(name, L".xci") ||
           HasExtension(name, L".nca");
}

bool IsKeysFile(const winrt::hstring& name) {
    return HasExtension(name, L".keys");
}

bool IsFirmwareFile(const winrt::hstring& name) {
    return HasExtension(name, L".nca");
}

/// Folders of the removable drive that may hold content: games\, Citron\games\, root.
std::vector<StorageFolder> ContentFolders(const StorageFolder& device) {
    std::vector<StorageFolder> folders;
    const auto try_sub = [](const StorageFolder& parent, const wchar_t* name) -> StorageFolder {
        try {
            const auto item = parent.TryGetItemAsync(name).get();
            if (item) {
                return item.try_as<StorageFolder>();
            }
        } catch (...) {
        }
        return nullptr;
    };

    if (auto games = try_sub(device, L"games")) {
        folders.push_back(games);
    }
    if (auto citron = try_sub(device, L"Citron")) {
        if (auto games = try_sub(citron, L"games")) {
            folders.push_back(games);
        }
    }
    folders.push_back(device);
    return folders;
}

std::vector<StorageFolder> KeysFolders(const StorageFolder& device) {
    std::vector<StorageFolder> folders;
    const auto try_sub = [](const StorageFolder& parent, const wchar_t* name) -> StorageFolder {
        try {
            const auto item = parent.TryGetItemAsync(name).get();
            if (item) {
                return item.try_as<StorageFolder>();
            }
        } catch (...) {
        }
        return nullptr;
    };

    if (auto keys = try_sub(device, L"keys")) {
        folders.push_back(keys);
    }
    if (auto citron = try_sub(device, L"Citron")) {
        if (auto keys = try_sub(citron, L"keys")) {
            folders.push_back(keys);
        }
    }
    folders.push_back(device);
    return folders;
}

/// Firmware search order. Accepts either the raw registered contents or a nand\system tree.
std::vector<std::pair<StorageFolder, std::wstring>> FirmwareFolders(const StorageFolder& device) {
    std::vector<std::pair<StorageFolder, std::wstring>> result;
    const auto resolve = [](const StorageFolder& root, const wchar_t* relative) -> StorageFolder {
        StorageFolder current = root;
        const std::wstring path{relative};
        size_t pos = 0;
        while (pos < path.size()) {
            size_t next = path.find(L'\\', pos);
            if (next == std::wstring::npos) {
                next = path.size();
            }
            const std::wstring part = path.substr(pos, next - pos);
            pos = next + 1;
            if (part.empty()) {
                continue;
            }
            try {
                const auto item = current.TryGetItemAsync(part).get();
                if (!item) {
                    return nullptr;
                }
                current = item.try_as<StorageFolder>();
                if (!current) {
                    return nullptr;
                }
            } catch (...) {
                return nullptr;
            }
        }
        return current;
    };

    const auto add = [&](const wchar_t* relative) {
        if (auto folder = resolve(device, relative)) {
            result.emplace_back(folder, relative);
        }
    };

    add(L"firmware\\registered");
    add(L"Citron\\firmware\\registered");
    add(L"firmware");
    add(L"Citron\\firmware");
    add(L"nand\\system\\Contents\\registered");
    add(L"Citron\\nand\\system\\Contents\\registered");
    add(L"system\\Contents\\registered");
    return result;
}

std::vector<StorageFile> ListFiles(const StorageFolder& folder) {
    std::vector<StorageFile> files;
    try {
        for (const auto& file : folder.GetFilesAsync().get()) {
            files.push_back(file);
        }
    } catch (const winrt::hresult_error& e) {
        LOG_WARNING(Frontend, "Failed to enumerate '{}': {}", winrt::to_string(folder.Path()),
                    winrt::to_string(e.message()));
    }
    return files;
}

std::vector<StorageFolder> ListFolders(const StorageFolder& folder) {
    std::vector<StorageFolder> folders;
    try {
        for (const auto& child : folder.GetFoldersAsync().get()) {
            folders.push_back(child);
        }
    } catch (...) {
    }
    return folders;
}

/// True when the emulator's regular file APIs can open the path directly (no copy needed).
bool HasDirectAccess(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

enum class ContentKind {
    Base,
    UpdateOrDlc,
    Unknown,
};

/// Tells the base game apart from an update or DLC package. Games arrive as several NSPs in the
/// same folder (base, updates, DLC); booting an update or DLC alone fails, so the base package is
/// what the frontend must launch. XCI/NCA are always treated as base candidates.
ContentKind ClassifyContent(const std::string& path, const winrt::hstring& name) {
    if (!HasExtension(name, L".nsp")) {
        return ContentKind::Base;
    }

    auto vfs = std::make_shared<FileSys::RealVfsFilesystem>();
    auto file = vfs->OpenFile(path, FileSys::OpenMode::Read);
    if (!file) {
        return ContentKind::Unknown;
    }

    const FileSys::NSP nsp(file);
    if (nsp.GetStatus() != Loader::ResultStatus::Success) {
        return ContentKind::Unknown;
    }
    if (nsp.GetProgramStatus() != Loader::ResultStatus::Success) {
        // DLC packages carry no Program NCA.
        return ContentKind::UpdateOrDlc;
    }

    // Update programs share the base title ID with the 0x800 bit set.
    const u64 title_id = nsp.GetProgramTitleID();
    return title_id != 0 && (title_id & 0x800) == 0 ? ContentKind::Base
                                                     : ContentKind::UpdateOrDlc;
}

std::string FindLocalContent(const fs::path& dir) {
    static constexpr const char* extensions[] = {".nsp", ".xci", ".nca"};

    std::vector<fs::path> candidates;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return {};
    }
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        for (const char* wanted : extensions) {
            if (ext == wanted) {
                candidates.push_back(entry.path());
                break;
            }
        }
    }
    if (candidates.empty()) {
        return {};
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.front().string();
}

bool CopyToLocalFolder(const StorageFile& file, const fs::path& dest_dir,
                       const wchar_t* dest_name = nullptr) {
    try {
        std::error_code ec;
        fs::create_directories(dest_dir, ec);
        const auto destination = StorageFolder::GetFolderFromPathAsync(dest_dir.wstring()).get();
        file.CopyAsync(destination, dest_name ? winrt::hstring{dest_name} : file.Name(),
                       NameCollisionOption::ReplaceExisting)
            .get();
        return true;
    } catch (const winrt::hresult_error& e) {
        LOG_WARNING(Frontend, "Failed to copy '{}': {}", winrt::to_string(file.Name()),
                    winrt::to_string(e.message()));
        return false;
    }
}

std::vector<StorageFolder> RemovableDevices() {
    std::vector<StorageFolder> devices;
    try {
        for (const auto& item : KnownFolders::RemovableDevices().GetFoldersAsync().get()) {
            const std::string path = winrt::to_string(item.Path());
            LOG_INFO(Frontend, "Removable device detected: {}", path.empty() ? "(no path)" : path);
            devices.push_back(item);
        }
    } catch (const winrt::hresult_error& e) {
        LOG_WARNING(Frontend, "Removable device enumeration failed: {}",
                    winrt::to_string(e.message()));
    }
    return devices;
}

void ImportKeys(const StorageFolder& device, const fs::path& keys_dir) {
    if (fs::exists(keys_dir / "prod.keys")) {
        return;
    }
    for (const auto& folder : KeysFolders(device)) {
        for (const auto& file : ListFiles(folder)) {
            if (!IsKeysFile(file.Name())) {
                continue;
            }
            if (CopyToLocalFolder(file, keys_dir, nullptr)) {
                LOG_INFO(Frontend, "Imported {} from '{}'", winrt::to_string(file.Name()),
                         winrt::to_string(file.Path()));
            }
        }
        if (fs::exists(keys_dir / "prod.keys")) {
            return;
        }
    }
}

void CopyFirmwareFiles(const StorageFolder& source, const fs::path& registered_dir, u32 depth,
                       u32& copied) {
    for (const auto& file : ListFiles(source)) {
        if (!IsFirmwareFile(file.Name())) {
            continue;
        }
        const std::wstring name{file.Name().c_str()};
        const fs::path dest_path = registered_dir / fs::path{name};
        if (fs::exists(dest_path)) {
            // Already imported (firmware import is a one-time first-run step).
            continue;
        }
        if (CopyToLocalFolder(file, registered_dir, name.c_str())) {
            ++copied;
        }
    }
    if (depth >= 2) {
        return;
    }
    for (const auto& folder : ListFolders(source)) {
        CopyFirmwareFiles(folder, registered_dir, depth + 1, copied);
    }
}

void ImportFirmware(const StorageFolder& device, const fs::path& registered_dir) {
    for (const auto& [folder, description] : FirmwareFolders(device)) {
        // Only accept a folder that actually contains firmware (an .nca file).
        const auto files = ListFiles(folder);
        const bool has_firmware = std::any_of(files.begin(), files.end(), [](const StorageFile& f) {
            return IsFirmwareFile(f.Name());
        });
        if (!has_firmware) {
            continue;
        }
        u32 copied = 0;
        CopyFirmwareFiles(folder, registered_dir, 0, copied);
        if (copied > 0) {
            LOG_INFO(Frontend, "Imported {} firmware file(s) from '{}' ({})", copied,
                     winrt::to_string(folder.Path()), winrt::to_string(description));
        }
        return;
    }
}

} // Anonymous namespace

void ImportExternalContent(const std::string& app_dir) {
    const fs::path root{app_dir};
    const fs::path keys_dir = root / "keys";
    const fs::path registered_dir = root / "nand" / "system" / "Contents" / "registered";

    std::error_code ec;
    const bool have_keys = fs::exists(keys_dir / "prod.keys");
    const bool have_firmware = fs::exists(registered_dir) && !fs::is_empty(registered_dir, ec);
    if (have_keys && have_firmware) {
        return;
    }

    for (const auto& device : RemovableDevices()) {
        if (!fs::exists(keys_dir / "prod.keys")) {
            ImportKeys(device, keys_dir);
        }
        std::error_code firmware_ec;
        if (!fs::exists(registered_dir) || fs::is_empty(registered_dir, firmware_ec)) {
            ImportFirmware(device, registered_dir);
        }
    }
}

std::string FindBootableContent(const std::string& app_dir) {
    const fs::path root{app_dir};

    std::string local = FindLocalContent(root / "games");
    if (local.empty()) {
        local = FindLocalContent(root);
    }
    if (!local.empty()) {
        LOG_INFO(Frontend, "Found content in app storage: {}", local);
        return local;
    }

    std::string unclassified_path;
    std::string update_or_dlc_path;

    for (const auto& device : RemovableDevices()) {
        for (const auto& folder : ContentFolders(device)) {
            auto files = ListFiles(folder);
            std::sort(files.begin(), files.end(), [](const StorageFile& a, const StorageFile& b) {
                return std::wstring{a.Name().c_str()} < std::wstring{b.Name().c_str()};
            });
            for (const auto& file : files) {
                if (!IsContentFile(file.Name())) {
                    continue;
                }
                const std::string path = winrt::to_string(file.Path());
                if (path.empty()) {
                    continue;
                }
                if (!HasDirectAccess(path)) {
                    // The drive did not grant direct access to the emulator's file APIs; copy the
                    // title into the app data folder instead.
                    LOG_INFO(Frontend,
                             "Direct access to '{}' unavailable; copying into app storage", path);
                    const fs::path games_dir = root / "games";
                    if (CopyToLocalFolder(file, games_dir, nullptr)) {
                        const std::string copied =
                            games_dir.string() + "\\" + winrt::to_string(file.Name());
                        LOG_INFO(Frontend, "Copied content to app storage: {}", copied);
                        return copied;
                    }
                    continue;
                }

                switch (ClassifyContent(path, file.Name())) {
                case ContentKind::Base:
                    LOG_INFO(Frontend, "Launching content from removable storage: {}", path);
                    return path;
                case ContentKind::UpdateOrDlc:
                    LOG_INFO(Frontend, "Skipping '{}' (update or DLC, not a base game)", path);
                    if (update_or_dlc_path.empty()) {
                        update_or_dlc_path = path;
                    }
                    break;
                case ContentKind::Unknown:
                    LOG_WARNING(Frontend, "Could not identify '{}'; deferring", path);
                    if (unclassified_path.empty()) {
                        unclassified_path = path;
                    }
                    break;
                }
            }
        }
    }

    // No base game was positively identified: prefer a package we could not parse over one that
    // is known to be an update/DLC (keys may be missing, which breaks classification).
    if (!unclassified_path.empty()) {
        LOG_WARNING(Frontend, "No base game identified; launching '{}'", unclassified_path);
        return unclassified_path;
    }
    if (!update_or_dlc_path.empty()) {
        LOG_WARNING(Frontend, "No base game found; launching '{}' as a last resort",
                    update_or_dlc_path);
    }
    return update_or_dlc_path;
}

} // namespace CitronUWP
