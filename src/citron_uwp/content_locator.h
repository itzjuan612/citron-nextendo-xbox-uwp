// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Removable-storage support for the UWP frontend.
//
// Xbox cannot use broadFileSystemAccess or the FolderPicker, so external drives are reached
// through the `removableStorage` capability and KnownFolders.RemovableDevices. That grant is
// filtered to the file types declared in the manifest (see the file type association in
// tools/uwp/AppxManifest.xml). Keys and firmware are copied into the app's LocalFolder once;
// games are launched in place when the drive grants direct access, otherwise copied.

#pragma once

#include <string>

namespace CitronUWP {

/// Looks for `prod.keys`/`title.keys` and firmware on attached removable devices and copies
/// them into the app data folder when not already present. Must run before the emulation
/// session initialises (KeyManager::ReloadKeys reads LocalFolder\keys).
void ImportExternalContent(const std::string& app_dir);

/// Finds the first bootable NSP/XCI/NCA: LocalFolder\games first, then the app data root,
/// then any attached removable device (games\ and Citron\games\ and the drive root).
/// Returns a host path usable by the emulator, or an empty string when nothing was found.
std::string FindBootableContent(const std::string& app_dir);

} // namespace CitronUWP
