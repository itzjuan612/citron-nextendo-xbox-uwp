// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <string>

namespace D3D12 {

/// TEMP DIAGNOSTIC (session 12): directory where the presented image is periodically captured
/// to a BMP file, so the current frame can be fetched and inspected without a TV. Capture is
/// gated by Settings::values.uwp_frame_capture and can be disabled at runtime by creating a
/// "capture_off" file in this directory; only the newest captures are kept.
void SetCaptureDirectory(std::string directory);

/// TEMP DIAGNOSTIC (session 13): the active capture directory, reused to write translated
/// shader (SPIR-V) dumps next to the BMP captures for offline inspection.
[[nodiscard]] std::string GetCaptureDirectory();

} // namespace D3D12
