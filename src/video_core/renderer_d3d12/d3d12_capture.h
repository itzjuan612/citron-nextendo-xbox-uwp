// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <string>

#include "video_core/renderer_d3d12/d3d12_command_list.h"

namespace D3D12 {

/// TEMP DIAGNOSTIC (session 12): directory where the presented image is periodically captured
/// to a BMP file, so the current frame can be fetched and inspected without a TV. Capture is
/// gated by Settings::values.uwp_frame_capture and can be disabled at runtime by creating a
/// "capture_off" file in this directory; only the newest captures are kept.
void SetCaptureDirectory(std::string directory);

/// TEMP DIAGNOSTIC (session 13): the active capture directory, reused to write translated
/// shader (SPIR-V) dumps next to the BMP captures for offline inspection.
[[nodiscard]] std::string GetCaptureDirectory();

/// TEMP DIAGNOSTIC (session 14): records a scene3d readback copy mid-frame, immediately
/// after a scene draw, so the copy executes before any post-composite clear of the RT.
void RecordScene3dCaptureMidFrame(ID3D12Device* d3d, CommandList& command_list,
                                  ID3D12Resource* src);

} // namespace D3D12
