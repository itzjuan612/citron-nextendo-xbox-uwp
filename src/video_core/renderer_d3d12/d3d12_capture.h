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

/// TEMP DIAGNOSTIC (session 15): records a full-mip0 readback of a sampled texture so its
/// content can be dumped to BMP (black scene diagnosis). One dump per image, total capped.
/// Returns true when a dump was armed.
bool RecordTextureContentDump(ID3D12Device* d3d, CommandList& command_list, ID3D12Resource* src,
                              u32 guest_format, u64 guest_addr, u32 image_index,
                              bool force = false);

/// TEMP DIAGNOSTIC (S18, Phase 0.2.1): guest frame counter, incremented once per guest present
/// in RasterizerD3D12::TickFrame (RendererD3D12::Composite calls it once per guest frame, while
/// the S17 "perf" probe counts HOST presents - the two rates differ by a large factor). Every
/// "Frame capture written" line carries it so captures can be sorted by guest phase: the
/// style-screen camera dolly advances on guest time, which makes wall-clock timestamps useless
/// for cross-run comparisons.
extern u64 g_s18_guest_frames;

/// TEMP DIAGNOSTIC (S18, Phase 0.2.1): the float4 of the newest 1x1 R32G32B32A32 exposure
/// readback, written by WriteTextureContentDumps and appended to every "Frame capture written"
/// line. The guest-memory read of that texture always returns zeros (GPU-authored - standing
/// trap), so the resource readback is the only ground truth for the value.
extern std::string g_s18_last_exposure;

} // namespace D3D12
