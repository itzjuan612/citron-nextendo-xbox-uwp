// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// TEMP DIAGNOSTIC (session 11): ring of recorded D3D12 command-list operations.
// Every GPU command recorded into a command list pushes a compact record here. When a
// device removal is detected (delayed check), the last records are dumped so the invalid
// command can be identified. Strip this header and every Diag::Push/Dump call before
// release; the real fixes it is protecting must stay.

#pragma once

#include <atomic>

#include "common/common_types.h"
#include "common/logging.h"

namespace D3D12::Diag {

enum class CmdKind : u32 {
    Marker = 0,
    Transition,
    UavBarrier,
    CopyBufferRegion,
    CopyTextureRegion,
    ClearRtv,
    ClearDsv,
    Draw,
    DrawIndexed,
    Dispatch,
    StateSignal,
};

struct CmdRecord {
    CmdKind kind;
    u64 a;
    u64 b;
    u64 c;
    u64 d;
};

inline constexpr u32 CMD_RING_SIZE = 4096;
inline std::atomic<u32> g_cmd_ring_index{0};
inline CmdRecord g_cmd_ring[CMD_RING_SIZE]{};

inline void Push(CmdKind kind, u64 a = 0, u64 b = 0, u64 c = 0, u64 d = 0) {
    const u32 idx = g_cmd_ring_index.fetch_add(1, std::memory_order_relaxed) % CMD_RING_SIZE;
    g_cmd_ring[idx] = CmdRecord{kind, a, b, c, d};
}

inline void Dump() {
    static std::atomic<bool> dumped{false};
    if (dumped.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    const u32 total = g_cmd_ring_index.load(std::memory_order_relaxed);
    const u32 count = total < CMD_RING_SIZE ? total : CMD_RING_SIZE;
    LOG_CRITICAL(Render_D3D12, "DIAG cmd ring (last {} of {}):", count, total);
    for (u32 i = 0; i < count; ++i) {
        const u32 idx = (total - count + i) % CMD_RING_SIZE;
        const CmdRecord& r = g_cmd_ring[idx];
        LOG_CRITICAL(Render_D3D12, "  cmd#{} kind={} a={:#x} b={:#x} c={:#x} d={:#x}", idx,
                     static_cast<u32>(r.kind), r.a, r.b, r.c, r.d);
    }
}

} // namespace D3D12::Diag
