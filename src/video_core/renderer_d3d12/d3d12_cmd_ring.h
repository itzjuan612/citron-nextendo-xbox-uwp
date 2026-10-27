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
#include <unordered_map>

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

// S25: batch bookkeeping for the Close-failure probe. MarkBatch() runs before every
// submit; LogBatchSummary() (called on a failed Close) counts the retained ops of the
// failing batch and correlates them with resources whose D3D12 objects were destroyed,
// because Xbox D3D12 fails Close with E_INVALIDARG when a recorded-but-unexecuted
// command references a destroyed resource (see RasterizerD3D12::TickFrame's flush
// comment and StagingBufferPool::EvictLRU).
inline std::atomic<u32> g_batch_start{0};

inline void MarkBatch() {
    g_batch_start.store(g_cmd_ring_index.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
}

inline constexpr u32 DESTROYED_RING_SIZE = 32;
inline std::atomic<u32> g_destroyed_index{0};
inline u64 g_destroyed_ptrs[DESTROYED_RING_SIZE]{};

/// S25: set by the first Close failure in a run. While armed, RasterizerD3D12::Draw
/// flushes the list after every draw, so the NEXT (recurring) Close failure pins the
/// exact draw/segment that carries the invalid op.
inline std::atomic<bool> g_arm_per_draw_flush{false};
inline std::atomic<u32> g_per_draw_flush_budget{0};

/// Records a resource pointer whose D3D12 object was just destroyed. Call from the
/// destroy site; LogBatchSummary scans recorded ops for these pointers.
inline void NoteDestroyedResource(u64 ptr) {
    if (ptr == 0) {
        return;
    }
    const u32 idx = g_destroyed_index.fetch_add(1, std::memory_order_relaxed);
    g_destroyed_ptrs[idx % DESTROYED_RING_SIZE] = ptr;
}

inline void LogBatchSummary() {
    const u32 total = g_cmd_ring_index.load(std::memory_order_relaxed);
    const u32 start = g_batch_start.load(std::memory_order_relaxed);
    const u32 window = total >= start ? total - start : 0;
    // Only the last CMD_RING_SIZE records survive the overwrite.
    const u32 scan = window > CMD_RING_SIZE ? CMD_RING_SIZE : window;
    u32 counts[11]{};
    for (u32 i = total - scan; i < total; ++i) {
        const CmdRecord& r = g_cmd_ring[i % CMD_RING_SIZE];
        if (static_cast<u32>(r.kind) < 11) {
            ++counts[static_cast<u32>(r.kind)];
        }
    }
    LOG_CRITICAL(Render_D3D12,
                 "DIAG batch: recorded={} retained={} draw={} drawIdx={} dispatch={} "
                 "cpyBuf={} cpyTex={} trans={} uav={} clear={} state/mark={}",
                 window, scan, counts[7], counts[8], counts[9], counts[3], counts[4],
                 counts[1], counts[2], counts[5] + counts[6], counts[10] + counts[0]);
    // S25 state simulator: track one implied state per resource across the failing batch
    // (the backend's invariant is COMMON between ops) and flag any transition whose
    // recorded StateBefore disagrees - an unbalanced enter/exit pair wedges a resource and
    // every later op claiming COMMON on it fails Close with E_INVALIDARG.
    {
        std::unordered_map<u64, u32> implied;
        implied.reserve(1024);
        u32 flagged = 0;
        for (u32 i = total - scan; i < total; ++i) {
            const CmdRecord& r = g_cmd_ring[i % CMD_RING_SIZE];
            if (r.kind != CmdKind::Transition || r.a == 0) {
                continue;
            }
            const auto [it, inserted] = implied.try_emplace(r.a, 0u);
            if (it->second != static_cast<u32>(r.b)) {
                if (flagged < 12) {
                    LOG_CRITICAL(Render_D3D12,
                                 "DIAG state mismatch: res={:#x} implied={:#x} recorded "
                                 "before={:#x} after={:#x} (record #{})",
                                 r.a, it->second, static_cast<u32>(r.b),
                                 static_cast<u32>(r.c), i);
                }
                ++flagged;
            }
            it->second = static_cast<u32>(r.c);
        }
        LOG_CRITICAL(Render_D3D12, "DIAG state sim: {} transition-state mismatches",
                     flagged);
    }
    const u32 destroyed_n = g_destroyed_index.load(std::memory_order_relaxed);
    if (destroyed_n == 0) {
        return;
    }
    u32 total_refs = 0;
    for (u32 k = 0; k < DESTROYED_RING_SIZE && k < destroyed_n; ++k) {
        const u64 ptr = g_destroyed_ptrs[k];
        if (ptr == 0) {
            continue;
        }
        u32 refs = 0;
        for (u32 i = total - scan; i < total; ++i) {
            const CmdRecord& r = g_cmd_ring[i % CMD_RING_SIZE];
            if (r.a == ptr || r.b == ptr) {
                ++refs;
            }
        }
        if (refs != 0 && total_refs < 4) {
            LOG_CRITICAL(Render_D3D12,
                         "DIAG destroyed resource {:#x} still referenced by {} ops in the "
                         "failing batch",
                         ptr, refs);
        }
        total_refs += refs;
    }
    LOG_CRITICAL(Render_D3D12, "DIAG destroyed ring: tracked={} refs_in_batch={}",
                 destroyed_n < DESTROYED_RING_SIZE ? destroyed_n : DESTROYED_RING_SIZE,
                 total_refs);
}

inline void Dump() {
    // S20: dump on EVERY Close failure (was once per run) - the run's systematic failures hit
    // *different* batches, and only the failing batch's tail names the invalid command.
    // S25: window = the WHOLE failing batch (since the last successful submit) instead of a
    // fixed 512 tail - the invalid op sat outside the old window, and the ring holds 4096.
    static std::atomic<u32> dumps_done{0};
    if (dumps_done.fetch_add(1, std::memory_order_relaxed) >= 8) {
        return;
    }
    const u32 total = g_cmd_ring_index.load(std::memory_order_relaxed);
    const u32 start = g_batch_start.load(std::memory_order_relaxed);
    const u32 window = total >= start ? total - start : 0;
    const u32 count = window > CMD_RING_SIZE ? CMD_RING_SIZE : window;
    LOG_CRITICAL(Render_D3D12, "DIAG cmd ring (failing batch, {} of {} recorded):", count,
                 window);
    for (u32 i = 0; i < count; ++i) {
        const u32 idx = (total - count + i) % CMD_RING_SIZE;
        const CmdRecord& r = g_cmd_ring[idx];
        LOG_CRITICAL(Render_D3D12, "  cmd#{} kind={} a={:#x} b={:#x} c={:#x} d={:#x}", idx,
                     static_cast<u32>(r.kind), r.a, r.b, r.c, r.d);
    }
}

} // namespace D3D12::Diag
