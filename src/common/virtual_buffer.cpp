// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "common/assert.h"
#include "common/common_funcs.h"
#include "common/logging.h"
#include "common/virtual_buffer.h"

namespace Common {

void* AllocateMemoryPages(std::size_t size) noexcept {
    if (size == 0) {
        return nullptr;
    }
#ifdef _WIN32
    void* base{VirtualAlloc(nullptr, size, MEM_COMMIT, PAGE_READWRITE)};
#else
    void* base{mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)};

    if (base == MAP_FAILED) {
        base = nullptr;
    }
#endif

    if (!base) {
        // Callers write through the result unconditionally, so a failed allocation cannot be
        // recovered from here; take the process down with the cause in the log.
        LOG_CRITICAL(Common_Memory, "AllocateMemoryPages failed to allocate {} bytes", size);
        Crash();
    }

    return base;
}

void FreeMemoryPages(void* base, [[maybe_unused]] std::size_t size) noexcept {
    if (!base) {
        return;
    }
#ifdef _WIN32
    ASSERT(VirtualFree(base, 0, MEM_RELEASE));
#else
    ASSERT(munmap(base, size) == 0);
#endif
}

#ifdef CITRON_UWP
void* ReserveMemoryPages(std::size_t size) noexcept {
    if (size == 0) {
        return nullptr;
    }
    void* base{VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE)};
    if (!base) {
        LOG_CRITICAL(Common_Memory, "ReserveMemoryPages failed to reserve {} bytes", size);
        Crash();
    }
    return base;
}

void* CommitMemoryPages(void* base, std::size_t size) noexcept {
    if (size == 0) {
        return base;
    }
    void* committed{VirtualAlloc(base, size, MEM_COMMIT, PAGE_READWRITE)};
    if (!committed) {
        LOG_CRITICAL(Common_Memory, "CommitMemoryPages failed to commit {} bytes at {:p}",
                     size, static_cast<const void*>(base));
        Crash();
    }
    return committed;
}
#endif

} // namespace Common
