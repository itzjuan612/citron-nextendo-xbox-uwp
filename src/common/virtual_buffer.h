// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

#ifdef CITRON_UWP
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>

#include "common/assert.h"
#include "common/common_types.h"
#endif

namespace Common {

void* AllocateMemoryPages(std::size_t size) noexcept;
void FreeMemoryPages(void* base, std::size_t size) noexcept;

template <typename T>
class VirtualBuffer final {
public:
    // TODO: Uncomment this and change Common::PageTable::PageInfo to be trivially constructible
    // using std::atomic_ref once libc++ has support for it
    // static_assert(
    //     std::is_trivially_constructible_v<T>,
    //     "T must be trivially constructible, as non-trivial constructors will not be executed "
    //     "with the current allocator");

    constexpr VirtualBuffer() = default;
    explicit VirtualBuffer(std::size_t count) : alloc_size{count * sizeof(T)} {
        base_ptr = reinterpret_cast<T*>(AllocateMemoryPages(alloc_size));
    }

    ~VirtualBuffer() noexcept {
        FreeMemoryPages(base_ptr, alloc_size);
    }

    VirtualBuffer(const VirtualBuffer&) = delete;
    VirtualBuffer& operator=(const VirtualBuffer&) = delete;

    VirtualBuffer(VirtualBuffer&& other) noexcept
        : alloc_size{std::exchange(other.alloc_size, 0)}, base_ptr{std::exchange(other.base_ptr,
                                                                   nullptr)} {}

    VirtualBuffer& operator=(VirtualBuffer&& other) noexcept {
        alloc_size = std::exchange(other.alloc_size, 0);
        base_ptr = std::exchange(other.base_ptr, nullptr);
        return *this;
    }

    void resize(std::size_t count) {
        const auto new_size = count * sizeof(T);
        if (new_size == alloc_size) {
            return;
        }

        FreeMemoryPages(base_ptr, alloc_size);

        alloc_size = new_size;
        base_ptr = reinterpret_cast<T*>(AllocateMemoryPages(alloc_size));
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const {
        return base_ptr[index];
    }

    [[nodiscard]] constexpr T& operator[](std::size_t index) {
        return base_ptr[index];
    }

    [[nodiscard]] constexpr T* data() {
        return base_ptr;
    }

    [[nodiscard]] constexpr const T* data() const {
        return base_ptr;
    }

    [[nodiscard]] constexpr std::size_t size() const {
        return alloc_size / sizeof(T);
    }

private:
    std::size_t alloc_size{};
    T* base_ptr{};
};

#ifdef CITRON_UWP

/// Reserves a virtual range without committing it (pages are inaccessible until committed).
void* ReserveMemoryPages(std::size_t size) noexcept;
/// Commits a previously reserved range; the OS zero-fills the pages.
void* CommitMemoryPages(void* base, std::size_t size) noexcept;

/// Default chunk initializer: freshly committed pages are already zero-filled by the OS.
struct SparseLazyZeroInitializer {
    template <typename T>
    void operator()(T*, T*) const noexcept {}
};

/**
 * Lazily committed storage for very large flat tables.
 *
 * A 39-bit address space page table (2^27 entries x 32 bytes = 4 GiB) or the device memory
 * manager entry table (2^28 entries x 16 bytes = 4 GiB) cannot be committed up front within
 * the Xbox game memory budget. This reserves the virtual range and commits 64 KiB chunks on
 * first access, so the tables cost roughly their share of the guest memory actually touched.
 *
 * OnCommit runs once per newly committed chunk (after the OS zero-fill) and can restore
 * non-zero defaults for types that need them.
 */
template <typename T, typename OnCommit = SparseLazyZeroInitializer>
class SparseLazyBuffer final {
public:
    SparseLazyBuffer() = default;
    explicit SparseLazyBuffer(std::size_t count) {
        resize(count);
    }
    ~SparseLazyBuffer();

    SparseLazyBuffer(const SparseLazyBuffer&) = delete;
    SparseLazyBuffer& operator=(const SparseLazyBuffer&) = delete;

    SparseLazyBuffer(SparseLazyBuffer&& other) noexcept;
    SparseLazyBuffer& operator=(SparseLazyBuffer&& other) noexcept;

    void resize(std::size_t count);

    [[nodiscard]] std::size_t size() const {
        return element_count;
    }

    [[nodiscard]] const T& operator[](std::size_t index) const {
        return at(index);
    }

    [[nodiscard]] T& operator[](std::size_t index) {
        return at(index);
    }

private:
    static constexpr std::size_t ChunkSize = 64 * 1024;
    static constexpr u32 ChunkUncommitted = 0;
    static constexpr u32 ChunkCommitting = 1;
    static constexpr u32 ChunkCommitted = 2;

    T& at(std::size_t index) const;
    void CommitChunk(std::size_t chunk) const;
    void Release();

    std::byte* base{};
    std::atomic<u32>* chunk_states{};
    std::size_t chunk_count{};
    std::size_t total_bytes{};
    std::size_t element_count{};
};

template <typename T, typename OnCommit>
SparseLazyBuffer<T, OnCommit>::~SparseLazyBuffer() {
    Release();
}

template <typename T, typename OnCommit>
SparseLazyBuffer<T, OnCommit>::SparseLazyBuffer(SparseLazyBuffer&& other) noexcept
    : base{std::exchange(other.base, nullptr)},
      chunk_states{std::exchange(other.chunk_states, nullptr)},
      chunk_count{std::exchange(other.chunk_count, 0)},
      total_bytes{std::exchange(other.total_bytes, 0)},
      element_count{std::exchange(other.element_count, 0)} {}

template <typename T, typename OnCommit>
SparseLazyBuffer<T, OnCommit>& SparseLazyBuffer<T, OnCommit>::operator=(
    SparseLazyBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    Release();
    base = std::exchange(other.base, nullptr);
    chunk_states = std::exchange(other.chunk_states, nullptr);
    chunk_count = std::exchange(other.chunk_count, 0);
    total_bytes = std::exchange(other.total_bytes, 0);
    element_count = std::exchange(other.element_count, 0);
    return *this;
}

template <typename T, typename OnCommit>
void SparseLazyBuffer<T, OnCommit>::resize(std::size_t count) {
    Release();

    element_count = count;
    total_bytes = count * sizeof(T);
    if (total_bytes == 0) {
        return;
    }

    chunk_count = (total_bytes + ChunkSize - 1) / ChunkSize;
    base = static_cast<std::byte*>(ReserveMemoryPages(total_bytes));
    ASSERT_MSG(base != nullptr, "SparseLazyBuffer: failed to reserve {} bytes", total_bytes);

    chunk_states = new std::atomic<u32>[chunk_count]{};
}

template <typename T, typename OnCommit>
T& SparseLazyBuffer<T, OnCommit>::at(std::size_t index) const {
    ASSERT_MSG(index < element_count,
               "SparseLazyBuffer out-of-range access: index={} element_count={}", index,
               element_count);

    const std::size_t chunk = (index * sizeof(T)) / ChunkSize;
    if (chunk_states[chunk].load(std::memory_order_acquire) != ChunkCommitted) {
        CommitChunk(chunk);
    }
    return reinterpret_cast<T*>(base)[index];
}

template <typename T, typename OnCommit>
void SparseLazyBuffer<T, OnCommit>::CommitChunk(std::size_t chunk) const {
    u32 expected = ChunkUncommitted;
    if (chunk_states[chunk].compare_exchange_strong(expected, ChunkCommitting,
                                                    std::memory_order_acq_rel)) {
        const std::size_t offset = chunk * ChunkSize;
        const std::size_t size = (std::min)(ChunkSize, total_bytes - offset);
        void* committed = CommitMemoryPages(base + offset, size);
        ASSERT_MSG(committed != nullptr, "SparseLazyBuffer: failed to commit {} bytes", size);
        OnCommit{}(reinterpret_cast<T*>(base + offset),
                   reinterpret_cast<T*>(base + offset + size));
        chunk_states[chunk].store(ChunkCommitted, std::memory_order_release);
        return;
    }

    // Another thread is committing this chunk; wait until it is usable.
    while (chunk_states[chunk].load(std::memory_order_acquire) != ChunkCommitted) {
        std::this_thread::yield();
    }
}

template <typename T, typename OnCommit>
void SparseLazyBuffer<T, OnCommit>::Release() {
    if (base != nullptr) {
        FreeMemoryPages(base, total_bytes);
        base = nullptr;
    }
    delete[] chunk_states;
    chunk_states = nullptr;
    chunk_count = 0;
    total_bytes = 0;
    element_count = 0;
}

#endif // CITRON_UWP

} // namespace Common
