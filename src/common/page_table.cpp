// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/page_table.h"
#include "common/scope_exit.h"

#ifdef CITRON_UWP
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <utility>

#include "common/assert.h"
#include "common/logging.h"
#endif

namespace Common {

PageTable::PageTable() = default;

PageTable::~PageTable() noexcept = default;

bool PageTable::BeginTraversal(TraversalEntry* out_entry, TraversalContext* out_context,
                               Common::ProcessAddress address) const {
    out_context->next_offset = GetInteger(address);
    out_context->next_page = address / page_size;

    return this->ContinueTraversal(out_entry, out_context);
}

bool PageTable::ContinueTraversal(TraversalEntry* out_entry, TraversalContext* context) const {
    // Setup invalid defaults.
    out_entry->phys_addr = 0;
    out_entry->block_size = page_size;

    // Regardless of whether the page was mapped, advance on exit.
    SCOPE_EXIT {
        context->next_page += 1;
        context->next_offset += page_size;
    };

    // Validate that we can read the actual entry.
    const auto page = context->next_page;
    if (page >= entries.size()) {
        return false;
    }

    // Validate that the entry is mapped.
    const auto phys_addr = entries[page].backing_addr;
    if (phys_addr == 0) {
        return false;
    }

    // Populate the results.
    out_entry->phys_addr = phys_addr + context->next_offset;

    return true;
}

void PageTable::Resize(std::size_t address_space_width_in_bits, std::size_t page_size_in_bits) {
    const std::size_t num_page_table_entries = 1ULL << (address_space_width_in_bits - page_size_in_bits);
    entries.resize(num_page_table_entries);
    current_address_space_width_in_bits = address_space_width_in_bits;
    page_size = 1ULL << page_size_in_bits;
}

#ifdef CITRON_UWP
namespace {
constexpr std::size_t SparseChunkSize = 64 * 1024;
constexpr u32 ChunkUncommitted = 0;
constexpr u32 ChunkCommitting = 1;
constexpr u32 ChunkCommitted = 2;
} // Anonymous namespace

template <typename T>
SparseLazyBuffer<T>::~SparseLazyBuffer() {
    Release();
}

template <typename T>
SparseLazyBuffer<T>::SparseLazyBuffer(SparseLazyBuffer&& other) noexcept
    : base{std::exchange(other.base, nullptr)},
      chunk_states{std::exchange(other.chunk_states, nullptr)},
      chunk_count{std::exchange(other.chunk_count, 0)},
      total_bytes{std::exchange(other.total_bytes, 0)},
      element_count{std::exchange(other.element_count, 0)} {}

template <typename T>
SparseLazyBuffer<T>& SparseLazyBuffer<T>::operator=(SparseLazyBuffer&& other) noexcept {
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

template <typename T>
void SparseLazyBuffer<T>::resize(std::size_t count) {
    Release();

    element_count = count;
    total_bytes = count * sizeof(T);
    if (total_bytes == 0) {
        return;
    }

    chunk_count = (total_bytes + SparseChunkSize - 1) / SparseChunkSize;
    base = static_cast<std::byte*>(VirtualAlloc(nullptr, total_bytes, MEM_RESERVE, PAGE_READWRITE));
    ASSERT_MSG(base != nullptr, "SparseLazyBuffer: failed to reserve {} bytes", total_bytes);

    chunk_states = new std::atomic<u32>[chunk_count]{};
}

template <typename T>
T& SparseLazyBuffer<T>::at(std::size_t index) const {
    if (index >= element_count) [[unlikely]] {
        LOG_CRITICAL(Common, "SparseLazyBuffer: out-of-range access index={} (element_count={})",
                     index, element_count);
        ASSERT_MSG(false, "SparseLazyBuffer out-of-range access");
    }
    const std::size_t chunk = (index * sizeof(T)) / SparseChunkSize;
    if (chunk_states[chunk].load(std::memory_order_acquire) != ChunkCommitted) {
        CommitChunk(chunk);
    }
    return reinterpret_cast<T*>(base)[index];
}

template <typename T>
void SparseLazyBuffer<T>::CommitChunk(std::size_t chunk) const {
    u32 expected = ChunkUncommitted;
    if (chunk_states[chunk].compare_exchange_strong(expected, ChunkCommitting,
                                                    std::memory_order_acq_rel)) {
        const std::size_t offset = chunk * SparseChunkSize;
        const std::size_t size = std::min(SparseChunkSize, total_bytes - offset);
        void* committed = VirtualAlloc(base + offset, size, MEM_COMMIT, PAGE_READWRITE);
        ASSERT_MSG(committed != nullptr, "SparseLazyBuffer: failed to commit {} bytes", size);
        chunk_states[chunk].store(ChunkCommitted, std::memory_order_release);
        return;
    }

    // Another thread is committing this chunk; wait until it is usable.
    while (chunk_states[chunk].load(std::memory_order_acquire) != ChunkCommitted) {
        YieldProcessor();
    }
}

template <typename T>
void SparseLazyBuffer<T>::Release() {
    if (base != nullptr) {
        ASSERT(VirtualFree(base, 0, MEM_RELEASE));
        base = nullptr;
    }
    delete[] chunk_states;
    chunk_states = nullptr;
    chunk_count = 0;
    total_bytes = 0;
    element_count = 0;
}

template class SparseLazyBuffer<PageTable::PageTableEntry>;
#endif

} // namespace Common
