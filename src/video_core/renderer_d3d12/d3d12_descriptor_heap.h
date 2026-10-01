// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include "common/common_types.h"

namespace D3D12 {

using Microsoft::WRL::ComPtr;

/// A D3D12 descriptor heap with a linear bump allocator.
///
/// The allocator is a bring-up implementation: descriptors are allocated in
/// contiguous runs and the whole heap can be reset once the GPU work that
/// referenced it has retired. Shader-visible heaps (CBV/SRV/UAV, samplers) are
/// used for descriptor tables; RTV/DSV heaps are CPU-only.
class DescriptorHeap {
public:
    static constexpr u32 INVALID_INDEX = 0xFFFFFFFFu;

    DescriptorHeap() = default;
    DescriptorHeap(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, u32 capacity,
                   bool shader_visible);
    ~DescriptorHeap();

    DescriptorHeap(const DescriptorHeap&) = delete;
    DescriptorHeap& operator=(const DescriptorHeap&) = delete;
    DescriptorHeap(DescriptorHeap&&) = default;
    DescriptorHeap& operator=(DescriptorHeap&&) = default;

    [[nodiscard]] bool IsValid() const {
        return heap != nullptr;
    }

    [[nodiscard]] ID3D12DescriptorHeap* Get() const {
        return heap.Get();
    }

    [[nodiscard]] D3D12_DESCRIPTOR_HEAP_TYPE GetType() const {
        return type;
    }

    [[nodiscard]] u32 GetDescriptorSize() const {
        return descriptor_size;
    }

    [[nodiscard]] u32 Capacity() const {
        return capacity;
    }

    [[nodiscard]] u32 Size() const {
        return size;
    }

    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(u32 index) const {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(index) * descriptor_size;
        return handle;
    }

    [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(u32 index) const {
        D3D12_GPU_DESCRIPTOR_HANDLE handle = heap->GetGPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<u64>(index) * descriptor_size;
        return handle;
    }

    /// Allocates `count` contiguous descriptors, returning the first index, or
    /// INVALID_INDEX when the heap is exhausted.
    [[nodiscard]] u32 Allocate(u32 count = 1) {
        if (size + count > capacity) {
            return INVALID_INDEX;
        }
        const u32 index = size;
        size += count;
        return index;
    }

    /// Releases every allocation. Only safe once no pending GPU work references them.
    void Reset() {
        size = 0;
    }

private:
    ComPtr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_TYPE type{};
    u32 descriptor_size{};
    u32 capacity{};
    u32 size{};
};

} // namespace D3D12
