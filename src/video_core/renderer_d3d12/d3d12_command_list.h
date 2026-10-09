// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include "common/common_types.h"

namespace D3D12 {

class Device;

using Microsoft::WRL::ComPtr;

/// A single D3D12 graphics command list plus its backing allocator.
///
/// Records a frame's worth of commands and submits them through the renderer's
/// direct queue. Command allocators can only be reset once the GPU has finished
/// executing the list, so the owner is responsible for calling `WaitForIdle`
/// (or tracking fences) before `Reset`.
class CommandList {
public:
    CommandList() = default;
    explicit CommandList(ID3D12Device* device);
    ~CommandList();

    CommandList(const CommandList&) = delete;
    CommandList& operator=(const CommandList&) = delete;
    CommandList(CommandList&&) = delete;
    CommandList& operator=(CommandList&&) = delete;

    [[nodiscard]] bool IsValid() const {
        return allocator != nullptr && list != nullptr;
    }

    [[nodiscard]] ID3D12GraphicsCommandList* Get() const {
        return list.Get();
    }

    ID3D12GraphicsCommandList* operator->() const {
        return list.Get();
    }

    /// Resets the allocator and the command list for a new recording.
    void Reset();

    /// Recovers the command list after a fatal failure. A list whose Close
    /// failed (e.g. E_INVALIDARG) can never be closed again, so Reset()
    /// cannot salvage it and every later submit would be dropped. Releases
    /// the allocator and list and recreates both exactly like the
    /// constructor, leaving the list in the same ready-to-record (closed)
    /// state a fresh construction provides.
    void Recover(ID3D12Device* device);

    /// Closes the command list (required before ExecuteCommandLists).
    /// Returns false (and logs) when Close fails; a list that failed to
    /// close must never be submitted through ExecuteCommandLists.
    bool Close();

    /// Closes and submits the recorded commands to the device's direct queue.
    void Execute(Device& device);

    // State and draw helpers. These are thin wrappers to keep call sites readable.

    void SetRootSignature(ID3D12RootSignature* root_signature);
    /// Binds a root signature for compute dispatches (mirrors `SetRootSignature`).
    void SetComputeRootSignature(ID3D12RootSignature* root_signature);
    void SetPipelineState(ID3D12PipelineState* pipeline);
    void SetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY topology);
    void SetViewport(const D3D12_VIEWPORT& viewport);
    void SetScissorRect(const D3D12_RECT& rect);

    void OMSetRenderTargets(u32 render_target_count,
                            const D3D12_CPU_DESCRIPTOR_HANDLE* render_targets);
    void ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE rtv, const f32 color[4]);
    void ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE dsv, D3D12_CLEAR_FLAGS flags, f32 depth,
                               u8 stencil);

    void Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after);
    void UAVBarrier(ID3D12Resource* resource);

    void SetGraphicsRoot32BitConstant(u32 index, u32 value, u32 offset = 0);
    void SetGraphicsRootConstantBufferView(u32 index, D3D12_GPU_VIRTUAL_ADDRESS address);
    void SetGraphicsRootDescriptorTable(u32 index, D3D12_GPU_DESCRIPTOR_HANDLE handle);
    void SetComputeRoot32BitConstant(u32 index, u32 value, u32 offset = 0);
    void SetComputeRootConstantBufferView(u32 index, D3D12_GPU_VIRTUAL_ADDRESS address);
    void SetComputeRootDescriptorTable(u32 index, D3D12_GPU_DESCRIPTOR_HANDLE handle);

    void Draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance);
    void DrawIndexed(u32 index_count, u32 instance_count, u32 first_index, s32 vertex_offset,
                     u32 first_instance);
    void Dispatch(u32 group_count_x, u32 group_count_y, u32 group_count_z);

    void CopyBufferRegion(ID3D12Resource* dst, u64 dst_offset, ID3D12Resource* src, u64 src_offset,
                          u64 size);

private:
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
};

} // namespace D3D12
