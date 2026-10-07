// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/d3d12_command_list.h"

#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_cmd_ring.h"
#include "video_core/renderer_d3d12/d3d12_device.h"

namespace D3D12 {

CommandList::CommandList(ID3D12Device* device) {
    HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_PPV_ARGS(&allocator));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateCommandAllocator failed: {:#x}", static_cast<u32>(hr));
        return;
    }
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                   IID_PPV_ARGS(&list));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateCommandList failed: {:#x}", static_cast<u32>(hr));
        return;
    }
    list->Close();
}

CommandList::~CommandList() = default;

void CommandList::Reset() {
    if (!IsValid()) {
        return;
    }
    allocator->Reset();
    list->Reset(allocator.Get(), nullptr);
}

void CommandList::Recover(ID3D12Device* device) {
    // A list whose Close failed stays wedged open: Close will keep returning
    // E_FAIL and Reset() cannot recover it. Drop both objects and recreate
    // them exactly like the constructor, ending in the same closed (ready)
    // state.
    allocator.Reset();
    list.Reset();
    HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_PPV_ARGS(&allocator));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateCommandAllocator failed: {:#x}", static_cast<u32>(hr));
        allocator.Reset();
        list.Reset();
        return;
    }
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                       IID_PPV_ARGS(&list));
    if (FAILED(hr)) {
        LOG_ERROR(Render_D3D12, "CreateCommandList failed: {:#x}", static_cast<u32>(hr));
        allocator.Reset();
        list.Reset();
        return;
    }
    list->Close();
}

bool CommandList::Close() {
    if (list) {
        const HRESULT hr = list->Close();
        if (FAILED(hr)) {
            LOG_ERROR(Render_D3D12, "CommandList::Close failed: {:#x}", static_cast<u32>(hr));
            return false;
        }
    }
    return true;
}

void CommandList::Execute(Device& device) {
    if (!IsValid()) {
        LOG_ERROR(Render_D3D12, "CommandList::Execute: invalid list/allocator");
        return;
    }
    if (!Close()) {
        LOG_ERROR(Render_D3D12, "CommandList::Execute: skipping submit, list failed Close");
        Diag::Dump();
        Recover(device.GetDevice());
        return;
    }
    ID3D12CommandList* lists[] = {list.Get()};
    device.GetQueue()->ExecuteCommandLists(1, lists);
}

void CommandList::SetRootSignature(ID3D12RootSignature* root_signature) {
    Diag::Push(Diag::CmdKind::StateSignal, 1, reinterpret_cast<u64>(root_signature));
    list->SetGraphicsRootSignature(root_signature);
}

void CommandList::SetPipelineState(ID3D12PipelineState* pipeline) {
    Diag::Push(Diag::CmdKind::StateSignal, 2, reinterpret_cast<u64>(pipeline));
    list->SetPipelineState(pipeline);
}

void CommandList::SetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY topology) {
    Diag::Push(Diag::CmdKind::StateSignal, 3, static_cast<u64>(topology));
    list->IASetPrimitiveTopology(topology);
}

void CommandList::SetViewport(const D3D12_VIEWPORT& viewport) {
    Diag::Push(Diag::CmdKind::StateSignal, 4);
    list->RSSetViewports(1, &viewport);
}

void CommandList::SetScissorRect(const D3D12_RECT& rect) {
    Diag::Push(Diag::CmdKind::StateSignal, 5);
    list->RSSetScissorRects(1, &rect);
}

void CommandList::OMSetRenderTargets(u32 render_target_count,
                                     const D3D12_CPU_DESCRIPTOR_HANDLE* render_targets) {
    Diag::Push(Diag::CmdKind::StateSignal, 6, render_target_count,
               render_targets != nullptr && render_target_count > 0 ? render_targets[0].ptr : 0);
    list->OMSetRenderTargets(render_target_count, render_targets, FALSE, nullptr);
}

void CommandList::ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE rtv, const f32 color[4]) {
    Diag::Push(Diag::CmdKind::ClearRtv, rtv.ptr);
    list->ClearRenderTargetView(rtv, color, 0, nullptr);
}

void CommandList::ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE dsv, D3D12_CLEAR_FLAGS flags,
                                        f32 depth, u8 stencil) {
    Diag::Push(Diag::CmdKind::ClearDsv, dsv.ptr, static_cast<u64>(flags));
    list->ClearDepthStencilView(dsv, flags, depth, stencil, 0, nullptr);
}

void CommandList::Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                             D3D12_RESOURCE_STATES after) {
    if (resource == nullptr) {
        return;
    }
    Diag::Push(Diag::CmdKind::Transition, reinterpret_cast<u64>(resource),
               static_cast<u64>(before), static_cast<u64>(after));
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

void CommandList::UAVBarrier(ID3D12Resource* resource) {
    Diag::Push(Diag::CmdKind::UavBarrier, reinterpret_cast<u64>(resource));
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    list->ResourceBarrier(1, &barrier);
}

void CommandList::SetGraphicsRoot32BitConstant(u32 index, u32 value, u32 offset) {
    Diag::Push(Diag::CmdKind::StateSignal, 7, index, value, offset);
    list->SetGraphicsRoot32BitConstant(index, value, offset);
}

void CommandList::SetGraphicsRootConstantBufferView(u32 index, D3D12_GPU_VIRTUAL_ADDRESS address) {
    Diag::Push(Diag::CmdKind::StateSignal, 8, index, address);
    list->SetGraphicsRootConstantBufferView(index, address);
}

void CommandList::SetGraphicsRootDescriptorTable(u32 index, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
    Diag::Push(Diag::CmdKind::StateSignal, 9, index, handle.ptr);
    list->SetGraphicsRootDescriptorTable(index, handle);
}

void CommandList::SetComputeRoot32BitConstant(u32 index, u32 value, u32 offset) {
    list->SetComputeRoot32BitConstant(index, value, offset);
}

void CommandList::SetComputeRootConstantBufferView(u32 index, D3D12_GPU_VIRTUAL_ADDRESS address) {
    list->SetComputeRootConstantBufferView(index, address);
}

void CommandList::SetComputeRootDescriptorTable(u32 index, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
    list->SetComputeRootDescriptorTable(index, handle);
}

void CommandList::Draw(u32 vertex_count, u32 instance_count, u32 first_vertex,
                       u32 first_instance) {
    Diag::Push(Diag::CmdKind::Draw, vertex_count, instance_count, first_vertex, first_instance);
    list->DrawInstanced(vertex_count, instance_count, first_vertex, first_instance);
}

void CommandList::DrawIndexed(u32 index_count, u32 instance_count, u32 first_index,
                              s32 vertex_offset, u32 first_instance) {
    Diag::Push(Diag::CmdKind::DrawIndexed, index_count, instance_count, first_index,
               static_cast<u64>(static_cast<s64>(vertex_offset)) << 32 | first_instance);
    list->DrawIndexedInstanced(index_count, instance_count, first_index, vertex_offset,
                               first_instance);
}

void CommandList::Dispatch(u32 group_count_x, u32 group_count_y, u32 group_count_z) {
    Diag::Push(Diag::CmdKind::Dispatch, group_count_x, group_count_y, group_count_z);
    list->Dispatch(group_count_x, group_count_y, group_count_z);
}

void CommandList::CopyBufferRegion(ID3D12Resource* dst, u64 dst_offset, ID3D12Resource* src,
                                   u64 src_offset, u64 size) {
    if (dst == nullptr || src == nullptr) {
        return;
    }
    Diag::Push(Diag::CmdKind::CopyBufferRegion, reinterpret_cast<u64>(dst),
               reinterpret_cast<u64>(src), size, dst_offset);
    list->CopyBufferRegion(dst, dst_offset, src, src_offset, size);
}

} // namespace D3D12
