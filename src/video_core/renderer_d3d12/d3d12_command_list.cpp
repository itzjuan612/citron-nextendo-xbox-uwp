// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/d3d12_command_list.h"

#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_cmd_ring.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_texture_cache.h"

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
        // S25: the failing batch's op counts + destroyed-resource correlation, before the
        // raw 512-record dump, so the summary survives log truncation. The batch is every
        // op recorded since the LAST successful submit (MarkBatch runs at the submit
        // points below, not here - marking here would window only the ops inside
        // Execute itself, which are none).
        Diag::LogBatchSummary();
        D3D12::DumpRecentTextureUploads();
        Diag::Dump();
        Recover(device.GetDevice());
        Diag::MarkBatch();
        // S25: arm per-draw flushing so the next (recurring) failure pins the exact draw.
        Diag::g_arm_per_draw_flush.store(true, std::memory_order_relaxed);
        Diag::g_per_draw_flush_budget.store(900, std::memory_order_relaxed);
        return;
    }
    ID3D12CommandList* lists[] = {list.Get()};
    device.GetQueue()->ExecuteCommandLists(1, lists);
    Diag::MarkBatch();
}

void CommandList::SetRootSignature(ID3D12RootSignature* root_signature) {
    Diag::Push(Diag::CmdKind::StateSignal, 1, reinterpret_cast<u64>(root_signature));
    list->SetGraphicsRootSignature(root_signature);
}

void CommandList::SetComputeRootSignature(ID3D12RootSignature* root_signature) {
    Diag::Push(Diag::CmdKind::StateSignal, 10, reinterpret_cast<u64>(root_signature));
    list->SetComputeRootSignature(root_signature);
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
    // S25: carry the values - a NaN/negative-width viewport is a Close E_INVALIDARG
    // candidate and the record must name it. a=(X,Y) b=(W,H) c=(MinDepth,MaxDepth), f32
    // bit-casts packed two per u64.
    const auto pack2 = [](f32 lo, f32 hi) {
        u32 lo_b = 0;
        u32 hi_b = 0;
        std::memcpy(&lo_b, &lo, 4);
        std::memcpy(&hi_b, &hi, 4);
        return (static_cast<u64>(hi_b) << 32) | lo_b;
    };
    Diag::Push(Diag::CmdKind::StateSignal, 4, pack2(viewport.TopLeftX, viewport.TopLeftY),
               pack2(viewport.Width, viewport.Height),
               pack2(viewport.MinDepth, viewport.MaxDepth));
    list->RSSetViewports(1, &viewport);
}

void CommandList::SetScissorRect(const D3D12_RECT& rect) {
    Diag::Push(Diag::CmdKind::StateSignal, (static_cast<u64>(rect.top) << 32) | static_cast<u32>(rect.left),
               (static_cast<u64>(rect.bottom) << 32) | static_cast<u32>(rect.right));
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
