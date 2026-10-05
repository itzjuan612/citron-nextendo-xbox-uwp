// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_d3d12/d3d12_graphics_pipeline.h"

#include <algorithm>
#include <cstring>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/gpu.h"
#include "video_core/surface.h"
#include "video_core/textures/texture.h"

namespace D3D12 {

namespace {

using Maxwell3D = Tegra::Engines::Maxwell3D;
namespace VideoSurface = VideoCore::Surface;

D3D12_BLEND BlendFactor(Maxwell3D::Regs::Blend::Factor factor) {
    switch (factor) {
    case Maxwell3D::Regs::Blend::Factor::Zero_D3D:
    case Maxwell3D::Regs::Blend::Factor::Zero_GL:
        return D3D12_BLEND_ZERO;
    case Maxwell3D::Regs::Blend::Factor::One_D3D:
    case Maxwell3D::Regs::Blend::Factor::One_GL:
        return D3D12_BLEND_ONE;
    case Maxwell3D::Regs::Blend::Factor::SourceColor_D3D:
    case Maxwell3D::Regs::Blend::Factor::SourceColor_GL:
        return D3D12_BLEND_SRC_COLOR;
    case Maxwell3D::Regs::Blend::Factor::OneMinusSourceColor_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusSourceColor_GL:
        return D3D12_BLEND_INV_SRC_COLOR;
    case Maxwell3D::Regs::Blend::Factor::SourceAlpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::SourceAlpha_GL:
        return D3D12_BLEND_SRC_ALPHA;
    case Maxwell3D::Regs::Blend::Factor::OneMinusSourceAlpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusSourceAlpha_GL:
        return D3D12_BLEND_INV_SRC_ALPHA;
    case Maxwell3D::Regs::Blend::Factor::DestAlpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::DestAlpha_GL:
        return D3D12_BLEND_DEST_ALPHA;
    case Maxwell3D::Regs::Blend::Factor::OneMinusDestAlpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusDestAlpha_GL:
        return D3D12_BLEND_INV_DEST_ALPHA;
    case Maxwell3D::Regs::Blend::Factor::DestColor_D3D:
    case Maxwell3D::Regs::Blend::Factor::DestColor_GL:
        return D3D12_BLEND_DEST_COLOR;
    case Maxwell3D::Regs::Blend::Factor::OneMinusDestColor_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusDestColor_GL:
        return D3D12_BLEND_INV_DEST_COLOR;
    case Maxwell3D::Regs::Blend::Factor::SourceAlphaSaturate_D3D:
    case Maxwell3D::Regs::Blend::Factor::SourceAlphaSaturate_GL:
        return D3D12_BLEND_SRC_ALPHA_SAT;
    case Maxwell3D::Regs::Blend::Factor::Source1Color_D3D:
    case Maxwell3D::Regs::Blend::Factor::Source1Color_GL:
        return D3D12_BLEND_SRC1_COLOR;
    case Maxwell3D::Regs::Blend::Factor::OneMinusSource1Color_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusSource1Color_GL:
        return D3D12_BLEND_INV_SRC1_COLOR;
    case Maxwell3D::Regs::Blend::Factor::Source1Alpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::Source1Alpha_GL:
        return D3D12_BLEND_SRC1_ALPHA;
    case Maxwell3D::Regs::Blend::Factor::OneMinusSource1Alpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusSource1Alpha_GL:
        return D3D12_BLEND_INV_SRC1_ALPHA;
    case Maxwell3D::Regs::Blend::Factor::BlendFactor_D3D:
    case Maxwell3D::Regs::Blend::Factor::ConstantColor_GL:
        return D3D12_BLEND_BLEND_FACTOR;
    case Maxwell3D::Regs::Blend::Factor::OneMinusBlendFactor_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusConstantColor_GL:
        return D3D12_BLEND_INV_BLEND_FACTOR;
    case Maxwell3D::Regs::Blend::Factor::BothSourceAlpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::ConstantAlpha_GL:
        return D3D12_BLEND_BLEND_FACTOR;
    case Maxwell3D::Regs::Blend::Factor::OneMinusBothSourceAlpha_D3D:
    case Maxwell3D::Regs::Blend::Factor::OneMinusConstantAlpha_GL:
        return D3D12_BLEND_INV_BLEND_FACTOR;
    }
    return D3D12_BLEND_ONE;
}

D3D12_BLEND_OP BlendEquation(Maxwell3D::Regs::Blend::Equation equation) {
    switch (equation) {
    case Maxwell3D::Regs::Blend::Equation::Add_D3D:
    case Maxwell3D::Regs::Blend::Equation::Add_GL:
        return D3D12_BLEND_OP_ADD;
    case Maxwell3D::Regs::Blend::Equation::Subtract_D3D:
    case Maxwell3D::Regs::Blend::Equation::Subtract_GL:
        return D3D12_BLEND_OP_SUBTRACT;
    case Maxwell3D::Regs::Blend::Equation::ReverseSubtract_D3D:
    case Maxwell3D::Regs::Blend::Equation::ReverseSubtract_GL:
        return D3D12_BLEND_OP_REV_SUBTRACT;
    case Maxwell3D::Regs::Blend::Equation::Min_D3D:
    case Maxwell3D::Regs::Blend::Equation::Min_GL:
        return D3D12_BLEND_OP_MIN;
    case Maxwell3D::Regs::Blend::Equation::Max_D3D:
    case Maxwell3D::Regs::Blend::Equation::Max_GL:
        return D3D12_BLEND_OP_MAX;
    }
    return D3D12_BLEND_OP_ADD;
}

D3D12_LOGIC_OP LogicOp(u32 packed) {
    return static_cast<D3D12_LOGIC_OP>(packed + 1);
}

D3D12_COMPARISON_FUNC ComparisonOp(u32 packed) {
    // The packing normalizes both OpenGL and D3D enums to 0..7 and D3D12_COMPARISON_FUNC
    // starts at 1 with the same ordering.
    return static_cast<D3D12_COMPARISON_FUNC>(packed + 1);
}

D3D12_STENCIL_OP StencilOp(u32 packed) {
    // Same property as the comparison ops: D3D12_STENCIL_OP starts at 1 with the same order.
    return static_cast<D3D12_STENCIL_OP>(packed + 1);
}

D3D12_CULL_MODE CullMode(const FixedPipelineState& state) {
    if (state.cull_enable == 0) {
        return D3D12_CULL_MODE_NONE;
    }
    switch (state.CullFaceMode()) {
    case Maxwell3D::Regs::CullFace::Front:
        return D3D12_CULL_MODE_FRONT;
    case Maxwell3D::Regs::CullFace::Back:
        return D3D12_CULL_MODE_BACK;
    case Maxwell3D::Regs::CullFace::FrontAndBack:
    default:
        return D3D12_CULL_MODE_NONE;
    }
}

DXGI_FORMAT VertexFormat(Maxwell3D::Regs::VertexAttribute::Type type,
                         Maxwell3D::Regs::VertexAttribute::Size size) {
    using Type = Maxwell3D::Regs::VertexAttribute::Type;
    using Size = Maxwell3D::Regs::VertexAttribute::Size;

    // D3D12 has no _USCALED/_SSCALED vertex formats. The shader profile disables scaled
    // attributes, so the guest shader performs the conversion itself and the layout uses
    // the integer formats.
    if (type == Type::UScaled) {
        type = Type::UInt;
    } else if (type == Type::SScaled) {
        type = Type::SInt;
    }

    // 3-component sizes. D3D12 has exact 3x32 formats, but no 3x16 or 3x8 formats; for
    // those the 4-component variant is used (the shader input signature ignores the extra
    // component, and out-of-bounds input fetch reads return 0), which is what other
    // backends do for the widths D3D12 cannot express.
    if (size == Size::Size_R32_G32_B32) {
        switch (type) {
        case Type::Float:
            return DXGI_FORMAT_R32G32B32_FLOAT;
        case Type::UInt:
            return DXGI_FORMAT_R32G32B32_UINT;
        case Type::SInt:
            return DXGI_FORMAT_R32G32B32_SINT;
        default:
            return DXGI_FORMAT_UNKNOWN;
        }
    }
    if (size == Size::Size_R16_G16_B16) {
        switch (type) {
        case Type::Float:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case Type::UNorm:
            return DXGI_FORMAT_R16G16B16A16_UNORM;
        case Type::SNorm:
            return DXGI_FORMAT_R16G16B16A16_SNORM;
        case Type::UInt:
            return DXGI_FORMAT_R16G16B16A16_UINT;
        case Type::SInt:
            return DXGI_FORMAT_R16G16B16A16_SINT;
        default:
            return DXGI_FORMAT_UNKNOWN;
        }
    }
    if (size == Size::Size_R8_G8_B8) {
        switch (type) {
        case Type::UNorm:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case Type::SNorm:
            return DXGI_FORMAT_R8G8B8A8_SNORM;
        case Type::UInt:
            return DXGI_FORMAT_R8G8B8A8_UINT;
        case Type::SInt:
            return DXGI_FORMAT_R8G8B8A8_SINT;
        default:
            return DXGI_FORMAT_UNKNOWN;
        }
    }

    // {R, RG, RGB(A), RGBA} for one component width and signedness.
    const auto pick = [&](DXGI_FORMAT r, DXGI_FORMAT rg, DXGI_FORMAT rgba) -> DXGI_FORMAT {
        switch (size) {
        case Size::Size_R8:
        case Size::Size_R16:
        case Size::Size_R32:
        case Size::Size_A8:
            return r;
        case Size::Size_R8_G8:
        case Size::Size_G8_R8:
        case Size::Size_R16_G16:
        case Size::Size_R32_G32:
            return rg;
        case Size::Size_R8_G8_B8_A8:
        case Size::Size_X8_B8_G8_R8:
        case Size::Size_R16_G16_B16_A16:
        case Size::Size_R32_G32_B32_A32:
            return rgba;
        case Size::Size_A2_B10_G10_R10:
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        case Size::Size_B10_G11_R11:
            return DXGI_FORMAT_R11G11B10_FLOAT;
        default:
            return DXGI_FORMAT_UNKNOWN;
        }
    };

    const bool is_16 =
        size == Size::Size_R16 || size == Size::Size_R16_G16 || size == Size::Size_R16_G16_B16 ||
        size == Size::Size_R16_G16_B16_A16;
    const bool is_32 = size == Size::Size_R32 || size == Size::Size_R32_G32 ||
                       size == Size::Size_R32_G32_B32 || size == Size::Size_R32_G32_B32_A32;

    switch (type) {
    case Type::UNorm:
        if (is_16) {
            return pick(DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM,
                        DXGI_FORMAT_R16G16B16A16_UNORM);
        }
        return pick(DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM);
    case Type::SNorm:
        if (is_16) {
            return pick(DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16G16_SNORM,
                        DXGI_FORMAT_R16G16B16A16_SNORM);
        }
        return pick(DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8G8_SNORM, DXGI_FORMAT_R8G8B8A8_SNORM);
    case Type::UInt:
        if (is_32) {
            return pick(DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT,
                        DXGI_FORMAT_R32G32B32A32_UINT);
        }
        if (is_16) {
            return pick(DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16_UINT,
                        DXGI_FORMAT_R16G16B16A16_UINT);
        }
        return pick(DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8B8A8_UINT);
    case Type::SInt:
        if (is_32) {
            return pick(DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32G32_SINT,
                        DXGI_FORMAT_R32G32B32A32_SINT);
        }
        if (is_16) {
            return pick(DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16G16_SINT,
                        DXGI_FORMAT_R16G16B16A16_SINT);
        }
        return pick(DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_R8G8B8A8_SINT);
    case Type::Float:
        return pick(DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT,
                    DXGI_FORMAT_R32G32B32A32_FLOAT);
    case Type::UnusedEnumDoNotUseBecauseItWillGoAway:
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType(Maxwell3D::Regs::PrimitiveTopology topology) {
    using Topology = Maxwell3D::Regs::PrimitiveTopology;
    switch (topology) {
    case Topology::Points:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    case Topology::Lines:
    case Topology::LineLoop:
    case Topology::LineStrip:
    case Topology::LinesAdjacency:
    case Topology::LineStripAdjacency:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    case Topology::Patches:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    default:
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }
}

u32 SampleCount(Tegra::Texture::MsaaMode mode) {
    switch (mode) {
    case Tegra::Texture::MsaaMode::Msaa1x1:
        return 1;
    case Tegra::Texture::MsaaMode::Msaa2x1:
    case Tegra::Texture::MsaaMode::Msaa2x1_D3D:
        return 2;
    case Tegra::Texture::MsaaMode::Msaa2x2:
    case Tegra::Texture::MsaaMode::Msaa2x2_VC4:
    case Tegra::Texture::MsaaMode::Msaa2x2_VC12:
        return 4;
    case Tegra::Texture::MsaaMode::Msaa4x2:
    case Tegra::Texture::MsaaMode::Msaa4x2_D3D:
    case Tegra::Texture::MsaaMode::Msaa4x2_VC8:
    case Tegra::Texture::MsaaMode::Msaa4x2_VC24:
        return 8;
    case Tegra::Texture::MsaaMode::Msaa4x4:
        return 16;
    default:
        return 1;
    }
}

// Maps a stage's compiled bytecode to D3D12_SHADER_BYTECODE. Unbound stages
// yield an empty blob ({nullptr, 0}) so the pipeline description never hands
// a dangling pointer to CreateGraphicsPipelineState.
const auto StageBytecode = [](const std::vector<u8>& bytecode)
    -> D3D12_SHADER_BYTECODE {
    if (bytecode.empty()) {
        return D3D12_SHADER_BYTECODE{nullptr, 0};
    }
    return D3D12_SHADER_BYTECODE{bytecode.data(), bytecode.size()};
};

} // Anonymous namespace

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(this), sizeof(*this)));
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    return std::memcmp(this, &rhs, sizeof(*this)) == 0;
}

DXGI_FORMAT SurfaceFormat(VideoSurface::PixelFormat format) {
    switch (format) {
    case VideoSurface::PixelFormat::A8B8G8R8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VideoSurface::PixelFormat::A8B8G8R8_SNORM:
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    case VideoSurface::PixelFormat::A8B8G8R8_SINT:
        return DXGI_FORMAT_R8G8B8A8_SINT;
    case VideoSurface::PixelFormat::A8B8G8R8_UINT:
        return DXGI_FORMAT_R8G8B8A8_UINT;
    case VideoSurface::PixelFormat::B8G8R8A8_UNORM:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case VideoSurface::PixelFormat::B8G8R8A8_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case VideoSurface::PixelFormat::A8B8G8R8_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case VideoSurface::PixelFormat::R16G16B16A16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case VideoSurface::PixelFormat::R16G16B16A16_UNORM:
        return DXGI_FORMAT_R16G16B16A16_UNORM;
    case VideoSurface::PixelFormat::R16G16B16A16_SNORM:
        return DXGI_FORMAT_R16G16B16A16_SNORM;
    case VideoSurface::PixelFormat::R16G16B16A16_SINT:
        return DXGI_FORMAT_R16G16B16A16_SINT;
    case VideoSurface::PixelFormat::R16G16B16A16_UINT:
        return DXGI_FORMAT_R16G16B16A16_UINT;
    case VideoSurface::PixelFormat::R16G16B16X16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case VideoSurface::PixelFormat::R32G32B32A32_FLOAT:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case VideoSurface::PixelFormat::R32G32B32A32_SINT:
        return DXGI_FORMAT_R32G32B32A32_SINT;
    case VideoSurface::PixelFormat::R32G32B32A32_UINT:
        return DXGI_FORMAT_R32G32B32A32_UINT;
    case VideoSurface::PixelFormat::R32G32_FLOAT:
        return DXGI_FORMAT_R32G32_FLOAT;
    case VideoSurface::PixelFormat::R32G32_SINT:
        return DXGI_FORMAT_R32G32_SINT;
    case VideoSurface::PixelFormat::R32G32_UINT:
        return DXGI_FORMAT_R32G32_UINT;
    case VideoSurface::PixelFormat::R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case VideoSurface::PixelFormat::R32_SINT:
        return DXGI_FORMAT_R32_SINT;
    case VideoSurface::PixelFormat::R32_UINT:
        return DXGI_FORMAT_R32_UINT;
    case VideoSurface::PixelFormat::R16_FLOAT:
        return DXGI_FORMAT_R16_FLOAT;
    case VideoSurface::PixelFormat::R16_UNORM:
        return DXGI_FORMAT_R16_UNORM;
    case VideoSurface::PixelFormat::R16_SNORM:
        return DXGI_FORMAT_R16_SNORM;
    case VideoSurface::PixelFormat::R16_UINT:
        return DXGI_FORMAT_R16_UINT;
    case VideoSurface::PixelFormat::R16_SINT:
        return DXGI_FORMAT_R16_SINT;
    case VideoSurface::PixelFormat::R16G16_FLOAT:
        return DXGI_FORMAT_R16G16_FLOAT;
    case VideoSurface::PixelFormat::R16G16_UNORM:
        return DXGI_FORMAT_R16G16_UNORM;
    case VideoSurface::PixelFormat::R16G16_SNORM:
        return DXGI_FORMAT_R16G16_SNORM;
    case VideoSurface::PixelFormat::R16G16_UINT:
        return DXGI_FORMAT_R16G16_UINT;
    case VideoSurface::PixelFormat::R16G16_SINT:
        return DXGI_FORMAT_R16G16_SINT;
    case VideoSurface::PixelFormat::R8_UNORM:
        return DXGI_FORMAT_R8_UNORM;
    case VideoSurface::PixelFormat::R8_SNORM:
        return DXGI_FORMAT_R8_SNORM;
    case VideoSurface::PixelFormat::R8_UINT:
        return DXGI_FORMAT_R8_UINT;
    case VideoSurface::PixelFormat::R8_SINT:
        return DXGI_FORMAT_R8_SINT;
    case VideoSurface::PixelFormat::R8G8_UNORM:
        return DXGI_FORMAT_R8G8_UNORM;
    case VideoSurface::PixelFormat::R8G8_SNORM:
        return DXGI_FORMAT_R8G8_SNORM;
    case VideoSurface::PixelFormat::R8G8_UINT:
        return DXGI_FORMAT_R8G8_UINT;
    case VideoSurface::PixelFormat::R8G8_SINT:
        return DXGI_FORMAT_R8G8_SINT;
    case VideoSurface::PixelFormat::B10G11R11_FLOAT:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case VideoSurface::PixelFormat::A2B10G10R10_UNORM:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case VideoSurface::PixelFormat::A2B10G10R10_UINT:
        return DXGI_FORMAT_R10G10B10A2_UINT;
    case VideoSurface::PixelFormat::A2R10G10B10_UNORM:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case VideoSurface::PixelFormat::E5B9G9R9_FLOAT:
        return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    case VideoSurface::PixelFormat::D32_FLOAT:
        return DXGI_FORMAT_D32_FLOAT;
    case VideoSurface::PixelFormat::D16_UNORM:
        return DXGI_FORMAT_D16_UNORM;
    case VideoSurface::PixelFormat::X8_D24_UNORM:
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case VideoSurface::PixelFormat::D24_UNORM_S8_UINT:
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case VideoSurface::PixelFormat::S8_UINT_D24_UNORM:
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case VideoSurface::PixelFormat::D32_FLOAT_S8_UINT:
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case VideoSurface::PixelFormat::R32G32B32_FLOAT:
        return DXGI_FORMAT_R32G32B32_FLOAT;
    case VideoSurface::PixelFormat::BC1_RGBA_SRGB:
        return DXGI_FORMAT_BC1_UNORM_SRGB;
    case VideoSurface::PixelFormat::BC2_SRGB:
        return DXGI_FORMAT_BC2_UNORM_SRGB;
    case VideoSurface::PixelFormat::BC3_SRGB:
        return DXGI_FORMAT_BC3_UNORM_SRGB;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

RootSignatureParams GraphicsPipeline::BuildRootSignatureParams(
    const std::array<const Shader::Info*, MAX_SHADER_STAGES>& infos,
    const std::array<bool, MAX_SHADER_STAGES>& needs_runtime_data) {
    RootSignatureParams params{};
    for (u32 i = 0; i < MAX_SHADER_STAGES; ++i) {
        const Shader::Info* info = infos[i];
        if (!info) {
            continue;
        }
        params.num_cbv += Shader::NumDescriptors(info->constant_buffer_descriptors);
        params.num_resources += Shader::NumDescriptors(info->storage_buffers_descriptors);
        params.num_resources += Shader::NumDescriptors(info->texture_buffer_descriptors);
        params.num_resources += Shader::NumDescriptors(info->image_buffer_descriptors);
        params.num_resources += Shader::NumDescriptors(info->texture_descriptors);
        params.num_resources += Shader::NumDescriptors(info->image_descriptors);
        params.needs_push_constants |= info->uses_render_area || info->uses_rescaling_uniform;
        params.needs_runtime_data |= needs_runtime_data[i];
    }
    return params;
}

GraphicsPipeline::GraphicsPipeline(
    ID3D12Device* device, const GraphicsPipelineCacheKey& key_,
    const std::array<std::vector<u8>, MAX_SHADER_STAGES>& stages,
    const std::array<const Shader::Info*, MAX_SHADER_STAGES>& infos,
    std::array<bool, MAX_SHADER_STAGES> needs_runtime_data_)
    : key{key_}, root_signature(device, BuildRootSignatureParams(infos, needs_runtime_data_)) {
    const FixedPipelineState& state = key.state;

    // Input layout: all generic attributes use the TEXCOORD semantic with the attribute index
    // as the semantic index, matching Mesa's DXIL signature emission.
    std::vector<D3D12_INPUT_ELEMENT_DESC> input_elements;
    u32 input_slot_count = 0;
    for (u32 i = 0; i < state.attributes.size(); ++i) {
        const auto& attribute = state.attributes[i];
        if (attribute.enabled == 0) {
            continue;
        }
        const DXGI_FORMAT format = VertexFormat(attribute.Type(), attribute.Size());
        if (format == DXGI_FORMAT_UNKNOWN) {
            LOG_WARNING(Render_D3D12, "Unsupported vertex attribute {} (type={}, size={})", i,
                        static_cast<u32>(attribute.Type()), static_cast<u32>(attribute.Size()));
            continue;
        }
        const u32 buffer = attribute.buffer.Value();
        input_slot_count = std::max(input_slot_count, buffer + 1);
        D3D12_INPUT_ELEMENT_DESC element{};
        element.SemanticName = "TEXCOORD";
        element.SemanticIndex = i;
        element.Format = format;
        element.InputSlot = buffer;
        element.AlignedByteOffset = attribute.offset.Value();
        element.InputSlotClass = state.binding_divisors[buffer] == 0
                                     ? D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA
                                     : D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
        element.InstanceDataStepRate = state.binding_divisors[buffer];
        input_elements.push_back(element);
    }

    D3D12_BLEND_DESC blend_desc{};
    blend_desc.AlphaToCoverageEnable = state.alpha_to_coverage_enabled != 0;
    blend_desc.IndependentBlendEnable = TRUE;
    for (u32 i = 0; i < state.attachments.size(); ++i) {
        const auto& attachment = state.attachments[i];
        auto& target = blend_desc.RenderTarget[i];
        if (attachment.mask_r) {
            target.RenderTargetWriteMask |= D3D12_COLOR_WRITE_ENABLE_RED;
        }
        if (attachment.mask_g) {
            target.RenderTargetWriteMask |= D3D12_COLOR_WRITE_ENABLE_GREEN;
        }
        if (attachment.mask_b) {
            target.RenderTargetWriteMask |= D3D12_COLOR_WRITE_ENABLE_BLUE;
        }
        if (attachment.mask_a) {
            target.RenderTargetWriteMask |= D3D12_COLOR_WRITE_ENABLE_ALPHA;
        }
        if (attachment.enable == 0) {
            continue;
        }
        target.BlendEnable = TRUE;
        target.SrcBlend = BlendFactor(attachment.SourceRGBFactor());
        target.DestBlend = BlendFactor(attachment.DestRGBFactor());
        target.BlendOp = BlendEquation(attachment.EquationRGB());
        target.SrcBlendAlpha = BlendFactor(attachment.SourceAlphaFactor());
        target.DestBlendAlpha = BlendFactor(attachment.DestAlphaFactor());
        target.BlendOpAlpha = BlendEquation(attachment.EquationAlpha());
    }
    blend_desc.RenderTarget[0].LogicOpEnable = state.logic_op_enable != 0;
    blend_desc.RenderTarget[0].LogicOp = LogicOp(state.logic_op.Value());

    D3D12_RASTERIZER_DESC rasterizer_desc{};
    switch (state.PolygonModeMode()) {
    case Maxwell3D::Regs::PolygonMode::Line:
        rasterizer_desc.FillMode = D3D12_FILL_MODE_WIREFRAME;
        break;
    case Maxwell3D::Regs::PolygonMode::Fill:
    case Maxwell3D::Regs::PolygonMode::Point:
    default:
        rasterizer_desc.FillMode = D3D12_FILL_MODE_SOLID;
        break;
    }
    rasterizer_desc.CullMode = CullMode(state);
    rasterizer_desc.FrontCounterClockwise =
        state.FrontFaceMode() == Maxwell3D::Regs::FrontFace::CounterClockWise;
    rasterizer_desc.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
    rasterizer_desc.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    rasterizer_desc.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    rasterizer_desc.DepthClipEnable = state.depth_clamp_disabled == 0;
    rasterizer_desc.MultisampleEnable = SampleCount(state.msaa_mode.Value()) > 1;
    rasterizer_desc.AntialiasedLineEnable = state.smooth_lines != 0;
    rasterizer_desc.ForcedSampleCount = 0;
    rasterizer_desc.ConservativeRaster = state.conservative_raster_enable != 0
                                             ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
                                             : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    D3D12_DEPTH_STENCIL_DESC depth_stencil_desc{};
    depth_stencil_desc.DepthEnable = state.depth_test_enable != 0;
    depth_stencil_desc.DepthWriteMask =
        state.depth_write_enable != 0 ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    depth_stencil_desc.DepthFunc = ComparisonOp(state.depth_test_func.Value());
    depth_stencil_desc.StencilEnable = state.stencil_enable != 0;
    depth_stencil_desc.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    depth_stencil_desc.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    depth_stencil_desc.FrontFace.StencilFailOp = StencilOp(state.front.action_stencil_fail.Value());
    depth_stencil_desc.FrontFace.StencilDepthFailOp =
        StencilOp(state.front.action_depth_fail.Value());
    depth_stencil_desc.FrontFace.StencilPassOp = StencilOp(state.front.action_depth_pass.Value());
    depth_stencil_desc.FrontFace.StencilFunc = ComparisonOp(state.front.test_func.Value());
    depth_stencil_desc.BackFace.StencilFailOp = StencilOp(state.back.action_stencil_fail.Value());
    depth_stencil_desc.BackFace.StencilDepthFailOp =
        StencilOp(state.back.action_depth_fail.Value());
    depth_stencil_desc.BackFace.StencilPassOp = StencilOp(state.back.action_depth_pass.Value());
    depth_stencil_desc.BackFace.StencilFunc = ComparisonOp(state.back.test_func.Value());

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root_signature.Get();
    desc.VS = StageBytecode(stages[0]);
    desc.HS = StageBytecode(stages[1]);
    desc.DS = StageBytecode(stages[2]);
    desc.GS = StageBytecode(stages[3]);
    desc.PS = StageBytecode(stages[4]);
    desc.BlendState = blend_desc;
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState = rasterizer_desc;
    desc.DepthStencilState = depth_stencil_desc;
    desc.InputLayout = {input_elements.empty() ? nullptr : input_elements.data(),
                        static_cast<UINT>(input_elements.size())};
    desc.IBStripCutValue = state.primitive_restart_enable != 0
                               ? D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
                               : D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
    desc.PrimitiveTopologyType = PrimitiveTopologyType(state.topology.Value());

    u32 render_target_count = 0;
    for (u32 i = 0; i < state.color_formats.size(); ++i) {
        if (state.color_formats[i] == 0) {
            continue;
        }
        const auto pixel_format = VideoSurface::PixelFormatFromRenderTargetFormat(
            static_cast<Tegra::RenderTargetFormat>(state.color_formats[i]));
        const DXGI_FORMAT format = SurfaceFormat(pixel_format);
        // D3D12 cannot bind a depth/stencil format as a render target;
        // the pipeline is unrepresentable, so skip creating the PSO
        // (same philosophy as the no-vertex-program skip).
        switch (format) {
        case DXGI_FORMAT_D16_UNORM:
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
        case DXGI_FORMAT_D32_FLOAT:
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_R32G8X24_TYPELESS: {
            static bool logged_depth_as_rtv = false;
            if (!logged_depth_as_rtv) {
                logged_depth_as_rtv = true;
                LOG_DEBUG(Render_D3D12,
                          "Depth/stencil format as render target "
                          "(slot {}, color_formats[{}]={}), "
                          "aborting pipeline creation",
                          i, i, state.color_formats[i]);
            }
            pipeline.Reset();
            return;
        }
        default:
            break;
        }
        desc.RTVFormats[i] = format;
        if (format != DXGI_FORMAT_UNKNOWN) {
            render_target_count = i + 1;
        }
    }
    desc.NumRenderTargets = render_target_count;

    if (state.depth_enabled != 0) {
        const auto pixel_format = VideoSurface::PixelFormatFromDepthFormat(
            static_cast<Tegra::DepthFormat>(state.depth_format.Value()));
        desc.DSVFormat = SurfaceFormat(pixel_format);
    } else {
        desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    }

    desc.SampleDesc.Count = SampleCount(state.msaa_mode.Value());
    desc.SampleDesc.Quality = 0;
    desc.NodeMask = 0;
    desc.CachedPSO = {};
    desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    const HRESULT hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline));
    if (FAILED(hr)) {
        // Dump the full state for the first few failures: E_INVALIDARG has no debug layer on
        // Xbox, so the invalid field must be identified from the values themselves.
        static u32 failure_dump_count = 0;
        if (failure_dump_count++ < 3) {
            for (const auto& element : input_elements) {
                LOG_ERROR(Render_D3D12,
                          "  input element: sem_idx={} fmt={:#x} slot={} offset={} class={} "
                          "step={}",
                          element.SemanticIndex, static_cast<u32>(element.Format),
                          element.InputSlot, element.AlignedByteOffset,
                          static_cast<u32>(element.InputSlotClass), element.InstanceDataStepRate);
            }
            LOG_ERROR(Render_D3D12,
                      "  state: depth={} write={} func={} stencil={} sfail={} dpfail={} "
                      "dppass={} sfunc={} blend0={} logic={} strip={} topotype={} fill={} "
                      "cull={} forced_samples={} conservative={}",
                      depth_stencil_desc.DepthEnable,
                      static_cast<u32>(depth_stencil_desc.DepthWriteMask),
                      static_cast<u32>(depth_stencil_desc.DepthFunc),
                      depth_stencil_desc.StencilEnable,
                      static_cast<u32>(depth_stencil_desc.FrontFace.StencilFailOp),
                      static_cast<u32>(depth_stencil_desc.FrontFace.StencilDepthFailOp),
                      static_cast<u32>(depth_stencil_desc.FrontFace.StencilPassOp),
                      static_cast<u32>(depth_stencil_desc.FrontFace.StencilFunc),
                      blend_desc.RenderTarget[0].BlendEnable,
                      blend_desc.RenderTarget[0].LogicOpEnable,
                      static_cast<u32>(desc.IBStripCutValue),
                      static_cast<u32>(desc.PrimitiveTopologyType),
                      static_cast<u32>(rasterizer_desc.FillMode),
                      static_cast<u32>(rasterizer_desc.CullMode), rasterizer_desc.ForcedSampleCount,
                      static_cast<u32>(rasterizer_desc.ConservativeRaster));
        }
        const auto rtv_hex = [&desc](u32 index) {
            return index < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT
                       ? static_cast<u32>(desc.RTVFormats[index])
                       : 0u;
        };
        LOG_ERROR(Render_D3D12,
                  "CreateGraphicsPipelineState failed: {:#x} (num_rt={}, "
                  "rtv[0..3]=[{:#x}, {:#x}, {:#x}, {:#x}], dsv={:#x}, "
                  "vs_size={}, ps_size={}, topology={}, samples={}, inputs={})",
                  static_cast<u32>(hr), desc.NumRenderTargets, rtv_hex(0),
                  rtv_hex(1), rtv_hex(2), rtv_hex(3),
                  static_cast<u32>(desc.DSVFormat), desc.VS.BytecodeLength,
                  desc.PS.BytecodeLength, static_cast<u32>(state.topology.Value()),
                  desc.SampleDesc.Count, desc.InputLayout.NumElements);
        pipeline.Reset();
        return;
    }

    LOG_DEBUG(Render_D3D12, "Graphics pipeline created (topology={}, rts={}, dsv={})",
              static_cast<u32>(state.topology.Value()), render_target_count,
              static_cast<u32>(desc.DSVFormat));
}

GraphicsPipeline::~GraphicsPipeline() = default;

} // namespace D3D12
