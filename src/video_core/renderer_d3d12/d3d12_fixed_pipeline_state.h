// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <type_traits>

#include "common/bit_field.h"
#include "common/common_types.h"

#include "video_core/engines/maxwell_3d.h"
#include "video_core/surface.h"

namespace D3D12 {

/// Packs the Maxwell fixed-function register state that has a direct D3D12 pipeline
/// equivalent. D3D12 exposes no dynamic state for most of these, so the whole struct is
/// part of the graphics pipeline cache key and a PSO is created per unique combination.
struct FixedPipelineState {
    static u32 PackComparisonOp(Tegra::Engines::Maxwell3D::Regs::ComparisonOp op) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::ComparisonOp UnpackComparisonOp(u32 packed) noexcept;

    static u32 PackStencilOp(Tegra::Engines::Maxwell3D::Regs::StencilOp::Op op) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::StencilOp::Op UnpackStencilOp(u32 packed) noexcept;

    static u32 PackCullFace(Tegra::Engines::Maxwell3D::Regs::CullFace cull) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::CullFace UnpackCullFace(u32 packed) noexcept;

    static u32 PackFrontFace(Tegra::Engines::Maxwell3D::Regs::FrontFace face) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::FrontFace UnpackFrontFace(u32 packed) noexcept;

    static u32 PackPolygonMode(Tegra::Engines::Maxwell3D::Regs::PolygonMode mode) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::PolygonMode UnpackPolygonMode(u32 packed) noexcept;

    static u32 PackLogicOp(Tegra::Engines::Maxwell3D::Regs::LogicOp::Op op) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::LogicOp::Op UnpackLogicOp(u32 packed) noexcept;

    static u32 PackBlendEquation(Tegra::Engines::Maxwell3D::Regs::Blend::Equation equation) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::Blend::Equation UnpackBlendEquation(
        u32 packed) noexcept;

    static u32 PackBlendFactor(Tegra::Engines::Maxwell3D::Regs::Blend::Factor factor) noexcept;
    static Tegra::Engines::Maxwell3D::Regs::Blend::Factor UnpackBlendFactor(u32 packed) noexcept;

    struct BlendingAttachment {
        union {
            u32 raw;
            BitField<0, 1, u32> mask_r;
            BitField<1, 1, u32> mask_g;
            BitField<2, 1, u32> mask_b;
            BitField<3, 1, u32> mask_a;
            BitField<4, 3, u32> equation_rgb;
            BitField<7, 3, u32> equation_a;
            BitField<10, 5, u32> factor_source_rgb;
            BitField<15, 5, u32> factor_dest_rgb;
            BitField<20, 5, u32> factor_source_a;
            BitField<25, 5, u32> factor_dest_a;
            BitField<30, 1, u32> enable;
        };

        void Refresh(const Tegra::Engines::Maxwell3D::Regs& regs, size_t index);

        [[nodiscard]] std::array<bool, 4> Mask() const noexcept {
            return {mask_r != 0, mask_g != 0, mask_b != 0, mask_a != 0};
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::Blend::Equation EquationRGB() const noexcept {
            return UnpackBlendEquation(equation_rgb.Value());
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::Blend::Equation
        EquationAlpha() const noexcept {
            return UnpackBlendEquation(equation_a.Value());
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::Blend::Factor SourceRGBFactor() const noexcept {
            return UnpackBlendFactor(factor_source_rgb.Value());
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::Blend::Factor DestRGBFactor() const noexcept {
            return UnpackBlendFactor(factor_dest_rgb.Value());
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::Blend::Factor
        SourceAlphaFactor() const noexcept {
            return UnpackBlendFactor(factor_source_a.Value());
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::Blend::Factor
        DestAlphaFactor() const noexcept {
            return UnpackBlendFactor(factor_dest_a.Value());
        }
    };

    union VertexAttribute {
        u32 raw;
        BitField<0, 1, u32> enabled;
        BitField<1, 5, u32> buffer;
        BitField<6, 14, u32> offset;
        BitField<20, 3, u32> type;
        BitField<23, 6, u32> size;

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type Type() const noexcept {
            return static_cast<Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type>(
                type.Value());
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Size Size() const noexcept {
            return static_cast<Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Size>(
                size.Value());
        }
    };

    template <size_t Position>
    union StencilFace {
        BitField<Position + 0, 3, u32> action_stencil_fail;
        BitField<Position + 3, 3, u32> action_depth_fail;
        BitField<Position + 6, 3, u32> action_depth_pass;
        BitField<Position + 9, 3, u32> test_func;

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::StencilOp::Op
        ActionStencilFail() const noexcept {
            return UnpackStencilOp(action_stencil_fail);
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::StencilOp::Op
        ActionDepthFail() const noexcept {
            return UnpackStencilOp(action_depth_fail);
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::StencilOp::Op
        ActionDepthPass() const noexcept {
            return UnpackStencilOp(action_depth_pass);
        }

        [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::ComparisonOp TestFunc() const noexcept {
            return UnpackComparisonOp(test_func);
        }
    };

    union {
        u32 raster1;
        BitField<0, 2, u32> polygon_mode;
        BitField<2, 1, u32> ndc_minus_one_to_one;
        BitField<3, 1, u32> y_negate;
        BitField<4, 1, u32> provoking_vertex_last;
        BitField<5, 4, Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology> topology;
        BitField<9, 4, Tegra::Texture::MsaaMode> msaa_mode;
        BitField<13, 1, u32> smooth_lines;
        BitField<14, 1, u32> conservative_raster_enable;
    };
    union {
        u32 raster2;
        BitField<0, 2, u32> cull_face;
        BitField<2, 1, u32> cull_enable;
        BitField<3, 1, u32> primitive_restart_enable;
        BitField<4, 1, u32> depth_bias_enable;
        BitField<5, 1, u32> rasterize_enable;
        BitField<6, 4, u32> logic_op;
        BitField<10, 1, u32> logic_op_enable;
        BitField<11, 1, u32> depth_clamp_disabled;
        BitField<12, 1, u32> front_face;
    };
    union {
        u32 depth1;
        StencilFace<0> front;
        StencilFace<12> back;
        BitField<24, 1, u32> stencil_enable;
        BitField<25, 1, u32> depth_write_enable;
        BitField<26, 1, u32> depth_bounds_enable;
        BitField<27, 1, u32> depth_test_enable;
        BitField<28, 3, u32> depth_test_func;
    };
    union {
        u32 depth2;
        BitField<0, 3, u32> alpha_test_func;
        BitField<3, 1, u32> alpha_test_enable;
        BitField<4, 1, u32> early_z;
        BitField<5, 1, u32> depth_enabled;
        BitField<6, 5, u32> depth_format;
        BitField<11, 1, u32> alpha_to_coverage_enabled;
        BitField<12, 1, u32> alpha_to_one_enabled;
        BitField<13, 3, Tegra::Engines::Maxwell3D::EngineHint> app_stage;
    };

    std::array<u8, Tegra::Engines::Maxwell3D::Regs::NumRenderTargets> color_formats{};
    std::array<BlendingAttachment, Tegra::Engines::Maxwell3D::Regs::NumRenderTargets> attachments{};
    std::array<VertexAttribute, Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes> attributes{};
    std::array<u32, Tegra::Engines::Maxwell3D::Regs::NumVertexArrays> binding_divisors{};
    std::array<u16, Tegra::Engines::Maxwell3D::Regs::NumVertexArrays> vertex_strides{};
    std::array<u16, Tegra::Engines::Maxwell3D::Regs::NumViewports> viewport_swizzles{};

    u32 alpha_test_ref{};
    u32 point_size{};

    /// Captures the current Maxwell fixed-function state.
    void Refresh(Tegra::Engines::Maxwell3D& maxwell3d);

    [[nodiscard]] size_t Hash() const noexcept;

    bool operator==(const FixedPipelineState& rhs) const noexcept {
        return std::memcmp(this, &rhs, sizeof(*this)) == 0;
    }

    bool operator!=(const FixedPipelineState& rhs) const noexcept {
        return !operator==(rhs);
    }

    [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::ComparisonOp DepthTestFunc() const noexcept {
        return UnpackComparisonOp(depth_test_func);
    }

    [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::CullFace CullFaceMode() const noexcept {
        return UnpackCullFace(cull_face.Value());
    }

    [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::FrontFace FrontFaceMode() const noexcept {
        return UnpackFrontFace(front_face.Value());
    }

    [[nodiscard]] Tegra::Engines::Maxwell3D::Regs::PolygonMode PolygonModeMode() const noexcept {
        return UnpackPolygonMode(polygon_mode.Value());
    }
};

static_assert(std::has_unique_object_representations_v<FixedPipelineState>);
static_assert(std::is_trivially_copyable_v<FixedPipelineState>);

} // namespace D3D12
