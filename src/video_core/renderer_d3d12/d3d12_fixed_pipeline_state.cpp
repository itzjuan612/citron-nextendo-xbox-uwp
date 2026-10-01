// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>

#include "common/bit_cast.h"
#include "common/cityhash.h"
#include "video_core/engines/draw_manager.h"
#include "video_core/renderer_d3d12/d3d12_fixed_pipeline_state.h"

namespace D3D12 {

void FixedPipelineState::Refresh(Tegra::Engines::Maxwell3D& maxwell3d) {
    const Tegra::Engines::Maxwell3D::Regs& regs = maxwell3d.regs;
    const auto topology_ = maxwell3d.draw_manager->GetDrawState().topology;

    raster1 = 0;
    polygon_mode.Assign(PackPolygonMode(regs.polygon_mode_front));
    ndc_minus_one_to_one.Assign(regs.depth_mode ==
                                        Tegra::Engines::Maxwell3D::Regs::DepthMode::MinusOneToOne
                                    ? 1
                                    : 0);
    y_negate.Assign(regs.window_origin.mode !=
                            Tegra::Engines::Maxwell3D::Regs::WindowOrigin::Mode::UpperLeft
                        ? 1
                        : 0);
    provoking_vertex_last.Assign(
        regs.provoking_vertex == Tegra::Engines::Maxwell3D::Regs::ProvokingVertex::Last ? 1 : 0);
    topology.Assign(topology_);
    msaa_mode.Assign(regs.anti_alias_samples_mode);
    smooth_lines.Assign(regs.line_anti_alias_enable != 0 ? 1 : 0);
    conservative_raster_enable.Assign(regs.conservative_raster_enable != 0 ? 1 : 0);

    raster2 = 0;
    cull_face.Assign(PackCullFace(regs.gl_cull_face));
    cull_enable.Assign(regs.gl_cull_test_enabled != 0 ? 1 : 0);
    primitive_restart_enable.Assign(regs.primitive_restart.enabled != 0 ? 1 : 0);
    rasterize_enable.Assign(regs.rasterize_enable != 0 ? 1 : 0);
    logic_op.Assign(PackLogicOp(regs.logic_op.op));
    logic_op_enable.Assign(regs.logic_op.enable != 0 ? 1 : 0);
    depth_clamp_disabled.Assign(
        regs.viewport_clip_control.geometry_clip ==
            Tegra::Engines::Maxwell3D::Regs::ViewportClipControl::GeometryClip::Passthrough ||
        regs.viewport_clip_control.geometry_clip ==
            Tegra::Engines::Maxwell3D::Regs::ViewportClipControl::GeometryClip::FrustumXYZ ||
        regs.viewport_clip_control.geometry_clip ==
            Tegra::Engines::Maxwell3D::Regs::ViewportClipControl::GeometryClip::FrustumZ);

    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        0u, // Points
        1u, // Lines
        1u, // LineLoop
        1u, // LineStrip
        2u, // Triangles
        2u, // TriangleStrip
        2u, // TriangleFan
        2u, // Quads
        2u, // QuadStrip
        2u, // Polygon
        1u, // LinesAdjacency
        1u, // LineStripAdjacency
        2u, // TrianglesAdjacency
        2u, // TriangleStripAdjacency
        2u, // Patches
    };
    depth_bias_enable.Assign(
        enabled_lut[POLYGON_OFFSET_ENABLE_LUT[static_cast<u32>(topology_)]] != 0 ? 1 : 0);

    u32 packed_front_face = PackFrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        // Flip front face
        packed_front_face = 1 - packed_front_face;
    }
    front_face.Assign(packed_front_face);

    depth1 = 0;
    front.action_stencil_fail.Assign(PackStencilOp(regs.stencil_front_op.fail));
    front.action_depth_fail.Assign(PackStencilOp(regs.stencil_front_op.zfail));
    front.action_depth_pass.Assign(PackStencilOp(regs.stencil_front_op.zpass));
    front.test_func.Assign(PackComparisonOp(regs.stencil_front_op.func));
    if (regs.stencil_two_side_enable) {
        back.action_stencil_fail.Assign(PackStencilOp(regs.stencil_back_op.fail));
        back.action_depth_fail.Assign(PackStencilOp(regs.stencil_back_op.zfail));
        back.action_depth_pass.Assign(PackStencilOp(regs.stencil_back_op.zpass));
        back.test_func.Assign(PackComparisonOp(regs.stencil_back_op.func));
    } else {
        back.action_stencil_fail.Assign(front.action_stencil_fail);
        back.action_depth_fail.Assign(front.action_depth_fail);
        back.action_depth_pass.Assign(front.action_depth_pass);
        back.test_func.Assign(front.test_func);
    }
    stencil_enable.Assign(regs.stencil_enable);
    depth_write_enable.Assign(regs.depth_write_enabled);
    depth_bounds_enable.Assign(regs.depth_bounds_enable);
    depth_test_enable.Assign(regs.depth_test_enable);
    depth_test_func.Assign(PackComparisonOp(regs.depth_test_func));

    depth2 = 0;
    const auto test_func = regs.alpha_test_enabled != 0
                               ? regs.alpha_test_func
                               : Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Always_GL;
    alpha_test_func.Assign(PackComparisonOp(test_func));
    alpha_test_enable.Assign(regs.alpha_test_enabled != 0 ? 1 : 0);
    early_z.Assign(regs.mandated_early_z != 0 ? 1 : 0);
    depth_enabled.Assign(regs.zeta_enable != 0 ? 1 : 0);
    depth_format.Assign(static_cast<u32>(regs.zeta.format));
    alpha_to_coverage_enabled.Assign(regs.anti_alias_alpha_control.alpha_to_coverage != 0 ? 1 : 0);
    alpha_to_one_enabled.Assign(regs.anti_alias_alpha_control.alpha_to_one != 0 ? 1 : 0);
    app_stage.Assign(maxwell3d.engine_state);

    for (size_t i = 0; i < regs.rt.size(); ++i) {
        color_formats[i] = static_cast<u8>(regs.rt[i].format);
    }
    for (size_t index = 0; index < attachments.size(); ++index) {
        attachments[index].Refresh(regs, index);
    }
    alpha_test_ref = Common::BitCast<u32>(regs.alpha_test_ref);
    point_size = Common::BitCast<u32>(regs.point_size);

    for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::NumVertexArrays; ++index) {
        const bool is_enabled = regs.vertex_stream_instances.IsInstancingEnabled(index);
        binding_divisors[index] = is_enabled ? regs.vertex_streams[index].frequency : 0;
        vertex_strides[index] = static_cast<u16>(regs.vertex_streams[index].stride.Value());
    }
    for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes; ++index) {
        const auto& input = regs.vertex_attrib_format[index];
        auto& attribute = attributes[index];
        attribute.raw = 0;
        attribute.enabled.Assign(input.constant ? 0 : 1);
        attribute.buffer.Assign(input.buffer);
        attribute.offset.Assign(input.offset);
        attribute.type.Assign(static_cast<u32>(input.type.Value()));
        attribute.size.Assign(static_cast<u32>(input.size.Value()));
    }
    for (size_t i = 0; i < regs.viewport_transform.size(); ++i) {
        viewport_swizzles[i] = static_cast<u16>(regs.viewport_transform[i].swizzle.raw);
    }
}

void FixedPipelineState::BlendingAttachment::Refresh(const Tegra::Engines::Maxwell3D::Regs& regs,
                                                     size_t index) {
    const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];

    raw = 0;
    mask_r.Assign(mask.R);
    mask_g.Assign(mask.G);
    mask_b.Assign(mask.B);
    mask_a.Assign(mask.A);

    if (!regs.blend.enable[index]) {
        return;
    }

    const auto setup_blend = [&]<typename T>(const T& src) {
        equation_rgb.Assign(PackBlendEquation(src.color_op));
        equation_a.Assign(PackBlendEquation(src.alpha_op));
        factor_source_rgb.Assign(PackBlendFactor(src.color_source));
        factor_dest_rgb.Assign(PackBlendFactor(src.color_dest));
        factor_source_a.Assign(PackBlendFactor(src.alpha_source));
        factor_dest_a.Assign(PackBlendFactor(src.alpha_dest));
        enable.Assign(1);
    };

    if (!regs.blend_per_target_enabled) {
        setup_blend(regs.blend);
        return;
    }
    setup_blend(regs.blend_per_target[index]);
}

size_t FixedPipelineState::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof(*this));
    return static_cast<size_t>(hash);
}

u32 FixedPipelineState::PackComparisonOp(
    Tegra::Engines::Maxwell3D::Regs::ComparisonOp op) noexcept {
    // OpenGL enums go from 0x200 to 0x207 and the others from 1 to 8.
    const u32 value = static_cast<u32>(op);
    return value - (value >= 0x200 ? 0x200 : 1);
}

Tegra::Engines::Maxwell3D::Regs::ComparisonOp FixedPipelineState::UnpackComparisonOp(
    u32 packed) noexcept {
    return static_cast<Tegra::Engines::Maxwell3D::Regs::ComparisonOp>(packed + 1);
}

u32 FixedPipelineState::PackStencilOp(Tegra::Engines::Maxwell3D::Regs::StencilOp::Op op) noexcept {
    switch (op) {
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Keep_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Keep_GL:
        return 0;
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Zero_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Zero_GL:
        return 1;
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Replace_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Replace_GL:
        return 2;
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::IncrSaturate_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::IncrSaturate_GL:
        return 3;
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::DecrSaturate_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::DecrSaturate_GL:
        return 4;
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Invert_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Invert_GL:
        return 5;
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Incr_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Incr_GL:
        return 6;
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Decr_D3D:
    case Tegra::Engines::Maxwell3D::Regs::StencilOp::Op::Decr_GL:
        return 7;
    }
    return 0;
}

Tegra::Engines::Maxwell3D::Regs::StencilOp::Op FixedPipelineState::UnpackStencilOp(
    u32 packed) noexcept {
    using Op = Tegra::Engines::Maxwell3D::Regs::StencilOp::Op;
    static constexpr std::array LUT = {Op::Keep_D3D,     Op::Zero_D3D,    Op::Replace_D3D,
                                       Op::IncrSaturate_D3D, Op::DecrSaturate_D3D, Op::Invert_D3D,
                                       Op::Incr_D3D,     Op::Decr_D3D};
    return LUT[packed];
}

u32 FixedPipelineState::PackCullFace(Tegra::Engines::Maxwell3D::Regs::CullFace cull) noexcept {
    const u32 value = static_cast<u32>(cull);
    return value - (value == 0x408 ? 0x406 : 0x404);
}

Tegra::Engines::Maxwell3D::Regs::CullFace FixedPipelineState::UnpackCullFace(u32 packed) noexcept {
    using CullFace = Tegra::Engines::Maxwell3D::Regs::CullFace;
    static constexpr std::array LUT = {CullFace::Front, CullFace::Back, CullFace::FrontAndBack};
    return LUT[packed];
}

u32 FixedPipelineState::PackFrontFace(Tegra::Engines::Maxwell3D::Regs::FrontFace face) noexcept {
    return static_cast<u32>(face) - 0x900;
}

Tegra::Engines::Maxwell3D::Regs::FrontFace FixedPipelineState::UnpackFrontFace(
    u32 packed) noexcept {
    return static_cast<Tegra::Engines::Maxwell3D::Regs::FrontFace>(packed + 0x900);
}

u32 FixedPipelineState::PackPolygonMode(
    Tegra::Engines::Maxwell3D::Regs::PolygonMode mode) noexcept {
    return static_cast<u32>(mode) - 0x1B00;
}

Tegra::Engines::Maxwell3D::Regs::PolygonMode FixedPipelineState::UnpackPolygonMode(
    u32 packed) noexcept {
    return static_cast<Tegra::Engines::Maxwell3D::Regs::PolygonMode>(packed + 0x1B00);
}

u32 FixedPipelineState::PackLogicOp(Tegra::Engines::Maxwell3D::Regs::LogicOp::Op op) noexcept {
    return static_cast<u32>(op) - 0x1500;
}

Tegra::Engines::Maxwell3D::Regs::LogicOp::Op FixedPipelineState::UnpackLogicOp(
    u32 packed) noexcept {
    return static_cast<Tegra::Engines::Maxwell3D::Regs::LogicOp::Op>(packed + 0x1500);
}

u32 FixedPipelineState::PackBlendEquation(
    Tegra::Engines::Maxwell3D::Regs::Blend::Equation equation) noexcept {
    using Equation = Tegra::Engines::Maxwell3D::Regs::Blend::Equation;
    switch (equation) {
    case Equation::Add_D3D:
    case Equation::Add_GL:
        return 0;
    case Equation::Subtract_D3D:
    case Equation::Subtract_GL:
        return 1;
    case Equation::ReverseSubtract_D3D:
    case Equation::ReverseSubtract_GL:
        return 2;
    case Equation::Min_D3D:
    case Equation::Min_GL:
        return 3;
    case Equation::Max_D3D:
    case Equation::Max_GL:
        return 4;
    }
    return 0;
}

Tegra::Engines::Maxwell3D::Regs::Blend::Equation FixedPipelineState::UnpackBlendEquation(
    u32 packed) noexcept {
    using Equation = Tegra::Engines::Maxwell3D::Regs::Blend::Equation;
    static constexpr std::array LUT = {Equation::Add_D3D, Equation::Subtract_D3D,
                                       Equation::ReverseSubtract_D3D, Equation::Min_D3D,
                                       Equation::Max_D3D};
    return LUT[packed];
}

u32 FixedPipelineState::PackBlendFactor(
    Tegra::Engines::Maxwell3D::Regs::Blend::Factor factor) noexcept {
    using Factor = Tegra::Engines::Maxwell3D::Regs::Blend::Factor;
    switch (factor) {
    case Factor::Zero_D3D:
    case Factor::Zero_GL:
        return 0;
    case Factor::One_D3D:
    case Factor::One_GL:
        return 1;
    case Factor::SourceColor_D3D:
    case Factor::SourceColor_GL:
        return 2;
    case Factor::OneMinusSourceColor_D3D:
    case Factor::OneMinusSourceColor_GL:
        return 3;
    case Factor::SourceAlpha_D3D:
    case Factor::SourceAlpha_GL:
        return 4;
    case Factor::OneMinusSourceAlpha_D3D:
    case Factor::OneMinusSourceAlpha_GL:
        return 5;
    case Factor::DestAlpha_D3D:
    case Factor::DestAlpha_GL:
        return 6;
    case Factor::OneMinusDestAlpha_D3D:
    case Factor::OneMinusDestAlpha_GL:
        return 7;
    case Factor::DestColor_D3D:
    case Factor::DestColor_GL:
        return 8;
    case Factor::OneMinusDestColor_D3D:
    case Factor::OneMinusDestColor_GL:
        return 9;
    case Factor::SourceAlphaSaturate_D3D:
    case Factor::SourceAlphaSaturate_GL:
        return 10;
    case Factor::Source1Color_D3D:
    case Factor::Source1Color_GL:
        return 11;
    case Factor::OneMinusSource1Color_D3D:
    case Factor::OneMinusSource1Color_GL:
        return 12;
    case Factor::Source1Alpha_D3D:
    case Factor::Source1Alpha_GL:
        return 13;
    case Factor::OneMinusSource1Alpha_D3D:
    case Factor::OneMinusSource1Alpha_GL:
        return 14;
    case Factor::BlendFactor_D3D:
    case Factor::ConstantColor_GL:
        return 15;
    case Factor::OneMinusBlendFactor_D3D:
    case Factor::OneMinusConstantColor_GL:
        return 16;
    case Factor::BothSourceAlpha_D3D:
    case Factor::ConstantAlpha_GL:
        return 17;
    case Factor::OneMinusBothSourceAlpha_D3D:
    case Factor::OneMinusConstantAlpha_GL:
        return 18;
    }
    return 0;
}

Tegra::Engines::Maxwell3D::Regs::Blend::Factor FixedPipelineState::UnpackBlendFactor(
    u32 packed) noexcept {
    using Factor = Tegra::Engines::Maxwell3D::Regs::Blend::Factor;
    static constexpr std::array LUT = {
        Factor::Zero_D3D,
        Factor::One_D3D,
        Factor::SourceColor_D3D,
        Factor::OneMinusSourceColor_D3D,
        Factor::SourceAlpha_D3D,
        Factor::OneMinusSourceAlpha_D3D,
        Factor::DestAlpha_D3D,
        Factor::OneMinusDestAlpha_D3D,
        Factor::DestColor_D3D,
        Factor::OneMinusDestColor_D3D,
        Factor::SourceAlphaSaturate_D3D,
        Factor::Source1Color_D3D,
        Factor::OneMinusSource1Color_D3D,
        Factor::Source1Alpha_D3D,
        Factor::OneMinusSource1Alpha_D3D,
        Factor::BlendFactor_D3D,
        Factor::OneMinusBlendFactor_D3D,
        Factor::BothSourceAlpha_D3D,
        Factor::OneMinusBothSourceAlpha_D3D,
    };
    return LUT[packed];
}

} // namespace D3D12
