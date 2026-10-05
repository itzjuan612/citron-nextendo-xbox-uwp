// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <new>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include "common/bit_cast.h"
#include "common/logging.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_pipeline_cache.h"
#include "video_core/renderer_d3d12/d3d12_texture_cache.h"
#include "video_core/surface.h"

namespace D3D12 {

namespace {

using Tegra::Engines::Maxwell3D;
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCore::Surface::PixelFormat;

/// A pipeline cannot be created when any colour attachment format has no render-target
/// representation in D3D12 (e.g. R11G11B10_FLOAT). Rejecting the key here avoids paying for
/// shader translation and a failing CreateGraphicsPipelineState on every draw.
[[nodiscard]] bool IsPipelineFormatRepresentable(const GraphicsPipelineCacheKey& key) {
    for (const u32 raw_format : key.state.color_formats) {
        const auto format = static_cast<Tegra::RenderTargetFormat>(raw_format);
        if (format == Tegra::RenderTargetFormat::NONE) {
            continue;
        }
        const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
        if (pixel_format == VideoCore::Surface::PixelFormat::Invalid ||
            !TextureCacheRuntime::IsRepresentableRenderTarget(pixel_format)) {
            return false;
        }
    }
    return true;
}

// The Xbox grants a fixed 5120 MB commit budget. Translating a large shader transiently
// commits hundreds of MB (captured OOM stacks land in the Maxwell->IR passes), which does
// not fit while the process already sits at ~4.5 GB. Skip compiling when the headroom is
// gone instead of letting the allocation kill the app; the pipeline is retried later.
constexpr u64 CommitBudgetBytes = 5120ull * 1024 * 1024;
constexpr u64 PipelineCompileReserveBytes = 640ull * 1024 * 1024;

[[nodiscard]] bool HasPipelineCompileHeadroom() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        return true;
    }
    const u64 committed = static_cast<u64>(pmc.PrivateUsage);
    return committed + PipelineCompileReserveBytes < CommitBudgetBytes;
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell3D::Regs::ComparisonOp comparison) {
    using Op = Maxwell3D::Regs::ComparisonOp;
    switch (comparison) {
    case Op::Never_D3D:
    case Op::Never_GL:
        return Shader::CompareFunction::Never;
    case Op::Less_D3D:
    case Op::Less_GL:
        return Shader::CompareFunction::Less;
    case Op::Equal_D3D:
    case Op::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Op::LessEqual_D3D:
    case Op::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Op::Greater_D3D:
    case Op::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Op::NotEqual_D3D:
    case Op::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Op::GreaterEqual_D3D:
    case Op::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Op::Always_D3D:
    case Op::Always_GL:
        return Shader::CompareFunction::Always;
    }
    return Shader::CompareFunction::Always;
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell3D::Regs::VertexAttribute::Type::SNorm:
    case Maxwell3D::Regs::VertexAttribute::Type::UNorm:
    case Maxwell3D::Regs::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell3D::Regs::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell3D::Regs::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell3D::Regs::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell3D::Regs::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    default:
        return Shader::AttributeType::Disabled;
    }
}

Shader::FragmentOutputType GetFragmentOutputType(u8 encoded_format) {
    const auto format{static_cast<Tegra::RenderTargetFormat>(encoded_format)};
    if (format == Tegra::RenderTargetFormat::NONE) {
        return Shader::FragmentOutputType::Float;
    }
    const auto pixel_format{VideoCore::Surface::PixelFormatFromRenderTargetFormat(format)};
    if (!VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
        return Shader::FragmentOutputType::Float;
    }
    return VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)
               ? Shader::FragmentOutputType::SignedInt
               : Shader::FragmentOutputType::UnsignedInt;
}

Shader::InputTopology MaxwellToInputTopology(Maxwell3D::Regs::PrimitiveTopology topology) {
    using Topology = Maxwell3D::Regs::PrimitiveTopology;
    switch (topology) {
    case Topology::Points:
        return Shader::InputTopology::Points;
    case Topology::Lines:
    case Topology::LineLoop:
    case Topology::LineStrip:
        return Shader::InputTopology::Lines;
    case Topology::LinesAdjacency:
    case Topology::LineStripAdjacency:
        return Shader::InputTopology::LinesAdjacency;
    case Topology::Triangles:
    case Topology::TriangleStrip:
    case Topology::TriangleFan:
        return Shader::InputTopology::Triangles;
    case Topology::TrianglesAdjacency:
    case Topology::TriangleStripAdjacency:
        return Shader::InputTopology::TrianglesAdjacency;
    case Topology::Quads:
    case Topology::QuadStrip:
    case Topology::Polygon:
        return Shader::InputTopology::Triangles;
    default:
        return Shader::InputTopology::Triangles;
    }
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell3D::Regs::PrimitiveTopology topology) {
    using Topology = Maxwell3D::Regs::PrimitiveTopology;
    switch (topology) {
    case Topology::Points:
        return Shader::OutputTopology::PointList;
    case Topology::Lines:
    case Topology::LineLoop:
    case Topology::LineStrip:
    case Topology::LinesAdjacency:
    case Topology::LineStripAdjacency:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping =
            previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{Common::BitCast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexA:
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell3D::Regs::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            info.convert_depth_mode = gl_ndc;
        }
        std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                               &CastAttributeType);
        break;
    case Shader::Stage::TessellationEval:
        // Tessellation fixed parameters are not tracked yet; use common defaults.
        info.tess_clockwise = true;
        info.tess_primitive = Shader::TessPrimitive::Triangles;
        info.tess_spacing = Shader::TessSpacing::Equal;
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        std::ranges::transform(key.state.color_formats, info.frag_color_types.begin(),
                               &GetFragmentOutputType);
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = Common::BitCast<float>(key.state.alpha_test_ref);
        break;
    }
    default:
        break;
    }
    info.input_topology = MaxwellToInputTopology(
        static_cast<Maxwell3D::Regs::PrimitiveTopology>(key.state.topology.Value()));
    return info;
}

D3D12::ShaderStage CompilerStage(Shader::Stage stage) {
    switch (stage) {
    case Shader::Stage::VertexA:
    case Shader::Stage::VertexB:
        return D3D12::ShaderStage::DXIL_SPIRV_SHADER_VERTEX;
    case Shader::Stage::TessellationControl:
        return D3D12::ShaderStage::DXIL_SPIRV_SHADER_TESS_CTRL;
    case Shader::Stage::TessellationEval:
        return D3D12::ShaderStage::DXIL_SPIRV_SHADER_TESS_EVAL;
    case Shader::Stage::Geometry:
        return D3D12::ShaderStage::DXIL_SPIRV_SHADER_GEOMETRY;
    case Shader::Stage::Fragment:
        return D3D12::ShaderStage::DXIL_SPIRV_SHADER_FRAGMENT;
    default:
        return D3D12::ShaderStage::DXIL_SPIRV_SHADER_NONE;
    }
}

} // Anonymous namespace

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory, Device& device_,
                             ShaderCompiler& compiler_)
    : VideoCommon::ShaderCache{device_memory}, device{device_}, compiler{compiler_} {
    profile = Shader::Profile{
        .supported_spirv = 0x00010600,
        .unified_descriptor_binding = false,
        .has_split_descriptor_sets = true,
        .support_descriptor_aliasing = true,
        .support_int8 = false,
        .support_int16 = false,
        .support_int64 = true,
        .support_vertex_instance_id = false,
        .support_float_controls = false,
        .support_separate_denorm_behavior = false,
        .support_separate_rounding_mode = false,
        .support_fp16_denorm_preserve = false,
        .support_fp32_denorm_preserve = false,
        .support_fp16_denorm_flush = false,
        .support_fp32_denorm_flush = false,
        .support_fp16_signed_zero_nan_preserve = false,
        .support_fp32_signed_zero_nan_preserve = false,
        .support_fp64_signed_zero_nan_preserve = false,
        .support_explicit_workgroup_layout = false,
        .support_vote = false,
        .support_viewport_index_layer_non_geometry = false,
        .support_viewport_mask = false,
        .support_typeless_image_loads = false,
        .support_demote_to_helper_invocation = false,
        .support_int64_atomics = false,
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = false,
        .support_native_ndc = false,
        .support_gl_nv_gpu_shader_5 = false,
        .support_gl_amd_gpu_shader_half_float = false,
        .support_gl_texture_shadow_lod = false,
        .support_gl_warp_intrinsics = false,
        .support_gl_variable_aoffi = false,
        .support_gl_sparse_textures = false,
        .support_gl_derivative_control = false,
        .support_scaled_attributes = false,
        .support_multi_viewport = true,
        .support_geometry_streams = false,
        .warp_size_potentially_larger_than_guest = true,
        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_fastmath_off = false,
        .force_fragment_relaxed_precision = false,
        .need_gather_subpixel_offset = false,
        .has_broken_spirv_clamp = false,
        .has_broken_spirv_position_input = false,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = false,
        .has_gl_component_indexing_bug = false,
        .has_gl_precise_bug = false,
        .has_gl_cbuf_ftou_bug = false,
        .has_gl_bool_ref_bug = false,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic = false,
        .gl_max_compute_smem_size = 0,
        .has_broken_robust = false,
        .min_ssbo_alignment = 4,
        .max_user_clip_distances = 8,
    };
    host_info = Shader::HostTranslateInfo{
        .support_float64 = true,
        .support_float16 = false,
        .support_int64 = true,
        .needs_demote_reorder = false,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = false,
        .min_ssbo_alignment = 4,
        .support_geometry_shader_passthrough = false,
        .support_conditional_barrier = false,
    };
}

PipelineCache::~PipelineCache() = default;

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    StoredPipeline* stored = CurrentStoredPipeline();
    return stored ? stored->pipeline.get() : nullptr;
}

PipelineCache::StoredPipeline* PipelineCache::CurrentStoredPipeline() {
    GraphicsPipelineCacheKey key{};
    if (!RefreshStages(key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    if (key.unique_hashes[0] == 0 && key.unique_hashes[1] == 0) {
        // No vertex program is bound, so a graphics pipeline is meaningless:
        // both vertex stages would be empty and CreateGraphicsPipelineState
        // would fail with E_INVALIDARG. A zero fragment hash alone is legal
        // (depth-only prepasses), so only skip when both vertex slots unset.
        static bool logged_no_vertex_program = false;
        if (!logged_no_vertex_program) {
            logged_no_vertex_program = true;
            LOG_DEBUG(Render_D3D12, "Skipping pipeline with no vertex program");
        }
        current_pipeline = nullptr;
        return nullptr;
    }
    key.state.Refresh(*maxwell3d);
    if (current_pipeline) {
        bool same = current_key.state == key.state;
        for (size_t i = 0; i < key.unique_hashes.size() && same; ++i) {
            same = current_key.unique_hashes[i] == key.unique_hashes[i];
        }
        if (same) {
            return current_pipeline;
        }
    }
    current_key = key;
    current_pipeline = CurrentPipelineSlowPath();
    return current_pipeline;
}

PipelineCache::StoredPipeline* PipelineCache::CurrentPipelineSlowPath() {
    const auto it = cache.find(current_key);
    if (it != cache.end()) {
        return it->second.get();
    }
    if (failed_keys.contains(current_key)) {
        return nullptr;
    }
    if (!IsPipelineFormatRepresentable(current_key)) {
        failed_keys.insert(current_key);
        static std::atomic<u32> unrepresentable_keys{0};
        const u32 count = unrepresentable_keys.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 8) {
            LOG_ERROR(Render_D3D12,
                      "Pipeline with unrepresentable render target format skipped (key {})",
                      count);
        }
        return nullptr;
    }
    auto stored = CreatePipeline(current_key);
    if (!stored) {
        return nullptr;
    }
    StoredPipeline* const result = stored.get();
    cache.emplace(current_key, std::move(stored));
    if (cache.size() % 16 == 0) {
        LOG_INFO(Render_D3D12,
                 "Pipeline cache: entries={}, compiled_stages={}, compiled_bytes={}",
                 cache.size(), compiled_stage_count, compiled_stage_bytes);
    }
    return result;
}

std::unique_ptr<PipelineCache::StoredPipeline> PipelineCache::CreatePipeline(
    const GraphicsPipelineCacheKey& key) try {
    if (!HasPipelineCompileHeadroom()) {
        static std::atomic<u32> skip_count{0};
        const u32 count = skip_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 8 || count % 64 == 0) {
            LOG_WARNING(Render_D3D12,
                        "Pipeline compile skipped: commit headroom below {} MB (count {})",
                        PipelineCompileReserveBytes / (1024 * 1024), count);
        }
        return nullptr;
    }
    VideoCommon::ShaderCache::GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, key.unique_hashes);

    pools.ReleaseContents();
    std::array<Shader::IR::Program, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    Shader::IR::Program* layer_source_program{};
    size_t env_index{0};
    for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram; ++index) {
        const bool is_emulated_stage =
            layer_source_program != nullptr &&
            index == static_cast<u32>(Tegra::Engines::Maxwell3D::Regs::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            const auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*environments.Span()[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }

    auto stored = std::make_unique<StoredPipeline>();
    std::array<std::vector<u8>, MAX_SHADER_STAGES> stages{};
    std::array<bool, MAX_SHADER_STAGES> runtime_data{};
    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    // Program 0 is VertexA: unlike the Vulkan backend (which asserts here), a lone
    // VertexA is a valid vertex stage and is emitted as stage 0.
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
         index < Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram; ++index) {
        const bool is_emulated_stage =
            layer_source_program != nullptr &&
            index == static_cast<u32>(Tegra::Engines::Maxwell3D::Regs::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index == 0 ? 0 : index - 1};

        const auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage)};
        ConvertLegacyToGeneric(program, runtime_info);
        std::vector<u32> code = EmitSPIRV(profile, runtime_info, program, binding);

        ShaderMetadata metadata{};
        std::vector<u8> dxil = compiler.Compile({code.data(), code.size()},
                                                CompilerStage(program.stage), metadata);
        if (dxil.empty()) {
            LOG_ERROR(Render_D3D12, "Draw pipeline stage {} compilation failed: {}", stage_index,
                      compiler.GetLastError());
            return nullptr;
        }
        stages[stage_index] = std::move(dxil);
        runtime_data[stage_index] = metadata.requires_runtime_data;
        compiled_stage_count += 1;
        compiled_stage_bytes += stages[stage_index].size();
        // Copy only the draw-time `info` out of the IR program; the block/syntax vectors
        // stay in the local `programs` array and are released when this function returns.
        stored->infos_storage[stage_index] = program.info;
        stored->infos[stage_index] = &stored->infos_storage[stage_index];
        previous_stage = &program;
    }
    stored->pipeline = std::make_unique<GraphicsPipeline>(device.GetDevice(), key, stages,
                                                          stored->infos, runtime_data);
    if (!stored->pipeline->IsValid()) {
        LOG_ERROR(Render_D3D12, "Draw graphics pipeline creation failed");
        return nullptr;
    }
    return stored;
} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_D3D12, "Draw shader translation failed: {}", exception.what());
    return nullptr;
} catch (const std::bad_alloc&) {
    // The fixed Xbox commit budget can run out while translating a large shader (the new
    // handler throws after logging). Skip this pipeline instead of killing the process;
    // the draw is retried and succeeds once there is headroom again.
    static std::atomic<u32> oom_count{0};
    const u32 count = oom_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 8 || count % 64 == 0) {
        LOG_ERROR(Render_D3D12,
                  "Pipeline compile ran out of memory; skipping pipeline (count {})", count);
    }
    return nullptr;
}

} // namespace D3D12
