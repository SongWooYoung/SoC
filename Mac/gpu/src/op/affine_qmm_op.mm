#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/affine_qmm_op.h"

#include "buffer/buffer_arena.h"
#include "metal/command_stream.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

bool UseExperimentalQ4MlpSpecialized() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_Q4_MLP_SPECIALIZED");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQ4LmheadSpecialized() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_Q4_LMHEAD_SPECIALIZED");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQ4DownprojSpecialized() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DOWNPROJ_SPECIALIZED");
    return value != nullptr && std::string(value) == "1";
}

bool IsMlpDecodeProfile(const char* profile_label) {
    if (profile_label == nullptr) {
        return false;
    }
    const std::string label(profile_label);
    return label == "GateProjDecodeQ4" || label == "UpProjDecodeQ4" || label == "DownProjDecodeQ4";
}

bool IsLmheadDecodeProfile(const char* profile_label) {
    if (profile_label == nullptr) {
        return false;
    }
    return std::string(profile_label) == "LMHeadDecodeQ4";
}

bool IsDownprojDecodeProfile(const char* profile_label) {
    if (profile_label == nullptr) {
        return false;
    }
    return std::string(profile_label) == "DownProjDecodeQ4";
}

struct MetalAffineQmmParams {
    std::uint32_t row_count;
    std::uint32_t inner_dim;
    std::uint32_t column_count;
    std::uint32_t output_row_stride;
    std::uint32_t packed_inner_dim;
    std::uint32_t groups_per_row;
    std::uint32_t group_size;
    std::uint32_t bits;
    std::uint32_t enable_silu;
    std::uint32_t add_residual;
    std::uint32_t padding;
};

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalAffineQmmParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalAffineQmmParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalAffineQmmParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params length:sizeof(MetalAffineQmmParams) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate affine qmm params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

}  // namespace

bool AffineQmmOp::Run(const MetalContext& context,
                      PipelineCache* pipeline_cache,
                      const DeviceTensor& lhs,
                      const AffineQmmWeight& weight,
                      const DeviceTensor* residual,
                      const DeviceTensor& output,
                      const AffineQmmParams& params,
                      BufferArena* temporary_arena,
                      CommandStream* stream,
                      std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }
    if (!lhs.IsValid() || !weight.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm requires valid tensors";
        }
        return false;
    }
    if (params.add_residual != (residual != nullptr)) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm params.add_residual must match whether a residual tensor is provided";
        }
        return false;
    }
    if (lhs.GetDesc().GetDataType() != DataType::kFloat32 ||
        weight.qweight.GetDesc().GetDataType() != DataType::kUInt32 ||
        weight.scales.GetDesc().GetDataType() != DataType::kFloat32 ||
        weight.qbiases.GetDesc().GetDataType() != DataType::kFloat32 ||
        output.GetDesc().GetDataType() != DataType::kFloat32) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm expects float32 lhs/scales/qbiases/output and uint32 packed weights";
        }
        return false;
    }
    if (lhs.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm expects rank-2 lhs and output";
        }
        return false;
    }
    if (params.row_count != lhs.GetDesc().GetShape()[0] || params.inner_dim != lhs.GetDesc().GetShape()[1]) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm lhs shape does not match params";
        }
        return false;
    }
    if (output.GetDesc().GetShape() != std::vector<std::size_t>{params.row_count, params.column_count}) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm output shape does not match params";
        }
        return false;
    }
    if (params.output_row_stride != output.GetDesc().GetShape()[1]) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm output row stride must match contiguous output column count";
        }
        return false;
    }
    if (residual != nullptr &&
        (!residual->IsValid() || residual->GetDesc().GetDataType() != DataType::kFloat32 ||
         residual->GetDesc().Rank() != 2 ||
         residual->GetDesc().GetShape() != output.GetDesc().GetShape())) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm residual must match output shape";
        }
        return false;
    }
    if (params.inner_dim > 4096u) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm inner_dim exceeds current kernel scratch capacity";
        }
        return false;
    }

    const std::uint32_t elems_per_int = 32u / weight.bits;
    if (params.inner_dim % elems_per_int != 0 || params.inner_dim % weight.group_size != 0) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm inner_dim must align to packed/group sizes";
        }
        return false;
    }
    const std::uint32_t packed_inner_dim = params.inner_dim / elems_per_int;
    const std::uint32_t groups_per_row = params.inner_dim / weight.group_size;
    if (weight.qweight.GetDesc().GetShape() != std::vector<std::size_t>{params.column_count, packed_inner_dim} ||
        weight.scales.GetDesc().GetShape() != std::vector<std::size_t>{params.column_count, groups_per_row} ||
        weight.qbiases.GetDesc().GetShape() != std::vector<std::size_t>{params.column_count, groups_per_row}) {
        if (error_message != nullptr) {
            *error_message = "AffineQmm weight tensor shapes do not match params";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kEmbedding;
    const bool use_downproj_specialized =
        params.row_count == 1 && UseExperimentalQ4DownprojSpecialized() && IsDownprojDecodeProfile(params.profile_label);
    const bool use_mlp_specialized =
        params.row_count == 1 && UseExperimentalQ4MlpSpecialized() && IsMlpDecodeProfile(params.profile_label) &&
        !use_downproj_specialized;
    const bool use_lmhead_specialized =
        params.row_count == 1 && UseExperimentalQ4LmheadSpecialized() && IsLmheadDecodeProfile(params.profile_label);
            key.function_name = use_downproj_specialized ? "affine_qmm_t_4bit_lmhead2"
                                  : (use_mlp_specialized ? "affine_qmm_t_4bit_mlp2"
                                  : ((use_lmhead_specialized || use_downproj_specialized)
                                      ? "affine_qmm_t_4bit_lmhead2"
                                      : "affine_qmm_t_4bit"));
            key.threadgroup_width = use_mlp_specialized ? 16 : 32;
    key.threadgroup_height = 1;
    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> lhs_buffer = (__bridge id<MTLBuffer>)lhs.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> qweight_buffer = (__bridge id<MTLBuffer>)weight.qweight.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> scales_buffer = (__bridge id<MTLBuffer>)weight.scales.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> qbiases_buffer = (__bridge id<MTLBuffer>)weight.qbiases.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> residual_buffer =
            residual == nullptr ? output_buffer : (__bridge id<MTLBuffer>)residual->GetBuffer()->GetNativeHandle();

        MetalAffineQmmParams metal_params = {params.row_count,
                                             params.inner_dim,
                                             params.column_count,
                                             params.output_row_stride,
                                             packed_inner_dim,
                                             groups_per_row,
                                             weight.group_size,
                                             weight.bits,
                                             params.enable_silu ? 1u : 0u,
                                             params.add_residual ? 1u : 0u,
                                             0u};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        id<MTLCommandBuffer> command_buffer = nil;
        if (stream != nullptr) {
            encoder = (__bridge id<MTLComputeCommandEncoder>)stream->GetOrCreateComputeEncoder();
        } else {
            id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
            command_buffer = [command_queue commandBuffer];
            encoder = [command_buffer computeCommandEncoder];
        }
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create affine qmm encoder";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs_buffer offset:lhs.GetByteOffset() atIndex:0];
        [encoder setBuffer:qweight_buffer offset:weight.qweight.GetByteOffset() atIndex:1];
        [encoder setBuffer:scales_buffer offset:weight.scales.GetByteOffset() atIndex:2];
        [encoder setBuffer:qbiases_buffer offset:weight.qbiases.GetByteOffset() atIndex:3];
        [encoder setBuffer:residual_buffer offset:(residual == nullptr ? output.GetByteOffset() : residual->GetByteOffset()) atIndex:4];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:5];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:6];

        const std::uint32_t outputs_per_thread =
            (use_mlp_specialized || use_lmhead_specialized || use_downproj_specialized) ? 2u : 1u;
        const MTLSize threadgroup_size = MTLSizeMake(key.threadgroup_width, 1, 1);
        const MTLSize threadgroups_per_grid =
            MTLSizeMake((params.column_count + (key.threadgroup_width * outputs_per_thread) - 1) /
                            (key.threadgroup_width * outputs_per_thread),
                        params.row_count,
                        1);
        [encoder dispatchThreadgroups:threadgroups_per_grid threadsPerThreadgroup:threadgroup_size];

        if (stream == nullptr) {
            [encoder endEncoding];
            const char* profile_label = params.profile_label == nullptr ? "AffineQmm" : params.profile_label;
            if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                               "AffineQmm command buffer failed",
                                               profile_label,
                                               1,
                                               error_message)) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace soc::gpu
