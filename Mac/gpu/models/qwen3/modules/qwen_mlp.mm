#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "models/qwen3/modules/qwen_mlp.h"

#include <cstdlib>
#include <string>

#include "buffer/buffer_arena.h"
#include "kernel/kernel_key.h"
#include "kernel/pipeline_cache.h"
#include "metal/command_stream.h"
#include "metal/metal_context.h"
#include "op/affine_qmm_op.h"
#include "op/elementwise_mul_op.h"
#include "op/linear_op.h"

namespace soc::gpu {
namespace {

struct MetalMatMulParams {
    std::uint32_t row_count;
    std::uint32_t inner_dim;
    std::uint32_t column_count;
    std::uint32_t lhs_row_stride;
    std::uint32_t rhs_row_stride;
    std::uint32_t output_row_stride;
};

bool IsProjectionWeightType(DataType data_type) {
    return data_type == DataType::kFloat32 || data_type == DataType::kFloat16;
}

bool AllocateTemporaryTensor(BufferArena* arena,
                             const TensorDesc& desc,
                             DeviceTensor* tensor,
                             std::string* error_message) {
    if (arena == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Temporary arena is required for QwenMLP";
        }
        return false;
    }
    BufferArenaSlice slice;
    if (!arena->Allocate(desc.ByteSize(), 256, &slice, error_message)) {
        return false;
    }
    *tensor = DeviceTensor(slice.buffer, slice.offset_bytes, desc);
    return true;
}

bool ValidateMlpIO(const DeviceTensor& input,
                   const DeviceTensor& output,
                   const QwenMlpWeights& weights,
                   std::size_t* row_count,
                   std::size_t* hidden_size,
                   std::size_t* intermediate_size,
                   std::string* error_message) {
    if (!input.IsValid() || !output.IsValid() || !weights.gate_proj_weight.IsValid() ||
        !weights.up_proj_weight.IsValid() || !weights.down_proj_weight.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP tensors must be valid";
        }
        return false;
    }
    if (input.GetDesc().GetDataType() != DataType::kFloat32 ||
        output.GetDesc().GetDataType() != DataType::kFloat32 ||
        !IsProjectionWeightType(weights.gate_proj_weight.GetDesc().GetDataType()) ||
        !IsProjectionWeightType(weights.up_proj_weight.GetDesc().GetDataType()) ||
        !IsProjectionWeightType(weights.down_proj_weight.GetDesc().GetDataType())) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP expects float32 activations and float32 or float16 projection weights";
        }
        return false;
    }
    if (input.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2 ||
        weights.gate_proj_weight.GetDesc().Rank() != 2 ||
        weights.up_proj_weight.GetDesc().Rank() != 2 ||
        weights.down_proj_weight.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP expects rank-2 tensors";
        }
        return false;
    }

    const auto& input_shape = input.GetDesc().GetShape();
    const auto& output_shape = output.GetDesc().GetShape();
    const auto& gate_shape = weights.gate_proj_weight.GetDesc().GetShape();
    const auto& up_shape = weights.up_proj_weight.GetDesc().GetShape();
    const auto& down_shape = weights.down_proj_weight.GetDesc().GetShape();
    if (output_shape != input_shape) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP output shape must match input shape";
        }
        return false;
    }
    if (gate_shape[1] != input_shape[1] || up_shape[1] != input_shape[1] || gate_shape[0] != up_shape[0]) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP gate/up projection weights do not match input hidden size";
        }
        return false;
    }
    if (down_shape[0] != input_shape[1] || down_shape[1] != gate_shape[0]) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP down projection weight shape does not match gate/up output size";
        }
        return false;
    }

    *row_count = input_shape[0];
    *hidden_size = input_shape[1];
    *intermediate_size = gate_shape[0];
    return true;
}

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalMatMulParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalMatMulParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalMatMulParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params length:sizeof(MetalMatMulParams) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate fused gate/up params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

bool UseExperimentalFusedGateUp() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_FUSED_GATE_UP");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQ4Decode() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalSafeDecodeBatch() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH");
    return value != nullptr && std::string(value) == "1";
}

bool ScratchMatches(const DeviceTensor& tensor, const TensorDesc& expected_desc) {
    return tensor.IsValid() && tensor.GetDesc().GetDataType() == expected_desc.GetDataType() &&
           tensor.GetDesc().GetShape() == expected_desc.GetShape();
}

bool RunDecodeProjection(const MetalContext& context,
                         PipelineCache* pipeline_cache,
                         const DeviceTensor& input,
                         const DeviceTensor* residual,
                         const DeviceTensor& dense_weight,
                         const AffineQmmWeight& q4_weight,
                         const char* dense_label,
                         const char* q4_label,
                         std::size_t inner_dim,
                         std::size_t column_count,
                         bool enable_silu,
                         bool add_residual,
                         const DeviceTensor& output,
                         BufferArena* temporary_arena,
                         CommandStream* stream,
                         std::string* error_message) {
    const bool use_q4 = input.GetDesc().GetShape()[0] == 1 && UseExperimentalQ4Decode() && q4_weight.IsValid();
    if (use_q4) {
        AffineQmmParams qmm_params;
        qmm_params.row_count = 1;
        qmm_params.inner_dim = static_cast<std::uint32_t>(inner_dim);
        qmm_params.column_count = static_cast<std::uint32_t>(column_count);
        qmm_params.output_row_stride = static_cast<std::uint32_t>(column_count);
        qmm_params.enable_silu = enable_silu;
        qmm_params.add_residual = add_residual;
        qmm_params.profile_label = q4_label;
        return AffineQmmOp::Run(context,
                                pipeline_cache,
                                input,
                                q4_weight,
                                residual,
                                output,
                                qmm_params,
                                temporary_arena,
                                stream,
                                error_message);
    }

    LinearParams params;
    params.activation = enable_silu ? LinearActivation::kSiLU : LinearActivation::kNone;
    params.add_residual = add_residual;
    params.matmul.row_count = 1;
    params.matmul.inner_dim = static_cast<std::uint32_t>(inner_dim);
    params.matmul.column_count = static_cast<std::uint32_t>(column_count);
    params.matmul.profile_label = dense_label;
    params.matmul.decode_mode = true;
    params.matmul.transpose_rhs = true;
    return LinearOp::Run(context,
                         pipeline_cache,
                         input,
                         dense_weight,
                         nullptr,
                         residual,
                         output,
                         params,
                         temporary_arena,
                         stream,
                         error_message);
}

bool RunFusedDecodeGateUp(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          const DeviceTensor& input,
                          const QwenMlpWeights& weights,
                          const DeviceTensor& gate_tensor,
                          const DeviceTensor& up_tensor,
                          std::size_t hidden_size,
                          std::size_t intermediate_size,
                          BufferArena* temporary_arena,
                          CommandStream* stream,
                          std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }
    const bool rhs_is_float16 = weights.gate_proj_weight.GetDesc().GetDataType() == DataType::kFloat16;
    if (weights.up_proj_weight.GetDesc().GetDataType() != weights.gate_proj_weight.GetDesc().GetDataType()) {
        if (error_message != nullptr) {
            *error_message = "Fused decode gate/up requires matching weight dtypes";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kElementwiseMul;
    key.function_name = rhs_is_float16 ? "dual_matmul_f32_f16rhs_decode_vec4" : "dual_matmul_f32_decode_vec4";
    key.threadgroup_width = 32;
    key.threadgroup_height = 1;

    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> lhs_buffer = (__bridge id<MTLBuffer>)input.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> rhs0_buffer = (__bridge id<MTLBuffer>)weights.gate_proj_weight.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> rhs1_buffer = (__bridge id<MTLBuffer>)weights.up_proj_weight.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output0_buffer = (__bridge id<MTLBuffer>)gate_tensor.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output1_buffer = (__bridge id<MTLBuffer>)up_tensor.GetBuffer()->GetNativeHandle();

        MetalMatMulParams metal_params = {1u,
                                          static_cast<std::uint32_t>(hidden_size),
                                          static_cast<std::uint32_t>(intermediate_size),
                                          static_cast<std::uint32_t>(hidden_size),
                                          static_cast<std::uint32_t>(weights.gate_proj_weight.GetDesc().GetShape()[1]),
                                          static_cast<std::uint32_t>(intermediate_size)};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        id<MTLCommandBuffer> command_buffer = nil;
        if (stream != nullptr) {
            encoder = (__bridge id<MTLComputeCommandEncoder>)stream->BeginEncoder();
        } else {
            id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
            command_buffer = [command_queue commandBuffer];
            encoder = [command_buffer computeCommandEncoder];
        }
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create fused gate/up compute encoder";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs_buffer offset:input.GetByteOffset() atIndex:0];
        [encoder setBuffer:rhs0_buffer offset:weights.gate_proj_weight.GetByteOffset() atIndex:1];
        [encoder setBuffer:rhs1_buffer offset:weights.up_proj_weight.GetByteOffset() atIndex:2];
        [encoder setBuffer:output0_buffer offset:gate_tensor.GetByteOffset() atIndex:3];
        [encoder setBuffer:output1_buffer offset:up_tensor.GetByteOffset() atIndex:4];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:5];

        const MTLSize threadgroup_size = MTLSizeMake(32, 1, 1);
        const MTLSize threadgroups_per_grid = MTLSizeMake((intermediate_size + 31) / 32, 1, 1);
        [encoder dispatchThreadgroups:threadgroups_per_grid threadsPerThreadgroup:threadgroup_size];

        if (stream != nullptr) {
            stream->EndEncoder();
        } else {
            [encoder endEncoding];
            if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                               "Fused gate/up command buffer failed",
                                               "GateUpDecode",
                                               1,
                                               error_message)) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace

bool QwenMLP::Run(const MetalContext& context,
                  PipelineCache* pipeline_cache,
                  const DeviceTensor& input,
                  const DeviceTensor* residual,
                  const QwenMlpWeights& weights,
                  const DeviceTensor& output,
                  const QwenMlpParams& params,
                  BufferArena* temporary_arena,
                  const QwenMlpScratch* scratch,
                  CommandStream* stream,
                  std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }

    std::size_t row_count = 0;
    std::size_t hidden_size = 0;
    std::size_t inferred_intermediate_size = 0;
    if (!ValidateMlpIO(input, output, weights, &row_count, &hidden_size, &inferred_intermediate_size, error_message)) {
        return false;
    }
    if (params.intermediate_size != 0 && params.intermediate_size != inferred_intermediate_size) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP intermediate size does not match weight shape";
        }
        return false;
    }
    if (params.add_residual && residual != nullptr) {
        if (!residual->IsValid() || residual->GetDesc().GetDataType() != DataType::kFloat32 ||
            residual->GetDesc().Rank() != 2 || residual->GetDesc().GetShape() != output.GetDesc().GetShape()) {
            if (error_message != nullptr) {
                *error_message = "QwenMLP residual must match output shape";
            }
            return false;
        }
    }

    BufferArenaMarkGuard arena_mark(temporary_arena, "QwenMLP");
    CommandStream local_stream;
    CommandStream* active_stream = stream;
    const bool use_local_decode_batch = row_count == 1 && stream == nullptr && UseExperimentalSafeDecodeBatch();

    DeviceTensor gate_tensor;
    DeviceTensor up_tensor;
    DeviceTensor fused_tensor;
    const TensorDesc intermediate_desc =
        TensorDesc::CreateContiguous(DataType::kFloat32, {row_count, inferred_intermediate_size});
    if (scratch != nullptr &&
        ScratchMatches(scratch->gate_tensor, intermediate_desc) &&
        ScratchMatches(scratch->up_tensor, intermediate_desc) &&
        ScratchMatches(scratch->fused_tensor, intermediate_desc)) {
        gate_tensor = scratch->gate_tensor;
        up_tensor = scratch->up_tensor;
        fused_tensor = scratch->fused_tensor;
    } else {
        if (!AllocateTemporaryTensor(temporary_arena, intermediate_desc, &gate_tensor, error_message) ||
            !AllocateTemporaryTensor(temporary_arena, intermediate_desc, &up_tensor, error_message) ||
            !AllocateTemporaryTensor(temporary_arena, intermediate_desc, &fused_tensor, error_message)) {
            return false;
        }
    }

    if (use_local_decode_batch) {
        if (!local_stream.Begin(context, error_message)) {
            return false;
        }
        active_stream = &local_stream;
    }

    if (row_count == 1 && UseExperimentalFusedGateUp()) {
        if (!RunFusedDecodeGateUp(context,
                                  pipeline_cache,
                                  input,
                                  weights,
                                  gate_tensor,
                                  up_tensor,
                                  hidden_size,
                                  inferred_intermediate_size,
                                  temporary_arena,
                                  active_stream,
                                  error_message)) {
            return false;
        }
    } else if (row_count == 1) {
        if (!RunDecodeProjection(context,
                                 pipeline_cache,
                                 input,
                                 nullptr,
                                 weights.gate_proj_weight,
                                 weights.gate_proj_q4_weight,
                                 "GateProjDecode",
                                 "GateProjDecodeQ4",
                                 hidden_size,
                                 inferred_intermediate_size,
                                 true,
                                 false,
                                 gate_tensor,
                                 temporary_arena,
                                 active_stream,
                                 error_message) ||
            !RunDecodeProjection(context,
                                 pipeline_cache,
                                 input,
                                 nullptr,
                                 weights.up_proj_weight,
                                 weights.up_proj_q4_weight,
                                 "UpProjDecode",
                                 "UpProjDecodeQ4",
                                 hidden_size,
                                 inferred_intermediate_size,
                                 false,
                                 false,
                                 up_tensor,
                                 temporary_arena,
                                 active_stream,
                                 error_message)) {
            return false;
        }
    } else {
        LinearParams gate_params;
        gate_params.activation = LinearActivation::kSiLU;
        gate_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
        gate_params.matmul.inner_dim = static_cast<std::uint32_t>(hidden_size);
        gate_params.matmul.column_count = static_cast<std::uint32_t>(inferred_intermediate_size);
        gate_params.matmul.profile_label = row_count == 1 ? "GateProjDecode" : "GateProjPrefill";
        gate_params.matmul.decode_mode = row_count == 1;
        gate_params.matmul.transpose_rhs = true;
        if (!LinearOp::Run(context,
                           pipeline_cache,
                           input,
                           weights.gate_proj_weight,
                           nullptr,
                           nullptr,
                           gate_tensor,
                           gate_params,
                           temporary_arena,
                           active_stream,
                           error_message)) {
            return false;
        }

        LinearParams up_params;
        up_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
        up_params.matmul.inner_dim = static_cast<std::uint32_t>(hidden_size);
        up_params.matmul.column_count = static_cast<std::uint32_t>(inferred_intermediate_size);
        up_params.matmul.profile_label = row_count == 1 ? "UpProjDecode" : "UpProjPrefill";
        up_params.matmul.decode_mode = row_count == 1;
        up_params.matmul.transpose_rhs = true;
        if (!LinearOp::Run(context,
                           pipeline_cache,
                           input,
                           weights.up_proj_weight,
                           nullptr,
                           nullptr,
                           up_tensor,
                           up_params,
                           temporary_arena,
                           active_stream,
                           error_message)) {
            return false;
        }
    }

    ElementwiseMulParams mul_params;
    mul_params.row_count = static_cast<std::uint32_t>(row_count);
    mul_params.row_size = static_cast<std::uint32_t>(inferred_intermediate_size);
    if (!ElementwiseMulOp::Run(context,
                               pipeline_cache,
                               gate_tensor,
                               up_tensor,
                               fused_tensor,
                               mul_params,
                               temporary_arena,
                               active_stream,
                               error_message)) {
        return false;
    }

    if (row_count == 1) {
        if (!RunDecodeProjection(context,
                                 pipeline_cache,
                                 fused_tensor,
                                 params.add_residual ? (residual == nullptr ? &input : residual) : nullptr,
                                 weights.down_proj_weight,
                                 weights.down_proj_q4_weight,
                                 "DownProjDecode",
                                 "DownProjDecodeQ4",
                                 inferred_intermediate_size,
                                 hidden_size,
                                 false,
                                 params.add_residual,
                                 output,
                                 temporary_arena,
                                 active_stream,
                                 error_message)) {
            return false;
        }
    } else {
        LinearParams down_params;
        down_params.add_residual = params.add_residual;
        down_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
        down_params.matmul.inner_dim = static_cast<std::uint32_t>(inferred_intermediate_size);
        down_params.matmul.column_count = static_cast<std::uint32_t>(hidden_size);
        down_params.matmul.profile_label = "DownProjPrefill";
        down_params.matmul.decode_mode = false;
        down_params.matmul.transpose_rhs = true;
        if (!LinearOp::Run(context,
                           pipeline_cache,
                           fused_tensor,
                           weights.down_proj_weight,
                           nullptr,
                           params.add_residual ? (residual == nullptr ? &input : residual) : nullptr,
                           output,
                           down_params,
                           temporary_arena,
                           active_stream,
                           error_message)) {
            return false;
        }
    }

    if (use_local_decode_batch) {
        if (!local_stream.Flush(context, "DecodeMlpBatch", error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu
