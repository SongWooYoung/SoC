#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/matmul_op.h"

#include <algorithm>

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"

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

struct ScopedArenaMark {
    explicit ScopedArenaMark(BufferArena* arena) : guard(arena) {}
    BufferArenaMarkGuard guard;
};

bool ValidateMatMulTensors(const DeviceTensor& lhs,
                           const DeviceTensor& rhs,
                           const DeviceTensor* bias,
                           const DeviceTensor* residual,
                           const DeviceTensor& output,
                           const MatMulParams* params,
                           std::uint32_t* row_count,
                           std::uint32_t* inner_dim,
                           std::uint32_t* column_count,
                           std::string* error_message) {
    if (!lhs.IsValid() || !rhs.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "MatMul tensors must be valid device tensors";
        }
        return false;
    }
    if (lhs.GetDesc().GetDataType() != DataType::kFloat32 ||
        rhs.GetDesc().GetDataType() != DataType::kFloat32 ||
        output.GetDesc().GetDataType() != DataType::kFloat32 ||
        (bias != nullptr && bias->GetDesc().GetDataType() != DataType::kFloat32)) {
        if (error_message != nullptr) {
            *error_message = "MatMul baseline currently supports float32 tensors only";
        }
        return false;
    }
    if (lhs.GetDesc().Rank() != 2 || rhs.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "MatMul baseline expects rank-2 lhs, rhs, and output tensors";
        }
        return false;
    }

    const auto& lhs_shape = lhs.GetDesc().GetShape();
    const auto& rhs_shape = rhs.GetDesc().GetShape();
    const auto& output_shape = output.GetDesc().GetShape();
    const std::size_t rhs_inner_dim = params != nullptr && params->transpose_rhs ? rhs_shape[1] : rhs_shape[0];
    const std::size_t rhs_column_count = params != nullptr && params->transpose_rhs ? rhs_shape[0] : rhs_shape[1];
    if (lhs_shape[1] != rhs_inner_dim) {
        if (error_message != nullptr) {
            *error_message = "MatMul inner dimensions must match";
        }
        return false;
    }
    if (output_shape[0] != lhs_shape[0] || output_shape[1] != rhs_column_count) {
        if (error_message != nullptr) {
            *error_message = "MatMul output shape must match lhs_rows x rhs_cols";
        }
        return false;
    }
    if (bias != nullptr) {
        if (!bias->IsValid() || bias->GetDesc().Rank() != 1 || bias->GetDesc().GetShape()[0] != rhs_column_count) {
            if (error_message != nullptr) {
                *error_message = "MatMul bias must be a valid rank-1 tensor matching output columns";
            }
            return false;
        }
    }
    if (residual != nullptr) {
        if (!residual->IsValid() || residual->GetDesc().GetDataType() != DataType::kFloat32 ||
            residual->GetDesc().Rank() != 2 || residual->GetDesc().GetShape() != output_shape) {
            if (error_message != nullptr) {
                *error_message = "MatMul residual must be a valid rank-2 float32 tensor matching the output shape";
            }
            return false;
        }
    }

    *row_count = static_cast<std::uint32_t>(lhs_shape[0]);
    *inner_dim = static_cast<std::uint32_t>(lhs_shape[1]);
    *column_count = static_cast<std::uint32_t>(rhs_column_count);
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
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params
                                               length:sizeof(MetalMatMulParams)
                                              options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate MatMul params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

}  // namespace

MatMulExecutionPolicy MatMulOp::SelectPolicy(const MatMulParams& params,
                                             std::uint32_t row_count,
                                             std::uint32_t column_count,
                                             std::uint32_t inner_dim) {
    MatMulExecutionPolicy policy;
    if (params.decode_mode) {
        if (column_count <= 4) {
            policy.threadgroup_width = std::max<std::uint32_t>(1, column_count);
        } else if (inner_dim >= 128 || column_count >= 16) {
            policy.threadgroup_width = std::min<std::uint32_t>(16, std::max<std::uint32_t>(1, column_count));
        } else {
            policy.threadgroup_width = std::min<std::uint32_t>(8, std::max<std::uint32_t>(1, column_count));
        }
        policy.threadgroup_height = 1;
        policy.tile_columns = policy.threadgroup_width;
        policy.tile_rows = 1;
        if (params.preferred_threadgroup_width != 0) {
            policy.threadgroup_width = std::min<std::uint32_t>(params.preferred_threadgroup_width,
                                                               std::max<std::uint32_t>(1, column_count));
            policy.tile_columns = policy.threadgroup_width;
        }
        if (params.preferred_tile_columns != 0) {
            policy.tile_columns = params.preferred_tile_columns;
        }
        policy.use_tiled_kernel = inner_dim >= 64 && column_count >= 64 && policy.tile_columns <= 32;
        return policy;
    }

    if (column_count >= 16 || inner_dim >= 128) {
        const std::uint32_t preferred_width = row_count <= 8 ? 8 : 16;
        policy.threadgroup_width = std::min<std::uint32_t>(preferred_width, std::max<std::uint32_t>(1, column_count));
        policy.threadgroup_height = std::min<std::uint32_t>(4, std::max<std::uint32_t>(1, row_count));
    } else if (column_count >= 8 || inner_dim >= 64) {
        policy.threadgroup_width = std::min<std::uint32_t>(8, std::max<std::uint32_t>(1, column_count));
        policy.threadgroup_height = std::min<std::uint32_t>(4, std::max<std::uint32_t>(1, row_count));
    } else {
        policy.threadgroup_width = std::min<std::uint32_t>(4, std::max<std::uint32_t>(1, column_count));
        policy.threadgroup_height = std::min<std::uint32_t>(2, std::max<std::uint32_t>(1, row_count));
    }

    if (params.preferred_threadgroup_width != 0) {
        policy.threadgroup_width = std::min<std::uint32_t>(params.preferred_threadgroup_width,
                                                           std::max<std::uint32_t>(1, column_count));
    }
    if (params.preferred_threadgroup_height != 0) {
        policy.threadgroup_height = std::min<std::uint32_t>(params.preferred_threadgroup_height,
                                                            std::max<std::uint32_t>(1, row_count));
    }
    policy.tile_columns = policy.threadgroup_width;
    policy.tile_rows = policy.threadgroup_height;
    if (params.preferred_tile_columns != 0) {
        policy.tile_columns = params.preferred_tile_columns;
    }
    if (params.preferred_tile_rows != 0) {
        policy.tile_rows = params.preferred_tile_rows;
    }
    policy.use_tiled_kernel = inner_dim >= 64 && column_count >= 64 && row_count >= 2 && policy.tile_columns <= 32 &&
                              policy.tile_rows <= 4;
    return policy;
}

bool MatMulOp::Run(const MetalContext& context,
                   PipelineCache* pipeline_cache,
                   const DeviceTensor& lhs,
                   const DeviceTensor& rhs,
                   const DeviceTensor* bias,
                   const DeviceTensor* residual,
                   const DeviceTensor& output,
                   const MatMulParams& params,
                   BufferArena* temporary_arena,
                   std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }

    std::uint32_t inferred_rows = 0;
    std::uint32_t inferred_inner_dim = 0;
    std::uint32_t inferred_columns = 0;
    if (!ValidateMatMulTensors(lhs,
                               rhs,
                               bias,
                               residual,
                               output,
                               &params,
                               &inferred_rows,
                               &inferred_inner_dim,
                               &inferred_columns,
                               error_message)) {
        return false;
    }

    const std::uint32_t row_count = params.row_count == 0 ? inferred_rows : params.row_count;
    const std::uint32_t inner_dim = params.inner_dim == 0 ? inferred_inner_dim : params.inner_dim;
    const std::uint32_t column_count = params.column_count == 0 ? inferred_columns : params.column_count;
    if (row_count != inferred_rows || inner_dim != inferred_inner_dim || column_count != inferred_columns) {
        if (error_message != nullptr) {
            *error_message = "Explicit MatMul params must match tensor shapes";
        }
        return false;
    }
    if (params.use_bias != (bias != nullptr)) {
        if (error_message != nullptr) {
            *error_message = "MatMul params.use_bias must match whether a bias tensor is provided";
        }
        return false;
    }
    if (params.add_residual != (residual != nullptr)) {
        if (error_message != nullptr) {
            *error_message = "MatMul params.add_residual must match whether a residual tensor is provided";
        }
        return false;
    }
    if (params.decode_mode && row_count != 1) {
        if (error_message != nullptr) {
            *error_message = "Decode-specialized MatMul currently requires row_count == 1";
        }
        return false;
    }

    const MatMulExecutionPolicy policy = SelectPolicy(params, row_count, column_count, inner_dim);
    if (policy.threadgroup_width == 0 || policy.threadgroup_height == 0 || policy.tile_columns == 0 ||
        policy.tile_rows == 0) {
        if (error_message != nullptr) {
            *error_message = "MatMul policy dimensions must be non-zero";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kMatMul;
    if (policy.use_tiled_kernel) {
        key.function_name = params.decode_mode ? "matmul_f32_decode_tiled" : "matmul_f32_tiled";
    } else {
        key.function_name = "matmul_f32_basic";
    }
    key.threadgroup_width = policy.threadgroup_width;
    key.threadgroup_height = policy.threadgroup_height;
    key.specialization.use_bias = params.use_bias;
    key.specialization.decode_mode = params.decode_mode;
    key.specialization.transpose_rhs = params.transpose_rhs;
    key.specialization.enable_silu = params.enable_silu;
    key.specialization.enable_residual = params.add_residual;
    key.specialization.tile_columns = policy.tile_columns;
    key.specialization.tile_rows = policy.tile_rows;

    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        BufferArenaMarkGuard arena_mark(temporary_arena, "MatMulOp");
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> lhs_buffer = (__bridge id<MTLBuffer>)lhs.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> rhs_buffer = (__bridge id<MTLBuffer>)rhs.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> bias_buffer = bias == nullptr
                                        ? output_buffer
                                        : (__bridge id<MTLBuffer>)bias->GetBuffer()->GetNativeHandle();
        id<MTLBuffer> residual_buffer = residual == nullptr
                            ? output_buffer
                            : (__bridge id<MTLBuffer>)residual->GetBuffer()->GetNativeHandle();

        MetalMatMulParams metal_params = {row_count,
                                          inner_dim,
                                          column_count,
                                          static_cast<std::uint32_t>(lhs.GetDesc().GetShape()[1]),
                                          static_cast<std::uint32_t>(rhs.GetDesc().GetShape()[1]),
                                          static_cast<std::uint32_t>(output.GetDesc().GetShape()[1])};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create MatMul command buffer";
            }
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create MatMul compute encoder";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs_buffer offset:lhs.GetByteOffset() atIndex:0];
        [encoder setBuffer:rhs_buffer offset:rhs.GetByteOffset() atIndex:1];
        [encoder setBuffer:bias_buffer offset:bias == nullptr ? 0 : bias->GetByteOffset() atIndex:2];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:3];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:4];
        [encoder setBuffer:residual_buffer offset:residual == nullptr ? 0 : residual->GetByteOffset() atIndex:5];

        const NSUInteger threadgroup_width = std::min<NSUInteger>(key.threadgroup_width, column_count);
        const NSUInteger threadgroup_height = std::min<NSUInteger>(key.threadgroup_height, row_count);
        const MTLSize threadgroup_size =
            MTLSizeMake(threadgroup_width == 0 ? 1 : threadgroup_width, threadgroup_height == 0 ? 1 : threadgroup_height, 1);
        if (policy.use_tiled_kernel) {
            const MTLSize threadgroups_per_grid =
                MTLSizeMake((column_count + policy.tile_columns - 1) / policy.tile_columns,
                            (row_count + policy.tile_rows - 1) / policy.tile_rows,
                            1);
            [encoder dispatchThreadgroups:threadgroups_per_grid threadsPerThreadgroup:threadgroup_size];
        } else {
            const MTLSize grid_size = MTLSizeMake(column_count, row_count, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }
        [encoder endEncoding];

        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "MatMul command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu