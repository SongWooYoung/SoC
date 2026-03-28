#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/rms_norm_op.h"

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"

namespace soc::gpu {

namespace {

struct MetalRmsNormParams {
    std::uint32_t row_size;
    std::uint32_t row_count;
    float epsilon;
    std::uint32_t padding;
};

bool ValidateRmsNormTensors(const DeviceTensor& input,
                            const DeviceTensor& weight,
                            const DeviceTensor& output,
                            std::uint32_t* row_size,
                            std::uint32_t* row_count,
                            std::string* error_message) {
    if (!input.IsValid() || !weight.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "RMSNorm tensors must be valid device tensors";
        }
        return false;
    }

    if (input.GetDesc().GetDataType() != DataType::kFloat32 ||
        weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        output.GetDesc().GetDataType() != DataType::kFloat32) {
        if (error_message != nullptr) {
            *error_message = "RMSNorm baseline currently supports float32 tensors only";
        }
        return false;
    }

    if (input.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2 || weight.GetDesc().Rank() != 1) {
        if (error_message != nullptr) {
            *error_message = "RMSNorm baseline expects input/output rank 2 and weight rank 1";
        }
        return false;
    }

    const auto& input_shape = input.GetDesc().GetShape();
    const auto& output_shape = output.GetDesc().GetShape();
    const auto& weight_shape = weight.GetDesc().GetShape();
    if (input_shape != output_shape) {
        if (error_message != nullptr) {
            *error_message = "RMSNorm input and output shapes must match";
        }
        return false;
    }
    if (weight_shape[0] != input_shape[1]) {
        if (error_message != nullptr) {
            *error_message = "RMSNorm weight size must match hidden size";
        }
        return false;
    }

    *row_count = static_cast<std::uint32_t>(input_shape[0]);
    *row_size = static_cast<std::uint32_t>(input_shape[1]);
    return true;
}

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalRmsNormParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalRmsNormParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalRmsNormParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> params_buffer_local = [device newBufferWithBytes:&params
                                                            length:sizeof(MetalRmsNormParams)
                                                           options:MTLResourceStorageModeShared];
    if (params_buffer_local == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate RMSNorm params buffer";
        }
        return false;
    }
    *params_buffer = params_buffer_local;
    *params_offset = 0;
    return true;
}

}  // namespace

bool RmsNormOp::Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const DeviceTensor& weight,
                    const DeviceTensor& output,
                    const RmsNormParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }

    std::uint32_t inferred_row_size = 0;
    std::uint32_t inferred_row_count = 0;
    if (!ValidateRmsNormTensors(input, weight, output, &inferred_row_size, &inferred_row_count, error_message)) {
        return false;
    }

    const std::uint32_t row_size = params.row_size == 0 ? inferred_row_size : params.row_size;
    const std::uint32_t row_count = params.row_count == 0 ? inferred_row_count : params.row_count;
    if (row_size != inferred_row_size || row_count != inferred_row_count) {
        if (error_message != nullptr) {
            *error_message = "Explicit RMSNorm params must match tensor shapes";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kRmsNorm;
    key.function_name = "rms_norm_f32_rowwise";
    key.threadgroup_width = 1;
    key.threadgroup_height = 1;

    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        BufferArenaMarkGuard arena_mark(temporary_arena, "RmsNormOp");
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> weight_buffer = (__bridge id<MTLBuffer>)weight.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();

        MetalRmsNormParams metal_params = {row_size, row_count, params.epsilon, 0};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create RMSNorm command buffer";
            }
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create RMSNorm compute encoder";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:input.GetByteOffset() atIndex:0];
        [encoder setBuffer:weight_buffer offset:weight.GetByteOffset() atIndex:1];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:2];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:3];

        const MTLSize grid_size = MTLSizeMake(row_count, 1, 1);
        const MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "RMSNorm command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu