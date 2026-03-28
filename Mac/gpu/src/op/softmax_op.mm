#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/softmax_op.h"

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

struct MetalSoftmaxParams {
    std::uint32_t row_count;
    std::uint32_t row_size;
};

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalSoftmaxParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalSoftmaxParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalSoftmaxParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params length:sizeof(MetalSoftmaxParams) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate softmax params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

}  // namespace

bool SoftmaxOp::Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const DeviceTensor& output,
                    const SoftmaxParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }
    if (!input.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "Softmax tensors must be valid";
        }
        return false;
    }
    if (input.GetDesc().GetDataType() != DataType::kFloat32 || output.GetDesc().GetDataType() != DataType::kFloat32 ||
        input.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2 || input.GetDesc().GetShape() != output.GetDesc().GetShape()) {
        if (error_message != nullptr) {
            *error_message = "Softmax expects matching rank-2 float32 tensors";
        }
        return false;
    }

    const std::uint32_t row_count = params.row_count == 0 ? static_cast<std::uint32_t>(input.GetDesc().GetShape()[0]) : params.row_count;
    const std::uint32_t row_size = params.row_size == 0 ? static_cast<std::uint32_t>(input.GetDesc().GetShape()[1]) : params.row_size;
    if (row_count != input.GetDesc().GetShape()[0] || row_size != input.GetDesc().GetShape()[1]) {
        if (error_message != nullptr) {
            *error_message = "Softmax params do not match tensor shape";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kSoftmax;
    key.function_name = "softmax_f32_rowwise";
    key.threadgroup_width = 1;
    key.threadgroup_height = 1;
    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        BufferArenaMarkGuard arena_mark(temporary_arena, "SoftmaxOp");
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();

        const MetalSoftmaxParams metal_params{row_count, row_size};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create softmax command objects";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:input.GetByteOffset() atIndex:0];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:1];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(row_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "Softmax command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu