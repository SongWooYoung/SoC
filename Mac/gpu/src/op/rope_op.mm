#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/rope_op.h"

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

struct MetalRopeParams {
    std::uint32_t row_count;
    std::uint32_t head_count;
    std::uint32_t head_dim;
    std::uint32_t rotary_dim;
    std::uint32_t position_offset;
    float rope_theta;
};

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalRopeParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalRopeParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalRopeParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params length:sizeof(MetalRopeParams) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate RoPE params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

}  // namespace

bool RopeOp::Run(const MetalContext& context,
                 PipelineCache* pipeline_cache,
                 const DeviceTensor& input,
                 const DeviceTensor& output,
                 const RopeParams& params,
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
            *error_message = "RoPE tensors must be valid";
        }
        return false;
    }
    if (input.GetDesc().GetDataType() != DataType::kFloat32 || output.GetDesc().GetDataType() != DataType::kFloat32) {
        if (error_message != nullptr) {
            *error_message = "RoPE baseline currently supports float32 only";
        }
        return false;
    }
    if (input.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2 || input.GetDesc().GetShape() != output.GetDesc().GetShape()) {
        if (error_message != nullptr) {
            *error_message = "RoPE expects matching rank-2 input/output tensors";
        }
        return false;
    }

    const std::uint32_t row_count = params.row_count == 0 ? static_cast<std::uint32_t>(input.GetDesc().GetShape()[0]) : params.row_count;
    const std::uint32_t hidden_size = static_cast<std::uint32_t>(input.GetDesc().GetShape()[1]);
    const std::uint32_t head_count = params.head_count;
    const std::uint32_t head_dim = params.head_dim;
    const std::uint32_t rotary_dim = params.rotary_dim == 0 ? head_dim : params.rotary_dim;
    if (row_count != input.GetDesc().GetShape()[0] || head_count == 0 || head_dim == 0 || head_count * head_dim != hidden_size ||
        rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        if (error_message != nullptr) {
            *error_message = "RoPE params do not match tensor shapes";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kRoPE;
    key.function_name = "rope_f32_qwen";
    key.threadgroup_width = std::min<std::uint32_t>(64, head_count * (rotary_dim / 2));
    key.threadgroup_height = 1;
    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        BufferArenaMarkGuard arena_mark(temporary_arena, "RopeOp");
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();

        const MetalRopeParams metal_params{row_count, head_count, head_dim, rotary_dim, params.position_offset, params.rope_theta};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create RoPE command objects";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:input.GetByteOffset() atIndex:0];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:1];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(head_count * (rotary_dim / 2), row_count, 1)
                 threadsPerThreadgroup:MTLSizeMake(key.threadgroup_width, 1, 1)];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "RoPE command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu