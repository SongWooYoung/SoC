#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/elementwise_mul_op.h"

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

struct MetalElementwiseMulParams {
    std::uint32_t row_count;
    std::uint32_t row_size;
};

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalElementwiseMulParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalElementwiseMulParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalElementwiseMulParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params
                                               length:sizeof(MetalElementwiseMulParams)
                                              options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate elementwise mul params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

}  // namespace

bool ElementwiseMulOp::Run(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           const DeviceTensor& lhs,
                           const DeviceTensor& rhs,
                           const DeviceTensor& output,
                           const ElementwiseMulParams& params,
                           BufferArena* temporary_arena,
                           std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }
    if (!lhs.IsValid() || !rhs.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "ElementwiseMul tensors must be valid";
        }
        return false;
    }
    if (lhs.GetDesc().GetDataType() != DataType::kFloat32 ||
        rhs.GetDesc().GetDataType() != DataType::kFloat32 ||
        output.GetDesc().GetDataType() != DataType::kFloat32 ||
        lhs.GetDesc().Rank() != 2 ||
        rhs.GetDesc().Rank() != 2 ||
        output.GetDesc().Rank() != 2 ||
        lhs.GetDesc().GetShape() != rhs.GetDesc().GetShape() ||
        lhs.GetDesc().GetShape() != output.GetDesc().GetShape()) {
        if (error_message != nullptr) {
            *error_message = "ElementwiseMul expects matching rank-2 float32 tensors";
        }
        return false;
    }

    const std::uint32_t row_count =
        params.row_count == 0 ? static_cast<std::uint32_t>(lhs.GetDesc().GetShape()[0]) : params.row_count;
    const std::uint32_t row_size =
        params.row_size == 0 ? static_cast<std::uint32_t>(lhs.GetDesc().GetShape()[1]) : params.row_size;
    if (row_count != lhs.GetDesc().GetShape()[0] || row_size != lhs.GetDesc().GetShape()[1]) {
        if (error_message != nullptr) {
            *error_message = "ElementwiseMul params do not match tensor shape";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kElementwiseMul;
    key.function_name = "elementwise_mul_f32";
    key.threadgroup_width = 8;
    key.threadgroup_height = 8;

    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        BufferArenaMarkGuard arena_mark(temporary_arena, "ElementwiseMulOp");
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> lhs_buffer = (__bridge id<MTLBuffer>)lhs.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> rhs_buffer = (__bridge id<MTLBuffer>)rhs.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();

        const MetalElementwiseMulParams metal_params{row_count, row_size};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create elementwise mul command objects";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs_buffer offset:lhs.GetByteOffset() atIndex:0];
        [encoder setBuffer:rhs_buffer offset:rhs.GetByteOffset() atIndex:1];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:2];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(row_size, row_count, 1)
                 threadsPerThreadgroup:MTLSizeMake(key.threadgroup_width, key.threadgroup_height, 1)];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "ElementwiseMul command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu