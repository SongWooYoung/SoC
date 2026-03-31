#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/sampler_topk_op.h"

#include "buffer/buffer_arena.h"
#include "metal/command_stream.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

struct MetalSamplerTopKParams {
    std::uint32_t row_count;
    std::uint32_t row_size;
    std::uint32_t top_k;
    std::uint32_t padding;
};

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalSamplerTopKParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalSamplerTopKParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalSamplerTopKParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params length:sizeof(MetalSamplerTopKParams) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate sampler top-k params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

}  // namespace

bool SamplerTopKOp::Run(const MetalContext& context,
                        PipelineCache* pipeline_cache,
                        const DeviceTensor& logits,
                        const DeviceTensor& top_values,
                        const DeviceTensor& top_indices,
                        const SamplerTopKParams& params,
                        BufferArena* temporary_arena,
                        CommandStream* stream,
                        std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }
    if (!logits.IsValid() || !top_values.IsValid() || !top_indices.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "SamplerTopKOp requires valid tensors";
        }
        return false;
    }
    if (logits.GetDesc().GetDataType() != DataType::kFloat32 || logits.GetDesc().Rank() != 2 ||
        top_values.GetDesc().GetDataType() != DataType::kFloat32 || top_values.GetDesc().Rank() != 2 ||
        top_indices.GetDesc().GetDataType() != DataType::kInt32 || top_indices.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "SamplerTopKOp expects logits=float32[rows,vocab], top_values=float32[rows,k], top_indices=int32[rows,k]";
        }
        return false;
    }

    const std::uint32_t row_count = params.row_count == 0 ? static_cast<std::uint32_t>(logits.GetDesc().GetShape()[0]) : params.row_count;
    const std::uint32_t row_size = params.row_size == 0 ? static_cast<std::uint32_t>(logits.GetDesc().GetShape()[1]) : params.row_size;
    const std::uint32_t top_k = params.top_k == 0 ? 1u : params.top_k;
    if (row_count != logits.GetDesc().GetShape()[0] || row_size != logits.GetDesc().GetShape()[1]) {
        if (error_message != nullptr) {
            *error_message = "SamplerTopKOp params do not match logits shape";
        }
        return false;
    }
    if (top_k > SamplerTopKParams::kMaxTopK || top_k > row_size) {
        if (error_message != nullptr) {
            *error_message = "SamplerTopKOp top_k exceeds supported range";
        }
        return false;
    }

    const std::vector<std::size_t> expected_output_shape{row_count, top_k};
    if (top_values.GetDesc().GetShape() != expected_output_shape || top_indices.GetDesc().GetShape() != expected_output_shape) {
        if (error_message != nullptr) {
            *error_message = "SamplerTopKOp output shapes must be [row_count, top_k]";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kSamplerTopK;
    key.function_name = "sampler_topk_f32_rowwise";
    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        BufferArenaMarkGuard arena_mark(stream != nullptr ? nullptr : temporary_arena, "SamplerTopKOp");
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> logits_buffer = (__bridge id<MTLBuffer>)logits.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> top_values_buffer = (__bridge id<MTLBuffer>)top_values.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> top_indices_buffer = (__bridge id<MTLBuffer>)top_indices.GetBuffer()->GetNativeHandle();

        const MetalSamplerTopKParams metal_params{row_count, row_size, top_k, 0u};
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
            if (command_buffer == nil) {
                if (error_message != nullptr) {
                    *error_message = "Failed to create sampler top-k command objects";
                }
                return false;
            }
            encoder = [command_buffer computeCommandEncoder];
        }
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create sampler top-k command objects";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:logits_buffer offset:logits.GetByteOffset() atIndex:0];
        [encoder setBuffer:top_values_buffer offset:top_values.GetByteOffset() atIndex:1];
        [encoder setBuffer:top_indices_buffer offset:top_indices.GetByteOffset() atIndex:2];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(row_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

        if (stream != nullptr) {
            stream->EndEncoder();
        } else {
            [encoder endEncoding];
            if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                               "Sampler top-k command buffer failed",
                                               error_message)) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace soc::gpu