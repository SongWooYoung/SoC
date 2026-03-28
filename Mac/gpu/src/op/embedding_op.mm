#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/embedding_op.h"

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

struct MetalEmbeddingParams {
    std::uint32_t token_count;
    std::uint32_t hidden_size;
    std::uint32_t vocab_size;
    std::uint32_t padding;
};

bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const MetalEmbeddingParams& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(MetalEmbeddingParams), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(MetalEmbeddingParams), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params length:sizeof(MetalEmbeddingParams) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate embedding params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

}  // namespace

bool EmbeddingOp::Run(const MetalContext& context,
                      PipelineCache* pipeline_cache,
                      const DeviceTensor& token_ids,
                      const DeviceTensor& embedding_table,
                      const DeviceTensor& output,
                      const EmbeddingParams& params,
                      BufferArena* temporary_arena,
                      std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }
    if (!token_ids.IsValid() || !embedding_table.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "Embedding tensors must be valid";
        }
        return false;
    }
    if (token_ids.GetDesc().GetDataType() != DataType::kInt32 || output.GetDesc().GetDataType() != DataType::kFloat32) {
        if (error_message != nullptr) {
            *error_message = "Embedding expects int32 token ids and float32 output";
        }
        return false;
    }
    if (embedding_table.GetDesc().GetDataType() != DataType::kFloat32 && embedding_table.GetDesc().GetDataType() != DataType::kFloat16) {
        if (error_message != nullptr) {
            *error_message = "Embedding table must be float16 or float32";
        }
        return false;
    }
    if (token_ids.GetDesc().Rank() != 1 || embedding_table.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "Embedding expects rank-1 ids, rank-2 table, and rank-2 output";
        }
        return false;
    }

    const std::uint32_t token_count = params.token_count == 0 ? static_cast<std::uint32_t>(token_ids.GetDesc().GetShape()[0]) : params.token_count;
    const std::uint32_t vocab_size = params.vocab_size == 0 ? static_cast<std::uint32_t>(embedding_table.GetDesc().GetShape()[0]) : params.vocab_size;
    const std::uint32_t hidden_size = params.hidden_size == 0 ? static_cast<std::uint32_t>(embedding_table.GetDesc().GetShape()[1]) : params.hidden_size;
    if (output.GetDesc().GetShape()[0] != token_count || output.GetDesc().GetShape()[1] != hidden_size ||
        token_ids.GetDesc().GetShape()[0] != token_count || embedding_table.GetDesc().GetShape()[0] != vocab_size ||
        embedding_table.GetDesc().GetShape()[1] != hidden_size) {
        if (error_message != nullptr) {
            *error_message = "Embedding tensor shapes do not match params";
        }
        return false;
    }

    KernelKey key;
    key.kind = KernelKind::kEmbedding;
    key.function_name = embedding_table.GetDesc().GetDataType() == DataType::kFloat16 ? "embedding_f16_lookup" : "embedding_f32_lookup";
    key.threadgroup_width = std::min<std::uint32_t>(32, hidden_size);
    key.threadgroup_height = 1;

    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        BufferArenaMarkGuard arena_mark(temporary_arena, "EmbeddingOp");
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> id_buffer = (__bridge id<MTLBuffer>)token_ids.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> table_buffer = (__bridge id<MTLBuffer>)embedding_table.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();

        const MetalEmbeddingParams metal_params{token_count, hidden_size, vocab_size, 0};
        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, metal_params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create embedding command objects";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:id_buffer offset:token_ids.GetByteOffset() atIndex:0];
        [encoder setBuffer:table_buffer offset:embedding_table.GetByteOffset() atIndex:1];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:2];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:3];

        [encoder dispatchThreads:MTLSizeMake(hidden_size, token_count, 1)
                 threadsPerThreadgroup:MTLSizeMake(key.threadgroup_width, 1, 1)];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "Embedding command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu