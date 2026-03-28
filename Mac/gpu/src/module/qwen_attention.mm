#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "module/qwen_attention.h"

#include <cmath>

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"
#include "op/linear_op.h"
#include "op/rms_norm_op.h"
#include "op/rope_op.h"
#include "op/softmax_op.h"

namespace soc::gpu {
namespace {

struct MetalAttentionScoreParams {
    std::uint32_t query_row_count;
    std::uint32_t key_row_count;
    std::uint32_t query_head_count;
    std::uint32_t key_value_head_count;
    std::uint32_t head_dim;
    std::uint32_t head_group_size;
    std::uint32_t query_position_base;
    std::uint32_t causal_mask;
    float scale;
};

struct MetalAttentionValueParams {
    std::uint32_t query_row_count;
    std::uint32_t key_row_count;
    std::uint32_t query_head_count;
    std::uint32_t key_value_head_count;
    std::uint32_t head_dim;
    std::uint32_t head_group_size;
};

bool AllocateTemporaryTensor(BufferArena* arena,
                             const TensorDesc& desc,
                             DeviceTensor* tensor,
                             std::string* error_message) {
    if (arena == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Temporary arena is required for QwenAttention";
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

template <typename ParamsT>
bool AllocateParamsBuffer(const MetalContext& context,
                          BufferArena* temporary_arena,
                          const ParamsT& params,
                          id<MTLBuffer>* params_buffer,
                          std::size_t* params_offset,
                          std::string* error_message) {
    if (temporary_arena != nullptr) {
        BufferArenaSlice slice;
        if (!temporary_arena->Allocate(sizeof(ParamsT), 256, &slice, error_message)) {
            return false;
        }
        if (!slice.buffer->Write(&params, sizeof(ParamsT), slice.offset_bytes, error_message)) {
            return false;
        }
        *params_buffer = (__bridge id<MTLBuffer>)slice.buffer->GetNativeHandle();
        *params_offset = slice.offset_bytes;
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:&params length:sizeof(ParamsT) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "Failed to allocate QwenAttention params buffer";
        }
        return false;
    }
    *params_buffer = buffer;
    *params_offset = 0;
    return true;
}

bool ValidateAttentionIO(const DeviceTensor& input,
                         const DeviceTensor& output,
                         const QwenAttentionWeights& weights,
                         const QwenAttentionParams& params,
                         std::size_t* row_count,
                         std::size_t* hidden_size,
                         std::size_t* query_hidden_size,
                         std::size_t* key_value_hidden_size,
                         std::string* error_message) {
    if (!input.IsValid() || !output.IsValid() || !weights.q_proj_weight.IsValid() || !weights.k_proj_weight.IsValid() ||
        !weights.v_proj_weight.IsValid() || !weights.o_proj_weight.IsValid() || !weights.q_norm_weight.IsValid() ||
        !weights.k_norm_weight.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention tensors must be valid";
        }
        return false;
    }
    if (input.GetDesc().GetDataType() != DataType::kFloat32 || output.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.q_proj_weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.k_proj_weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.v_proj_weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.o_proj_weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.q_norm_weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.k_norm_weight.GetDesc().GetDataType() != DataType::kFloat32) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention baseline currently supports float32 tensors only";
        }
        return false;
    }
    if (input.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2 ||
        weights.q_proj_weight.GetDesc().Rank() != 2 || weights.k_proj_weight.GetDesc().Rank() != 2 ||
        weights.v_proj_weight.GetDesc().Rank() != 2 || weights.o_proj_weight.GetDesc().Rank() != 2 ||
        weights.q_norm_weight.GetDesc().Rank() != 1 || weights.k_norm_weight.GetDesc().Rank() != 1) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention expects rank-2 activations/weights and rank-1 norm weights";
        }
        return false;
    }
    if (params.num_attention_heads == 0 || params.num_key_value_heads == 0 || params.head_dim == 0 ||
        (params.num_attention_heads % params.num_key_value_heads) != 0) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention head configuration is invalid";
        }
        return false;
    }

    const auto& input_shape = input.GetDesc().GetShape();
    const auto& output_shape = output.GetDesc().GetShape();
    const auto& q_shape = weights.q_proj_weight.GetDesc().GetShape();
    const auto& k_shape = weights.k_proj_weight.GetDesc().GetShape();
    const auto& v_shape = weights.v_proj_weight.GetDesc().GetShape();
    const auto& o_shape = weights.o_proj_weight.GetDesc().GetShape();
    const auto& q_norm_shape = weights.q_norm_weight.GetDesc().GetShape();
    const auto& k_norm_shape = weights.k_norm_weight.GetDesc().GetShape();

    const std::size_t inferred_hidden_size = input_shape[1];
    const std::size_t inferred_query_hidden = params.num_attention_heads * params.head_dim;
    const std::size_t inferred_key_value_hidden = params.num_key_value_heads * params.head_dim;
    const std::size_t rotary_dim = params.rotary_dim == 0 ? params.head_dim : params.rotary_dim;

    if (output_shape != input_shape) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention output shape must match input shape";
        }
        return false;
    }
    if (q_shape[0] != inferred_query_hidden || q_shape[1] != inferred_hidden_size ||
        k_shape[0] != inferred_key_value_hidden || k_shape[1] != inferred_hidden_size ||
        v_shape[0] != inferred_key_value_hidden || v_shape[1] != inferred_hidden_size ||
        o_shape[0] != inferred_hidden_size || o_shape[1] != inferred_query_hidden) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention projection weight shapes do not match the configured head layout";
        }
        return false;
    }
    if (q_norm_shape[0] != params.head_dim || k_norm_shape[0] != params.head_dim || rotary_dim > params.head_dim ||
        (rotary_dim % 2) != 0) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention norm or rotary parameters do not match head_dim";
        }
        return false;
    }

    *row_count = input_shape[0];
    *hidden_size = inferred_hidden_size;
    *query_hidden_size = inferred_query_hidden;
    *key_value_hidden_size = inferred_key_value_hidden;
    return true;
}

bool DispatchAttentionScores(const MetalContext& context,
                             PipelineCache* pipeline_cache,
                             const DeviceTensor& queries,
                             const DeviceTensor& keys,
                             const DeviceTensor& scores,
                             const MetalAttentionScoreParams& params,
                             BufferArena* temporary_arena,
                             std::string* error_message) {
    KernelKey key;
    key.kind = KernelKind::kAttentionScores;
    key.function_name = "attention_scores_f32_qwen";
    key.threadgroup_width = 8;
    key.threadgroup_height = 8;
    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> query_buffer = (__bridge id<MTLBuffer>)queries.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> key_buffer = (__bridge id<MTLBuffer>)keys.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> score_buffer = (__bridge id<MTLBuffer>)scores.GetBuffer()->GetNativeHandle();

        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create attention score command objects";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:query_buffer offset:queries.GetByteOffset() atIndex:0];
        [encoder setBuffer:key_buffer offset:keys.GetByteOffset() atIndex:1];
        [encoder setBuffer:score_buffer offset:scores.GetByteOffset() atIndex:2];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(params.key_row_count, params.query_row_count * params.query_head_count, 1)
                 threadsPerThreadgroup:MTLSizeMake(key.threadgroup_width, key.threadgroup_height, 1)];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "Attention score command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

bool DispatchAttentionValues(const MetalContext& context,
                             PipelineCache* pipeline_cache,
                             const DeviceTensor& probabilities,
                             const DeviceTensor& values,
                             const DeviceTensor& output,
                             const MetalAttentionValueParams& params,
                             BufferArena* temporary_arena,
                             std::string* error_message) {
    KernelKey key;
    key.kind = KernelKind::kAttentionValues;
    key.function_name = "attention_values_f32_qwen";
    key.threadgroup_width = 8;
    key.threadgroup_height = 8;
    const void* pipeline_handle = pipeline_cache->GetOrCreatePipeline(key, error_message);
    if (pipeline_handle == nullptr) {
        return false;
    }

    @autoreleasepool {
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
        id<MTLBuffer> probability_buffer = (__bridge id<MTLBuffer>)probabilities.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> value_buffer = (__bridge id<MTLBuffer>)values.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.GetBuffer()->GetNativeHandle();

        id<MTLBuffer> params_buffer = nil;
        std::size_t params_offset = 0;
        if (!AllocateParamsBuffer(context, temporary_arena, params, &params_buffer, &params_offset, error_message)) {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create attention value command objects";
            }
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:probability_buffer offset:probabilities.GetByteOffset() atIndex:0];
        [encoder setBuffer:value_buffer offset:values.GetByteOffset() atIndex:1];
        [encoder setBuffer:output_buffer offset:output.GetByteOffset() atIndex:2];
        [encoder setBuffer:params_buffer offset:params_offset atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(params.head_dim, params.query_row_count * params.query_head_count, 1)
                 threadsPerThreadgroup:MTLSizeMake(key.threadgroup_width, key.threadgroup_height, 1)];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "Attention value command buffer failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace

bool RunAttentionInternal(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          const DeviceTensor& input,
                          const DeviceTensor* residual,
                          const QwenAttentionWeights& weights,
                          KVCache* kv_cache,
                          std::size_t layer_index,
                          bool decode_mode,
                          const DeviceTensor& output,
                          const QwenAttentionParams& params,
                          BufferArena* temporary_arena,
                          std::string* error_message);

bool QwenAttention::Run(const MetalContext& context,
                        PipelineCache* pipeline_cache,
                        const DeviceTensor& input,
                        const DeviceTensor* residual,
                        const QwenAttentionWeights& weights,
                        const DeviceTensor& output,
                        const QwenAttentionParams& params,
                        BufferArena* temporary_arena,
                        std::string* error_message) {
    return RunAttentionInternal(context,
                                pipeline_cache,
                                input,
                                residual,
                                weights,
                                nullptr,
                                0,
                                input.GetDesc().Rank() == 2 && input.GetDesc().GetShape()[0] == 1,
                                output,
                                params,
                                temporary_arena,
                                error_message);
}

bool QwenAttention::RunPrefill(const MetalContext& context,
                               PipelineCache* pipeline_cache,
                               const DeviceTensor& input,
                               const DeviceTensor* residual,
                               const QwenAttentionWeights& weights,
                               KVCache* kv_cache,
                               std::size_t layer_index,
                               const DeviceTensor& output,
                               const QwenAttentionParams& params,
                               BufferArena* temporary_arena,
                               std::string* error_message) {
    return RunAttentionInternal(context,
                                pipeline_cache,
                                input,
                                residual,
                                weights,
                                kv_cache,
                                layer_index,
                                false,
                                output,
                                params,
                                temporary_arena,
                                error_message);
}

bool QwenAttention::RunDecode(const MetalContext& context,
                              PipelineCache* pipeline_cache,
                              const DeviceTensor& input,
                              const DeviceTensor* residual,
                              const QwenAttentionWeights& weights,
                              KVCache* kv_cache,
                              std::size_t layer_index,
                              const DeviceTensor& output,
                              const QwenAttentionParams& params,
                              BufferArena* temporary_arena,
                              std::string* error_message) {
    return RunAttentionInternal(context,
                                pipeline_cache,
                                input,
                                residual,
                                weights,
                                kv_cache,
                                layer_index,
                                true,
                                output,
                                params,
                                temporary_arena,
                                error_message);
}

bool RunAttentionInternal(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          const DeviceTensor& input,
                          const DeviceTensor* residual,
                          const QwenAttentionWeights& weights,
                          KVCache* kv_cache,
                          std::size_t layer_index,
                          bool decode_mode,
                          const DeviceTensor& output,
                          const QwenAttentionParams& params,
                          BufferArena* temporary_arena,
                          std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }

    std::size_t row_count = 0;
    std::size_t hidden_size = 0;
    std::size_t query_hidden_size = 0;
    std::size_t key_value_hidden_size = 0;
    if (!ValidateAttentionIO(input,
                             output,
                             weights,
                             params,
                             &row_count,
                             &hidden_size,
                             &query_hidden_size,
                             &key_value_hidden_size,
                             error_message)) {
        return false;
    }
    if (params.add_residual && residual != nullptr) {
        if (!residual->IsValid() || residual->GetDesc().GetDataType() != DataType::kFloat32 ||
            residual->GetDesc().Rank() != 2 || residual->GetDesc().GetShape() != output.GetDesc().GetShape()) {
            if (error_message != nullptr) {
                *error_message = "QwenAttention residual must match output shape";
            }
            return false;
        }
    }
    if (decode_mode && row_count != 1) {
        if (error_message != nullptr) {
            *error_message = "QwenAttention decode path expects a single-token input";
        }
        return false;
    }
    if (kv_cache != nullptr) {
        if (layer_index >= kv_cache->GetLayerCount()) {
            if (error_message != nullptr) {
                *error_message = "QwenAttention KV cache layer index is out of range";
            }
            return false;
        }
        if (kv_cache->GetKeyValueHeadCount() != params.num_key_value_heads || kv_cache->GetHeadDim() != params.head_dim) {
            if (error_message != nullptr) {
                *error_message = "QwenAttention KV cache layout does not match attention parameters";
            }
            return false;
        }
    }

    BufferArenaMarkGuard arena_mark(temporary_arena, "QwenAttention");

    const std::size_t rotary_dim = params.rotary_dim == 0 ? params.head_dim : params.rotary_dim;
    const TensorDesc query_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {row_count, query_hidden_size});
    const TensorDesc key_value_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {row_count, key_value_hidden_size});
    DeviceTensor query_proj;
    DeviceTensor key_proj;
    DeviceTensor value_proj;
    DeviceTensor query_norm;
    DeviceTensor key_norm;
    DeviceTensor query_rope;
    DeviceTensor key_rope;
    DeviceTensor attention_scores;
    DeviceTensor attention_probs;
    DeviceTensor attention_context;
    if (!AllocateTemporaryTensor(temporary_arena, query_desc, &query_proj, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, key_value_desc, &key_proj, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, key_value_desc, &value_proj, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, query_desc, &query_norm, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, key_value_desc, &key_norm, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, query_desc, &query_rope, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, key_value_desc, &key_rope, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, query_desc, &attention_context, error_message)) {
        return false;
    }

    LinearParams q_params;
    q_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
    q_params.matmul.inner_dim = static_cast<std::uint32_t>(hidden_size);
    q_params.matmul.column_count = static_cast<std::uint32_t>(query_hidden_size);
    q_params.matmul.decode_mode = decode_mode;
    q_params.matmul.transpose_rhs = true;
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       input,
                       weights.q_proj_weight,
                       nullptr,
                       nullptr,
                       query_proj,
                       q_params,
                       temporary_arena,
                       error_message)) {
        return false;
    }

    LinearParams k_params = q_params;
    k_params.matmul.column_count = static_cast<std::uint32_t>(key_value_hidden_size);
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       input,
                       weights.k_proj_weight,
                       nullptr,
                       nullptr,
                       key_proj,
                       k_params,
                       temporary_arena,
                       error_message) ||
        !LinearOp::Run(context,
                       pipeline_cache,
                       input,
                       weights.v_proj_weight,
                       nullptr,
                       nullptr,
                       value_proj,
                       k_params,
                       temporary_arena,
                       error_message)) {
        return false;
    }

    const DeviceTensor query_heads(query_norm.GetBuffer(),
                                   query_norm.GetByteOffset(),
                                   TensorDesc::CreateContiguous(DataType::kFloat32,
                                                                {row_count * params.num_attention_heads, params.head_dim}));
    const DeviceTensor key_heads(key_norm.GetBuffer(),
                                 key_norm.GetByteOffset(),
                                 TensorDesc::CreateContiguous(DataType::kFloat32,
                                                              {row_count * params.num_key_value_heads, params.head_dim}));
    const DeviceTensor query_proj_heads(query_proj.GetBuffer(),
                                        query_proj.GetByteOffset(),
                                        TensorDesc::CreateContiguous(DataType::kFloat32,
                                                                     {row_count * params.num_attention_heads, params.head_dim}));
    const DeviceTensor key_proj_heads(key_proj.GetBuffer(),
                                      key_proj.GetByteOffset(),
                                      TensorDesc::CreateContiguous(DataType::kFloat32,
                                                                   {row_count * params.num_key_value_heads, params.head_dim}));

    RmsNormParams q_norm_params;
    q_norm_params.epsilon = params.rms_epsilon;
    q_norm_params.row_count = static_cast<std::uint32_t>(row_count * params.num_attention_heads);
    q_norm_params.row_size = static_cast<std::uint32_t>(params.head_dim);
    if (!RmsNormOp::Run(context,
                        pipeline_cache,
                        query_proj_heads,
                        weights.q_norm_weight,
                        query_heads,
                        q_norm_params,
                        temporary_arena,
                        error_message)) {
        return false;
    }

    RmsNormParams k_norm_params;
    k_norm_params.epsilon = params.rms_epsilon;
    k_norm_params.row_count = static_cast<std::uint32_t>(row_count * params.num_key_value_heads);
    k_norm_params.row_size = static_cast<std::uint32_t>(params.head_dim);
    if (!RmsNormOp::Run(context,
                        pipeline_cache,
                        key_proj_heads,
                        weights.k_norm_weight,
                        key_heads,
                        k_norm_params,
                        temporary_arena,
                        error_message)) {
        return false;
    }

    RopeParams q_rope_params;
    q_rope_params.row_count = static_cast<std::uint32_t>(row_count);
    q_rope_params.head_count = static_cast<std::uint32_t>(params.num_attention_heads);
    q_rope_params.head_dim = static_cast<std::uint32_t>(params.head_dim);
    q_rope_params.rotary_dim = static_cast<std::uint32_t>(rotary_dim);
    q_rope_params.position_offset = static_cast<std::uint32_t>(params.position_offset);
    q_rope_params.rope_theta = params.rope_theta;
    if (!RopeOp::Run(context,
                     pipeline_cache,
                     query_norm,
                     query_rope,
                     q_rope_params,
                     temporary_arena,
                     error_message)) {
        return false;
    }

    RopeParams k_rope_params = q_rope_params;
    k_rope_params.head_count = static_cast<std::uint32_t>(params.num_key_value_heads);
    if (!RopeOp::Run(context,
                     pipeline_cache,
                     key_norm,
                     key_rope,
                     k_rope_params,
                     temporary_arena,
                     error_message)) {
        return false;
    }

    std::size_t sequence_length_before_append = 0;
    DeviceTensor score_keys = key_rope;
    DeviceTensor score_values = value_proj;
    std::size_t total_sequence_length = row_count;
    if (kv_cache != nullptr) {
        const KVCacheLayerView existing_view = kv_cache->ViewForLayer(layer_index);
        sequence_length_before_append = existing_view.sequence_length;
        if (decode_mode) {
            if (!kv_cache->AppendDecodeToken(context, layer_index, key_rope, value_proj, error_message)) {
                return false;
            }
        } else {
            if (!kv_cache->AppendPrefill(context, layer_index, key_rope, value_proj, error_message)) {
                return false;
            }
        }
        const KVCacheLayerView full_view = kv_cache->ViewForLayer(layer_index);
        score_keys = full_view.keys;
        score_values = full_view.values;
        total_sequence_length = full_view.sequence_length;
    }

    const TensorDesc score_runtime_desc =
        TensorDesc::CreateContiguous(DataType::kFloat32, {row_count * params.num_attention_heads, total_sequence_length});
    if (!AllocateTemporaryTensor(temporary_arena, score_runtime_desc, &attention_scores, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, score_runtime_desc, &attention_probs, error_message)) {
        return false;
    }

    MetalAttentionScoreParams score_params;
    score_params.query_row_count = static_cast<std::uint32_t>(row_count);
    score_params.key_row_count = static_cast<std::uint32_t>(total_sequence_length);
    score_params.query_head_count = static_cast<std::uint32_t>(params.num_attention_heads);
    score_params.key_value_head_count = static_cast<std::uint32_t>(params.num_key_value_heads);
    score_params.head_dim = static_cast<std::uint32_t>(params.head_dim);
    score_params.head_group_size = static_cast<std::uint32_t>(params.num_attention_heads / params.num_key_value_heads);
    score_params.query_position_base = static_cast<std::uint32_t>(sequence_length_before_append);
    score_params.causal_mask = 1;
    score_params.scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    if (!DispatchAttentionScores(context,
                                 pipeline_cache,
                                 query_rope,
                                 score_keys,
                                 attention_scores,
                                 score_params,
                                 temporary_arena,
                                 error_message)) {
        return false;
    }

    SoftmaxParams softmax_params;
    softmax_params.row_count = static_cast<std::uint32_t>(row_count * params.num_attention_heads);
    softmax_params.row_size = static_cast<std::uint32_t>(total_sequence_length);
    if (!SoftmaxOp::Run(context,
                        pipeline_cache,
                        attention_scores,
                        attention_probs,
                        softmax_params,
                        temporary_arena,
                        error_message)) {
        return false;
    }

    MetalAttentionValueParams value_params;
    value_params.query_row_count = static_cast<std::uint32_t>(row_count);
    value_params.key_row_count = static_cast<std::uint32_t>(total_sequence_length);
    value_params.query_head_count = static_cast<std::uint32_t>(params.num_attention_heads);
    value_params.key_value_head_count = static_cast<std::uint32_t>(params.num_key_value_heads);
    value_params.head_dim = static_cast<std::uint32_t>(params.head_dim);
    value_params.head_group_size = static_cast<std::uint32_t>(params.num_attention_heads / params.num_key_value_heads);
    if (!DispatchAttentionValues(context,
                                 pipeline_cache,
                                 attention_probs,
                                 score_values,
                                 attention_context,
                                 value_params,
                                 temporary_arena,
                                 error_message)) {
        return false;
    }

    LinearParams o_params;
    o_params.add_residual = params.add_residual;
    o_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
    o_params.matmul.inner_dim = static_cast<std::uint32_t>(query_hidden_size);
    o_params.matmul.column_count = static_cast<std::uint32_t>(hidden_size);
    o_params.matmul.decode_mode = row_count == 1;
    o_params.matmul.transpose_rhs = true;
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       attention_context,
                       weights.o_proj_weight,
                       nullptr,
                       params.add_residual ? (residual == nullptr ? &input : residual) : nullptr,
                       output,
                       o_params,
                       temporary_arena,
                       error_message)) {
        return false;
    }

    return true;
}

}  // namespace soc::gpu