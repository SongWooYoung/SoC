#include "model/qwen_causal_lm.h"

#include "buffer/buffer_arena.h"
#include "op/embedding_op.h"
#include "op/linear_op.h"
#include "op/rms_norm_op.h"

namespace soc::gpu {
namespace {

bool AllocateTemporaryTensor(BufferArena* arena,
                             const TensorDesc& desc,
                             DeviceTensor* tensor,
                             std::string* error_message) {
    if (arena == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Temporary arena is required for QwenCausalLM";
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

}  // namespace

QwenCausalLM::QwenCausalLM(QwenCausalLMWeights weights, QwenCausalLMParams params)
    : weights_(std::move(weights)), params_(std::move(params)) {}

std::size_t QwenCausalLM::num_layers() const { return weights_.blocks.size(); }
std::size_t QwenCausalLM::num_key_value_heads() const { return params_.num_key_value_heads; }
std::size_t QwenCausalLM::head_dim() const { return params_.head_dim; }
std::size_t QwenCausalLM::vocab_size() const { return params_.vocab_size; }
std::size_t QwenCausalLM::max_position_embeddings() const { return params_.max_position_embeddings; }

bool QwenCausalLM::ForwardHidden(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const DeviceTensor& token_ids,
                                 const DeviceTensor& output,
                                 BufferArena* temporary_arena,
                                 std::size_t position_offset,
                                 std::string* error_message) const {
    if (!token_ids.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM hidden forward expects valid tensors";
        }
        return false;
    }
    if (token_ids.GetDesc().GetDataType() != DataType::kInt32 || token_ids.GetDesc().Rank() != 1 ||
        output.GetDesc().GetDataType() != DataType::kFloat32 || output.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM hidden forward expects rank-1 int32 token ids and rank-2 float32 output";
        }
        return false;
    }
    const std::size_t token_count = token_ids.GetDesc().GetShape()[0];
    if (output.GetDesc().GetShape() != std::vector<std::size_t>{token_count, params_.hidden_size}) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM hidden output shape must be [token_count, hidden_size]";
        }
        return false;
    }

    BufferArenaMarkGuard arena_mark(temporary_arena, "QwenCausalLMHidden");

    DeviceTensor hidden_states;
    DeviceTensor next_hidden_states;
    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {token_count, params_.hidden_size});
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &hidden_states, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, hidden_desc, &next_hidden_states, error_message)) {
        return false;
    }

    EmbeddingParams embedding_params;
    embedding_params.token_count = static_cast<std::uint32_t>(token_count);
    embedding_params.hidden_size = static_cast<std::uint32_t>(params_.hidden_size);
    embedding_params.vocab_size = static_cast<std::uint32_t>(params_.vocab_size);
    if (!EmbeddingOp::Run(context,
                          pipeline_cache,
                          token_ids,
                          weights_.embed_tokens_weight,
                          hidden_states,
                          embedding_params,
                          temporary_arena,
                          error_message)) {
        return false;
    }

    for (std::size_t layer_index = 0; layer_index < weights_.blocks.size(); ++layer_index) {
        QwenBlockParams block_params;
        block_params.attention.num_attention_heads = params_.num_attention_heads;
        block_params.attention.num_key_value_heads = params_.num_key_value_heads;
        block_params.attention.head_dim = params_.head_dim;
        block_params.attention.rotary_dim = params_.head_dim;
        block_params.attention.position_offset = position_offset;
        block_params.attention.rope_theta = params_.rope_theta;
        block_params.attention.rms_epsilon = params_.rms_norm_eps;
        block_params.mlp.intermediate_size = params_.intermediate_size;
        block_params.rms_epsilon = params_.rms_norm_eps;
        if (!QwenBlock::Run(context,
                            pipeline_cache,
                            hidden_states,
                            weights_.blocks[layer_index],
                            next_hidden_states,
                            block_params,
                            temporary_arena,
                            error_message)) {
            return false;
        }
        std::swap(hidden_states, next_hidden_states);
    }

    RmsNormParams final_norm_params;
    final_norm_params.epsilon = params_.rms_norm_eps;
    final_norm_params.row_count = static_cast<std::uint32_t>(token_count);
    final_norm_params.row_size = static_cast<std::uint32_t>(params_.hidden_size);
    return RmsNormOp::Run(context,
                          pipeline_cache,
                          hidden_states,
                          weights_.final_norm_weight,
                          output,
                          final_norm_params,
                          temporary_arena,
                          error_message);
}

bool QwenCausalLM::ForwardHiddenCached(const MetalContext& context,
                                       PipelineCache* pipeline_cache,
                                       const DeviceTensor& token_ids,
                                       KVCache* kv_cache,
                                       const DeviceTensor& output,
                                       BufferArena* temporary_arena,
                                       std::size_t position_offset,
                                       std::string* error_message) const {
    if (kv_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM cached forward requires a KVCache";
        }
        return false;
    }
    if (!token_ids.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM cached hidden forward expects valid tensors";
        }
        return false;
    }

    const std::size_t token_count = token_ids.GetDesc().GetShape()[0];
    BufferArenaMarkGuard arena_mark(temporary_arena, token_count == 1 ? "QwenCausalLMDecode" : "QwenCausalLMPrefill");

    DeviceTensor hidden_states;
    DeviceTensor next_hidden_states;
    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {token_count, params_.hidden_size});
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &hidden_states, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, hidden_desc, &next_hidden_states, error_message)) {
        return false;
    }

    EmbeddingParams embedding_params;
    embedding_params.token_count = static_cast<std::uint32_t>(token_count);
    embedding_params.hidden_size = static_cast<std::uint32_t>(params_.hidden_size);
    embedding_params.vocab_size = static_cast<std::uint32_t>(params_.vocab_size);
    if (!EmbeddingOp::Run(context,
                          pipeline_cache,
                          token_ids,
                          weights_.embed_tokens_weight,
                          hidden_states,
                          embedding_params,
                          temporary_arena,
                          error_message)) {
        return false;
    }

    const bool decode_mode = token_count == 1;
    for (std::size_t layer_index = 0; layer_index < weights_.blocks.size(); ++layer_index) {
        QwenBlockParams block_params;
        block_params.attention.num_attention_heads = params_.num_attention_heads;
        block_params.attention.num_key_value_heads = params_.num_key_value_heads;
        block_params.attention.head_dim = params_.head_dim;
        block_params.attention.rotary_dim = params_.head_dim;
        block_params.attention.position_offset = position_offset;
        block_params.attention.rope_theta = params_.rope_theta;
        block_params.attention.rms_epsilon = params_.rms_norm_eps;
        block_params.mlp.intermediate_size = params_.intermediate_size;
        block_params.rms_epsilon = params_.rms_norm_eps;
        const bool block_ok = decode_mode
            ? QwenBlock::RunDecode(context,
                                   pipeline_cache,
                                   hidden_states,
                                   weights_.blocks[layer_index],
                                   kv_cache,
                                   layer_index,
                                   next_hidden_states,
                                   block_params,
                                   temporary_arena,
                                   error_message)
            : QwenBlock::RunPrefill(context,
                                    pipeline_cache,
                                    hidden_states,
                                    weights_.blocks[layer_index],
                                    kv_cache,
                                    layer_index,
                                    next_hidden_states,
                                    block_params,
                                    temporary_arena,
                                    error_message);
        if (!block_ok) {
            return false;
        }
        std::swap(hidden_states, next_hidden_states);
    }

    RmsNormParams final_norm_params;
    final_norm_params.epsilon = params_.rms_norm_eps;
    final_norm_params.row_count = static_cast<std::uint32_t>(token_count);
    final_norm_params.row_size = static_cast<std::uint32_t>(params_.hidden_size);
    return RmsNormOp::Run(context,
                          pipeline_cache,
                          hidden_states,
                          weights_.final_norm_weight,
                          output,
                          final_norm_params,
                          temporary_arena,
                          error_message);
}

bool QwenCausalLM::ForwardLogits(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const DeviceTensor& token_ids,
                                 const DeviceTensor& logits_output,
                                 BufferArena* temporary_arena,
                                 std::size_t position_offset,
                                 std::string* error_message) const {
    DeviceTensor hidden_states;
    const std::size_t token_count = token_ids.GetDesc().GetShape()[0];
    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {token_count, params_.hidden_size});
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &hidden_states, error_message)) {
        return false;
    }
    if (!ForwardHidden(context, pipeline_cache, token_ids, hidden_states, temporary_arena, position_offset, error_message)) {
        return false;
    }
    return ComputeLogitsFromHidden(context, pipeline_cache, hidden_states, logits_output, temporary_arena, error_message);
}

bool QwenCausalLM::ForwardLogitsCached(const MetalContext& context,
                                       PipelineCache* pipeline_cache,
                                       const DeviceTensor& token_ids,
                                       KVCache* kv_cache,
                                       const DeviceTensor& logits_output,
                                       BufferArena* temporary_arena,
                                       std::size_t position_offset,
                                       std::string* error_message) const {
    DeviceTensor hidden_states;
    const std::size_t token_count = token_ids.GetDesc().GetShape()[0];
    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {token_count, params_.hidden_size});
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &hidden_states, error_message)) {
        return false;
    }
    if (!ForwardHiddenCached(context, pipeline_cache, token_ids, kv_cache, hidden_states, temporary_arena, position_offset, error_message)) {
        return false;
    }
    return ComputeLogitsFromHidden(context, pipeline_cache, hidden_states, logits_output, temporary_arena, error_message);
}

const QwenCausalLMWeights& QwenCausalLM::weights() const { return weights_; }
const QwenCausalLMParams& QwenCausalLM::params() const { return params_; }

bool QwenCausalLM::ComputeLogitsFromHidden(const MetalContext& context,
                                           PipelineCache* pipeline_cache,
                                           const DeviceTensor& hidden_states,
                                           const DeviceTensor& logits_output,
                                           BufferArena* temporary_arena,
                                           std::string* error_message) const {
    if (!hidden_states.IsValid() || !logits_output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM logits computation expects valid tensors";
        }
        return false;
    }
    const std::size_t token_count = hidden_states.GetDesc().GetShape()[0];
    if (logits_output.GetDesc().GetShape() != std::vector<std::size_t>{token_count, params_.vocab_size}) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM logits output shape must be [token_count, vocab_size]";
        }
        return false;
    }

    LinearParams logits_params;
    logits_params.matmul.row_count = static_cast<std::uint32_t>(token_count);
    logits_params.matmul.inner_dim = static_cast<std::uint32_t>(params_.hidden_size);
    logits_params.matmul.column_count = static_cast<std::uint32_t>(params_.vocab_size);
    logits_params.matmul.decode_mode = token_count == 1;
    logits_params.matmul.transpose_rhs = true;
    const DeviceTensor& projection_weight = weights_.tie_word_embeddings ? weights_.embed_tokens_weight : weights_.lm_head_weight;
    return LinearOp::Run(context,
                         pipeline_cache,
                         hidden_states,
                         projection_weight,
                         nullptr,
                         nullptr,
                         logits_output,
                         logits_params,
                         temporary_arena,
                         error_message);
}

}  // namespace soc::gpu