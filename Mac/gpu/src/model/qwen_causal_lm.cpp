#include "model/qwen_causal_lm.h"

#include "buffer/buffer_arena.h"
#include "metal/command_stream.h"
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

bool CopyTensorViaHost(const DeviceTensor& input,
                      const DeviceTensor& output,
                      std::string* error_message) {
    if (input.GetDesc().ByteSize() != output.GetDesc().ByteSize()) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM copy requires matching tensor byte sizes";
        }
        return false;
    }
    std::vector<std::byte> bytes(input.GetDesc().ByteSize());
    if (!input.GetBuffer()->Read(bytes.data(), bytes.size(), input.GetByteOffset(), error_message)) {
        return false;
    }
    return output.GetBuffer()->Write(bytes.data(), bytes.size(), output.GetByteOffset(), error_message);
}

bool ValidateLayerRange(std::size_t start_layer,
                        std::size_t end_layer,
                        std::size_t total_layers,
                        std::string* error_message) {
    if (start_layer > end_layer || end_layer > total_layers) {
        if (error_message != nullptr) {
            *error_message = "invalid qwen layer range";
        }
        return false;
    }
    return true;
}

bool RunBlockRange(const MetalContext& context,
                   PipelineCache* pipeline_cache,
                   const QwenCausalLMWeights& weights,
                   const QwenCausalLMParams& params,
                   const DeviceTensor& input_hidden,
                   KVCache* kv_cache,
                   const DeviceTensor& output,
                   BufferArena* temporary_arena,
                   std::size_t position_offset,
                   std::size_t start_layer,
                   std::size_t end_layer,
                   bool apply_final_norm,
                   std::string* error_message) {
    if (!ValidateLayerRange(start_layer, end_layer, weights.blocks.size(), error_message)) {
        return false;
    }
    if (!input_hidden.IsValid() || !output.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM block range expects valid tensors";
        }
        return false;
    }
    if (input_hidden.GetDesc().GetDataType() != DataType::kFloat32 || input_hidden.GetDesc().Rank() != 2 ||
        output.GetDesc().GetDataType() != DataType::kFloat32 || output.GetDesc().Rank() != 2 ||
        input_hidden.GetDesc().GetShape() != output.GetDesc().GetShape()) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM block range expects matching rank-2 float32 hidden tensors";
        }
        return false;
    }

    const std::size_t token_count = input_hidden.GetDesc().GetShape()[0];
    if (start_layer == end_layer) {
        if (apply_final_norm) {
            RmsNormParams final_norm_params;
            final_norm_params.epsilon = params.rms_norm_eps;
            final_norm_params.row_count = static_cast<std::uint32_t>(token_count);
            final_norm_params.row_size = static_cast<std::uint32_t>(params.hidden_size);
            return RmsNormOp::Run(context,
                                  pipeline_cache,
                                  input_hidden,
                                  weights.final_norm_weight,
                                  output,
                                  final_norm_params,
                                  temporary_arena,
                                  nullptr,
                                  error_message);
        }
        return CopyTensorViaHost(input_hidden, output, error_message);
    }

    BufferArenaMarkGuard arena_mark(temporary_arena, kv_cache == nullptr ? "QwenCausalLMHiddenRange" : (token_count == 1 ? "QwenCausalLMDecodeRange" : "QwenCausalLMPrefillRange"));

    DeviceTensor current_hidden = input_hidden;
    DeviceTensor scratch_hidden;
    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {token_count, params.hidden_size});
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &scratch_hidden, error_message)) {
        return false;
    }

    const bool decode_mode = token_count == 1;

    CommandStream stream;
    if (!stream.Begin(context, error_message)) {
        return false;
    }

    for (std::size_t layer_index = start_layer; layer_index < end_layer; ++layer_index) {
        const bool is_last_layer = layer_index + 1 == end_layer;
        const DeviceTensor layer_output = (!apply_final_norm && is_last_layer) ? output : scratch_hidden;

        QwenBlockParams block_params;
        block_params.attention.num_attention_heads = params.num_attention_heads;
        block_params.attention.num_key_value_heads = params.num_key_value_heads;
        block_params.attention.head_dim = params.head_dim;
        block_params.attention.rotary_dim = params.head_dim;
        block_params.attention.position_offset = position_offset;
        block_params.attention.rope_theta = params.rope_theta;
        block_params.attention.rms_epsilon = params.rms_norm_eps;
        block_params.mlp.intermediate_size = params.intermediate_size;
        block_params.rms_epsilon = params.rms_norm_eps;

        const bool block_ok = kv_cache == nullptr
            ? QwenBlock::Run(context,
                             pipeline_cache,
                             current_hidden,
                             weights.blocks[layer_index],
                             layer_output,
                             block_params,
                             temporary_arena,
                             &stream,
                             error_message)
            : (decode_mode
                ? QwenBlock::RunDecode(context,
                                       pipeline_cache,
                                       current_hidden,
                                       weights.blocks[layer_index],
                                       kv_cache,
                                       layer_index,
                                       layer_output,
                                       block_params,
                                       temporary_arena,
                                       &stream,
                                       error_message)
                : QwenBlock::RunPrefill(context,
                                        pipeline_cache,
                                        current_hidden,
                                        weights.blocks[layer_index],
                                        kv_cache,
                                        layer_index,
                                        layer_output,
                                        block_params,
                                        temporary_arena,
                                        &stream,
                                        error_message));
        if (!block_ok) {
            return false;
        }

        if (!is_last_layer || apply_final_norm) {
            current_hidden = layer_output;
            if (!(is_last_layer && apply_final_norm)) {
                if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &scratch_hidden, error_message)) {
                    return false;
                }
            }
        }
    }

    if (!apply_final_norm) {
        return stream.Flush(context, error_message);
    }

    RmsNormParams final_norm_params;
    final_norm_params.epsilon = params.rms_norm_eps;
    final_norm_params.row_count = static_cast<std::uint32_t>(token_count);
    final_norm_params.row_size = static_cast<std::uint32_t>(params.hidden_size);
    if (!RmsNormOp::Run(context,
                          pipeline_cache,
                          current_hidden,
                          weights.final_norm_weight,
                          output,
                          final_norm_params,
                          temporary_arena,
                          &stream,
                          error_message)) {
        return false;
    }
    return stream.Flush(context, error_message);
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
    return ForwardHiddenRange(context,
                              pipeline_cache,
                              token_ids,
                              output,
                              temporary_arena,
                              position_offset,
                              0,
                              weights_.blocks.size(),
                              true,
                              error_message);
}

bool QwenCausalLM::ForwardHiddenRange(const MetalContext& context,
                                      PipelineCache* pipeline_cache,
                                      const DeviceTensor& token_ids,
                                      const DeviceTensor& output,
                                      BufferArena* temporary_arena,
                                      std::size_t position_offset,
                                      std::size_t start_layer,
                                      std::size_t end_layer,
                                      bool apply_final_norm,
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
    if (!ValidateLayerRange(start_layer, end_layer, weights_.blocks.size(), error_message)) {
        return false;
    }
    if (start_layer != 0) {
        if (error_message != nullptr) {
            *error_message = "token-based qwen layer ranges must start at layer 0";
        }
        return false;
    }

    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {token_count, params_.hidden_size});
    DeviceTensor hidden_states;
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &hidden_states, error_message)) {
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
                          nullptr,
                          error_message)) {
        return false;
    }
    return RunBlockRange(context,
                         pipeline_cache,
                         weights_,
                         params_,
                         hidden_states,
                         nullptr,
                         output,
                         temporary_arena,
                         position_offset,
                         start_layer,
                         end_layer,
                         apply_final_norm,
                         error_message);
}

bool QwenCausalLM::ForwardHiddenFromStatesRange(const MetalContext& context,
                                                PipelineCache* pipeline_cache,
                                                const DeviceTensor& hidden_states,
                                                const DeviceTensor& output,
                                                BufferArena* temporary_arena,
                                                std::size_t position_offset,
                                                std::size_t start_layer,
                                                std::size_t end_layer,
                                                bool apply_final_norm,
                                                std::string* error_message) const {
    return RunBlockRange(context,
                         pipeline_cache,
                         weights_,
                         params_,
                         hidden_states,
                         nullptr,
                         output,
                         temporary_arena,
                         position_offset,
                         start_layer,
                         end_layer,
                         apply_final_norm,
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
    return ForwardHiddenCachedRange(context,
                                    pipeline_cache,
                                    token_ids,
                                    kv_cache,
                                    output,
                                    temporary_arena,
                                    position_offset,
                                    0,
                                    weights_.blocks.size(),
                                    true,
                                    error_message);
}

bool QwenCausalLM::ForwardHiddenCachedRange(const MetalContext& context,
                                            PipelineCache* pipeline_cache,
                                            const DeviceTensor& token_ids,
                                            KVCache* kv_cache,
                                            const DeviceTensor& output,
                                            BufferArena* temporary_arena,
                                            std::size_t position_offset,
                                            std::size_t start_layer,
                                            std::size_t end_layer,
                                            bool apply_final_norm,
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
    if (!ValidateLayerRange(start_layer, end_layer, weights_.blocks.size(), error_message)) {
        return false;
    }
    if (start_layer != 0) {
        if (error_message != nullptr) {
            *error_message = "token-based qwen layer ranges must start at layer 0";
        }
        return false;
    }

    const std::size_t token_count = token_ids.GetDesc().GetShape()[0];
    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {token_count, params_.hidden_size});
    DeviceTensor hidden_states;
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &hidden_states, error_message)) {
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
                          nullptr,
                          error_message)) {
        return false;
    }
    return RunBlockRange(context,
                         pipeline_cache,
                         weights_,
                         params_,
                         hidden_states,
                         kv_cache,
                         output,
                         temporary_arena,
                         position_offset,
                         start_layer,
                         end_layer,
                         apply_final_norm,
                         error_message);
}

bool QwenCausalLM::ForwardHiddenFromStatesCachedRange(const MetalContext& context,
                                                      PipelineCache* pipeline_cache,
                                                      const DeviceTensor& hidden_states,
                                                      KVCache* kv_cache,
                                                      const DeviceTensor& output,
                                                      BufferArena* temporary_arena,
                                                      std::size_t position_offset,
                                                      std::size_t start_layer,
                                                      std::size_t end_layer,
                                                      bool apply_final_norm,
                                                      std::string* error_message) const {
    if (kv_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "QwenCausalLM cached forward requires a KVCache";
        }
        return false;
    }
    return RunBlockRange(context,
                         pipeline_cache,
                         weights_,
                         params_,
                         hidden_states,
                         kv_cache,
                         output,
                         temporary_arena,
                         position_offset,
                         start_layer,
                         end_layer,
                         apply_final_norm,
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
    return ForwardLogitsFromHidden(context, pipeline_cache, hidden_states, logits_output, temporary_arena, error_message);
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
    return ForwardLogitsFromHidden(context, pipeline_cache, hidden_states, logits_output, temporary_arena, error_message);
}

bool QwenCausalLM::ForwardLogitsFromHidden(const MetalContext& context,
                                           PipelineCache* pipeline_cache,
                                           const DeviceTensor& hidden_states,
                                           const DeviceTensor& logits_output,
                                           BufferArena* temporary_arena,
                                           std::string* error_message) const {
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
                         nullptr,
                         error_message);
}

}  // namespace soc::gpu