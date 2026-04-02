#pragma once

#include "header/tensor.h"
#include "models/qwen3_5/qwen3_5_architecture.h"
#include "models/qwen3_5/qwen3_5_state_layout.h"
#include "models/qwen3_5/modules/qwen3_5_block.h"
#include "runtime/model_runner.h"

namespace soc::gpu::models::qwen3_5 {

struct Qwen3_5Weights {
    DeviceTensor embed_tokens_weight;
    std::vector<Qwen3_5BlockWeights> blocks;
    DeviceTensor final_norm_weight;
    DeviceTensor lm_head_weight;
    bool tie_word_embeddings = false;
};

struct Qwen3_5HostGatedAttentionWeights {
    Tensor q_proj_weight;
    Tensor k_proj_weight;
    Tensor v_proj_weight;
    Tensor o_proj_weight;
    Tensor q_norm_weight;
    Tensor k_norm_weight;
};

struct Qwen3_5HostGatedDeltaNetWeights {
    Tensor norm_weight;
    Tensor in_proj_qkv_weight;
    Tensor in_proj_z_weight;
    Tensor in_proj_a_weight;
    Tensor in_proj_b_weight;
    Tensor out_proj_weight;
    Tensor conv1d_weight;
    Tensor a_log;
    Tensor dt_bias;
};

struct Qwen3_5HostMlpWeights {
    Tensor gate_proj_weight;
    Tensor up_proj_weight;
    Tensor down_proj_weight;
};

struct Qwen3_5HostBlockWeights {
    Tensor input_layernorm_weight;
    Qwen3_5HostGatedDeltaNetWeights linear;
    Qwen3_5HostGatedAttentionWeights attention;
    Tensor post_attention_layernorm_weight;
    Qwen3_5HostMlpWeights mlp;
};

struct Qwen3_5HostWeights {
    Tensor embed_tokens_weight;
    std::vector<Qwen3_5HostBlockWeights> blocks;
    Tensor final_norm_weight;
    Tensor lm_head_weight;
    bool tie_word_embeddings = false;
};

struct Qwen3_5AttentionDecodeCache {
    std::vector<float> key_values;
    std::vector<float> value_values;
    std::size_t sequence_length = 0;
};

struct Qwen3_5DeltaDecodeCache {
    std::vector<float> recurrent_state;
    std::vector<float> conv_history;
    std::vector<float> conv_sequence_scratch;
    std::size_t conv_history_tokens = 0;
};

struct Qwen3_5DecodeRuntimeState {
    bool ready = false;
    std::size_t cached_sequence_length = 0;
    std::vector<int> token_ids;
    std::vector<Qwen3_5AttentionDecodeCache> attention_layers;
    std::vector<Qwen3_5DeltaDecodeCache> deltanet_layers;
};

struct Qwen3_5TopLogitEntry {
    int token_id = -1;
    float logit = 0.0f;
};

struct Qwen3_5BoundaryProbe {
    int full_prompt_argmax_id = -1;
    int replay_warm_argmax_id = -1;
    float max_abs_logit_diff = 0.0f;
    float mean_abs_logit_diff = 0.0f;
    std::vector<Qwen3_5TopLogitEntry> full_prompt_top_logits;
    std::vector<Qwen3_5TopLogitEntry> replay_warm_top_logits;
    std::vector<std::size_t> attention_cache_lengths;
    std::vector<float> deltanet_state_l2;
};

class Qwen3_5Runner final : public ModelRunner {
public:
    explicit Qwen3_5Runner(Qwen3_5ArchitectureSpec spec);
    Qwen3_5Runner(Qwen3_5ArchitectureSpec spec,
                  Qwen3_5Weights weights,
                  Qwen3_5HostWeights host_weights,
                  Qwen3_5StateLayout state_layout);

    std::size_t num_layers() const override;
    std::size_t hidden_size() const override;
    std::size_t num_key_value_heads() const override;
    std::size_t head_dim() const override;
    std::size_t vocab_size() const override;
    std::size_t max_position_embeddings() const override;

    bool ForwardLogitsCached(const MetalContext& context,
                             PipelineCache* pipeline_cache,
                             const DeviceTensor& token_ids,
                             KVCache* kv_cache,
                             const DeviceTensor& logits_output,
                             BufferArena* temporary_arena,
                             std::size_t position_offset,
                             std::string* error_message) const override;

    bool ForwardLogitsFromHidden(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const DeviceTensor& hidden_states,
                                 const DeviceTensor& logits_output,
                                 BufferArena* temporary_arena,
                                 std::string* error_message) const override;

    DecodePlanner* GetDecodePlanner() override;
    const DecodePlanner* GetDecodePlanner() const override;

    const Qwen3_5ArchitectureSpec& spec() const;
    const Qwen3_5Weights& weights() const;
    const Qwen3_5HostWeights& host_weights() const;
    const Qwen3_5StateLayout& state_layout() const;
    bool has_loaded_weights() const;
    bool DebugBoundaryProbe(const MetalContext& context,
                            PipelineCache* pipeline_cache,
                            BufferArena* temporary_arena,
                            const std::vector<int32_t>& prompt_token_ids,
                            std::size_t top_k,
                            Qwen3_5BoundaryProbe* output,
                            std::string* error_message) const;

private:
    Qwen3_5ArchitectureSpec spec_;
    Qwen3_5Weights weights_;
    Qwen3_5HostWeights host_weights_;
    Qwen3_5StateLayout state_layout_;
    mutable Qwen3_5DecodeRuntimeState decode_state_;
};

}  // namespace soc::gpu::models::qwen3_5
