#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace soc::gpu::models::qwen3_5 {

enum class Qwen3_5LayerType {
    kGatedDeltaNet,
    kGatedAttention,
};

struct Qwen3_5RopeSpec {
    bool mrope_interleaved = true;
    std::vector<std::size_t> mrope_section = {11, 11, 10};
    std::string rope_type = "default";
    double rope_theta = 10000000.0;
    double partial_rotary_factor = 0.25;
};

struct Qwen3_5ArchitectureSpec {
    std::string model_name = "Qwen3.5-9B";
    std::string top_level_architecture = "Qwen3_5ForConditionalGeneration";
    std::string text_model_type = "qwen3_5_text";
    std::size_t vocab_size = 248320;
    std::size_t hidden_size = 4096;
    std::size_t intermediate_size = 12288;
    std::size_t num_hidden_layers = 32;
    std::size_t num_attention_heads = 16;
    std::size_t num_key_value_heads = 4;
    std::size_t attention_head_dim = 256;
    std::size_t rotary_dim = 64;
    std::size_t linear_num_key_heads = 16;
    std::size_t linear_num_value_heads = 32;
    std::size_t linear_key_head_dim = 128;
    std::size_t linear_value_head_dim = 128;
    std::size_t linear_conv_kernel_dim = 4;
    std::size_t full_attention_interval = 4;
    std::size_t mtp_num_hidden_layers = 1;
    std::size_t max_position_embeddings = 262144;
    std::size_t advertised_native_context = 262144;
    bool attn_output_gate = true;
    bool tie_word_embeddings = false;
    std::string weights_dtype = "bfloat16";
    std::string recurrent_state_dtype = "float32";
    float rms_norm_eps = 1.0e-6f;
    Qwen3_5RopeSpec rope;
    std::vector<Qwen3_5LayerType> layer_types;
};

Qwen3_5ArchitectureSpec BuildQwen3_5_9BReferenceSpec();

}  // namespace soc::gpu::models::qwen3_5
