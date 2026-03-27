#ifndef QWEN3_CONFIG_H
#define QWEN3_CONFIG_H

#include <cstddef>
#include <string>

#include "header/json_value.h"

struct Qwen3Config {
    std::string model_type;
    std::size_t hidden_size;
    std::size_t intermediate_size;
    std::size_t num_hidden_layers;
    std::size_t num_attention_heads;
    std::size_t num_key_value_heads;
    std::size_t head_dim;
    double rms_norm_eps;
    double rope_theta;
    bool tie_word_embeddings;
    std::string torch_dtype;
    std::size_t vocab_size;

    static Qwen3Config FromJson(const JsonValue& config_json);
    std::size_t HiddenSizePerAttentionHead() const;
    void Validate() const;
};

#endif