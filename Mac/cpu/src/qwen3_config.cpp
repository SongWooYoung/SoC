#include "header/qwen3_config.h"

#include <cmath>
#include <stdexcept>

namespace {
std::size_t RequireSize(const JsonValue& value, const std::string& key) {
    return static_cast<std::size_t>(value.at(key).as_int64());
}

double RequireDouble(const JsonValue& value, const std::string& key) {
    return value.at(key).as_number();
}

bool RequireBool(const JsonValue& value, const std::string& key) {
    return value.at(key).as_bool();
}
}

Qwen3Config Qwen3Config::FromJson(const JsonValue& config_json) {
    Qwen3Config config;
    config.model_type = config_json.at("model_type").as_string();
    config.hidden_size = RequireSize(config_json, "hidden_size");
    config.intermediate_size = RequireSize(config_json, "intermediate_size");
    config.num_hidden_layers = RequireSize(config_json, "num_hidden_layers");
    config.num_attention_heads = RequireSize(config_json, "num_attention_heads");
    config.num_key_value_heads = RequireSize(config_json, "num_key_value_heads");
    config.head_dim = RequireSize(config_json, "head_dim");
    config.rms_norm_eps = RequireDouble(config_json, "rms_norm_eps");
    config.rope_theta = RequireDouble(config_json, "rope_theta");
    config.tie_word_embeddings = RequireBool(config_json, "tie_word_embeddings");
    config.torch_dtype = config_json.at("torch_dtype").as_string();
    config.vocab_size = RequireSize(config_json, "vocab_size");
    config.Validate();
    return config;
}

std::size_t Qwen3Config::HiddenSizePerAttentionHead() const {
    return hidden_size / num_attention_heads;
}

void Qwen3Config::Validate() const {
    if (model_type != "qwen3") {
        throw std::runtime_error("config model_type must be qwen3");
    }
    if (hidden_size == 0 || intermediate_size == 0 || num_hidden_layers == 0) {
        throw std::runtime_error("qwen3 config dimensions must be non-zero");
    }
    if (num_attention_heads == 0 || num_key_value_heads == 0 || head_dim == 0) {
        throw std::runtime_error("qwen3 attention dimensions must be non-zero");
    }
    if (hidden_size % num_attention_heads != 0) {
        throw std::runtime_error("hidden_size must be divisible by num_attention_heads");
    }
    if (num_attention_heads % num_key_value_heads != 0) {
        throw std::runtime_error("num_attention_heads must be divisible by num_key_value_heads");
    }
    if (rms_norm_eps <= 0.0 || rope_theta <= 0.0) {
        throw std::runtime_error("qwen3 normalization and rope parameters must be positive");
    }
    if (vocab_size == 0) {
        throw std::runtime_error("vocab_size must be non-zero");
    }
    if (torch_dtype.empty()) {
        throw std::runtime_error("torch_dtype must not be empty");
    }
}