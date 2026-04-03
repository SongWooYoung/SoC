#pragma once

#include "utils/json.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ── RoPE parameters ─────────────────────────────────────────────────────────

struct RopeParameters {
    std::string rope_type = "default";
    double rope_theta = 10000000.0;
    bool mrope_interleaved = true;
    std::vector<int> mrope_section = {11, 11, 10, 0};
    float partial_rotary_factor = 0.25f;

    static RopeParameters from_json(const JsonValue& j) {
        RopeParameters r;
        if (auto* v = j.find("rope_type"))        r.rope_type = v->as_string();
        if (auto* v = j.find("rope_theta"))        r.rope_theta = v->as_number();
        if (auto* v = j.find("mrope_interleaved")) r.mrope_interleaved = v->as_bool();
        if (auto* v = j.find("mrope_section")) {
            r.mrope_section.clear();
            for (auto& e : v->as_array()) r.mrope_section.push_back(e.as_int());
        }
        if (auto* v = j.find("partial_rotary_factor")) r.partial_rotary_factor = static_cast<float>(v->as_number());
        return r;
    }
};

// ── Text config ─────────────────────────────────────────────────────────────

enum class LayerType : uint8_t { LinearAttention, FullAttention };

struct Qwen3_5TextConfig {
    int vocab_size             = 248320;
    int hidden_size            = 4096;
    int intermediate_size      = 12288;
    int num_hidden_layers      = 32;
    int num_attention_heads    = 16;
    int num_key_value_heads    = 4;
    std::string hidden_act     = "silu";
    int max_position_embeddings = 32768;
    float rms_norm_eps         = 1e-6f;
    bool attention_bias        = false;
    float attention_dropout    = 0.0f;
    bool tie_word_embeddings   = false;
    int head_dim               = 256;

    // Linear attention (GatedDeltaNet)
    int linear_conv_kernel_dim  = 4;
    int linear_key_head_dim     = 128;
    int linear_value_head_dim   = 128;
    int linear_num_key_heads    = 16;
    int linear_num_value_heads  = 32;

    // RoPE
    RopeParameters rope_parameters;

    // Per-layer type pattern
    std::vector<LayerType> layer_types;

    // Derived
    int rotary_dim() const {
        return static_cast<int>(head_dim * rope_parameters.partial_rotary_factor);
    }

    static Qwen3_5TextConfig from_json(const JsonValue& j) {
        Qwen3_5TextConfig c;
        if (auto* v = j.find("vocab_size"))             c.vocab_size = v->as_int();
        if (auto* v = j.find("hidden_size"))             c.hidden_size = v->as_int();
        if (auto* v = j.find("intermediate_size"))       c.intermediate_size = v->as_int();
        if (auto* v = j.find("num_hidden_layers"))       c.num_hidden_layers = v->as_int();
        if (auto* v = j.find("num_attention_heads"))     c.num_attention_heads = v->as_int();
        if (auto* v = j.find("num_key_value_heads"))     c.num_key_value_heads = v->as_int();
        if (auto* v = j.find("hidden_act"))              c.hidden_act = v->as_string();
        if (auto* v = j.find("max_position_embeddings")) c.max_position_embeddings = v->as_int();
        if (auto* v = j.find("rms_norm_eps"))            c.rms_norm_eps = static_cast<float>(v->as_number());
        if (auto* v = j.find("attention_bias"))          c.attention_bias = v->as_bool();
        if (auto* v = j.find("attention_dropout"))       c.attention_dropout = static_cast<float>(v->as_number());
        if (auto* v = j.find("tie_word_embeddings"))     c.tie_word_embeddings = v->as_bool();
        if (auto* v = j.find("head_dim"))                c.head_dim = v->as_int();
        if (auto* v = j.find("linear_conv_kernel_dim"))  c.linear_conv_kernel_dim = v->as_int();
        if (auto* v = j.find("linear_key_head_dim"))     c.linear_key_head_dim = v->as_int();
        if (auto* v = j.find("linear_value_head_dim"))   c.linear_value_head_dim = v->as_int();
        if (auto* v = j.find("linear_num_key_heads"))    c.linear_num_key_heads = v->as_int();
        if (auto* v = j.find("linear_num_value_heads"))  c.linear_num_value_heads = v->as_int();

        if (auto* v = j.find("rope_parameters"))
            c.rope_parameters = RopeParameters::from_json(*v);

        // layer_types: explicit array or generate from full_attention_interval
        if (auto* v = j.find("layer_types")) {
            c.layer_types.clear();
            for (auto& e : v->as_array()) {
                const auto& s = e.as_string();
                if (s == "full_attention")
                    c.layer_types.push_back(LayerType::FullAttention);
                else
                    c.layer_types.push_back(LayerType::LinearAttention);
            }
        } else {
            int interval = 4;
            if (auto* vi = j.find("full_attention_interval"))
                interval = vi->as_int();
            c.layer_types.resize(c.num_hidden_layers);
            for (int i = 0; i < c.num_hidden_layers; ++i) {
                c.layer_types[i] = ((i + 1) % interval == 0)
                    ? LayerType::FullAttention
                    : LayerType::LinearAttention;
            }
        }

        return c;
    }
};

// ── Vision config (placeholder for Phase 3v) ────────────────────────────────

struct Qwen3_5VisionConfig {
    int depth              = 27;
    int hidden_size        = 1152;
    std::string hidden_act = "gelu_pytorch_tanh";
    int intermediate_size  = 4304;
    int num_heads          = 16;
    int in_channels        = 3;
    int patch_size         = 16;
    int spatial_merge_size = 2;
    int temporal_patch_size = 2;
    int out_hidden_size    = 3584;
    int num_position_embeddings = 2304;

    static Qwen3_5VisionConfig from_json(const JsonValue& j) {
        Qwen3_5VisionConfig c;
        if (auto* v = j.find("depth"))              c.depth = v->as_int();
        if (auto* v = j.find("hidden_size"))         c.hidden_size = v->as_int();
        if (auto* v = j.find("hidden_act"))          c.hidden_act = v->as_string();
        if (auto* v = j.find("intermediate_size"))   c.intermediate_size = v->as_int();
        if (auto* v = j.find("num_heads"))           c.num_heads = v->as_int();
        if (auto* v = j.find("in_channels"))         c.in_channels = v->as_int();
        if (auto* v = j.find("patch_size"))          c.patch_size = v->as_int();
        if (auto* v = j.find("spatial_merge_size"))  c.spatial_merge_size = v->as_int();
        if (auto* v = j.find("temporal_patch_size")) c.temporal_patch_size = v->as_int();
        if (auto* v = j.find("out_hidden_size"))     c.out_hidden_size = v->as_int();
        if (auto* v = j.find("num_position_embeddings")) c.num_position_embeddings = v->as_int();
        return c;
    }
};

// ── Top-level config ────────────────────────────────────────────────────────

struct Qwen3_5Config {
    Qwen3_5TextConfig text_config;
    Qwen3_5VisionConfig vision_config;
    int image_token_id       = 248056;
    int video_token_id       = 248057;
    int vision_start_token_id = 248053;
    int vision_end_token_id  = 248054;
    bool tie_word_embeddings = false;

    static Qwen3_5Config from_file(const std::string& path) {
        JsonValue root = JsonParser::parse_file(path);
        return from_json(root);
    }

    static Qwen3_5Config from_json(const JsonValue& j) {
        Qwen3_5Config c;
        if (auto* v = j.find("text_config"))
            c.text_config = Qwen3_5TextConfig::from_json(*v);
        if (auto* v = j.find("vision_config"))
            c.vision_config = Qwen3_5VisionConfig::from_json(*v);
        if (auto* v = j.find("image_token_id"))       c.image_token_id = v->as_int();
        if (auto* v = j.find("video_token_id"))        c.video_token_id = v->as_int();
        if (auto* v = j.find("vision_start_token_id")) c.vision_start_token_id = v->as_int();
        if (auto* v = j.find("vision_end_token_id"))   c.vision_end_token_id = v->as_int();
        if (auto* v = j.find("tie_word_embeddings"))   c.tie_word_embeddings = v->as_bool();
        return c;
    }
};
