// Test: MLX config.h reuses py_cpp config and correctly loads Qwen3.5 config.json
//
// Verifies that including models/qwen3_5_mlx/config.h provides the same
// Qwen3_5Config structs and that all MLX-relevant fields are accessible.

#include "../models/qwen3_5_mlx/config.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n  %s:%d\n", msg, __FILE__, __LINE__); ++fail_count; } \
    else { ++pass_count; } \
} while (0)

#define CHECK_EQ(a, b, name)  CHECK((a) == (b), name)
#define CHECK_FEQ(a, b, name) CHECK(std::fabs((a) - (b)) < 1e-9, name)

int main(int argc, char** argv) {
    int pass_count = 0, fail_count = 0;

    const char* config_path = (argc >= 2) ? argv[1] : nullptr;
    if (!config_path) {
        std::fprintf(stderr, "Usage: test_mlx_config <config.json>\n");
        return 1;
    }

    std::printf("[mlx_config] Loading: %s\n", config_path);
    Qwen3_5Config cfg;
    try {
        cfg = Qwen3_5Config::from_file(config_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load: %s\n", e.what());
        return 1;
    }

    const auto& t = cfg.text_config;

    // ── MLX config.py field coverage ─────────────────────────────────────
    // These are the fields that MLX's TextConfig dataclass requires.

    CHECK_EQ(t.hidden_size, 2560, "hidden_size");
    CHECK_EQ(t.intermediate_size, 9216, "intermediate_size");
    CHECK_EQ(t.num_hidden_layers, 32, "num_hidden_layers");
    CHECK_EQ(t.num_attention_heads, 16, "num_attention_heads");
    CHECK_EQ(t.num_key_value_heads, 4, "num_key_value_heads");
    CHECK_EQ(t.vocab_size, 248320, "vocab_size");
    CHECK_FEQ(t.rms_norm_eps, 1e-6, "rms_norm_eps");
    CHECK_EQ(t.head_dim, 256, "head_dim");
    CHECK_EQ(t.tie_word_embeddings, true, "tie_word_embeddings");
    CHECK_EQ(t.attention_bias, false, "attention_bias");

    // GDN-specific fields (MLX: linear_num_value_heads, linear_key_head_dim, etc.)
    CHECK_EQ(t.linear_num_value_heads, 32, "linear_num_value_heads");
    CHECK_EQ(t.linear_num_key_heads, 16, "linear_num_key_heads");
    CHECK_EQ(t.linear_key_head_dim, 128, "linear_key_head_dim");
    CHECK_EQ(t.linear_value_head_dim, 128, "linear_value_head_dim");
    CHECK_EQ(t.linear_conv_kernel_dim, 4, "linear_conv_kernel_dim");

    // RoPE parameters (MLX config.py uses rope_parameters dict)
    CHECK_FEQ(t.rope_parameters.rope_theta, 1e7, "rope_theta");
    CHECK_EQ(t.rope_parameters.mrope_interleaved, true, "mrope_interleaved");
    CHECK_FEQ(t.rope_parameters.partial_rotary_factor, 0.25, "partial_rotary_factor");
    CHECK_EQ((int)t.rope_parameters.mrope_section.size(), 3, "mrope_section.size");
    CHECK_EQ(t.rope_parameters.mrope_section[0], 11, "mrope_section[0]");
    CHECK_EQ(t.rope_parameters.mrope_section[1], 11, "mrope_section[1]");
    CHECK_EQ(t.rope_parameters.mrope_section[2], 10, "mrope_section[2]");

    // Derived: rotary_dim = head_dim * partial_rotary_factor = 256 * 0.25 = 64
    CHECK_EQ(t.rotary_dim(), 64, "rotary_dim");

    // layer_types: MLX computes from full_attention_interval=4
    // Pattern: [lin, lin, lin, full] × 8
    CHECK_EQ((int)t.layer_types.size(), 32, "layer_types.size");
    int full_count = 0, linear_count = 0;
    for (int i = 0; i < 32; ++i) {
        bool expect_full = ((i + 1) % 4 == 0);
        LayerType expected = expect_full ? LayerType::FullAttention : LayerType::LinearAttention;
        CHECK_EQ(t.layer_types[i], expected, ("layer_types[" + std::to_string(i) + "]").c_str());
        if (t.layer_types[i] == LayerType::FullAttention) ++full_count;
        else ++linear_count;
    }
    CHECK_EQ(full_count, 8, "full_attention_layer_count");
    CHECK_EQ(linear_count, 24, "linear_attention_layer_count");

    // ── Test full_attention_interval generation (like MLX __post_init__) ──
    {
        std::string json_str = R"({
            "num_hidden_layers": 12,
            "full_attention_interval": 4,
            "hidden_size": 1024
        })";
        JsonValue j = JsonParser::parse(json_str);
        auto tc = Qwen3_5TextConfig::from_json(j);
        CHECK_EQ((int)tc.layer_types.size(), 12, "gen12: size");
        // i=3,7,11 → full
        CHECK_EQ(tc.layer_types[3], LayerType::FullAttention, "gen12[3]=full");
        CHECK_EQ(tc.layer_types[7], LayerType::FullAttention, "gen12[7]=full");
        CHECK_EQ(tc.layer_types[11], LayerType::FullAttention, "gen12[11]=full");
        CHECK_EQ(tc.layer_types[0], LayerType::LinearAttention, "gen12[0]=linear");
    }

    // ── Vision config ────────────────────────────────────────────────────
    const auto& v = cfg.vision_config;
    CHECK_EQ(v.depth, 24, "vision.depth");
    CHECK_EQ(v.hidden_size, 1024, "vision.hidden_size");
    CHECK_EQ(v.out_hidden_size, 2560, "vision.out_hidden_size");

    std::printf("\n[mlx_config] %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
