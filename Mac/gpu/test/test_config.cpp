#include "../models/qwen3_5_py_cpp/config.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n  %s:%d\n", msg, __FILE__, __LINE__); ++fail_count; } \
    else { ++pass_count; } \
} while (0)

#define CHECK_EQ(a, b, name)  CHECK((a) == (b), name)
#define CHECK_FEQ(a, b, name) CHECK(std::fabs((a) - (b)) < 1e-9, name)
#define CHECK_STR(a, b, name) CHECK((a) == (b), name)

int main(int argc, char** argv) {
    int pass_count = 0, fail_count = 0;

    // ── Test 1: load real config.json ────────────────────────────────────
    const char* config_path = (argc >= 2) ? argv[1] : nullptr;
    if (!config_path) {
        std::fprintf(stderr, "Usage: test_config <config.json>\n");
        return 1;
    }

    std::printf("Loading config from: %s\n", config_path);
    Qwen3_5Config cfg;
    try {
        cfg = Qwen3_5Config::from_file(config_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load config: %s\n", e.what());
        return 1;
    }

    const auto& t = cfg.text_config;

    // ── Text config fields (Qwen3.5-4B) ─────────────────────────────────
    CHECK_EQ(t.vocab_size,             248320,  "vocab_size");
    CHECK_EQ(t.hidden_size,            2560,    "hidden_size");
    CHECK_EQ(t.intermediate_size,      9216,    "intermediate_size");
    CHECK_EQ(t.num_hidden_layers,      32,      "num_hidden_layers");
    CHECK_EQ(t.num_attention_heads,    16,      "num_attention_heads");
    CHECK_EQ(t.num_key_value_heads,    4,       "num_key_value_heads");
    CHECK_EQ(t.head_dim,               256,     "head_dim");
    CHECK_EQ(t.max_position_embeddings, 262144, "max_position_embeddings");
    CHECK_FEQ(t.rms_norm_eps,          1e-6,    "rms_norm_eps");
    CHECK_STR(t.hidden_act,            "silu",  "hidden_act");
    CHECK_EQ(t.attention_bias,         false,   "attention_bias");
    CHECK_EQ(t.tie_word_embeddings,    true,    "text.tie_word_embeddings");

    // Linear attention fields
    CHECK_EQ(t.linear_conv_kernel_dim,  4,   "linear_conv_kernel_dim");
    CHECK_EQ(t.linear_key_head_dim,     128, "linear_key_head_dim");
    CHECK_EQ(t.linear_value_head_dim,   128, "linear_value_head_dim");
    CHECK_EQ(t.linear_num_key_heads,    16,  "linear_num_key_heads");
    CHECK_EQ(t.linear_num_value_heads,  32,  "linear_num_value_heads");

    // RoPE
    CHECK_STR(t.rope_parameters.rope_type,          "default", "rope_type");
    CHECK_FEQ(t.rope_parameters.rope_theta,          1e7,      "rope_theta");
    CHECK_EQ(t.rope_parameters.mrope_interleaved,    true,     "mrope_interleaved");
    CHECK_FEQ(t.rope_parameters.partial_rotary_factor, 0.25,   "partial_rotary_factor");
    CHECK_EQ((int)t.rope_parameters.mrope_section.size(), 3,   "mrope_section.size");
    CHECK_EQ(t.rope_parameters.mrope_section[0],     11,       "mrope_section[0]");
    CHECK_EQ(t.rope_parameters.mrope_section[1],     11,       "mrope_section[1]");
    CHECK_EQ(t.rope_parameters.mrope_section[2],     10,       "mrope_section[2]");

    // Derived
    CHECK_EQ(t.rotary_dim(),            64,  "rotary_dim (256 * 0.25)");

    // layer_types: 32 layers, pattern [lin, lin, lin, full] × 8
    CHECK_EQ((int)t.layer_types.size(), 32, "layer_types.size");
    for (int i = 0; i < 32; ++i) {
        bool expect_full = ((i + 1) % 4 == 0);
        LayerType expected = expect_full ? LayerType::FullAttention : LayerType::LinearAttention;
        CHECK_EQ(t.layer_types[i], expected, ("layer_types[" + std::to_string(i) + "]").c_str());
    }

    // ── Top-level config ─────────────────────────────────────────────────
    CHECK_EQ(cfg.tie_word_embeddings, true,  "top.tie_word_embeddings");
    CHECK_EQ(cfg.image_token_id,      248056, "image_token_id");
    CHECK_EQ(cfg.video_token_id,      248057, "video_token_id");

    // ── Vision config ────────────────────────────────────────────────────
    const auto& v = cfg.vision_config;
    CHECK_EQ(v.depth,              24,   "vision.depth");
    CHECK_EQ(v.hidden_size,        1024, "vision.hidden_size");
    CHECK_EQ(v.intermediate_size,  4096, "vision.intermediate_size");
    CHECK_EQ(v.num_heads,          16,   "vision.num_heads");
    CHECK_EQ(v.out_hidden_size,    2560, "vision.out_hidden_size");

    // ── Test 2: layer_types generation from full_attention_interval ──────
    {
        std::string json_str = R"({
            "num_hidden_layers": 8,
            "full_attention_interval": 3,
            "hidden_size": 512
        })";
        JsonValue j = JsonParser::parse(json_str);
        auto tc = Qwen3_5TextConfig::from_json(j);
        CHECK_EQ((int)tc.layer_types.size(), 8, "gen: layer_types.size");
        // Pattern with interval=3: layers 0,1 → lin, 2 → full, 3,4 → lin, 5 → full, 6,7 → lin
        // (i+1)%3==0 → i=2,5
        CHECK_EQ(tc.layer_types[0], LayerType::LinearAttention, "gen[0]");
        CHECK_EQ(tc.layer_types[1], LayerType::LinearAttention, "gen[1]");
        CHECK_EQ(tc.layer_types[2], LayerType::FullAttention,   "gen[2]");
        CHECK_EQ(tc.layer_types[3], LayerType::LinearAttention, "gen[3]");
        CHECK_EQ(tc.layer_types[4], LayerType::LinearAttention, "gen[4]");
        CHECK_EQ(tc.layer_types[5], LayerType::FullAttention,   "gen[5]");
        CHECK_EQ(tc.layer_types[6], LayerType::LinearAttention, "gen[6]");
        CHECK_EQ(tc.layer_types[7], LayerType::LinearAttention, "gen[7]");
    }

    // ── Summary ──────────────────────────────────────────────────────────
    std::printf("\n%d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
