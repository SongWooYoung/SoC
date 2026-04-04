// test_vision.cpp — Smoke test for Qwen3.5 vision model loading and inference
// Usage: ./test_vision <model_dir>
// model_dir should contain config.json and model.safetensors-*.safetensors

#include "models/qwen3_5_py_cpp/vision.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int pass_count = 0;
static int fail_count = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) { ++pass_count; }                                         \
        else { ++fail_count; fprintf(stderr, "FAIL: %s\n", msg); }         \
    } while (0)

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]);
        return 1;
    }

    std::string model_dir = argv[1];
    std::string config_path = model_dir + "/config.json";

    // ═════════════════════════════════════════════════════════════════════
    // 1. Load config
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading config...\n");
    Qwen3_5Config config = Qwen3_5Config::from_file(config_path);
    CHECK(config.vision_config.depth == 24, "vision depth should be 24");
    CHECK(config.vision_config.hidden_size == 1024, "vision hidden should be 1024");
    CHECK(config.vision_config.intermediate_size == 4096, "vision intermediate should be 4096");
    CHECK(config.vision_config.num_heads == 16, "vision num_heads should be 16");
    CHECK(config.vision_config.out_hidden_size == 2560, "vision out_hidden should be 2560");
    CHECK(config.vision_config.spatial_merge_size == 2, "spatial_merge_size should be 2");
    CHECK(config.image_token_id == 248056, "image_token_id should be 248056");
    printf("   Vision config OK: depth=%d, hidden=%d, out=%d\n",
           config.vision_config.depth,
           config.vision_config.hidden_size,
           config.vision_config.out_hidden_size);

    // ═════════════════════════════════════════════════════════════════════
    // 2. Load safetensors
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading safetensors...\n");
    SafetensorsBundle bundle;
    for (auto& entry : fs::directory_iterator(model_dir)) {
        auto p = entry.path();
        if (p.extension() == ".safetensors") {
            printf("   Loading %s...\n", p.filename().c_str());
            bundle.add_file(p.string());
        }
    }

    CHECK(bundle.has("model.visual.patch_embed.proj.weight"), "has patch_embed.proj.weight");
    CHECK(bundle.has("model.visual.pos_embed.weight"), "has pos_embed.weight");
    CHECK(bundle.has("model.visual.blocks.0.attn.qkv.weight"), "has block 0 attn");
    CHECK(bundle.has("model.visual.blocks.23.mlp.linear_fc1.weight"), "has block 23 mlp");
    CHECK(bundle.has("model.visual.merger.linear_fc1.weight"), "has merger fc1");
    CHECK(bundle.has("model.visual.merger.norm.weight"), "has merger norm");

    {
        auto& pe = bundle.get("model.visual.patch_embed.proj.weight");
        CHECK(pe.shape.size() == 5, "patch_embed weight is 5D");
        CHECK(pe.shape[0] == 1024 && pe.shape[1] == 3 &&
              pe.shape[2] == 2 && pe.shape[3] == 16 && pe.shape[4] == 16,
              "patch_embed shape [1024,3,2,16,16]");
    }
    {
        auto& pos = bundle.get("model.visual.pos_embed.weight");
        CHECK(pos.shape[0] == 2304 && pos.shape[1] == 1024,
              "pos_embed shape [2304,1024]");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 3. Test VisionRotaryEmbedding
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing VisionRotaryEmbedding...\n");
    {
        int head_dim = config.vision_config.hidden_size / config.vision_config.num_heads;  // 64
        int rope_dim = head_dim / 2;  // 32
        CHECK(head_dim == 64, "vision head_dim should be 64");

        Qwen3_5VisionRotaryEmbedding rope;
        rope.init(rope_dim);
        CHECK(rope.dim == 32, "rope dim should be 32");
        CHECK(static_cast<int>(rope.inv_freq.size()) == 16, "inv_freq size should be 16");

        auto freqs = rope.compute_freqs(10);
        CHECK(std::abs(freqs[0] - 0.0f) < 1e-6f, "freqs[0,0] should be 0");
        CHECK(freqs[1 * 16 + 0] > 0.0f, "freqs[1,0] should be positive");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 4. Load full vision model
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading vision model...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    Qwen3_5VisionModel vision;
    vision.load(bundle, "model.visual", config.vision_config);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("   Vision model loaded in %.1f ms\n", load_ms);

    CHECK(static_cast<int>(vision.blocks.size()) == 24, "24 vision blocks");
    CHECK(vision.config.hidden_size == 1024, "loaded hidden_size 1024");
    CHECK(vision.merger.out_hidden_size == 2560, "merger out_hidden 2560");
    CHECK(vision.merger.merged_dim == 4096, "merger merged_dim 4096");

    // ═════════════════════════════════════════════════════════════════════
    // 5. Test position embeddings
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing position embeddings...\n");
    {
        // Simulate a 224×224 image: T=1, H=14, W=14 after patching
        // (224 / 16 = 14 patches per side)
        std::vector<std::array<int, 3>> grid_thw = {{1, 14, 14}};
        int total_tokens = 1 * 14 * 14;  // 196

        auto pos = vision.fast_pos_embed_interpolate(grid_thw);
        CHECK(static_cast<int>(pos.size()) == total_tokens * 1024,
              "pos_embed size 196*1024");

        // Check that embeddings are non-zero
        float sum = 0.0f;
        for (auto v : pos) sum += std::abs(v);
        CHECK(sum > 0.0f, "pos embeddings are non-zero");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 6. Test rotary position embeddings
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing rot_pos_emb...\n");
    {
        std::vector<std::array<int, 3>> grid_thw = {{1, 14, 14}};
        int total = 1 * 14 * 14;
        int head_dim = 64;

        auto emb = vision.rot_pos_emb(grid_thw);
        CHECK(static_cast<int>(emb.size()) == total * head_dim,
              "rot_pos_emb size 196*64");

        // First token (0,0) should have zero freqs
        bool first_ok = true;
        for (int d = 0; d < head_dim; d++) {
            if (std::abs(emb[d]) > 1e-6f) { first_ok = false; break; }
        }
        CHECK(first_ok, "first token (0,0) has zero freqs");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 7. Test vision forward (small synthetic input)
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing vision forward (small)...\n");
    {
        // Use 4×4 spatial → 16 tokens → 4 merged tokens
        // T=1, H=4, W=4 (tiny, just for shape check)
        std::vector<std::array<int, 3>> grid_thw = {{1, 4, 4}};
        int total_patches = 1 * 4 * 4;  // 16
        int merge = config.vision_config.spatial_merge_size;
        int total_merged = total_patches / (merge * merge);  // 4

        // Create dummy pixel input
        // Each patch is C_in * temporal * patch_h * patch_w = 3 * 2 * 16 * 16 = 1536
        int patch_vol = 3 * 2 * 16 * 16;
        std::vector<float> pixels(total_patches * patch_vol, 0.01f);

        int out_dim = config.vision_config.out_hidden_size;
        std::vector<float> out(total_merged * out_dim);

        Scratch scratch;
        t0 = std::chrono::high_resolution_clock::now();
        int nm = vision.forward(out.data(), pixels.data(), grid_thw, scratch);
        t1 = std::chrono::high_resolution_clock::now();
        double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("   Vision forward (16 patches → %d merged): %.1f ms\n", nm, fwd_ms);

        CHECK(nm == 4, "merged tokens should be 4");

        // Check output is finite and non-zero
        float out_sum = 0.0f;
        bool all_finite = true;
        for (auto v : out) {
            out_sum += std::abs(v);
            if (!std::isfinite(v)) all_finite = false;
        }
        CHECK(all_finite, "vision output all finite");
        CHECK(out_sum > 0.0f, "vision output non-zero");
        printf("   Output L1 norm: %.4f\n", out_sum / out.size());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 8. Load full VLM (ForConditionalGeneration)
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading ForConditionalGeneration...\n");
    t0 = std::chrono::high_resolution_clock::now();

    Qwen3_5ForConditionalGeneration vlm;
    vlm.load(bundle, "model", config);

    t1 = std::chrono::high_resolution_clock::now();
    load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("   VLM loaded in %.1f ms\n", load_ms);

    CHECK(vlm.vocab_size == 248320, "vlm vocab_size");
    CHECK(vlm.hidden_size == 2560, "vlm hidden_size");
    CHECK(vlm.tie_embeddings == true, "vlm tie_embeddings");

    // ═════════════════════════════════════════════════════════════════════
    // 9. Test text-only forward through VLM
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing VLM text-only forward...\n");
    {
        std::vector<int> ids = {1, 2, 3};
        Scratch scratch;
        auto cache = vlm.create_cache();
        std::vector<float> logits(vlm.vocab_size);

        t0 = std::chrono::high_resolution_clock::now();
        vlm.forward(logits.data(), ids.data(), 3, 1,
                     nullptr, nullptr, {}, {}, cache, scratch);
        t1 = std::chrono::high_resolution_clock::now();
        double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("   VLM text-only prefill (3 tokens): %.1f ms\n", fwd_ms);

        // Check logits are finite
        bool finite = true;
        for (auto v : logits) if (!std::isfinite(v)) { finite = false; break; }
        CHECK(finite, "text-only logits finite");

        // Check argmax is reasonable
        int argmax = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        CHECK(argmax >= 0 && argmax < vlm.vocab_size, "argmax in range");
        printf("   argmax = %d\n", argmax);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 10. Test decode step through VLM
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing VLM decode step...\n");
    {
        // Continue from previous cache (3 tokens already processed)
        // Use a fresh cache for isolation
        Scratch scratch;
        auto cache = vlm.create_cache();
        std::vector<float> logits(vlm.vocab_size);

        // Prefill
        std::vector<int> ids = {100, 200};
        vlm.forward(logits.data(), ids.data(), 2, 1,
                     nullptr, nullptr, {}, {}, cache, scratch);

        // Decode
        int next = 300;
        vlm.forward_decode(logits.data(), &next, cache, scratch);

        bool finite = true;
        for (auto v : logits) if (!std::isfinite(v)) { finite = false; break; }
        CHECK(finite, "decode logits finite");
        CHECK(cache.seq_offset == 3, "seq_offset after 2+1=3");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Summary
    // ═════════════════════════════════════════════════════════════════════
    printf("\n══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("══════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
