// test_modeling.cpp — Smoke test for Qwen3.5 model loading and inference
// Usage: ./test_modeling <model_dir>
// model_dir should contain config.json and model.safetensors-*.safetensors

#include "models/qwen3_5_py_cpp/modeling.h"
#include "models/qwen3_5_py_cpp/tokenization.h"

#include <chrono>
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
    std::string tokenizer_path = model_dir + "/tokenizer.json";

    // ═════════════════════════════════════════════════════════════════════
    // 1. Load config
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading config...\n");
    Qwen3_5Config config = Qwen3_5Config::from_file(config_path);
    CHECK(config.text_config.hidden_size == 2560,
          "hidden_size should be 2560");
    CHECK(config.text_config.num_hidden_layers == 32,
          "num_hidden_layers should be 32");
    printf("   Config OK: hidden=%d, layers=%d, vocab=%d\n",
           config.text_config.hidden_size,
           config.text_config.num_hidden_layers,
           config.text_config.vocab_size);

    // ═════════════════════════════════════════════════════════════════════
    // 2. Load safetensors
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading safetensors...\n");
    SafetensorsBundle bundle;

    // Find and load all safetensors shards
    for (auto& entry : fs::directory_iterator(model_dir)) {
        auto p = entry.path();
        if (p.extension() == ".safetensors") {
            printf("   Loading %s...\n", p.filename().c_str());
            bundle.add_file(p.string());
        }
    }

    CHECK(bundle.has("model.language_model.embed_tokens.weight"),
          "should have embed_tokens");
    CHECK(bundle.has("model.language_model.layers.0.input_layernorm.weight"),
          "should have layer 0 input_layernorm");
    CHECK(bundle.has("model.language_model.layers.3.self_attn.q_proj.weight"),
          "should have layer 3 self_attn (full attention)");
    CHECK(bundle.has("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
          "should have layer 0 linear_attn (GDN)");

    // Check shapes
    {
        auto& emb = bundle.get("model.language_model.embed_tokens.weight");
        CHECK(emb.shape[0] == 248320 && emb.shape[1] == 2560,
              "embed_tokens shape [248320, 2560]");
        CHECK(emb.dtype == DType::BF16, "embed_tokens dtype BF16");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 3. Load model
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading model weights...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    Qwen3_5ForCausalLM model;
    model.load(bundle, "model", config);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("   Model loaded in %.1f ms\n", load_ms);

    CHECK(model.vocab_size == 248320, "vocab_size");
    CHECK(model.hidden_size == 2560, "hidden_size");
    CHECK(model.tie_embeddings == true, "tie_word_embeddings");

    // ═════════════════════════════════════════════════════════════════════
    // 4. Test single forward pass (prefill)
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing forward pass (3 tokens)...\n");
    // Use simple token IDs: [1, 2, 3]  (arbitrary, just for shape check)
    std::vector<int> test_ids = {1, 2, 3};
    Scratch scratch;
    ModelCache cache = model.model.create_cache();

    std::vector<float> logits(model.vocab_size);
    t0 = std::chrono::high_resolution_clock::now();
    model.forward(logits.data(), test_ids.data(),
                  static_cast<int>(test_ids.size()), 1, cache, scratch);
    t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Check logits are not all zero
    float sum = 0.0f;
    for (int i = 0; i < model.vocab_size; i++)
        sum += std::abs(logits[i]);
    CHECK(sum > 0.0f, "logits should not be all zero");

    // Check cache state
    CHECK(cache.seq_offset == 3, "seq_offset should be 3 after prefill");

    // Find top prediction
    int top_id = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    printf("   Prefill done in %.1f ms, top token: %d, logits sum: %.1f\n",
           prefill_ms, top_id, sum);

    // ═════════════════════════════════════════════════════════════════════
    // 5. Test decode step
    // ═════════════════════════════════════════════════════════════════════
    printf("── Testing decode step...\n");
    std::vector<float> decode_logits(model.vocab_size);
    t0 = std::chrono::high_resolution_clock::now();
    model.forward(decode_logits.data(), &top_id, 1, 1, cache, scratch);
    t1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CHECK(cache.seq_offset == 4, "seq_offset should be 4 after decode");

    int decode_top = static_cast<int>(
        std::max_element(decode_logits.begin(), decode_logits.end())
        - decode_logits.begin());
    printf("   Decode step in %.1f ms, top token: %d\n",
           decode_ms, decode_top);

    // ═════════════════════════════════════════════════════════════════════
    // 6. Full generation test (if tokenizer available)
    // ═════════════════════════════════════════════════════════════════════
    if (fs::exists(tokenizer_path)) {
        printf("── Testing generation with tokenizer...\n");
        Qwen3_5Tokenizer tokenizer = Qwen3_5Tokenizer::from_file(tokenizer_path);

        std::string prompt = "Hello";
        auto tokens = tokenizer.encode(prompt);
        printf("   Prompt: \"%s\" → %zu tokens\n",
               prompt.c_str(), tokens.size());

        t0 = std::chrono::high_resolution_clock::now();
        auto output = model.generate(tokens, 10);
        t1 = std::chrono::high_resolution_clock::now();
        double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::string decoded = tokenizer.decode(output);
        printf("   Generated %zu tokens in %.1f ms: \"%s\"\n",
               output.size(), gen_ms, decoded.c_str());

        CHECK(output.size() > tokens.size(),
              "should generate at least one new token");
    }

    // ═════════════════════════════════════════════════════════════════════
    printf("\n════════════════════════════════════════\n");
    printf("  PASS: %d   FAIL: %d\n", pass_count, fail_count);
    printf("════════════════════════════════════════\n");
    return fail_count > 0 ? 1 : 0;
}
