// test_validation.cpp — Phase 4: validate C++ output against Python reference
//
// Usage: ./test_validation <model_dir> <reference_json>
//
// Reads reference.json produced by gen_reference.py and compares:
//   1. Prefill argmax (first generated token)
//   2. Full token sequence (greedy generation)
//   3. Performance timing

#include "models/qwen3_5_py_cpp/modeling.h"
#include "models/qwen3_5_py_cpp/tokenization.h"
#include "utils/json.h"

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

// ── Chat template: non-thinking mode ─────────────────────────────────
// <|im_start|>system\n{system_message}<|im_end|>\n (optional)
// <|im_start|>user\n{message}<|im_end|>\n
// <|im_start|>assistant\n<think>\n\n</think>\n\n
std::vector<int> apply_chat_template_nothink(
    const std::string& user_message,
    const Qwen3_5Tokenizer& tokenizer)
{
    // Special token IDs
    const int IM_START  = 248045;
    const int IM_END    = 248046;
    const int THINK     = 248068;
    const int THINK_END = 248069;

    // Encode role strings and separators explicitly rather than relying on a
    // tokenizer-side chat template. The MLX export may not ship one.
    auto user_tok       = tokenizer.encode("user");
    auto newline_tok    = tokenizer.encode("\n");
    auto msg_tok        = tokenizer.encode(user_message);
    auto assistant_tok  = tokenizer.encode("assistant");
    auto dbl_nl_tok     = tokenizer.encode("\n\n");

    std::vector<int> ids;
    ids.push_back(IM_START);
    ids.insert(ids.end(), user_tok.begin(), user_tok.end());
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.insert(ids.end(), msg_tok.begin(), msg_tok.end());
    ids.push_back(IM_END);
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.push_back(IM_START);
    ids.insert(ids.end(), assistant_tok.begin(), assistant_tok.end());
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.push_back(THINK);
    ids.insert(ids.end(), dbl_nl_tok.begin(), dbl_nl_tok.end());
    ids.push_back(THINK_END);
    ids.insert(ids.end(), dbl_nl_tok.begin(), dbl_nl_tok.end());

    return ids;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <reference_json>\n", argv[0]);
        return 1;
    }

    std::string model_dir = argv[1];
    std::string ref_path  = argv[2];
    std::string config_path = model_dir + "/config.json";
    std::string tokenizer_path = model_dir + "/tokenizer.json";

    const int EOS_IM_END = 248046;

    // ═════════════════════════════════════════════════════════════════════
    // 1. Load reference data
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading reference data...\n");
    auto ref_json = JsonParser::parse_file(ref_path);
    const auto& ref_arr = ref_json.as_array();
    printf("   %zu test cases loaded\n", ref_arr.size());

    // ═════════════════════════════════════════════════════════════════════
    // 2. Load config + tokenizer + model
    // ═════════════════════════════════════════════════════════════════════
    printf("── Loading config...\n");
    Qwen3_5Config config = Qwen3_5Config::from_file(config_path);

    printf("── Loading tokenizer...\n");
    Qwen3_5Tokenizer tokenizer = Qwen3_5Tokenizer::from_file(tokenizer_path);

    printf("── Loading safetensors...\n");
    SafetensorsBundle bundle;
    for (auto& entry : fs::directory_iterator(model_dir)) {
        auto p = entry.path();
        if (p.extension() == ".safetensors") {
            printf("   %s\n", p.filename().c_str());
            bundle.add_file(p.string());
        }
    }

    printf("── Loading model...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    Qwen3_5ForCausalLM model;
    model.load(bundle, "model", config);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("   Model loaded in %.1f ms\n", load_ms);

    // ═════════════════════════════════════════════════════════════════════
    // 3. Validate each test case
    // ═════════════════════════════════════════════════════════════════════
    for (size_t tc = 0; tc < ref_arr.size(); tc++) {
        const auto& ref = ref_arr[tc].as_object();
        std::string prompt_text = ref.at("prompt_text").as_string();
        int ref_argmax = ref.at("prefill_logits_argmax").as_int();

        // Extract reference tokens
        std::vector<int> ref_prompt_tokens;
        for (auto& v : ref.at("prompt_tokens").as_array())
            ref_prompt_tokens.push_back(v.as_int());

        std::vector<int> ref_new_tokens;
        for (auto& v : ref.at("new_tokens").as_array())
            ref_new_tokens.push_back(v.as_int());

        std::string ref_text = ref.at("generated_text").as_string();
        int max_new = static_cast<int>(ref_new_tokens.size());

        printf("\n═══ Test case %zu: \"%s\" ═══\n", tc + 1, prompt_text.c_str());

        // ── 3a. Chat template tokenization check ─────────────────────
        auto cpp_prompt = apply_chat_template_nothink(prompt_text, tokenizer);
        printf("   Prompt tokens (C++): %zu, (Python): %zu\n",
               cpp_prompt.size(), ref_prompt_tokens.size());

        bool tokens_match = (cpp_prompt.size() == ref_prompt_tokens.size());
        if (tokens_match) {
            for (size_t i = 0; i < cpp_prompt.size(); i++) {
                if (cpp_prompt[i] != ref_prompt_tokens[i]) {
                    tokens_match = false;
                    printf("   MISMATCH at pos %zu: C++=%d, Py=%d\n",
                           i, cpp_prompt[i], ref_prompt_tokens[i]);
                    break;
                }
            }
        } else {
            printf("   C++: ");
            for (auto id : cpp_prompt) printf("%d ", id);
            printf("\n   Py:  ");
            for (auto id : ref_prompt_tokens) printf("%d ", id);
            printf("\n");
        }
        CHECK(tokens_match, ("template tokens match: " + prompt_text).c_str());

        // Use Python tokens as input (to isolate model from tokenizer bugs)
        auto& input_ids = ref_prompt_tokens;

        // ── 3b. Prefill argmax check ─────────────────────────────────
        printf("   Prefill (%zu tokens)...\n", input_ids.size());
        Scratch scratch;
        ModelCache cache = model.model.create_cache();
        std::vector<float> logits(model.vocab_size);

        t0 = std::chrono::high_resolution_clock::now();
        model.forward(logits.data(), input_ids.data(),
                      static_cast<int>(input_ids.size()), 1, cache, scratch);
        t1 = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int cpp_argmax = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());

        printf("   Prefill: %.1f ms, argmax C++=%d, Python=%d\n",
               prefill_ms, cpp_argmax, ref_argmax);
        CHECK(cpp_argmax == ref_argmax,
              ("prefill argmax match: " + prompt_text).c_str());

        // ── 3c. Full greedy decode from the existing prefill cache ───
        printf("   Generating (max %d new tokens)...\n", max_new);

        std::vector<int> cpp_new;
        cpp_new.reserve(static_cast<size_t>(max_new));

        int next_token = cpp_argmax;
        cpp_new.push_back(next_token);

        t0 = std::chrono::high_resolution_clock::now();
        for (int step = 1; step < max_new; ++step) {
            model.forward(logits.data(), &next_token, 1, 1, cache, scratch);
            next_token = static_cast<int>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            cpp_new.push_back(next_token);
            if (next_token == EOS_IM_END) {
                break;
            }
        }
        t1 = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double total_ms = prefill_ms + decode_ms;
        std::string cpp_text = tokenizer.decode(cpp_new);

        double tok_per_s = (cpp_new.size() > 1) ?
            (cpp_new.size() - 1) / (decode_ms / 1000.0) : 0.0;

        printf("   Generated %zu tokens in %.1f ms (%.1f tok/s)\n",
               cpp_new.size(), total_ms, tok_per_s);
        printf("   C++ text: \"%s\"\n", cpp_text.c_str());
        printf("   Py  text: \"%s\"\n", ref_text.c_str());

        // Compare token sequences
        bool seq_match = (cpp_new.size() == ref_new_tokens.size());
        int first_diff = -1;
        if (seq_match) {
            for (size_t i = 0; i < cpp_new.size(); i++) {
                if (cpp_new[i] != ref_new_tokens[i]) {
                    seq_match = false;
                    first_diff = static_cast<int>(i);
                    break;
                }
            }
        }

        if (!seq_match) {
            printf("   Token sequence MISMATCH");
            if (first_diff >= 0) {
                printf(" at position %d: C++=%d, Py=%d",
                       first_diff, cpp_new[first_diff], ref_new_tokens[first_diff]);
            } else {
                printf(" (length: C++=%zu, Py=%zu)",
                       cpp_new.size(), ref_new_tokens.size());
            }
            printf("\n");
            printf("   C++ tokens: ");
            for (auto id : cpp_new) printf("%d ", id);
            printf("\n   Py  tokens: ");
            for (auto id : ref_new_tokens) printf("%d ", id);
            printf("\n");
        }
        CHECK(seq_match,
              ("token sequence match: " + prompt_text).c_str());

        // ── 3d. Timing summary ───────────────────────────────────────
        double decode_ms_per_tok = (cpp_new.size() > 1) ?
            decode_ms / (cpp_new.size() - 1) : 0.0;
        printf("   Timing: prefill=%.1fms, decode=%.1fms/tok, total=%.1fms\n",
               prefill_ms, decode_ms_per_tok, total_ms);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Summary
    // ═════════════════════════════════════════════════════════════════════
    printf("\n══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("══════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
