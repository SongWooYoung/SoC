#include "../models/qwen3_5/tokenization.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static int pass_count = 0, fail_count = 0;

static void check_ids(const std::string& name, const std::vector<int>& got, const std::vector<int>& expected) {
    bool ok = (got == expected);
    if (!ok) {
        std::fprintf(stderr, "FAIL [encode] %s\n  expected: [", name.c_str());
        for (size_t i = 0; i < expected.size(); i++) std::fprintf(stderr, "%s%d", i?",":"", expected[i]);
        std::fprintf(stderr, "]\n  got:      [");
        for (size_t i = 0; i < got.size(); i++) std::fprintf(stderr, "%s%d", i?",":"", got[i]);
        std::fprintf(stderr, "]\n");
        ++fail_count;
    } else {
        ++pass_count;
    }
}

static void check_decode(const std::string& name, const Qwen3_5Tokenizer& tok,
                         const std::vector<int>& ids, const std::string& expected) {
    std::string got = tok.decode(ids);
    if (got != expected) {
        std::fprintf(stderr, "FAIL [decode] %s\n  expected: \"%s\"\n  got:      \"%s\"\n",
                     name.c_str(), expected.c_str(), got.c_str());
        ++fail_count;
    } else {
        ++pass_count;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test_tokenizer <tokenizer.json>\n");
        return 1;
    }

    std::printf("Loading tokenizer from: %s\n", argv[1]);
    Qwen3_5Tokenizer tok;
    try {
        tok = Qwen3_5Tokenizer::from_file(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load tokenizer: %s\n", e.what());
        return 1;
    }

    std::printf("Vocab size: %d\n\n", tok.vocab_size());

    // ── Encode tests (ground truth from Python tokenizers library) ────────

    check_ids("Hello, world!",
              tok.encode("Hello, world!"),
              {9419, 11, 1814, 0});

    check_ids("The quick brown fox",
              tok.encode("The quick brown fox jumps over the lazy dog."),
              {760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13});

    check_ids("Korean",
              tok.encode("안녕하세요"),
              {148924, 154982, 88005});

    check_ids("Hello + CJK + emoji",
              tok.encode("Hello 世界! 🌍"),
              {9419, 220, 96748, 0, 10838, 234, 235});

    check_ids("multiple spaces",
              tok.encode("  multiple   spaces  "),
              {220, 5081, 256, 12258, 256});

    check_ids("single char",
              tok.encode("a"),
              {64});

    check_ids("empty",
              tok.encode(""),
              {});

    check_ids("code",
              tok.encode("def fibonacci(n):"),
              {727, 73111, 1393, 1590});

    check_ids("arithmetic",
              tok.encode("1+2=3"),
              {16, 10, 17, 28, 18});

    check_ids("contraction",
              tok.encode("I've got it"),
              {40, 2908, 2597, 424});

    // ── Decode tests ─────────────────────────────────────────────────────

    check_decode("Hello, world!",    tok, {9419, 11, 1814, 0},                       "Hello, world!");
    check_decode("Korean",           tok, {148924, 154982, 88005},                     "안녕하세요");
    check_decode("single char",      tok, {64},                                        "a");
    check_decode("empty",            tok, {},                                          "");
    check_decode("code",             tok, {727, 73111, 1393, 1590},                    "def fibonacci(n):");
    check_decode("contraction",      tok, {40, 2908, 2597, 424},                       "I've got it");
    check_decode("Hello + CJK + emoji", tok, {9419, 220, 96748, 0, 10838, 234, 235},  "Hello 世界! 🌍");

    // ── Roundtrip tests ──────────────────────────────────────────────────
    {
        std::vector<std::string> roundtrip_texts = {
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "안녕하세요",
            "def fibonacci(n):",
            "1+2=3",
            "I've got it",
        };
        for (auto& text : roundtrip_texts) {
            auto ids = tok.encode(text);
            auto decoded = tok.decode(ids);
            if (decoded != text) {
                std::fprintf(stderr, "FAIL [roundtrip] \"%s\" -> \"%s\"\n", text.c_str(), decoded.c_str());
                ++fail_count;
            } else {
                ++pass_count;
            }
        }
    }

    std::printf("\n%d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
