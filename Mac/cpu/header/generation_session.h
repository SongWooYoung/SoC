#ifndef GENERATION_SESSION_H
#define GENERATION_SESSION_H

#include <optional>
#include <string>
#include <vector>

#include "header/kv_cache.h"
#include "header/qwen_causal_lm.h"
#include "header/sampler.h"
#include "header/tokenizer_runtime.h"

struct GenerationResult {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string generated_text;
};

class GenerationSession {
public:
    GenerationSession(QwenCausalLM model, TokenizerRuntime tokenizer, Sampler sampler, std::size_t max_sequence_length = 256);

    std::vector<int> Prefill(const std::string& prompt);
    int DecodeNextToken(const std::vector<int>& token_ids);
    GenerationResult Generate(const std::string& prompt, std::size_t max_new_tokens, int eos_token_id = -1);

private:
    Tensor TokenIdsToTensor(const std::vector<int>& token_ids) const;
    void ResetCache();

    QwenCausalLM model_;
    TokenizerRuntime tokenizer_;
    Sampler sampler_;
    std::optional<TensorKVCache> cache_;
};

#endif