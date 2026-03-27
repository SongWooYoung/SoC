#include "header/generation_session.h"

#include <stdexcept>

GenerationSession::GenerationSession(QwenCausalLM model, TokenizerRuntime tokenizer, Sampler sampler, std::size_t max_sequence_length)
    : model_(std::move(model)), tokenizer_(std::move(tokenizer)), sampler_(std::move(sampler)) {
    if (model_.num_layers() != 0) {
        cache_.emplace(model_.num_layers(), 1, model_.num_key_value_heads(), model_.head_dim(), max_sequence_length);
    }
}

std::vector<int> GenerationSession::Prefill(const std::string& prompt) {
    return tokenizer_.Encode(prompt);
}

int GenerationSession::DecodeNextToken(const std::vector<int>& token_ids) {
    if (token_ids.empty()) {
        throw std::runtime_error("decode requires at least one token id");
    }

    if (!cache_.has_value()) {
        const Tensor token_tensor = TokenIdsToTensor(token_ids);
        const Tensor logits = model_.ForwardLogits(token_tensor, 0);
        return sampler_.SampleFromLogits(logits, 0, token_ids.size() - 1);
    }

    const std::size_t required_prefix = token_ids.size() - 1;
    if (cache_->length(0) > required_prefix) {
        ResetCache();
    }
    if (cache_->length(0) < required_prefix) {
        ResetCache();
        if (required_prefix != 0) {
            const std::vector<int> prefix(token_ids.begin(), token_ids.end() - 1);
            const Tensor prefix_tensor = TokenIdsToTensor(prefix);
            static_cast<void>(model_.ForwardLogitsCached(prefix_tensor, *cache_, 0));
        }
    }

    const std::vector<int> last_token = {token_ids.back()};
    const Tensor last_token_tensor = TokenIdsToTensor(last_token);
    const Tensor logits = model_.ForwardLogitsCached(last_token_tensor, *cache_, required_prefix);
    return sampler_.SampleFromLogits(logits, 0, 0);
}

GenerationResult GenerationSession::Generate(const std::string& prompt, std::size_t max_new_tokens, int eos_token_id) {
    GenerationResult result;
    ResetCache();
    result.prompt_token_ids = Prefill(prompt);

    std::vector<int> running_token_ids = result.prompt_token_ids;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        const int token_id = DecodeNextToken(running_token_ids);
        result.generated_token_ids.push_back(token_id);
        running_token_ids.push_back(token_id);

        if (eos_token_id >= 0 && token_id == eos_token_id) {
            break;
        }
    }

    result.generated_text = tokenizer_.Decode(result.generated_token_ids);
    return result;
}

Tensor GenerationSession::TokenIdsToTensor(const std::vector<int>& token_ids) const {
    return Tensor(
        Storage::FromOwnedCopy(token_ids.data(), token_ids.size() * sizeof(int32_t)),
        DType::Int32,
        {1, token_ids.size()});
}

void GenerationSession::ResetCache() {
    if (!cache_.has_value()) {
        return;
    }
    cache_->ClearSequence(0);
}