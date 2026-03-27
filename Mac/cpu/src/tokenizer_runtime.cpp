#include "header/tokenizer_runtime.h"

#include <algorithm>
#include <stdexcept>

TokenizerRuntime::TokenizerRuntime(TokenizerRuntimeData runtime_data) : runtime_data_(std::move(runtime_data)) {
    for (const AddedTokenRecord& token : runtime_data_.added_tokens) {
        token_to_id_.emplace(token.content, token.id);
        id_to_token_.emplace(token.id, token.content);
        ordered_tokens_.push_back(token.content);
    }

    for (const VocabEntry& token : runtime_data_.vocab) {
        token_to_id_.emplace(token.token, token.id);
        id_to_token_.emplace(token.id, token.token);
        ordered_tokens_.push_back(token.token);
    }

    std::sort(ordered_tokens_.begin(), ordered_tokens_.end(), [](const std::string& left, const std::string& right) {
        if (left.size() != right.size()) {
            return left.size() > right.size();
        }
        return left < right;
    });

    ordered_tokens_.erase(std::unique(ordered_tokens_.begin(), ordered_tokens_.end()), ordered_tokens_.end());
}

const TokenizerRuntimeData& TokenizerRuntime::runtime_data() const {
    return runtime_data_;
}

std::vector<int> TokenizerRuntime::Encode(const std::string& text) const {
    std::vector<int> token_ids;
    std::size_t offset = 0;

    while (offset < text.size()) {
        const std::string matched = MatchLongestToken(text, offset);
        if (matched.empty()) {
            throw std::runtime_error("tokenizer runtime cannot encode input at byte offset " + std::to_string(offset));
        }

        token_ids.push_back(TokenToId(matched));
        offset += matched.size();
    }

    return token_ids;
}

std::string TokenizerRuntime::Decode(const std::vector<int>& token_ids) const {
    std::string text;
    for (int token_id : token_ids) {
        text += IdToToken(token_id);
    }
    return text;
}

int TokenizerRuntime::TokenToId(const std::string& token) const {
    const auto iterator = token_to_id_.find(token);
    if (iterator == token_to_id_.end()) {
        throw std::runtime_error("unknown token content: " + token);
    }
    return iterator->second;
}

std::string TokenizerRuntime::IdToToken(int token_id) const {
    const auto iterator = id_to_token_.find(token_id);
    if (iterator == id_to_token_.end()) {
        throw std::runtime_error("unknown token id: " + std::to_string(token_id));
    }
    return iterator->second;
}

std::string TokenizerRuntime::MatchLongestToken(const std::string& text, std::size_t offset) const {
    for (const std::string& token : ordered_tokens_) {
        if (token.empty()) {
            continue;
        }
        if (offset + token.size() > text.size()) {
            continue;
        }
        if (text.compare(offset, token.size(), token) == 0) {
            return token;
        }
    }

    return std::string();
}