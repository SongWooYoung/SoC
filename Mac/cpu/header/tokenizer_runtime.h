#ifndef TOKENIZER_RUNTIME_H
#define TOKENIZER_RUNTIME_H

#include <string>
#include <unordered_map>
#include <vector>

#include "header/runtime_assets.h"

class TokenizerRuntime {
public:
    explicit TokenizerRuntime(TokenizerRuntimeData runtime_data);

    const TokenizerRuntimeData& runtime_data() const;
    std::vector<int> Encode(const std::string& text) const;
    std::string Decode(const std::vector<int>& token_ids) const;
    int TokenToId(const std::string& token) const;
    std::string IdToToken(int token_id) const;

private:
    std::string MatchLongestToken(const std::string& text, std::size_t offset) const;

    TokenizerRuntimeData runtime_data_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<std::string> ordered_tokens_;
};

#endif