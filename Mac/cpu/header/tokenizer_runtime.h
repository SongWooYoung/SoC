#ifndef TOKENIZER_RUNTIME_H
#define TOKENIZER_RUNTIME_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
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
    std::string MatchLongestProtectedToken(const std::string& text, std::size_t offset) const;
    std::string MatchLongestToken(const std::string& text, std::size_t offset) const;
    std::size_t FindNextProtectedTokenOffset(const std::string& text, std::size_t offset) const;
    std::vector<int> EncodeWithBPE(const std::string& text) const;
    std::vector<std::string> PreTokenizeText(const std::string& text) const;
    std::vector<std::string> EncodePieceToSymbols(const std::string& text) const;
    std::vector<std::string> ApplyBPEMerges(std::vector<std::string> symbols) const;
    std::string DecodeModelToken(const std::string& token, bool first_model_token) const;
    std::string DecodeByteLevelToken(const std::string& token) const;
    static std::size_t Utf8CodepointWidth(unsigned char lead_byte);
    static bool StartsWith(const std::string& value, const std::string& prefix);
    static bool EndsWith(const std::string& value, const std::string& suffix);
    static std::string MakeMergeKey(const std::string& left, const std::string& right);

    TokenizerRuntimeData runtime_data_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<std::string> ordered_tokens_;
    std::vector<std::string> protected_tokens_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    std::unordered_map<std::string, unsigned char> unicode_to_byte_;
    std::unordered_set<int> added_token_ids_;
};

#endif