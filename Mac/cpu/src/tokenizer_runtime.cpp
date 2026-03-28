#include "header/tokenizer_runtime.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

TokenizerRuntime::TokenizerRuntime(TokenizerRuntimeData runtime_data) : runtime_data_(std::move(runtime_data)) {
    for (const AddedTokenRecord& token : runtime_data_.added_tokens) {
        token_to_id_.emplace(token.content, token.id);
        id_to_token_.emplace(token.id, token.content);
        ordered_tokens_.push_back(token.content);
        protected_tokens_.push_back(token.content);
        added_token_ids_.insert(token.id);
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

    std::sort(protected_tokens_.begin(), protected_tokens_.end(), [](const std::string& left, const std::string& right) {
        if (left.size() != right.size()) {
            return left.size() > right.size();
        }
        return left < right;
    });
    protected_tokens_.erase(std::unique(protected_tokens_.begin(), protected_tokens_.end()), protected_tokens_.end());

    for (std::size_t index = 0; index < runtime_data_.bpe_model.merges.size(); ++index) {
        const BPEMergeRecord& merge = runtime_data_.bpe_model.merges[index];
        merge_ranks_.emplace(MakeMergeKey(merge.left, merge.right), index);
    }

    for (std::size_t byte_value = 0; byte_value < runtime_data_.decoder.byte_level.byte_to_unicode.size(); ++byte_value) {
        unicode_to_byte_.emplace(runtime_data_.decoder.byte_level.byte_to_unicode[byte_value], static_cast<unsigned char>(byte_value));
    }
}

const TokenizerRuntimeData& TokenizerRuntime::runtime_data() const {
    return runtime_data_;
}

std::vector<int> TokenizerRuntime::Encode(const std::string& text) const {
    std::vector<int> token_ids;
    std::size_t offset = 0;

    while (offset < text.size()) {
        const std::string matched = runtime_data_.bpe_model.enabled
            ? MatchLongestProtectedToken(text, offset)
            : MatchLongestToken(text, offset);
        if (!matched.empty()) {
            token_ids.push_back(TokenToId(matched));
            offset += matched.size();
            continue;
        }

        if (!runtime_data_.bpe_model.enabled) {
            throw std::runtime_error("tokenizer runtime cannot encode input at byte offset " + std::to_string(offset));
        }

        const std::size_t next_boundary = FindNextProtectedTokenOffset(text, offset);
        const std::string segment = text.substr(offset, next_boundary - offset);
        const std::vector<int> encoded_segment = EncodeWithBPE(segment);
        token_ids.insert(token_ids.end(), encoded_segment.begin(), encoded_segment.end());
        offset = next_boundary;
    }

    return token_ids;
}

std::string TokenizerRuntime::Decode(const std::vector<int>& token_ids) const {
    std::string text;
    bool first_model_token = true;
    for (int token_id : token_ids) {
        const std::string token = IdToToken(token_id);
        if (added_token_ids_.find(token_id) != added_token_ids_.end()) {
            text += token;
            continue;
        }

        text += DecodeModelToken(token, first_model_token);
        first_model_token = false;
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

std::string TokenizerRuntime::MatchLongestProtectedToken(const std::string& text, std::size_t offset) const {
    for (const std::string& token : protected_tokens_) {
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

std::size_t TokenizerRuntime::FindNextProtectedTokenOffset(const std::string& text, std::size_t offset) const {
    std::size_t next_offset = text.size();
    for (const std::string& token : protected_tokens_) {
        if (token.empty()) {
            continue;
        }
        const std::size_t position = text.find(token, offset);
        if (position != std::string::npos && position < next_offset) {
            next_offset = position;
        }
    }
    return next_offset;
}

std::vector<int> TokenizerRuntime::EncodeWithBPE(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> token_ids;
    const std::vector<std::string> pieces = PreTokenizeText(text);

    for (const std::string& piece : pieces) {
        const std::vector<std::string> symbols = ApplyBPEMerges(EncodePieceToSymbols(piece));
        token_ids.reserve(token_ids.size() + symbols.size());
        for (const std::string& symbol : symbols) {
            const auto iterator = token_to_id_.find(symbol);
            if (iterator != token_to_id_.end()) {
                token_ids.push_back(iterator->second);
                continue;
            }

            if (!runtime_data_.bpe_model.unk_token.empty()) {
                token_ids.push_back(TokenToId(runtime_data_.bpe_model.unk_token));
                continue;
            }

            throw std::runtime_error("tokenizer runtime cannot encode BPE symbol: " + symbol);
        }
    }

    return token_ids;
}

std::vector<std::string> TokenizerRuntime::PreTokenizeText(const std::string& text) const {
    if (!runtime_data_.pre_tokenizer.enabled || !runtime_data_.pre_tokenizer.byte_level.enabled) {
        return {text};
    }

    std::vector<std::string> pieces;
    if (!runtime_data_.pre_tokenizer.byte_level.use_regex) {
        pieces.push_back(text);
    } else {
        std::size_t index = 0;
        while (index < text.size()) {
            const std::size_t piece_start = index;
            while (index < text.size() && std::isspace(static_cast<unsigned char>(text[index])) != 0) {
                ++index;
            }
            if (index == text.size()) {
                pieces.push_back(text.substr(piece_start));
                break;
            }

            if (std::isalpha(static_cast<unsigned char>(text[index])) != 0) {
                while (index < text.size() && std::isalpha(static_cast<unsigned char>(text[index])) != 0) {
                    ++index;
                }
            } else if (std::isdigit(static_cast<unsigned char>(text[index])) != 0) {
                while (index < text.size() && std::isdigit(static_cast<unsigned char>(text[index])) != 0) {
                    ++index;
                }
            } else {
                while (index < text.size() &&
                       std::isspace(static_cast<unsigned char>(text[index])) == 0 &&
                       std::isalnum(static_cast<unsigned char>(text[index])) == 0) {
                    ++index;
                }
            }

            pieces.push_back(text.substr(piece_start, index - piece_start));
        }
    }

    if (!pieces.empty() && runtime_data_.pre_tokenizer.byte_level.add_prefix_space) {
        if (!pieces.front().empty() && std::isspace(static_cast<unsigned char>(pieces.front().front())) == 0) {
            pieces.front().insert(pieces.front().begin(), ' ');
        }
    }

    return pieces;
}

std::vector<std::string> TokenizerRuntime::EncodePieceToSymbols(const std::string& text) const {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());

    const bool use_byte_level = runtime_data_.pre_tokenizer.byte_level.enabled &&
        !runtime_data_.decoder.byte_level.byte_to_unicode.empty();

    if (use_byte_level) {
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text.data());
        for (std::size_t index = 0; index < text.size(); ++index) {
            std::string symbol = runtime_data_.decoder.byte_level.byte_to_unicode[bytes[index]];
            if (!runtime_data_.bpe_model.continuing_subword_prefix.empty() && !symbols.empty()) {
                symbol = runtime_data_.bpe_model.continuing_subword_prefix + symbol;
            }
            if (!runtime_data_.bpe_model.end_of_word_suffix.empty() && index + 1 == text.size()) {
                symbol += runtime_data_.bpe_model.end_of_word_suffix;
            }
            symbols.push_back(std::move(symbol));
        }
        return symbols;
    }

    for (std::size_t index = 0; index < text.size(); ++index) {
        std::string symbol(1, text[index]);
        if (!runtime_data_.bpe_model.continuing_subword_prefix.empty() && !symbols.empty()) {
            symbol = runtime_data_.bpe_model.continuing_subword_prefix + symbol;
        }
        if (!runtime_data_.bpe_model.end_of_word_suffix.empty() && index + 1 == text.size()) {
            symbol += runtime_data_.bpe_model.end_of_word_suffix;
        }
        symbols.push_back(std::move(symbol));
    }

    return symbols;
}

std::vector<std::string> TokenizerRuntime::ApplyBPEMerges(std::vector<std::string> symbols) const {
    if (symbols.size() < 2 || merge_ranks_.empty()) {
        return symbols;
    }

    while (symbols.size() > 1) {
        std::size_t best_index = symbols.size();
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();

        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto iterator = merge_ranks_.find(MakeMergeKey(symbols[index], symbols[index + 1]));
            if (iterator == merge_ranks_.end()) {
                continue;
            }
            if (iterator->second < best_rank) {
                best_rank = iterator->second;
                best_index = index;
            }
        }

        if (best_index == symbols.size()) {
            break;
        }

        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    return symbols;
}

std::string TokenizerRuntime::DecodeModelToken(const std::string& token, bool first_model_token) const {
    std::string decoded = token;
    bool append_space = false;

    if (runtime_data_.decoder.bpe.enabled &&
        !runtime_data_.decoder.bpe.suffix.empty() &&
        EndsWith(decoded, runtime_data_.decoder.bpe.suffix)) {
        decoded.erase(decoded.size() - runtime_data_.decoder.bpe.suffix.size());
        append_space = true;
    }

    if (!first_model_token &&
        !runtime_data_.bpe_model.continuing_subword_prefix.empty() &&
        StartsWith(decoded, runtime_data_.bpe_model.continuing_subword_prefix)) {
        decoded.erase(0, runtime_data_.bpe_model.continuing_subword_prefix.size());
    }

    if (runtime_data_.decoder.byte_level.enabled && !unicode_to_byte_.empty()) {
        decoded = DecodeByteLevelToken(decoded);
    }

    if (append_space) {
        decoded.push_back(' ');
    }

    return decoded;
}

std::string TokenizerRuntime::DecodeByteLevelToken(const std::string& token) const {
    std::string bytes;
    bytes.reserve(token.size());

    std::size_t offset = 0;
    while (offset < token.size()) {
        const std::size_t width = Utf8CodepointWidth(static_cast<unsigned char>(token[offset]));
        const std::string codepoint = token.substr(offset, width);
        const auto iterator = unicode_to_byte_.find(codepoint);
        if (iterator != unicode_to_byte_.end()) {
            bytes.push_back(static_cast<char>(iterator->second));
        } else {
            bytes.append(codepoint);
        }
        offset += width;
    }

    return bytes;
}

std::size_t TokenizerRuntime::Utf8CodepointWidth(unsigned char lead_byte) {
    if ((lead_byte & 0x80u) == 0) {
        return 1;
    }
    if ((lead_byte & 0xE0u) == 0xC0u) {
        return 2;
    }
    if ((lead_byte & 0xF0u) == 0xE0u) {
        return 3;
    }
    if ((lead_byte & 0xF8u) == 0xF0u) {
        return 4;
    }
    return 1;
}

bool TokenizerRuntime::StartsWith(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool TokenizerRuntime::EndsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string TokenizerRuntime::MakeMergeKey(const std::string& left, const std::string& right) {
    return left + '\n' + right;
}