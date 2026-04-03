#pragma once

#include "utils/json.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// ── UTF-8 helpers ───────────────────────────────────────────────────────────

namespace utf8 {

// Decode one codepoint from UTF-8, advance pos. Returns 0xFFFD on error.
inline uint32_t decode(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return 0xFFFD;
    uint8_t b0 = static_cast<uint8_t>(s[pos]);
    if (b0 < 0x80) { pos += 1; return b0; }
    if ((b0 & 0xE0) == 0xC0) {
        if (pos + 1 >= s.size()) { pos = s.size(); return 0xFFFD; }
        uint32_t cp = ((b0 & 0x1F) << 6) | (static_cast<uint8_t>(s[pos+1]) & 0x3F);
        pos += 2; return cp;
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (pos + 2 >= s.size()) { pos = s.size(); return 0xFFFD; }
        uint32_t cp = ((b0 & 0x0F) << 12) | ((static_cast<uint8_t>(s[pos+1]) & 0x3F) << 6)
                     | (static_cast<uint8_t>(s[pos+2]) & 0x3F);
        pos += 3; return cp;
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (pos + 3 >= s.size()) { pos = s.size(); return 0xFFFD; }
        uint32_t cp = ((b0 & 0x07) << 18) | ((static_cast<uint8_t>(s[pos+1]) & 0x3F) << 12)
                     | ((static_cast<uint8_t>(s[pos+2]) & 0x3F) << 6)
                     | (static_cast<uint8_t>(s[pos+3]) & 0x3F);
        pos += 4; return cp;
    }
    pos += 1; return 0xFFFD;
}

inline std::string encode(uint32_t cp) {
    std::string r;
    if (cp < 0x80)        { r += static_cast<char>(cp); }
    else if (cp < 0x800)  { r += static_cast<char>(0xC0 | (cp >> 6));
                            r += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000){ r += static_cast<char>(0xE0 | (cp >> 12));
                            r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            r += static_cast<char>(0x80 | (cp & 0x3F)); }
    else                  { r += static_cast<char>(0xF0 | (cp >> 18));
                            r += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                            r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            r += static_cast<char>(0x80 | (cp & 0x3F)); }
    return r;
}

// Byte length of UTF-8 sequence starting at s[pos]
inline int seq_len(const std::string& s, size_t pos) {
    if (pos >= s.size()) return 0;
    uint8_t b = static_cast<uint8_t>(s[pos]);
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

} // namespace utf8

// ── Unicode property classification (simplified) ────────────────────────────

namespace uniprops {

inline bool is_letter(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp < 0xC0) return false;
    // Latin Extended, Greek, Cyrillic, etc
    if (cp >= 0x00C0 && cp <= 0x024F) return true;  // Latin Extended
    if (cp >= 0x0370 && cp <= 0x03FF) return true;  // Greek
    if (cp >= 0x0400 && cp <= 0x04FF) return true;  // Cyrillic
    if (cp >= 0x0500 && cp <= 0x052F) return true;  // Cyrillic Supplement
    if (cp >= 0x0530 && cp <= 0x058F) return true;  // Armenian
    if (cp >= 0x0590 && cp <= 0x05FF) return true;  // Hebrew
    if (cp >= 0x0600 && cp <= 0x06FF) return true;  // Arabic
    if (cp >= 0x0900 && cp <= 0x097F) return true;  // Devanagari
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;  // Thai
    if (cp >= 0x1100 && cp <= 0x11FF) return true;  // Hangul Jamo
    if (cp >= 0x3000 && cp <= 0x303F) return false; // CJK Symbols (punctuation)
    if (cp >= 0x3040 && cp <= 0x309F) return true;  // Hiragana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;  // Katakana
    if (cp >= 0x3100 && cp <= 0x312F) return true;  // Bopomofo
    if (cp >= 0x3130 && cp <= 0x318F) return true;  // Hangul Compat Jamo
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;  // CJK Unified Ext A
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;  // CJK Unified
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;  // Hangul Syllables
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;  // CJK Compat
    if (cp >= 0xFB50 && cp <= 0xFDFF) return true;  // Arabic Pres Forms A
    if (cp >= 0x10000 && cp <= 0x1007F) return true; // Linear B Syllabary
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true; // CJK Unified Ext B
    if (cp >= 0x2A700 && cp <= 0x2CEAF) return true; // CJK Unified Ext C/D/E
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return true; // CJK Unified Ext F
    if (cp >= 0x30000 && cp <= 0x3134F) return true; // CJK Unified Ext G
    return false;
}

inline bool is_mark(uint32_t cp) {
    // Combining marks: Mn, Mc, Me
    if (cp >= 0x0300 && cp <= 0x036F) return true;  // Combining Diacritical
    if (cp >= 0x0591 && cp <= 0x05C7) return true;  // Hebrew accents
    if (cp >= 0x0610 && cp <= 0x061A) return true;  // Arabic
    if (cp >= 0x064B && cp <= 0x065F) return true;  // Arabic
    if (cp >= 0x0900 && cp <= 0x0903) return true;  // Devanagari
    if (cp >= 0xFE20 && cp <= 0xFE2F) return true;  // Combining Half Marks
    return false;
}

inline bool is_number(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    // Fullwidth digits, other numeral systems
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    if (cp >= 0x0660 && cp <= 0x0669) return true;  // Arabic-Indic
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;  // Extended Arabic-Indic
    return false;
}

inline bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v'
        || cp == 0x00A0 || cp == 0x2000 || cp == 0x2001 || cp == 0x2002
        || cp == 0x2003 || cp == 0x2004 || cp == 0x2005 || cp == 0x2006
        || cp == 0x2007 || cp == 0x2008 || cp == 0x2009 || cp == 0x200A
        || cp == 0x200B || cp == 0x2028 || cp == 0x2029 || cp == 0x202F
        || cp == 0x205F || cp == 0x3000 || cp == 0xFEFF;
}

inline bool is_newline(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

} // namespace uniprops

// ── GPT-2 Byte-level BPE pre-tokenizer regex ───────────────────────────────
// Pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

namespace pretokenize {

// Read codepoints from a UTF-8 string
struct Codepoints {
    std::vector<uint32_t> cps;
    std::vector<size_t> byte_offsets; // byte offset of each codepoint
    size_t total_bytes;

    static Codepoints from_utf8(const std::string& s) {
        Codepoints result;
        size_t pos = 0;
        while (pos < s.size()) {
            result.byte_offsets.push_back(pos);
            result.cps.push_back(utf8::decode(s, pos));
        }
        result.total_bytes = s.size();
        return result;
    }

    std::string substr(const std::string& s, size_t cp_start, size_t cp_end) const {
        size_t b_start = byte_offsets[cp_start];
        size_t b_end = (cp_end < cps.size()) ? byte_offsets[cp_end] : total_bytes;
        return s.substr(b_start, b_end - b_start);
    }
};

// Check contraction suffix (case insensitive): 's 't 're 've 'm 'll 'd
inline int match_contraction(const std::vector<uint32_t>& cps, size_t i) {
    if (i >= cps.size()) return 0;
    uint32_t c = cps[i];
    if (c != '\'' && c != 0x2019) return 0; // ' or right single quote
    if (i + 1 >= cps.size()) return 0;
    uint32_t c1 = cps[i+1] | 0x20; // to lowercase
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (i + 2 >= cps.size()) return 0;
    uint32_t c2 = cps[i+2] | 0x20;
    if (c1 == 'r' && c2 == 'e') return 3;
    if (c1 == 'v' && c2 == 'e') return 3;
    if (c1 == 'l' && c2 == 'l') return 3;
    return 0;
}

inline std::vector<std::string> split(const std::string& text) {
    // Implements the GPT-4 / Qwen pre-tokenizer regex:
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
    // | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    // Alternatives are tried in order; first match wins.
    if (text.empty()) return {};

    auto data = Codepoints::from_utf8(text);
    const auto& cps = data.cps;
    size_t n = cps.size();
    std::vector<std::string> pieces;
    size_t i = 0;

    while (i < n) {
        // Alt 1: Contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
        {
            int clen = match_contraction(cps, i);
            if (clen > 0) {
                pieces.push_back(data.substr(text, i, i + clen));
                i += clen;
                continue;
            }
        }

        // Alt 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        {
            size_t j = i;
            uint32_t c = cps[j];
            if (!uniprops::is_newline(c) && !uniprops::is_letter(c) && !uniprops::is_number(c)) {
                j++;
            }
            size_t letter_start = j;
            while (j < n && (uniprops::is_letter(cps[j]) || uniprops::is_mark(cps[j]))) {
                j++;
            }
            if (j > letter_start) {
                pieces.push_back(data.substr(text, i, j));
                i = j;
                continue;
            }
        }

        // Alt 3: \p{N}
        if (uniprops::is_number(cps[i])) {
            pieces.push_back(data.substr(text, i, i + 1));
            i++;
            continue;
        }

        // Alt 4: ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            size_t j = i;
            // Optional leading space
            if (j < n && cps[j] == ' ') j++;
            size_t sym_start = j;
            while (j < n && !uniprops::is_whitespace(cps[j]) && !uniprops::is_letter(cps[j])
                   && !uniprops::is_mark(cps[j]) && !uniprops::is_number(cps[j])) {
                j++;
            }
            if (j > sym_start) {
                // Trailing newlines
                while (j < n && uniprops::is_newline(cps[j])) j++;
                pieces.push_back(data.substr(text, i, j));
                i = j;
                continue;
            }
        }

        // Alt 5: \s*[\r\n]+
        {
            size_t j = i;
            while (j < n && uniprops::is_whitespace(cps[j]) && !uniprops::is_newline(cps[j])) j++;
            if (j < n && uniprops::is_newline(cps[j])) {
                while (j < n && uniprops::is_newline(cps[j])) j++;
                pieces.push_back(data.substr(text, i, j));
                i = j;
                continue;
            }
        }

        // Alt 6: \s+(?!\S) — whitespace NOT followed by non-whitespace
        if (uniprops::is_whitespace(cps[i])) {
            size_t j = i;
            while (j < n && uniprops::is_whitespace(cps[j])) j++;
            // Try greedy first, then backtrack for (?!\S) lookahead
            // (?!\S) succeeds when next char is whitespace or end-of-string
            size_t end = j;
            while (end > i + 1) {
                // Check if char at position 'end' is NOT \S (i.e. is whitespace or end)
                if (end >= n || uniprops::is_whitespace(cps[end])) break;
                end--;
            }
            if (end >= n || uniprops::is_whitespace(cps[end]) || end == n) {
                // (?!\S) succeeded at position end
                if (end > i) {
                    pieces.push_back(data.substr(text, i, end));
                    i = end;
                    continue;
                }
            }

            // Alt 7: \s+ (fallback, consume all whitespace)
            pieces.push_back(data.substr(text, i, j));
            i = j;
            continue;
        }

        // Fallback: single codepoint
        pieces.push_back(data.substr(text, i, i + 1));
        i++;
    }

    return pieces;
}

} // namespace pretokenize

// ── Byte-to-Unicode mapping (GPT-2 style) ──────────────────────────────────

namespace byte_encoding {

inline void init_tables(uint32_t byte_to_unicode[256], std::unordered_map<uint32_t, uint8_t>& unicode_to_byte) {
    // Printable ranges that map identity: !-~ (33-126), ¡-¬ (161-172), ®-ÿ (174-255)
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            byte_to_unicode[b] = static_cast<uint32_t>(b);
        } else {
            byte_to_unicode[b] = static_cast<uint32_t>(256 + n);
            n++;
        }
    }
    for (int b = 0; b < 256; b++) {
        unicode_to_byte[byte_to_unicode[b]] = static_cast<uint8_t>(b);
    }
}

// Convert raw bytes to BPE-alphabet string
inline std::string bytes_to_bpe_str(const std::string& raw, const uint32_t byte_to_unicode[256]) {
    std::string result;
    for (uint8_t b : raw) {
        result += utf8::encode(byte_to_unicode[b]);
    }
    return result;
}

// Convert BPE-alphabet string back to raw bytes
inline std::string bpe_str_to_bytes(const std::string& bpe, const std::unordered_map<uint32_t, uint8_t>& unicode_to_byte) {
    std::string result;
    size_t pos = 0;
    while (pos < bpe.size()) {
        uint32_t cp = utf8::decode(bpe, pos);
        auto it = unicode_to_byte.find(cp);
        if (it != unicode_to_byte.end()) {
            result += static_cast<char>(it->second);
        }
    }
    return result;
}

} // namespace byte_encoding

// ── Tokenizer ───────────────────────────────────────────────────────────────

class Qwen3_5Tokenizer {
public:
    static Qwen3_5Tokenizer from_file(const std::string& path) {
        Qwen3_5Tokenizer tok;
        JsonValue root = JsonParser::parse_file(path);
        tok.load(root);
        return tok;
    }

    std::vector<int> encode(const std::string& text) const {
        if (text.empty()) return {};

        std::vector<int> ids;

        // Check for added tokens first (protected tokens)
        size_t pos = 0;
        while (pos < text.size()) {
            // Try to match longest added token at current position
            int best_len = 0;
            int best_id = -1;
            for (auto& [token_str, token_id] : added_token_map_) {
                if (text.compare(pos, token_str.size(), token_str) == 0) {
                    if (static_cast<int>(token_str.size()) > best_len) {
                        best_len = static_cast<int>(token_str.size());
                        best_id = token_id;
                    }
                }
            }
            if (best_len > 0) {
                // Encode any text before this added token
                // (shouldn't happen if we process left-to-right)
                ids.push_back(best_id);
                pos += best_len;
                continue;
            }

            // Find next added token
            size_t next_added = text.size();
            for (auto& [token_str, _] : added_token_map_) {
                size_t found = text.find(token_str, pos);
                if (found != std::string::npos && found < next_added) {
                    next_added = found;
                }
            }

            // BPE-encode the text between pos and next_added
            std::string chunk = text.substr(pos, next_added - pos);
            auto chunk_ids = encode_chunk(chunk);
            ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
            pos = next_added;
        }

        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string bpe_str;
        for (int id : ids) {
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                bpe_str += it->second;
            } else {
                auto ait = added_id_to_token_.find(id);
                if (ait != added_id_to_token_.end()) {
                    // Added tokens: convert current bpe_str to bytes first
                    std::string result = byte_encoding::bpe_str_to_bytes(bpe_str, unicode_to_byte_);
                    bpe_str.clear();
                    result += ait->second;
                    bpe_str.clear();
                    // Actually we need to accumulate differently for mixed
                    // Let's just decode everything at the end
                    bpe_str += ait->second; // added tokens are literal
                }
            }
        }
        // Convert BPE unicode string back to raw bytes
        // But added tokens are already raw text in the output...
        // We need to handle added vs normal tokens differently
        return decode_bpe_mixed(ids);
    }

    int vocab_size() const { return static_cast<int>(token_to_id_.size() + added_token_map_.size()); }

private:
    // Vocab
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;

    // Merges
    std::unordered_map<std::string, int> merge_ranks_; // "left\nright" → rank

    // Added tokens (special tokens)
    std::unordered_map<std::string, int> added_token_map_;
    std::unordered_map<int, std::string> added_id_to_token_;
    std::unordered_map<int, bool> added_is_special_;

    // Byte encoding
    uint32_t byte_to_unicode_[256]{};
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;

    void load(const JsonValue& root) {
        byte_encoding::init_tables(byte_to_unicode_, unicode_to_byte_);

        // Load model section
        const auto& model = root.at("model");

        // Vocab
        const auto& vocab = model.at("vocab").as_object();
        for (auto& [token, id_val] : vocab) {
            int id = id_val.as_int();
            token_to_id_[token] = id;
            id_to_token_[id] = token;
        }

        // Merges
        const auto& merges = model.at("merges").as_array();
        for (size_t i = 0; i < merges.size(); i++) {
            const std::string& merge_str = merges[i].as_string();
            merge_ranks_[merge_str] = static_cast<int>(i);
        }

        // Added tokens
        if (auto* added = root.find("added_tokens")) {
            for (auto& at : added->as_array()) {
                int id = at.at("id").as_int();
                std::string content = at.at("content").as_string();
                bool special = false;
                if (auto* sp = at.find("special")) special = sp->as_bool();
                added_token_map_[content] = id;
                added_id_to_token_[id] = content;
                added_is_special_[id] = special;
            }
        }
    }

    std::vector<int> encode_chunk(const std::string& text) const {
        if (text.empty()) return {};

        std::vector<int> all_ids;

        // Pre-tokenize with regex
        auto pieces = pretokenize::split(text);

        for (auto& piece : pieces) {
            // Convert piece bytes to BPE-alphabet string
            std::string bpe_input = byte_encoding::bytes_to_bpe_str(piece, byte_to_unicode_);

            // Split into individual BPE-alphabet characters (UTF-8 codepoints)
            std::vector<std::string> symbols;
            size_t pos = 0;
            while (pos < bpe_input.size()) {
                int len = utf8::seq_len(bpe_input, pos);
                symbols.push_back(bpe_input.substr(pos, len));
                pos += len;
            }

            // Apply BPE merges
            apply_bpe(symbols);

            // Look up each symbol in vocab
            for (auto& sym : symbols) {
                auto it = token_to_id_.find(sym);
                if (it != token_to_id_.end()) {
                    all_ids.push_back(it->second);
                }
                // If not found, skip (shouldn't happen with byte-level BPE)
            }
        }

        return all_ids;
    }

    void apply_bpe(std::vector<std::string>& symbols) const {
        if (symbols.size() <= 1) return;

        while (true) {
            // Find the merge with lowest rank among all adjacent pairs
            int best_rank = INT32_MAX;
            size_t best_pos = 0;

            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                std::string key = symbols[i] + " " + symbols[i+1];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                }
            }

            if (best_rank == INT32_MAX) break; // No more merges

            // Apply the merge at best_pos
            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);

            if (symbols.size() <= 1) break;
        }
    }

    std::string decode_bpe_mixed(const std::vector<int>& ids) const {
        std::string result;
        std::string bpe_buf; // accumulate BPE tokens

        for (int id : ids) {
            auto ait = added_id_to_token_.find(id);
            if (ait != added_id_to_token_.end()) {
                // Flush BPE buffer
                if (!bpe_buf.empty()) {
                    result += byte_encoding::bpe_str_to_bytes(bpe_buf, unicode_to_byte_);
                    bpe_buf.clear();
                }
                result += ait->second;
            } else {
                auto it = id_to_token_.find(id);
                if (it != id_to_token_.end()) {
                    bpe_buf += it->second;
                }
            }
        }
        // Flush remaining
        if (!bpe_buf.empty()) {
            result += byte_encoding::bpe_str_to_bytes(bpe_buf, unicode_to_byte_);
        }
        return result;
    }
};
