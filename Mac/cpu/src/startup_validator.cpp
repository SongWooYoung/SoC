#include "header/startup_validator.h"

#include <stdexcept>

Qwen3Config StartupValidator::ValidateManifest(const ManifestData& manifest) {
    if (manifest.format != "soc.cpp.llm_export") {
        throw std::runtime_error("manifest format must be soc.cpp.llm_export");
    }
    if (manifest.format_version < 2) {
        throw std::runtime_error("manifest format_version must be at least 2");
    }
    if (!manifest.config.is_object()) {
        throw std::runtime_error("manifest config must be a JSON object");
    }
    if (manifest.tensors.empty()) {
        throw std::runtime_error("manifest must contain at least one tensor");
    }
    if (manifest.tokenizer_runtime_file.empty()) {
        throw std::runtime_error("manifest must reference tokenizer_runtime_file");
    }

    return Qwen3Config::FromJson(manifest.config);
}

void StartupValidator::ValidateTokenizerRuntime(const TokenizerRuntimeData& tokenizer_runtime) {
    if (tokenizer_runtime.format != "soc.cpp.tokenizer_runtime") {
        throw std::runtime_error("tokenizer runtime format must be soc.cpp.tokenizer_runtime");
    }
    if (tokenizer_runtime.format_version != 1) {
        throw std::runtime_error("tokenizer runtime format_version must be 1");
    }
    if (tokenizer_runtime.vocab_size <= 0) {
        throw std::runtime_error("tokenizer runtime vocab_size must be positive");
    }
    if (tokenizer_runtime.template_runtime.type != "qwen3") {
        throw std::runtime_error("tokenizer template runtime type must be qwen3");
    }
    if (tokenizer_runtime.template_runtime.im_start.empty() || tokenizer_runtime.template_runtime.im_end.empty()) {
        throw std::runtime_error("tokenizer runtime must define chat boundary tokens");
    }
    if (FindTokenIdByContent(tokenizer_runtime, tokenizer_runtime.template_runtime.im_start) < 0) {
        throw std::runtime_error("tokenizer runtime is missing im_start token id");
    }
    if (FindTokenIdByContent(tokenizer_runtime, tokenizer_runtime.template_runtime.im_end) < 0) {
        throw std::runtime_error("tokenizer runtime is missing im_end token id");
    }
    if (FindTokenIdByContent(tokenizer_runtime, tokenizer_runtime.template_runtime.think_start) < 0) {
        throw std::runtime_error("tokenizer runtime is missing think_start token id");
    }
    if (FindTokenIdByContent(tokenizer_runtime, tokenizer_runtime.template_runtime.think_end) < 0) {
        throw std::runtime_error("tokenizer runtime is missing think_end token id");
    }
    if (tokenizer_runtime.bpe_model.enabled) {
        if (tokenizer_runtime.bpe_model.type != "bpe") {
            throw std::runtime_error("tokenizer bpe_model type must be bpe");
        }
        if (tokenizer_runtime.bpe_model.merges.empty()) {
            throw std::runtime_error("tokenizer bpe_model must define merges");
        }
        if (!tokenizer_runtime.bpe_model.unk_token.empty() &&
            !ContainsTokenContent(tokenizer_runtime, tokenizer_runtime.bpe_model.unk_token)) {
            throw std::runtime_error("tokenizer bpe_model unk_token must exist in tokenizer vocab");
        }
    }
    if (tokenizer_runtime.pre_tokenizer.enabled && tokenizer_runtime.pre_tokenizer.byte_level.enabled) {
        if (tokenizer_runtime.pre_tokenizer.type.empty()) {
            throw std::runtime_error("tokenizer pre_tokenizer must define a type when enabled");
        }
    }
    if (tokenizer_runtime.decoder.enabled && tokenizer_runtime.decoder.byte_level.enabled &&
        !tokenizer_runtime.decoder.byte_level.byte_to_unicode.empty() &&
        tokenizer_runtime.decoder.byte_level.byte_to_unicode.size() != 256) {
        throw std::runtime_error("tokenizer decoder byte_to_unicode mapping must contain 256 entries when provided");
    }
}

void StartupValidator::ValidateManifestAndTokenizer(const ManifestData& manifest, const TokenizerRuntimeData& tokenizer_runtime) {
    const Qwen3Config config = ValidateManifest(manifest);
    ValidateTokenizerRuntime(tokenizer_runtime);

    if (tokenizer_runtime.vocab_size <= 0) {
        throw std::runtime_error("tokenizer vocab_size must be positive");
    }
    if (static_cast<std::size_t>(tokenizer_runtime.vocab_size) > config.vocab_size) {
        throw std::runtime_error("tokenizer vocab_size must not exceed qwen3 config vocab_size");
    }
    int max_token_id = -1;
    for (const AddedTokenRecord& token : tokenizer_runtime.added_tokens) {
        if (token.id > max_token_id) {
            max_token_id = token.id;
        }
    }
    for (const VocabEntry& token : tokenizer_runtime.vocab) {
        if (token.id > max_token_id) {
            max_token_id = token.id;
        }
    }
    if (max_token_id < 0 || static_cast<std::size_t>(max_token_id) >= config.vocab_size) {
        throw std::runtime_error("tokenizer token ids must fit within qwen3 config vocab_size");
    }
}

int StartupValidator::FindTokenIdByContent(const TokenizerRuntimeData& tokenizer_runtime, const std::string& token_content) {
    for (const AddedTokenRecord& token : tokenizer_runtime.added_tokens) {
        if (token.content == token_content) {
            return token.id;
        }
    }
    for (const VocabEntry& token : tokenizer_runtime.vocab) {
        if (token.token == token_content) {
            return token.id;
        }
    }
    return -1;
}

bool StartupValidator::ContainsTokenContent(const TokenizerRuntimeData& tokenizer_runtime, const std::string& token_content) {
    return FindTokenIdByContent(tokenizer_runtime, token_content) >= 0;
}