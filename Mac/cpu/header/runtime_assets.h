#ifndef RUNTIME_ASSETS_H
#define RUNTIME_ASSETS_H

#include <cstddef>
#include <string>
#include <vector>

#include "header/json_value.h"

struct TensorRecord {
    std::string name;
    std::string file;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t byte_size;
    std::string source_shard;
};

struct ManifestData {
    std::string format;
    int format_version;
    std::string model_id;
    std::string source_dir;
    std::string export_dtype;
    std::string manifest_dir;
    std::string tokenizer_runtime_file;
    JsonValue config;
    JsonValue generation_config;
    std::vector<TensorRecord> tensors;

    const TensorRecord& FindTensor(const std::string& name) const;
};

struct AddedTokenRecord {
    int id;
    std::string content;
    bool special;
    bool single_word;
    bool lstrip;
    bool rstrip;
    bool normalized;
};

struct TemplateRuntimeData {
    std::string type;
    std::string im_start;
    std::string im_end;
    std::string think_start;
    std::string think_end;
    std::string default_system_prompt;
};

struct VocabEntry {
    int id;
    std::string token;
};

struct TokenizerRuntimeData {
    std::string format;
    int format_version;
    std::string tokenizer_class;
    int vocab_size;
    std::size_t model_max_length;
    std::string chat_template;
    TemplateRuntimeData template_runtime;
    std::vector<AddedTokenRecord> added_tokens;
    std::vector<VocabEntry> vocab;
    JsonValue special_tokens_map;
    JsonValue special_token_ids;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

class ManifestLoader {
public:
    static ManifestData LoadFromFile(const std::string& manifest_path);
};

class TokenizerRuntimeLoader {
public:
    static TokenizerRuntimeData LoadFromFile(const std::string& runtime_path);
};

class Qwen3TemplateBuilder {
public:
    explicit Qwen3TemplateBuilder(TemplateRuntimeData template_runtime);

    std::string BuildPrompt(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt,
        bool enable_thinking,
        const std::string& system_prompt_override = "") const;

private:
    TemplateRuntimeData template_runtime_;
};

#endif