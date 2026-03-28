#include "header/runtime_assets.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "header/json_parser.h"

namespace {
std::string RequireString(const JsonValue& object, const std::string& key) {
    return object.at(key).as_string();
}

int RequireInt(const JsonValue& object, const std::string& key) {
    return object.at(key).as_int();
}

std::size_t RequireSize(const JsonValue& object, const std::string& key) {
    return static_cast<std::size_t>(object.at(key).as_int64());
}

std::vector<std::size_t> ParseShape(const JsonValue& shape_value) {
    std::vector<std::size_t> shape;
    for (const JsonValue& value : shape_value.as_array()) {
        shape.push_back(static_cast<std::size_t>(value.as_int64()));
    }
    return shape;
}

TemplateRuntimeData ParseTemplateRuntime(const JsonValue& value) {
    return TemplateRuntimeData{
        RequireString(value, "type"),
        RequireString(value, "im_start"),
        RequireString(value, "im_end"),
        RequireString(value, "think_start"),
        RequireString(value, "think_end"),
        RequireString(value, "default_system_prompt"),
    };
}

BPETokenizerModelData ParseBPETokenizerModel(const JsonValue& value) {
    BPETokenizerModelData model;
    model.enabled = true;
    model.type = RequireString(value, "type");
    model.unk_token = value.contains("unk_token") && !value.at("unk_token").is_null()
        ? value.at("unk_token").as_string()
        : "";
    model.continuing_subword_prefix = value.contains("continuing_subword_prefix") && !value.at("continuing_subword_prefix").is_null()
        ? value.at("continuing_subword_prefix").as_string()
        : "";
    model.end_of_word_suffix = value.contains("end_of_word_suffix") && !value.at("end_of_word_suffix").is_null()
        ? value.at("end_of_word_suffix").as_string()
        : "";

    for (const JsonValue& merge_value : value.at("merges").as_array()) {
        model.merges.push_back(BPEMergeRecord{
            RequireString(merge_value, "left"),
            RequireString(merge_value, "right"),
        });
    }

    return model;
}

PreTokenizerRuntimeData ParsePreTokenizerRuntime(const JsonValue& value) {
    PreTokenizerRuntimeData pre_tokenizer;
    pre_tokenizer.enabled = true;
    pre_tokenizer.type = value.contains("type") && !value.at("type").is_null()
        ? value.at("type").as_string()
        : "";

    if (value.contains("byte_level") && !value.at("byte_level").is_null()) {
        const JsonValue& byte_level = value.at("byte_level");
        pre_tokenizer.byte_level.enabled = byte_level.contains("enabled") ? byte_level.at("enabled").as_bool() : true;
        pre_tokenizer.byte_level.add_prefix_space = byte_level.contains("add_prefix_space")
            ? byte_level.at("add_prefix_space").as_bool()
            : false;
        pre_tokenizer.byte_level.use_regex = byte_level.contains("use_regex")
            ? byte_level.at("use_regex").as_bool()
            : true;
    }

    return pre_tokenizer;
}

DecoderRuntimeData ParseDecoderRuntime(const JsonValue& value) {
    DecoderRuntimeData decoder;
    decoder.enabled = true;
    decoder.type = value.contains("type") && !value.at("type").is_null()
        ? value.at("type").as_string()
        : "";

    if (value.contains("byte_level") && !value.at("byte_level").is_null()) {
        const JsonValue& byte_level = value.at("byte_level");
        decoder.byte_level.enabled = byte_level.contains("enabled") ? byte_level.at("enabled").as_bool() : true;
        decoder.byte_level.add_prefix_space = byte_level.contains("add_prefix_space")
            ? byte_level.at("add_prefix_space").as_bool()
            : false;
        decoder.byte_level.trim_offsets = byte_level.contains("trim_offsets")
            ? byte_level.at("trim_offsets").as_bool()
            : false;
        decoder.byte_level.use_regex = byte_level.contains("use_regex")
            ? byte_level.at("use_regex").as_bool()
            : true;
        if (byte_level.contains("byte_to_unicode") && !byte_level.at("byte_to_unicode").is_null()) {
            for (const JsonValue& mapping : byte_level.at("byte_to_unicode").as_array()) {
                decoder.byte_level.byte_to_unicode.push_back(mapping.as_string());
            }
        }
    }

    if (value.contains("bpe") && !value.at("bpe").is_null()) {
        const JsonValue& bpe = value.at("bpe");
        decoder.bpe.enabled = bpe.contains("enabled") ? bpe.at("enabled").as_bool() : true;
        decoder.bpe.suffix = bpe.contains("suffix") && !bpe.at("suffix").is_null()
            ? bpe.at("suffix").as_string()
            : "";
    }

    return decoder;
}

std::string ResolveRelativePath(const std::filesystem::path& base_dir, const std::string& relative_path) {
    return (base_dir / relative_path).lexically_normal().string();
}
}

const TensorRecord& ManifestData::FindTensor(const std::string& name) const {
    for (const TensorRecord& tensor : tensors) {
        if (tensor.name == name) {
            return tensor;
        }
    }

    throw std::runtime_error("missing tensor in manifest: " + name);
}

ManifestData ManifestLoader::LoadFromFile(const std::string& manifest_path) {
    const JsonValue root = JsonParser::ParseFile(manifest_path);
    const std::filesystem::path manifest_file = std::filesystem::path(manifest_path).lexically_normal();
    const std::filesystem::path manifest_dir = manifest_file.parent_path();

    ManifestData manifest;
    manifest.format = RequireString(root, "format");
    manifest.format_version = RequireInt(root, "format_version");
    manifest.model_id = root.contains("model_id") && !root.at("model_id").is_null() ? root.at("model_id").as_string() : "";
    manifest.source_dir = root.contains("source_dir") ? root.at("source_dir").as_string() : "";
    manifest.export_dtype = RequireString(root, "export_dtype");
    manifest.manifest_dir = manifest_dir.string();
    manifest.tokenizer_runtime_file = root.contains("tokenizer_runtime_file")
        ? ResolveRelativePath(manifest_dir, root.at("tokenizer_runtime_file").as_string())
        : "";
    manifest.config = root.contains("config") ? root.at("config") : JsonValue();
    manifest.generation_config = root.contains("generation_config") ? root.at("generation_config") : JsonValue();

    for (const JsonValue& tensor_value : root.at("tensors").as_array()) {
        TensorRecord tensor;
        tensor.name = RequireString(tensor_value, "name");
        tensor.file = ResolveRelativePath(manifest_dir, RequireString(tensor_value, "file"));
        tensor.dtype = RequireString(tensor_value, "dtype");
        tensor.shape = ParseShape(tensor_value.at("shape"));
        tensor.byte_size = RequireSize(tensor_value, "byte_size");
        tensor.source_shard = RequireString(tensor_value, "source_shard");
        manifest.tensors.push_back(std::move(tensor));
    }

    return manifest;
}

TokenizerRuntimeData TokenizerRuntimeLoader::LoadFromFile(const std::string& runtime_path) {
    const JsonValue root = JsonParser::ParseFile(runtime_path);

    TokenizerRuntimeData tokenizer_runtime;
    tokenizer_runtime.format = RequireString(root, "format");
    tokenizer_runtime.format_version = RequireInt(root, "format_version");
    tokenizer_runtime.tokenizer_class = RequireString(root, "tokenizer_class");
    tokenizer_runtime.vocab_size = RequireInt(root, "vocab_size");
    tokenizer_runtime.model_max_length = RequireSize(root, "model_max_length");
    tokenizer_runtime.chat_template = root.contains("chat_template") && !root.at("chat_template").is_null()
        ? root.at("chat_template").as_string()
        : "";
    tokenizer_runtime.template_runtime = ParseTemplateRuntime(root.at("template_runtime"));
    tokenizer_runtime.special_tokens_map = root.contains("special_tokens_map") ? root.at("special_tokens_map") : JsonValue();
    tokenizer_runtime.special_token_ids = root.contains("special_token_ids") ? root.at("special_token_ids") : JsonValue();

    for (const JsonValue& token_value : root.at("added_tokens").as_array()) {
        tokenizer_runtime.added_tokens.push_back(AddedTokenRecord{
            RequireInt(token_value, "id"),
            RequireString(token_value, "content"),
            token_value.at("special").as_bool(),
            token_value.at("single_word").as_bool(),
            token_value.at("lstrip").as_bool(),
            token_value.at("rstrip").as_bool(),
            token_value.at("normalized").as_bool(),
        });
    }

    for (const JsonValue& vocab_value : root.at("vocab").as_array()) {
        tokenizer_runtime.vocab.push_back(VocabEntry{
            RequireInt(vocab_value, "id"),
            RequireString(vocab_value, "token"),
        });
    }

    tokenizer_runtime.bpe_model = root.contains("bpe_model") && !root.at("bpe_model").is_null()
        ? ParseBPETokenizerModel(root.at("bpe_model"))
        : BPETokenizerModelData{};
    tokenizer_runtime.pre_tokenizer = root.contains("pre_tokenizer") && !root.at("pre_tokenizer").is_null()
        ? ParsePreTokenizerRuntime(root.at("pre_tokenizer"))
        : PreTokenizerRuntimeData{};
    tokenizer_runtime.decoder = root.contains("decoder") && !root.at("decoder").is_null()
        ? ParseDecoderRuntime(root.at("decoder"))
        : DecoderRuntimeData{};

    return tokenizer_runtime;
}

Qwen3TemplateBuilder::Qwen3TemplateBuilder(TemplateRuntimeData template_runtime)
    : template_runtime_(std::move(template_runtime)) {}

std::string Qwen3TemplateBuilder::BuildPrompt(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string& system_prompt_override) const {
    std::ostringstream prompt;
    bool has_system_message = false;

    for (const ChatMessage& message : messages) {
        if (message.role == "system") {
            has_system_message = true;
        }
        prompt << template_runtime_.im_start << message.role << '\n';
        prompt << message.content << template_runtime_.im_end << '\n';
    }

    if (!has_system_message) {
        const std::string& system_prompt = system_prompt_override.empty()
            ? template_runtime_.default_system_prompt
            : system_prompt_override;
        std::ostringstream prefixed_prompt;
        prefixed_prompt << template_runtime_.im_start << "system\n";
        prefixed_prompt << system_prompt << template_runtime_.im_end << '\n';
        prefixed_prompt << prompt.str();
        prompt.str(std::string());
        prompt.clear();
        prompt << prefixed_prompt.str();
    }

    if (add_generation_prompt) {
        prompt << template_runtime_.im_start << "assistant\n";
        if (enable_thinking) {
            prompt << template_runtime_.think_start << '\n';
        }
    }

    return prompt.str();
}