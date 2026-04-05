#include "models/qwen3_5_mlx/language.h"
#include "models/qwen3_5_py_cpp/tokenization.h"
#include "utils/json.h"

#include <mlx/io.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace {

constexpr int kEosImEnd = 248046;

struct QuantizationConfig {
    int group_size = 64;
    int bits = 8;
    std::string mode = "affine";
};

using TensorMap = std::unordered_map<std::string, mx::array>;

bool parse_bool_env(const char* value, bool default_value = false) {
    if (value == nullptr) {
        return default_value;
    }
    const std::string text(value);
    return !(text == "0" || text == "false" || text == "off" || text == "no");
}

std::vector<int> apply_chat_template_nothink(
    const std::string& user_message,
    const Qwen3_5Tokenizer& tokenizer) {
    const int im_start = 248045;
    const int im_end = 248046;
    const int think = 248068;
    const int think_end = 248069;

    auto user_tok = tokenizer.encode("user");
    auto newline_tok = tokenizer.encode("\n");
    auto msg_tok = tokenizer.encode(user_message);
    auto assistant_tok = tokenizer.encode("assistant");
    auto dbl_nl_tok = tokenizer.encode("\n\n");

    std::vector<int> ids;
    ids.push_back(im_start);
    ids.insert(ids.end(), user_tok.begin(), user_tok.end());
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.insert(ids.end(), msg_tok.begin(), msg_tok.end());
    ids.push_back(im_end);
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.push_back(im_start);
    ids.insert(ids.end(), assistant_tok.begin(), assistant_tok.end());
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.push_back(think);
    ids.insert(ids.end(), dbl_nl_tok.begin(), dbl_nl_tok.end());
    ids.push_back(think_end);
    ids.insert(ids.end(), dbl_nl_tok.begin(), dbl_nl_tok.end());
    return ids;
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c))
                        << std::dec << std::setfill(' ');
                } else {
                    out << c;
                }
                break;
        }
    }
    return out.str();
}

std::string json_array_ints(const std::vector<int>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << values[i];
    }
    out << "]";
    return out.str();
}

std::string json_number(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

std::string json_stage_map(
    const std::unordered_map<std::string, qwen3_5_mlx::stage_trace::StageStats>& stage_map) {
    std::vector<std::string> names;
    names.reserve(stage_map.size());
    for (const auto& [name, _stats] : stage_map) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());

    std::ostringstream out;
    out << "{";
    for (size_t index = 0; index < names.size(); ++index) {
        const auto& name = names[index];
        const auto& stats = stage_map.at(name);
        if (index != 0) {
            out << ", ";
        }
        out << "\"" << json_escape(name) << "\": {"
            << "\"calls\": " << stats.calls << ", "
            << "\"dispatch_ms\": " << json_number(stats.dispatch_ms) << ", "
            << "\"sync_ms\": " << json_number(stats.sync_ms)
            << "}";
    }
    out << "}";
    return out.str();
}

std::string json_prompt_trace(const qwen3_5_mlx::stage_trace::PromptTrace& trace) {
    std::ostringstream out;
    out << "{\n";
    out << "        \"prefill\": " << json_stage_map(trace.prefill) << ",\n";
    out << "        \"decode\": " << json_stage_map(trace.decode) << "\n";
    out << "      }";
    return out.str();
}

int sample_argmax_with_trace(const mx::array& logits) {
    using namespace qwen3_5_mlx::stage_trace;

    mark_call("sampler_sync(argmax/item)");
    auto dispatch_start = std::chrono::steady_clock::now();
    auto current = mx::argmax(logits, -1);
    add_dispatch_ms("sampler_sync(argmax/item)", elapsed_ms(dispatch_start));

    auto sync_start = std::chrono::steady_clock::now();
    mx::eval(current);
    mx::synchronize();
    int token = current.item<int>();
    add_sync_ms("sampler_sync(argmax/item)", elapsed_ms(sync_start));
    return token;
}

QuantizationConfig load_quant_config(const std::string& config_path) {
    QuantizationConfig qc;
    auto root = JsonParser::parse_file(config_path);
    const JsonValue* quant = root.find("quantization");
    if (!quant) {
        quant = root.find("quantization_config");
    }
    if (quant) {
        if (auto* value = quant->find("group_size")) qc.group_size = value->as_int();
        if (auto* value = quant->find("bits")) qc.bits = value->as_int();
        if (auto* value = quant->find("mode")) qc.mode = value->as_string();
    }
    return qc;
}

TensorMap load_weight_map(const std::string& model_dir) {
    std::vector<fs::path> shard_paths;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        auto path = entry.path();
        if (path.extension() == ".safetensors") {
            shard_paths.push_back(path);
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());

    TensorMap weights;
    for (const auto& shard_path : shard_paths) {
        auto [arrays, _metadata] = mx::load_safetensors(shard_path.string());
        for (auto& [name, array] : arrays) {
            weights.insert_or_assign(name, std::move(array));
        }
    }
    return weights;
}

const mx::array& require_array(
    const TensorMap& weights,
    const std::vector<std::string>& names) {
    for (const auto& name : names) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            return it->second;
        }
    }

    std::ostringstream message;
    message << "Missing tensor. Tried:";
    for (const auto& name : names) {
        message << " " << name;
    }
    throw std::runtime_error(message.str());
}

std::optional<mx::array> find_optional_array(
    const TensorMap& weights,
    const std::vector<std::string>& names) {
    for (const auto& name : names) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            return it->second;
        }
    }
    return std::nullopt;
}

mx::array load_dense(
    const TensorMap& weights,
    const std::vector<std::string>& names,
    bool reorder_conv = false) {
    auto tensor = require_array(weights, names);
    if (reorder_conv && tensor.ndim() == 3 && tensor.shape(2) != 1) {
        return mx::transpose(tensor, {0, 2, 1});
    }
    return tensor;
}

qwen3_5_mlx::mlx_helpers::TensorParam load_param(
    const TensorMap& weights,
    const std::vector<std::string>& base_names,
    const QuantizationConfig& quant_config,
    bool reorder_conv = false) {
    for (const auto& base_name : base_names) {
        auto weight_it = weights.find(base_name + ".weight");
        if (weight_it == weights.end()) {
            continue;
        }

        qwen3_5_mlx::mlx_helpers::TensorParam param;
        param.weight = weight_it->second;
        if (reorder_conv && param.weight.ndim() == 3 && param.weight.shape(2) != 1) {
            param.weight = mx::transpose(param.weight, {0, 2, 1});
        }
        if (auto scales = find_optional_array(weights, {base_name + ".scales"}); scales.has_value()) {
            param.scales = *scales;
            param.biases = find_optional_array(weights, {base_name + ".biases"});
            param.group_size = quant_config.group_size;
            param.bits = quant_config.bits;
            param.mode = quant_config.mode;
        }
        return param;
    }

    std::ostringstream message;
    message << "Missing weight parameter. Tried bases:";
    for (const auto& base_name : base_names) {
        message << " " << base_name;
    }
    throw std::runtime_error(message.str());
}

qwen3_5_mlx::LanguageModel load_language_model(
    const TensorMap& weights,
    const Qwen3_5Config& config,
    const QuantizationConfig& quant_config) {
    using namespace qwen3_5_mlx;

    LanguageModel language_model(config.text_config, config);
    auto& text_model = language_model.model;
    const auto& text_cfg = config.text_config;

    text_model.config = text_cfg;
    text_model.norm_eps = text_cfg.rms_norm_eps;
    text_model.ssm_idx = 0;
    text_model.fa_idx = std::max(0, default_full_attention_interval(text_cfg) - 1);
    text_model.embed_w = load_param(
        weights,
        {
            "language_model.model.embed_tokens",
            "model.language_model.embed_tokens",
        },
        quant_config);
    text_model.norm_w = load_dense(
        weights,
        {
            "language_model.model.norm.weight",
            "model.language_model.norm.weight",
        });
    text_model.layers.resize(static_cast<size_t>(text_cfg.num_hidden_layers));

    for (int layer_index = 0; layer_index < text_cfg.num_hidden_layers; ++layer_index) {
        auto& layer = text_model.layers[static_cast<size_t>(layer_index)];
        const std::string mlx_prefix = "language_model.model.layers." + std::to_string(layer_index);
        const std::string raw_prefix = "model.language_model.layers." + std::to_string(layer_index);

        layer.is_linear = is_linear_layer(text_cfg, layer_index);
        layer.ln_eps = text_cfg.rms_norm_eps;
        layer.input_ln_w = load_dense(
            weights,
            {
                mlx_prefix + ".input_layernorm.weight",
                raw_prefix + ".input_layernorm.weight",
            });
        layer.post_attn_ln_w = load_dense(
            weights,
            {
                mlx_prefix + ".post_attention_layernorm.weight",
                raw_prefix + ".post_attention_layernorm.weight",
            });
        layer.mlp.gate_proj_w = load_param(
            weights,
            {
                mlx_prefix + ".mlp.gate_proj",
                raw_prefix + ".mlp.gate_proj",
            },
            quant_config);
        layer.mlp.up_proj_w = load_param(
            weights,
            {
                mlx_prefix + ".mlp.up_proj",
                raw_prefix + ".mlp.up_proj",
            },
            quant_config);
        layer.mlp.down_proj_w = load_param(
            weights,
            {
                mlx_prefix + ".mlp.down_proj",
                raw_prefix + ".mlp.down_proj",
            },
            quant_config);

        if (layer.is_linear) {
            auto& linear = layer.linear_attn;
            linear.num_k_heads = text_cfg.linear_num_key_heads;
            linear.num_v_heads = text_cfg.linear_num_value_heads;
            linear.head_k_dim = text_cfg.linear_key_head_dim;
            linear.head_v_dim = text_cfg.linear_value_head_dim;
            linear.key_dim = linear.num_k_heads * linear.head_k_dim;
            linear.value_dim = linear.num_v_heads * linear.head_v_dim;
            linear.conv_dim = linear.key_dim * 2 + linear.value_dim;
            linear.conv_kernel_size = text_cfg.linear_conv_kernel_dim;
            linear.norm_eps = text_cfg.rms_norm_eps;
            linear.conv1d_w = load_dense(
                weights,
                {
                    mlx_prefix + ".linear_attn.conv1d.weight",
                    raw_prefix + ".linear_attn.conv1d.weight",
                },
                true);
            linear.in_proj_qkv_w = load_param(
                weights,
                {
                    mlx_prefix + ".linear_attn.in_proj_qkv",
                    raw_prefix + ".linear_attn.in_proj_qkv",
                },
                quant_config);
            linear.in_proj_z_w = load_param(
                weights,
                {
                    mlx_prefix + ".linear_attn.in_proj_z",
                    raw_prefix + ".linear_attn.in_proj_z",
                },
                quant_config);
            linear.in_proj_b_w = load_param(
                weights,
                {
                    mlx_prefix + ".linear_attn.in_proj_b",
                    raw_prefix + ".linear_attn.in_proj_b",
                },
                quant_config);
            linear.in_proj_a_w = load_param(
                weights,
                {
                    mlx_prefix + ".linear_attn.in_proj_a",
                    raw_prefix + ".linear_attn.in_proj_a",
                },
                quant_config);
            linear.dt_bias = load_dense(
                weights,
                {
                    mlx_prefix + ".linear_attn.dt_bias",
                    raw_prefix + ".linear_attn.dt_bias",
                });
            linear.A_log = load_dense(
                weights,
                {
                    mlx_prefix + ".linear_attn.A_log",
                    raw_prefix + ".linear_attn.A_log",
                });
            linear.norm_w = load_dense(
                weights,
                {
                    mlx_prefix + ".linear_attn.norm.weight",
                    raw_prefix + ".linear_attn.norm.weight",
                });
            linear.out_proj_w = load_param(
                weights,
                {
                    mlx_prefix + ".linear_attn.out_proj",
                    raw_prefix + ".linear_attn.out_proj",
                },
                quant_config);
        } else {
            auto& attn = layer.self_attn;
            attn.num_heads = text_cfg.num_attention_heads;
            attn.num_kv_heads = text_cfg.num_key_value_heads;
            attn.head_dim = text_cfg.head_dim;
            attn.scale = 1.0f / std::sqrt(static_cast<float>(attn.head_dim));
            attn.norm_eps = text_cfg.rms_norm_eps;
            attn.q_proj_w = load_param(
                weights,
                {
                    mlx_prefix + ".self_attn.q_proj",
                    raw_prefix + ".self_attn.q_proj",
                },
                quant_config);
            attn.k_proj_w = load_param(
                weights,
                {
                    mlx_prefix + ".self_attn.k_proj",
                    raw_prefix + ".self_attn.k_proj",
                },
                quant_config);
            attn.v_proj_w = load_param(
                weights,
                {
                    mlx_prefix + ".self_attn.v_proj",
                    raw_prefix + ".self_attn.v_proj",
                },
                quant_config);
            attn.o_proj_w = load_param(
                weights,
                {
                    mlx_prefix + ".self_attn.o_proj",
                    raw_prefix + ".self_attn.o_proj",
                },
                quant_config);
            attn.q_norm_w = load_dense(
                weights,
                {
                    mlx_prefix + ".self_attn.q_norm.weight",
                    raw_prefix + ".self_attn.q_norm.weight",
                });
            attn.k_norm_w = load_dense(
                weights,
                {
                    mlx_prefix + ".self_attn.k_norm.weight",
                    raw_prefix + ".self_attn.k_norm.weight",
                });
            attn.rope = RotaryEmbedding(
                text_cfg.rotary_dim(),
                static_cast<float>(text_cfg.rope_parameters.rope_theta),
                text_cfg.rope_parameters.mrope_section);
        }
    }

    if (weights.count("language_model.lm_head.weight") > 0 || weights.count("model.lm_head.weight") > 0) {
        language_model.lm_head_w = load_param(
            weights,
            {
                "language_model.lm_head",
                "model.lm_head",
                "lm_head",
            },
            quant_config);
    }

    return language_model;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s <model_dir> <prompt_suite.json> <output_json> [max_new_tokens]\n", argv[0]);
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string prompt_suite_path = argv[2];
    const std::string output_path = argv[3];
    const int max_new_tokens = (argc >= 5) ? std::max(1, std::atoi(argv[4])) : 64;

    const std::string config_path = model_dir + "/config.json";
    const std::string tokenizer_path = model_dir + "/tokenizer.json";

    const char* cache_mode_env = std::getenv("QWEN3_5_MLX_LINEAR_CACHE_MODE");
    if (cache_mode_env != nullptr && std::string(cache_mode_env) == "arrays") {
        qwen3_5_mlx::runtime_options::set_linear_cache_mode(
            qwen3_5_mlx::runtime_options::LinearCacheMode::ArraysStyle);
    } else {
        qwen3_5_mlx::runtime_options::set_linear_cache_mode(
            qwen3_5_mlx::runtime_options::LinearCacheMode::Legacy);
    }

    const char* full_attention_cache_mode_env = std::getenv("QWEN3_5_MLX_FULL_ATTENTION_CACHE_MODE");
    if (full_attention_cache_mode_env != nullptr && std::string(full_attention_cache_mode_env) == "step_buffer") {
        qwen3_5_mlx::runtime_options::set_full_attention_cache_mode(
            qwen3_5_mlx::runtime_options::FullAttentionCacheMode::StepBuffer);
    } else {
        qwen3_5_mlx::runtime_options::set_full_attention_cache_mode(
            qwen3_5_mlx::runtime_options::FullAttentionCacheMode::Legacy);
    }

    const char* gated_delta_mode_env = std::getenv("QWEN3_5_MLX_GATED_DELTA_MODE");
    if (gated_delta_mode_env != nullptr && std::string(gated_delta_mode_env) == "compiled_ops") {
        qwen3_5_mlx::runtime_options::set_gated_delta_mode(
            qwen3_5_mlx::runtime_options::GatedDeltaMode::CompiledOps);
    } else if (gated_delta_mode_env != nullptr && std::string(gated_delta_mode_env) == "metal_kernel") {
        qwen3_5_mlx::runtime_options::set_gated_delta_mode(
            qwen3_5_mlx::runtime_options::GatedDeltaMode::MetalKernel);
    } else {
        qwen3_5_mlx::runtime_options::set_gated_delta_mode(
            qwen3_5_mlx::runtime_options::GatedDeltaMode::Ops);
    }

    const bool trace_enabled = parse_bool_env(std::getenv("QWEN3_5_MLX_STAGE_TRACE"), false);
    qwen3_5_mlx::stage_trace::set_enabled(trace_enabled);

    auto suite_json = JsonParser::parse_file(prompt_suite_path);
    const auto& prompts = suite_json.as_array();

    Qwen3_5Config config = Qwen3_5Config::from_file(config_path);
    QuantizationConfig quant_config = load_quant_config(config_path);
    Qwen3_5Tokenizer tokenizer = Qwen3_5Tokenizer::from_file(tokenizer_path);
    auto weights = load_weight_map(model_dir);
    auto language_model = load_language_model(weights, config, quant_config);

    if (qwen3_5_mlx::runtime_options::get_gated_delta_mode() !=
        qwen3_5_mlx::runtime_options::GatedDeltaMode::Ops) {
        auto warmup_tokens = apply_chat_template_nothink("Warm up gated delta compile path.", tokenizer);
        auto warmup_input = mx::array(
            warmup_tokens.data(),
            mx::Shape{1, static_cast<int>(warmup_tokens.size())},
            mx::int32);
        auto warmup_cache = qwen3_5_mlx::LanguageModel::make_cache(language_model.config);
        auto warmup_logits = language_model.forward(warmup_input, warmup_cache);
        mx::eval(warmup_logits);
        mx::synchronize();

        int warmup_token = mx::argmax(
            qwen3_5_mlx::mlx_helpers::slice_axis(
                warmup_logits,
                1,
                warmup_logits.shape(1) - 1,
                warmup_logits.shape(1)),
            -1).item<int>();
        auto decode_input = mx::array(&warmup_token, mx::Shape{1, 1}, mx::int32);
        auto decode_logits = language_model.forward(decode_input, warmup_cache);
        mx::eval(decode_logits);
        mx::synchronize();
    }

    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);
    if (!out) {
        std::fprintf(stderr, "Failed to open output: %s\n", output_path.c_str());
        return 1;
    }

    out << "{\n";
    out << "  \"model_dir\": \"" << json_escape(model_dir) << "\",\n";
    out << "  \"mode\": \"mlx_custom_quantized\",\n";
    out << "  \"linear_cache_mode\": \""
        << qwen3_5_mlx::runtime_options::linear_cache_mode_name(
               qwen3_5_mlx::runtime_options::get_linear_cache_mode())
        << "\",\n";
    out << "  \"full_attention_cache_mode\": \""
        << qwen3_5_mlx::runtime_options::full_attention_cache_mode_name(
               qwen3_5_mlx::runtime_options::get_full_attention_cache_mode())
        << "\",\n";
    out << "  \"gated_delta_mode\": \""
        << qwen3_5_mlx::runtime_options::gated_delta_mode_name(
               qwen3_5_mlx::runtime_options::get_gated_delta_mode())
        << "\",\n";
    out << "  \"trace_enabled\": " << (trace_enabled ? "true" : "false") << ",\n";
    out << "  \"max_new_tokens\": " << max_new_tokens << ",\n";
    out << "  \"rows\": [\n";

    qwen3_5_mlx::stage_trace::PromptTrace prompt_trace;
    qwen3_5_mlx::stage_trace::set_active_prompt_trace(trace_enabled ? &prompt_trace : nullptr);

    for (size_t i = 0; i < prompts.size(); ++i) {
        const auto& obj = prompts[i].as_object();
        const std::string id = obj.at("id").as_string();
        const std::string kind = obj.at("kind").as_string();
        const std::string prompt_text = obj.at("prompt_text").as_string();

        if (trace_enabled) {
            prompt_trace.reset();
        }
        auto prompt_tokens = apply_chat_template_nothink(prompt_text, tokenizer);
        auto cache = qwen3_5_mlx::LanguageModel::make_cache(language_model.config);

        auto prompt_input = mx::array(
            prompt_tokens.data(),
            mx::Shape{1, static_cast<int>(prompt_tokens.size())},
            mx::int32);

        auto t_wall0 = std::chrono::high_resolution_clock::now();
        auto t0 = t_wall0;
        qwen3_5_mlx::stage_trace::set_phase(qwen3_5_mlx::stage_trace::Phase::Prefill);
        auto logits = language_model.forward(prompt_input, cache);
        mx::eval(logits);
        mx::synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        int token = sample_argmax_with_trace(
            qwen3_5_mlx::mlx_helpers::slice_axis(logits, 1, logits.shape(1) - 1, logits.shape(1)));

        std::vector<int> generated_tokens;
        generated_tokens.reserve(static_cast<size_t>(max_new_tokens));
        generated_tokens.push_back(token);

        auto t_decode0 = std::chrono::high_resolution_clock::now();
        while (generated_tokens.size() < static_cast<size_t>(max_new_tokens) && token != kEosImEnd) {
            qwen3_5_mlx::stage_trace::set_phase(qwen3_5_mlx::stage_trace::Phase::Decode);
            auto decode_input = mx::array(&token, mx::Shape{1, 1}, mx::int32);
            logits = language_model.forward(decode_input, cache);
            mx::eval(logits);
            mx::synchronize();
            token = sample_argmax_with_trace(logits);
            generated_tokens.push_back(token);
        }
        auto t_decode1 = std::chrono::high_resolution_clock::now();

        const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double decode_total_ms = std::chrono::duration<double, std::milli>(t_decode1 - t_decode0).count();
        const double wall_ms = std::chrono::duration<double, std::milli>(t_decode1 - t_wall0).count();
        const double decode_ms = generated_tokens.empty() ? 0.0 : decode_total_ms / generated_tokens.size();
        const double throughput = generated_tokens.empty() ? 0.0 : (generated_tokens.size() * 1000.0 / wall_ms);
        const std::string output_text = tokenizer.decode(generated_tokens);

        out << "    {\n";
        out << "      \"id\": \"" << json_escape(id) << "\",\n";
        out << "      \"kind\": \"" << json_escape(kind) << "\",\n";
        out << "      \"prompt_text\": \"" << json_escape(prompt_text) << "\",\n";
        out << "      \"prompt_tokens\": " << json_array_ints(prompt_tokens) << ",\n";
        out << "      \"generated_tokens\": " << json_array_ints(generated_tokens) << ",\n";
        out << "      \"generated_token_count\": " << generated_tokens.size() << ",\n";
        out << "      \"output_text\": \"" << json_escape(output_text) << "\",\n";
        out << "      \"prefill_ms\": " << json_number(prefill_ms) << ",\n";
        out << "      \"decode_ms\": " << json_number(decode_ms) << ",\n";
        out << "      \"wall_ms\": " << json_number(wall_ms) << ",\n";
        out << "      \"throughput\": " << json_number(throughput) << ",\n";
        out << "      \"peak_memory_gb\": 0.000";
        if (trace_enabled) {
            out << ",\n";
            out << "      \"stage_trace\": " << json_prompt_trace(prompt_trace) << "\n";
        } else {
            out << "\n";
        }
        out << "    }";
        if (i + 1 != prompts.size()) out << ",";
        out << "\n";
        out.flush();

        std::fprintf(
            stderr,
            "[mlx_custom_quantized] %s done: %zu tokens, prefill=%.1fms, decode=%.1fms/tok, wall=%.1fms, throughput=%.2f tok/s\n",
            id.c_str(),
            generated_tokens.size(),
            prefill_ms,
            decode_ms,
            wall_ms,
            throughput);
    }

    qwen3_5_mlx::stage_trace::set_active_prompt_trace(nullptr);

    out << "  ]\n";
    out << "}\n";
    out.flush();
    return 0;
}