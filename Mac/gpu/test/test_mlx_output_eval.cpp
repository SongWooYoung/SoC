#include "models/qwen3_5_mlx/language.h"
#include "models/qwen3_5_py_cpp/tokenization.h"
#include "utils/json.h"
#include "utils/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace {

constexpr int kEosImEnd = 248046;

std::vector<int> apply_chat_template_nothink(
    const std::string& user_message,
    const Qwen3_5Tokenizer& tokenizer)
{
    const int IM_START = 248045;
    const int IM_END = 248046;
    const int THINK = 248068;
    const int THINK_END = 248069;

    auto user_tok = tokenizer.encode("user");
    auto newline_tok = tokenizer.encode("\n");
    auto msg_tok = tokenizer.encode(user_message);
    auto assistant_tok = tokenizer.encode("assistant");
    auto dbl_nl_tok = tokenizer.encode("\n\n");

    std::vector<int> ids;
    ids.push_back(IM_START);
    ids.insert(ids.end(), user_tok.begin(), user_tok.end());
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.insert(ids.end(), msg_tok.begin(), msg_tok.end());
    ids.push_back(IM_END);
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.push_back(IM_START);
    ids.insert(ids.end(), assistant_tok.begin(), assistant_tok.end());
    ids.insert(ids.end(), newline_tok.begin(), newline_tok.end());
    ids.push_back(THINK);
    ids.insert(ids.end(), dbl_nl_tok.begin(), dbl_nl_tok.end());
    ids.push_back(THINK_END);
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

mx::Shape to_shape(const std::vector<int64_t>& dims) {
    mx::Shape shape;
    shape.reserve(dims.size());
    for (int64_t dim : dims) {
        shape.push_back(static_cast<int>(dim));
    }
    return shape;
}

mx::array load_tensor_f32(
    const SafetensorsBundle& bundle,
    const std::string& name,
    bool reorder_conv = false) {
    const auto& meta = bundle.get(name);
    auto values = bundle.load_f32(name);
    auto shape = to_shape(meta.shape);

    if (reorder_conv && shape.size() == 3 && shape[2] != 1) {
        const int dim0 = shape[0];
        const int dim1 = shape[1];
        const int dim2 = shape[2];
        std::vector<float> reordered(values.size());
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                for (int k = 0; k < dim2; ++k) {
                    reordered[((i * dim2) + k) * dim1 + j] =
                        values[((i * dim1) + j) * dim2 + k];
                }
            }
        }
        values.swap(reordered);
        shape = {dim0, dim2, dim1};
    }

    return mx::array(values.data(), shape, mx::float32);
}

qwen3_5_mlx::LanguageModel load_language_model(
    const SafetensorsBundle& bundle,
    const Qwen3_5Config& cfg) {
    using namespace qwen3_5_mlx;

    LanguageModel language_model(cfg.text_config, cfg);
    auto& text_model = language_model.model;
    const auto& text_cfg = cfg.text_config;

    text_model.config = text_cfg;
    text_model.norm_eps = text_cfg.rms_norm_eps;
    text_model.ssm_idx = 0;
    text_model.fa_idx = std::max(0, default_full_attention_interval(text_cfg) - 1);
    text_model.embed_w = load_tensor_f32(bundle, "model.language_model.embed_tokens.weight");
    text_model.norm_w = load_tensor_f32(bundle, "model.language_model.norm.weight");
    text_model.layers.resize(static_cast<size_t>(text_cfg.num_hidden_layers));

    for (int layer_index = 0; layer_index < text_cfg.num_hidden_layers; ++layer_index) {
        auto& layer = text_model.layers[static_cast<size_t>(layer_index)];
        const std::string prefix = "model.language_model.layers." + std::to_string(layer_index);

        layer.is_linear = is_linear_layer(text_cfg, layer_index);
        layer.ln_eps = text_cfg.rms_norm_eps;
        layer.input_ln_w = load_tensor_f32(bundle, prefix + ".input_layernorm.weight");
        layer.post_attn_ln_w = load_tensor_f32(bundle, prefix + ".post_attention_layernorm.weight");
        layer.mlp.gate_proj_w = load_tensor_f32(bundle, prefix + ".mlp.gate_proj.weight");
        layer.mlp.up_proj_w = load_tensor_f32(bundle, prefix + ".mlp.up_proj.weight");
        layer.mlp.down_proj_w = load_tensor_f32(bundle, prefix + ".mlp.down_proj.weight");

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
            linear.conv1d_w = load_tensor_f32(bundle, prefix + ".linear_attn.conv1d.weight", true);
            linear.in_proj_qkv_w = load_tensor_f32(bundle, prefix + ".linear_attn.in_proj_qkv.weight");
            linear.in_proj_z_w = load_tensor_f32(bundle, prefix + ".linear_attn.in_proj_z.weight");
            linear.in_proj_b_w = load_tensor_f32(bundle, prefix + ".linear_attn.in_proj_b.weight");
            linear.in_proj_a_w = load_tensor_f32(bundle, prefix + ".linear_attn.in_proj_a.weight");
            linear.dt_bias = load_tensor_f32(bundle, prefix + ".linear_attn.dt_bias");
            linear.A_log = load_tensor_f32(bundle, prefix + ".linear_attn.A_log");
            linear.norm_w = load_tensor_f32(bundle, prefix + ".linear_attn.norm.weight");
            linear.out_proj_w = load_tensor_f32(bundle, prefix + ".linear_attn.out_proj.weight");
        } else {
            auto& attn = layer.self_attn;
            attn.num_heads = text_cfg.num_attention_heads;
            attn.num_kv_heads = text_cfg.num_key_value_heads;
            attn.head_dim = text_cfg.head_dim;
            attn.scale = 1.0f / std::sqrt(static_cast<float>(attn.head_dim));
            attn.norm_eps = text_cfg.rms_norm_eps;
            attn.q_proj_w = load_tensor_f32(bundle, prefix + ".self_attn.q_proj.weight");
            attn.k_proj_w = load_tensor_f32(bundle, prefix + ".self_attn.k_proj.weight");
            attn.v_proj_w = load_tensor_f32(bundle, prefix + ".self_attn.v_proj.weight");
            attn.o_proj_w = load_tensor_f32(bundle, prefix + ".self_attn.o_proj.weight");
            attn.q_norm_w = load_tensor_f32(bundle, prefix + ".self_attn.q_norm.weight");
            attn.k_norm_w = load_tensor_f32(bundle, prefix + ".self_attn.k_norm.weight");
            attn.rope = RotaryEmbedding(
                text_cfg.rotary_dim(),
                static_cast<float>(text_cfg.rope_parameters.rope_theta),
                text_cfg.rope_parameters.mrope_section);
        }
    }

    if (bundle.has("model.lm_head.weight")) {
        language_model.lm_head_w = load_tensor_f32(bundle, "model.lm_head.weight");
    } else if (bundle.has("lm_head.weight")) {
        language_model.lm_head_w = load_tensor_f32(bundle, "lm_head.weight");
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

    auto suite_json = JsonParser::parse_file(prompt_suite_path);
    const auto& prompts = suite_json.as_array();

    Qwen3_5Config config = Qwen3_5Config::from_file(config_path);
    Qwen3_5Tokenizer tokenizer = Qwen3_5Tokenizer::from_file(tokenizer_path);

    SafetensorsBundle bundle;
    for (auto& entry : fs::directory_iterator(model_dir)) {
        auto p = entry.path();
        if (p.extension() == ".safetensors") {
            bundle.add_file(p.string());
        }
    }

    auto language_model = load_language_model(bundle, config);

    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);
    if (!out) {
        std::fprintf(stderr, "Failed to open output: %s\n", output_path.c_str());
        return 1;
    }

    out << "{\n";
    out << "  \"model_dir\": \"" << json_escape(model_dir) << "\",\n";
    out << "  \"mode\": \"mlx_custom\",\n";
    out << "  \"max_new_tokens\": " << max_new_tokens << ",\n";
    out << "  \"rows\": [\n";

    for (size_t i = 0; i < prompts.size(); ++i) {
        const auto& obj = prompts[i].as_object();
        const std::string id = obj.at("id").as_string();
        const std::string kind = obj.at("kind").as_string();
        const std::string prompt_text = obj.at("prompt_text").as_string();

        auto prompt_tokens = apply_chat_template_nothink(prompt_text, tokenizer);
        auto cache = qwen3_5_mlx::LanguageModel::make_cache(language_model.config);

        auto prompt_input = mx::array(
            prompt_tokens.data(),
            mx::Shape{1, static_cast<int>(prompt_tokens.size())},
            mx::int32);

        auto t_wall0 = std::chrono::high_resolution_clock::now();
        auto t0 = t_wall0;
        auto logits = language_model.forward(prompt_input, cache);
        mx::eval(logits);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto current = mx::argmax(
            qwen3_5_mlx::mlx_helpers::slice_axis(logits, 1, logits.shape(1) - 1, logits.shape(1)),
            -1);
        mx::eval(current);
        int token = current.item<int>();

        std::vector<int> generated_tokens;
        generated_tokens.reserve(static_cast<size_t>(max_new_tokens));
        generated_tokens.push_back(token);

        auto t_decode0 = std::chrono::high_resolution_clock::now();
        while (generated_tokens.size() < static_cast<size_t>(max_new_tokens) && token != kEosImEnd) {
            auto decode_input = mx::array(&token, mx::Shape{1, 1}, mx::int32);
            logits = language_model.forward(decode_input, cache);
            mx::eval(logits);
            current = mx::argmax(logits, -1);
            mx::eval(current);
            token = current.item<int>();
            generated_tokens.push_back(token);
        }
        auto t_decode1 = std::chrono::high_resolution_clock::now();
        auto t_wall1 = t_decode1;

        const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double decode_total_ms = std::chrono::duration<double, std::milli>(t_decode1 - t_decode0).count();
        const double wall_ms = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
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
        out << "      \"throughput\": " << json_number(throughput) << "\n";
        out << "    }";
        if (i + 1 != prompts.size()) out << ",";
        out << "\n";
        out.flush();

        std::fprintf(
            stderr,
            "[mlx_custom] %s done: %zu tokens, prefill=%.1fms, decode=%.1fms/tok, wall=%.1fms, throughput=%.2f tok/s\n",
            id.c_str(),
            generated_tokens.size(),
            prefill_ms,
            decode_ms,
            wall_ms,
            throughput);
    }

    out << "  ]\n";
    out << "}\n";
    out.flush();
    return 0;
}