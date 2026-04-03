#include "models/qwen3_5/modeling.h"
#include "models/qwen3_5/tokenization.h"
#include "utils/json.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s <model_dir> <prompt_suite.json> <output_json> [max_new_tokens]\n", argv[0]);
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string prompt_suite_path = argv[2];
    const std::string output_path = argv[3];
    const int max_new_tokens = (argc >= 5) ? std::max(1, std::atoi(argv[4])) : 256;

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

    Qwen3_5ForCausalLM model;
    model.load(bundle, "model", config);

    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);
    if (!out) {
        std::fprintf(stderr, "Failed to open output: %s\n", output_path.c_str());
        return 1;
    }

    out << "{\n";
    out << "  \"model_dir\": \"" << json_escape(model_dir) << "\",\n";
    out << "  \"mode\": \"cpp\",\n";
    out << "  \"max_new_tokens\": " << max_new_tokens << ",\n";
    out << "  \"rows\": [\n";

    for (size_t i = 0; i < prompts.size(); ++i) {
        const auto& obj = prompts[i].as_object();
        const std::string id = obj.at("id").as_string();
        const std::string kind = obj.at("kind").as_string();
        const std::string prompt_text = obj.at("prompt_text").as_string();

        auto prompt_tokens = apply_chat_template_nothink(prompt_text, tokenizer);
        Scratch scratch;
        ModelCache cache = model.model.create_cache();
        std::vector<float> logits(model.vocab_size);

        auto t_wall0 = std::chrono::high_resolution_clock::now();
        auto t0 = t_wall0;
        model.forward(logits.data(), prompt_tokens.data(), static_cast<int>(prompt_tokens.size()), 1, cache, scratch);
        auto t1 = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int next_token = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::vector<int> generated_tokens;
        generated_tokens.reserve(static_cast<size_t>(max_new_tokens));
        generated_tokens.push_back(next_token);

        auto t_decode0 = std::chrono::high_resolution_clock::now();
        while (generated_tokens.size() < static_cast<size_t>(max_new_tokens) &&
               next_token != kEosImEnd) {
            model.forward(logits.data(), &next_token, 1, 1, cache, scratch);
            next_token = static_cast<int>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            generated_tokens.push_back(next_token);
        }
        auto t_decode1 = std::chrono::high_resolution_clock::now();
        auto t_wall1 = t_decode1;

        double decode_total_ms = std::chrono::duration<double, std::milli>(t_decode1 - t_decode0).count();
        double wall_ms = std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
        double decode_ms = generated_tokens.empty() ? 0.0 : decode_total_ms / generated_tokens.size();
        double throughput = generated_tokens.empty() ? 0.0 : (generated_tokens.size() * 1000.0 / wall_ms);

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

        std::fprintf(stderr, "[cpp] %s done: %zu tokens, prefill=%.1fms, decode=%.1fms/tok, wall=%.1fms, throughput=%.2f tok/s\n",
                     id.c_str(), generated_tokens.size(), prefill_ms, decode_ms, wall_ms, throughput);
    }

    out << "  ]\n";
    out << "}\n";
    out.flush();
    return 0;
}
