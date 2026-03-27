#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "header/runtime_pipeline.h"

namespace {
struct CliOptions {
    std::string manifest_path;
    std::string prompt;
    RuntimeGenerationOptions generation;
};

[[noreturn]] void PrintUsageAndExit(const char* executable, int exit_code) {
    std::cerr
        << "Usage: " << executable << " --manifest <manifest.json> --prompt <text> [options]\n"
        << "Options:\n"
        << "  --max-new-tokens <n>   Number of tokens to generate (default: 32)\n"
        << "  --temperature <value>  Sampler temperature (default: 1.0)\n"
        << "  --top-k <n>            Sampler top-k (default: 1)\n"
        << "  --eos-token-id <id>    Stop when this token id is generated (default: disabled)\n"
        << "  --max-seq-len <n>      KV cache capacity for generation (default: 256)\n";
    std::exit(exit_code);
}

CliOptions ParseArgs(int argc, char** argv) {
    CliOptions options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto require_value = [&](const char* name) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for argument: ") + name);
            }
            ++index;
            return argv[index];
        };

        if (argument == "--manifest") {
            options.manifest_path = require_value("--manifest");
        } else if (argument == "--prompt") {
            options.prompt = require_value("--prompt");
        } else if (argument == "--max-new-tokens") {
            options.generation.max_new_tokens = static_cast<std::size_t>(std::stoull(require_value("--max-new-tokens")));
        } else if (argument == "--temperature") {
            options.generation.sampler.temperature = std::stof(require_value("--temperature"));
        } else if (argument == "--top-k") {
            options.generation.sampler.top_k = static_cast<std::size_t>(std::stoull(require_value("--top-k")));
        } else if (argument == "--eos-token-id") {
            options.generation.eos_token_id = std::stoi(require_value("--eos-token-id"));
        } else if (argument == "--max-seq-len") {
            options.generation.max_sequence_length = static_cast<std::size_t>(std::stoull(require_value("--max-seq-len")));
        } else if (argument == "--help" || argument == "-h") {
            PrintUsageAndExit(argv[0], 0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }

    if (options.manifest_path.empty()) {
        throw std::runtime_error("--manifest is required");
    }
    if (options.prompt.empty()) {
        throw std::runtime_error("--prompt is required");
    }

    return options;
}

void PrintTokenIds(const char* label, const std::vector<int>& token_ids) {
    std::cout << label << ": [";
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << token_ids[index];
    }
    std::cout << "]\n";
}
}

int main(int argc, char** argv) {
    try {
        const CliOptions options = ParseArgs(argc, argv);
        const GenerationResult result = RuntimePipeline::Generate(options.manifest_path, options.prompt, options.generation);

        std::cout << "prompt: " << options.prompt << "\n";
        PrintTokenIds("prompt_token_ids", result.prompt_token_ids);
        PrintTokenIds("generated_token_ids", result.generated_token_ids);
        std::cout << "generated_text: " << result.generated_text << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "runtime error: " << error.what() << "\n";
        PrintUsageAndExit(argv[0], 1);
    }
}