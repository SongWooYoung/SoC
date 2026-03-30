#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "asset/runtime_assets.h"
#include "buffer/buffer_arena.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "model/qwen_causal_lm.h"
#include "model/qwen_model_loader.h"
#include "runtime/command_scheduler.h"
#include "runtime/generation_context.h"
#include "runtime/sampler.h"

#include "header/runtime_pipeline.h"
#include "header/tokenizer_runtime.h"

namespace {

struct CliOptions {
    std::string manifest_path = "../../models/cpp/qwen3-0.6b/manifest.json";
    std::string metallib_path = "build/shaders/gpu.metallib";
    std::string shader_source_path = "shaders/gpu_kernels.metal";
    std::string prompt;
    std::string prompt_file;
    RuntimeGenerationOptions generation;
    RuntimePromptOptions prompt_options;
    std::size_t temporary_arena_bytes = 1ull << 26;
    bool override_max_new_tokens = false;
    bool override_temperature = false;
    bool override_top_k = false;
    bool override_eos_token_id = false;
    bool override_max_sequence_length = false;
    bool verbose = false;
};

[[noreturn]] void PrintUsageAndExit(const char* executable, int exit_code) {
    std::cerr
        << "Usage: " << executable << " [--manifest <manifest.json>] (--prompt <text> | --prompt-file <path>) [options]\n"
        << "Options:\n"
        << "  --manifest <path>          Model manifest path (default: ../../models/cpp/qwen3-0.6b/manifest.json)\n"
        << "  --prompt <text>            Prompt text to generate from\n"
        << "  --prompt-file <path>       Read prompt text from a UTF-8 text file\n"
        << "  --max-new-tokens <n>       Number of tokens to generate\n"
        << "  --temperature <value>      Sampler temperature\n"
        << "  --top-k <n>                Sampler top-k\n"
        << "  --eos-token-id <id>        Stop when this token id is generated\n"
        << "  --max-seq-len <n>          KV cache capacity for generation\n"
        << "  --apply-chat-template      Wrap prompt as a single user chat turn\n"
        << "  --disable-thinking         Omit the opening <think> tag in chat template mode\n"
        << "  --system-prompt <text>     Override the default chat template system prompt\n"
        << "  --metallib <path>          Override compiled metallib path\n"
        << "  --shader-source <path>     Override Metal shader source path\n"
        << "  --temp-arena-bytes <n>     Temporary arena allocation size in bytes\n"
        << "  --verbose                  Print generation metadata to stderr\n"
        << "  --help, -h                 Show this help text\n";
    std::exit(exit_code);
}

std::string ReadTextFile(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open prompt file: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
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
        } else if (argument == "--prompt-file") {
            options.prompt_file = require_value("--prompt-file");
        } else if (argument == "--max-new-tokens") {
            options.generation.max_new_tokens = static_cast<std::size_t>(std::stoull(require_value("--max-new-tokens")));
            options.override_max_new_tokens = true;
        } else if (argument == "--temperature") {
            options.generation.sampler.temperature = std::stof(require_value("--temperature"));
            options.override_temperature = true;
        } else if (argument == "--top-k") {
            options.generation.sampler.top_k = static_cast<std::size_t>(std::stoull(require_value("--top-k")));
            options.override_top_k = true;
        } else if (argument == "--eos-token-id") {
            options.generation.eos_token_id = std::stoi(require_value("--eos-token-id"));
            options.override_eos_token_id = true;
        } else if (argument == "--max-seq-len") {
            options.generation.max_sequence_length = static_cast<std::size_t>(std::stoull(require_value("--max-seq-len")));
            options.override_max_sequence_length = true;
        } else if (argument == "--apply-chat-template") {
            options.prompt_options.apply_chat_template = true;
        } else if (argument == "--disable-thinking") {
            options.prompt_options.enable_thinking = false;
        } else if (argument == "--system-prompt") {
            options.prompt_options.system_prompt = require_value("--system-prompt");
        } else if (argument == "--metallib") {
            options.metallib_path = require_value("--metallib");
        } else if (argument == "--shader-source") {
            options.shader_source_path = require_value("--shader-source");
        } else if (argument == "--temp-arena-bytes") {
            options.temporary_arena_bytes = static_cast<std::size_t>(std::stoull(require_value("--temp-arena-bytes")));
        } else if (argument == "--verbose") {
            options.verbose = true;
        } else if (argument == "--help" || argument == "-h") {
            PrintUsageAndExit(argv[0], 0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }

    if (options.prompt.empty() == options.prompt_file.empty()) {
        throw std::runtime_error("provide exactly one of --prompt or --prompt-file");
    }
    if (options.temporary_arena_bytes == 0) {
        throw std::runtime_error("--temp-arena-bytes must be greater than zero");
    }

    return options;
}

RuntimeGenerationOptions ResolveGenerationOptions(const CliOptions& cli, const ManifestData& manifest) {
    RuntimeGenerationOptions resolved = RuntimePipeline::GenerationOptionsFromManifest(manifest);
    if (cli.override_max_new_tokens) {
        resolved.max_new_tokens = cli.generation.max_new_tokens;
    }
    if (cli.override_temperature) {
        resolved.sampler.temperature = cli.generation.sampler.temperature;
    }
    if (cli.override_top_k) {
        resolved.sampler.top_k = cli.generation.sampler.top_k;
    }
    if (cli.override_eos_token_id) {
        resolved.eos_token_id = cli.generation.eos_token_id;
    }
    if (cli.override_max_sequence_length) {
        resolved.max_sequence_length = cli.generation.max_sequence_length;
    }
    return resolved;
}

std::string ResolvePromptText(const CliOptions& options) {
    if (!options.prompt.empty()) {
        return options.prompt;
    }
    return ReadTextFile(options.prompt_file);
}

void PrintTokenIds(std::ostream& stream, const char* label, const std::vector<int>& token_ids) {
    stream << label << ": [";
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << token_ids[index];
    }
    stream << "]\n";
}

void PrintVerboseSummary(const CliOptions& options,
                         const std::string& prepared_prompt,
                         const std::vector<int>& prompt_token_ids,
                         const std::vector<int>& generated_token_ids,
                         const soc::gpu::MetalDeviceInfo& device_info,
                         double gpu_ms) {
    std::cerr << "manifest=" << options.manifest_path << "\n";
    if (!options.prompt_file.empty()) {
        std::cerr << "prompt_file=" << options.prompt_file << "\n";
    }
    std::cerr << "device=" << device_info.name << "\n";
    std::cerr << "max_new_tokens=" << generated_token_ids.size() << "\n";
    if (prepared_prompt != options.prompt && options.prompt_file.empty()) {
        std::cerr << "serialized_prompt=" << prepared_prompt << "\n";
    }
    PrintTokenIds(std::cerr, "prompt_token_ids", prompt_token_ids);
    PrintTokenIds(std::cerr, "generated_token_ids", generated_token_ids);
    std::cerr << "gpu_ms=" << gpu_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = ParseArgs(argc, argv);
        const std::string prompt_text = ResolvePromptText(options);

        const ManifestData manifest = ManifestLoader::LoadFromFile(options.manifest_path);
        const TokenizerRuntimeData tokenizer_runtime = TokenizerRuntimeLoader::LoadFromFile(manifest.tokenizer_runtime_file);
        const RuntimeGenerationOptions generation = ResolveGenerationOptions(options, manifest);
        const std::string prepared_prompt = RuntimePipeline::PreparePrompt(tokenizer_runtime, prompt_text, options.prompt_options);
        const TokenizerRuntime tokenizer(tokenizer_runtime);
        const std::vector<int> prompt_token_ids = tokenizer.Encode(prepared_prompt);

        std::string error_message;
        auto context = soc::gpu::MetalContext::CreateDefault(options.metallib_path,
                                                             options.shader_source_path,
                                                             &error_message);
        if (context == nullptr) {
            std::cerr << "failed to create Metal context: " << error_message << '\n';
            return 1;
        }

        soc::gpu::PipelineCache pipeline_cache(*context);
        auto temporary_arena = soc::gpu::BufferArena::CreateShared(*context,
                                                                   options.temporary_arena_bytes,
                                                                   "infer_temp",
                                                                   &error_message);
        if (temporary_arena == nullptr) {
            std::cerr << "failed to create temporary arena: " << error_message << '\n';
            return 1;
        }

        soc::gpu::QwenCausalLM gpu_model({{}, {}, {}, {}, true}, {});
        if (!soc::gpu::QwenModelLoader::LoadModelFromFile(*context, options.manifest_path, &gpu_model, &error_message)) {
            std::cerr << "failed to load GPU model: " << error_message << '\n';
            return 1;
        }

        soc::gpu::SamplerConfig sampler_config;
        sampler_config.temperature = generation.sampler.temperature;
        sampler_config.top_k = generation.sampler.top_k;
        const std::size_t sequence_capacity = std::max<std::size_t>(generation.max_sequence_length,
                                                                    prompt_token_ids.size() + generation.max_new_tokens + 8);
        soc::gpu::GenerationContext generation_context(std::move(gpu_model),
                                                       soc::gpu::Sampler(sampler_config),
                                                       soc::gpu::CommandScheduler(),
                                                       sequence_capacity);

        std::vector<int> generated_token_ids;
        context->ResetProfiling();
        if (!generation_context.Generate(*context,
                                         &pipeline_cache,
                                         prompt_token_ids,
                                         generation.max_new_tokens,
                                         generation.eos_token_id,
                                         temporary_arena.get(),
                                         &generated_token_ids,
                                         &error_message)) {
            std::cerr << "generation failed: " << error_message << '\n';
            return 1;
        }

        const soc::gpu::MetalProfilingSnapshot profiling = context->GetProfilingSnapshot();
        const std::string generated_text = tokenizer.Decode(generated_token_ids);
        std::cout << generated_text << '\n';

        if (options.verbose) {
            PrintVerboseSummary(options,
                                prepared_prompt,
                                prompt_token_ids,
                                generated_token_ids,
                                context->GetDeviceInfo(),
                                profiling.gpu_ms);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "runtime error: " << error.what() << "\n";
        PrintUsageAndExit(argv[0], 1);
    }
}