#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
#include "runtime/runtime_policy.h"
#include "runtime/sampler.h"

#include "header/generation_session.h"
#include "header/kv_cache.h"
#include "header/qwen_model_loader.h"
#include "header/sampler.h"
#include "header/runtime_pipeline.h"
#include "header/tokenizer_runtime.h"

namespace {

using Clock = std::chrono::steady_clock;

struct CliOptions {
    std::string manifest_path = "../../models/cpp/qwen3-0.6b/manifest.json";
    std::string metallib_path = "build/shaders/gpu.metallib";
    std::string shader_source_path = "shaders/gpu_kernels.metal";
    std::string prompt;
    std::string prompt_file;
    std::string output_file;
    std::string prompt_cache_artifact_save;
    std::string prompt_cache_artifact_load;
    RuntimeGenerationOptions generation;
    RuntimePromptOptions prompt_options;
    std::size_t temporary_arena_bytes = 1ull << 26;
    bool override_max_new_tokens = false;
    bool override_temperature = false;
    bool override_top_k = false;
    bool override_eos_token_id = false;
    bool override_max_sequence_length = false;
    bool json_output = false;
    int layer = -1;
    bool verbose = false;
};

struct ExecutionPlan {
    int requested_layer = -1;
    std::size_t resolved_gpu_layers = 0;
    std::size_t model_layer_count = 0;
    std::string mode;
};

struct InferenceRunResult {
    std::vector<int> generated_token_ids;
    std::string generated_text;
    double wall_ms = 0.0;
    double gpu_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double prompt_cache_load_ms = 0.0;
    std::string prompt_cache_mode = "disabled";
    soc::gpu::MetalProfilingSnapshot profile;
    soc::gpu::DecodePlanRunStats decode_plan_stats;
    soc::gpu::RuntimePolicy runtime_policy;
};

[[noreturn]] void PrintUsageAndExit(const char* executable, int exit_code) {
    std::cerr
        << "Usage: " << executable << " [--manifest <manifest.json>] (--prompt <text> | --prompt-file <path>) [options]\n"
        << "Options:\n"
        << "  --manifest <path>          Model manifest path (default: ../../models/cpp/qwen3-0.6b/manifest.json)\n"
        << "  --prompt <text>            Prompt text to generate from\n"
        << "  --prompt-file <path>       Read prompt text from a UTF-8 text file ('-' reads stdin)\n"
        << "  --output-file <path>       Write the primary result payload to a file instead of stdout\n"
        << "  --prompt-cache-artifact-save <path>  Save a prompt cache artifact after prefill\n"
        << "  --prompt-cache-artifact-load <path>  Load a prompt cache artifact instead of running prefill\n"
        << "  --json                     Emit a JSON object instead of plain text\n"
        << "  --layer <k>                GPU prefix layers: 0=full CPU, 1..N-1=hybrid, N or -1=full GPU\n"
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
    if (path == "-") {
        std::ostringstream buffer;
        buffer << std::cin.rdbuf();
        return buffer.str();
    }
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
        } else if (argument == "--output-file") {
            options.output_file = require_value("--output-file");
        } else if (argument == "--prompt-cache-artifact-save") {
            options.prompt_cache_artifact_save = require_value("--prompt-cache-artifact-save");
        } else if (argument == "--prompt-cache-artifact-load") {
            options.prompt_cache_artifact_load = require_value("--prompt-cache-artifact-load");
        } else if (argument == "--json") {
            options.json_output = true;
        } else if (argument == "--layer") {
            options.layer = std::stoi(require_value("--layer"));
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
    if (!options.prompt_cache_artifact_save.empty() && !options.prompt_cache_artifact_load.empty()) {
        throw std::runtime_error("--prompt-cache-artifact-save and --prompt-cache-artifact-load are mutually exclusive");
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

std::size_t ReadManifestLayerCount(const ManifestData& manifest) {
    if (!manifest.config.is_object() || !manifest.config.contains("num_hidden_layers")) {
        throw std::runtime_error("manifest config is missing num_hidden_layers");
    }
    return static_cast<std::size_t>(manifest.config.at("num_hidden_layers").as_int64());
}

ExecutionPlan ResolveExecutionPlan(const CliOptions& options, const ManifestData& manifest) {
    ExecutionPlan plan;
    plan.requested_layer = options.layer;
    plan.model_layer_count = ReadManifestLayerCount(manifest);

    if (options.layer < -1) {
        throw std::runtime_error("--layer must be -1 or a non-negative integer");
    }
    if (options.layer == -1) {
        plan.resolved_gpu_layers = plan.model_layer_count;
        plan.mode = "full-gpu";
        return plan;
    }

    const std::size_t requested_layers = static_cast<std::size_t>(options.layer);
    if (requested_layers > plan.model_layer_count) {
        throw std::runtime_error("--layer exceeds manifest num_hidden_layers");
    }
    plan.resolved_gpu_layers = requested_layers;
    if (requested_layers == 0) {
        plan.mode = "full-cpu";
    } else if (requested_layers == plan.model_layer_count) {
        plan.mode = "full-gpu";
    } else {
        plan.mode = "hybrid";
    }
    return plan;
}

double DurationMilliseconds(const Clock::time_point& start_time, const Clock::time_point& end_time) {
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

soc::gpu::DeviceTensor UploadGpuTokenIds(const soc::gpu::MetalContext& context,
                                         const std::vector<int>& token_ids,
                                         const std::string& label,
                                         std::string* error_message) {
    auto token_buffer = soc::gpu::MetalBuffer::CreateForTensorClass(context,
                                                                    token_ids.size() * sizeof(std::int32_t),
                                                                    label,
                                                                    soc::gpu::TensorClass::kTokenMetadata,
                                                                    error_message);
    if (token_buffer == nullptr) {
        return {};
    }
    if (!token_buffer->Write(token_ids.data(), token_ids.size() * sizeof(std::int32_t), 0, error_message)) {
        return {};
    }
    return soc::gpu::DeviceTensor(token_buffer,
                                  0,
                                  soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kInt32, {token_ids.size()}));
}

soc::gpu::DeviceTensor CreateGpuFloatTensor(const soc::gpu::MetalContext& context,
                                            const std::vector<std::size_t>& shape,
                                            const std::string& label,
                                            std::string* error_message) {
    std::size_t element_count = 1;
    for (std::size_t dim : shape) {
        element_count *= dim;
    }
    auto buffer = soc::gpu::MetalBuffer::CreateForTensorClass(context,
                                                              element_count * sizeof(float),
                                                              label,
                                                              soc::gpu::TensorClass::kTemporary,
                                                              error_message);
    if (buffer == nullptr) {
        return {};
    }
    return soc::gpu::DeviceTensor(buffer,
                                  0,
                                  soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, shape));
}

Tensor ReadGpuHiddenToCpuTensor(const soc::gpu::DeviceTensor& hidden_states, std::string* error_message) {
    if (!hidden_states.IsValid() || hidden_states.GetDesc().GetDataType() != soc::gpu::DataType::kFloat32 || hidden_states.GetDesc().Rank() != 2) {
        throw std::runtime_error("expected rank-2 float32 GPU hidden tensor");
    }
    const std::size_t token_count = hidden_states.GetDesc().GetShape()[0];
    const std::size_t hidden_size = hidden_states.GetDesc().GetShape()[1];
    std::vector<float> values(token_count * hidden_size, 0.0f);
    if (!hidden_states.GetBuffer()->Read(values.data(),
                                         values.size() * sizeof(float),
                                         hidden_states.GetByteOffset(),
                                         error_message)) {
        throw std::runtime_error(error_message != nullptr ? *error_message : "failed to read GPU hidden tensor");
    }
    return Tensor(Storage::FromOwnedCopy(values.data(), values.size() * sizeof(float)),
                  DType::Float32,
                  {1, token_count, hidden_size});
}

std::vector<std::string> BuildCpuStageDescriptions(const ExecutionPlan& plan) {
    std::vector<std::string> stages = {
        "manifest/tokenizer file IO and JSON parsing",
        "prompt serialization and tokenizer encode/decode",
        "CLI stdout/file emission",
    };
    if (plan.mode == "full-cpu") {
        stages.push_back("transformer embedding + layers [0," + std::to_string(plan.model_layer_count) + ") and sampler");
    } else if (plan.mode == "hybrid") {
        stages.push_back("transformer suffix layers [" + std::to_string(plan.resolved_gpu_layers) + "," + std::to_string(plan.model_layer_count) + ")");
        stages.push_back("next-token sampler on CPU logits");
    } else {
        stages.push_back("next-token decode from token ids to UTF-8 text");
    }
    return stages;
}

std::vector<std::string> BuildGpuStageDescriptions(const ExecutionPlan& plan) {
    if (plan.mode == "full-cpu") {
        return {};
    }
    if (plan.mode == "full-gpu") {
        return {"embedding lookup + transformer layers [0," + std::to_string(plan.model_layer_count) + ")", "KV cache update", "GPU top-k reduction before readback"};
    }
    return {"embedding lookup + transformer prefix layers [0," + std::to_string(plan.resolved_gpu_layers) + ")", "GPU KV cache update for prefix layers", "hidden-state handoff to CPU suffix"};
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

std::string JsonEscape(const std::string& value) {
    std::ostringstream stream;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\':
                stream << "\\\\";
                break;
            case '"':
                stream << "\\\"";
                break;
            case '\b':
                stream << "\\b";
                break;
            case '\f':
                stream << "\\f";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(ch) << std::dec << std::setfill(' ');
                } else {
                    stream << static_cast<char>(ch);
                }
                break;
        }
    }
    return stream.str();
}

std::string JsonArray(const std::vector<int>& values) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << values[index];
    }
    stream << ']';
    return stream.str();
}

std::string JsonStringArray(const std::vector<std::string>& values) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << '"' << JsonEscape(values[index]) << '"';
    }
    stream << ']';
    return stream.str();
}

std::string JsonSizeArray(const std::vector<std::size_t>& values) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << values[index];
    }
    stream << ']';
    return stream.str();
}

const char* DecodePlanBufferKindName(soc::gpu::DecodePlanBufferKind buffer_kind) {
    switch (buffer_kind) {
        case soc::gpu::DecodePlanBufferKind::kHiddenSlot0:
            return "hidden_slot0";
        case soc::gpu::DecodePlanBufferKind::kHiddenSlot1:
            return "hidden_slot1";
        case soc::gpu::DecodePlanBufferKind::kLogits:
            return "logits";
        case soc::gpu::DecodePlanBufferKind::kKvKeys:
            return "kv_keys";
        case soc::gpu::DecodePlanBufferKind::kKvValues:
            return "kv_values";
    }
    return "unknown";
}

const char* DecodePlanHazardKindName(bool prior_write, bool current_write) {
    if (prior_write && current_write) {
        return "write_after_write";
    }
    if (prior_write) {
        return "read_after_write";
    }
    return "write_after_read";
}

std::string JsonDecodePlanStageBlockers(const std::vector<soc::gpu::DecodePlanRunStats::StageBlocker>& blockers) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < blockers.size(); ++index) {
        const auto& blocker = blockers[index];
        if (index != 0) {
            stream << ", ";
        }
        stream << "{\"stage_index\": " << blocker.stage_index
               << ", \"stage_label\": \"" << JsonEscape(blocker.stage_label) << "\""
               << ", \"prior_stage_index\": " << blocker.prior_stage_index
               << ", \"prior_stage_label\": \"" << JsonEscape(blocker.prior_stage_label) << "\""
               << ", \"buffer_kind\": \"" << DecodePlanBufferKindName(blocker.buffer_kind) << "\""
               << ", \"hazard_kind\": \"" << DecodePlanHazardKindName(blocker.prior_write, blocker.current_write) << "\""
               << '}';
    }
    stream << ']';
    return stream.str();
}

std::string JsonProfilingEntries(const soc::gpu::MetalProfilingSnapshot& snapshot) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < snapshot.entries.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        const auto& entry = snapshot.entries[index];
        stream << "{"
               << "\"label\":\"" << JsonEscape(entry.label) << "\", "
               << "\"gpu_ms\":" << entry.gpu_ms << ", "
               << "\"wait_ms\":" << entry.wait_ms << ", "
               << "\"command_buffer_count\":" << entry.command_buffer_count << ", "
               << "\"encoder_count\":" << entry.encoder_count
               << "}";
    }
    stream << ']';
    return stream.str();
}

InferenceRunResult RunFullGpuInference(const soc::gpu::MetalContext& context,
                                       soc::gpu::PipelineCache* pipeline_cache,
                                       soc::gpu::BufferArena* temporary_arena,
                                       soc::gpu::QwenCausalLM gpu_model,
                                       const RuntimeGenerationOptions& generation,
                                       const std::vector<int>& prompt_token_ids,
                                       const std::string& prompt_cache_artifact_save,
                                       const std::string& prompt_cache_artifact_load,
                                       const TokenizerRuntimeData& tokenizer_runtime,
                                       std::string* error_message) {
    InferenceRunResult run;
    soc::gpu::SamplerConfig sampler_config;
    sampler_config.temperature = generation.sampler.temperature;
    sampler_config.top_k = generation.sampler.top_k;
    const std::size_t sequence_capacity = std::max<std::size_t>(generation.max_sequence_length,
                                                                prompt_token_ids.size() + generation.max_new_tokens + 8);
    soc::gpu::GenerationContext generation_context(std::move(gpu_model),
                                                   soc::gpu::Sampler(sampler_config),
                                                   soc::gpu::CommandScheduler(),
                                                   sequence_capacity);

    const Clock::time_point start_time = Clock::now();
    context.ResetProfiling();
    if (prompt_cache_artifact_load.empty()) {
        const Clock::time_point prefill_start_time = Clock::now();
        if (!generation_context.Prefill(context,
                                        pipeline_cache,
                                        prompt_token_ids,
                                        temporary_arena,
                                        error_message)) {
            return run;
        }
        run.prefill_ms = DurationMilliseconds(prefill_start_time, Clock::now());
        run.prompt_cache_mode = prompt_cache_artifact_save.empty() ? "disabled" : "artifact-save";

        if (!prompt_cache_artifact_save.empty() &&
            !generation_context.SavePromptCacheArtifact(context, prompt_cache_artifact_save, error_message)) {
            return run;
        }
    } else {
        const Clock::time_point load_start_time = Clock::now();
        if (!generation_context.LoadPromptCacheArtifact(context, prompt_cache_artifact_load, error_message)) {
            return run;
        }
        run.prompt_cache_load_ms = DurationMilliseconds(load_start_time, Clock::now());
        run.prefill_ms = run.prompt_cache_load_ms;
        run.prompt_cache_mode = "artifact-load";
        if (generation_context.prompt_token_ids() != prompt_token_ids) {
            if (error_message != nullptr) {
                *error_message = "loaded prompt cache artifact does not match prompt token ids";
            }
            return run;
        }
    }

    const Clock::time_point decode_start_time = Clock::now();
    if (!generation_context.GenerateFromLoadedPromptCache(context,
                                                          pipeline_cache,
                                                          generation.max_new_tokens,
                                                          generation.eos_token_id,
                                                          temporary_arena,
                                                          &run.generated_token_ids,
                                                          error_message)) {
        return run;
    }
    run.decode_ms = DurationMilliseconds(decode_start_time, Clock::now());
    run.wall_ms = DurationMilliseconds(start_time, Clock::now());
    run.profile = context.GetProfilingSnapshot();
    run.gpu_ms = run.profile.gpu_ms;
    run.decode_plan_stats = generation_context.scheduler().last_decode_plan_run_stats();
    run.runtime_policy = generation_context.runtime_policy();
    run.generated_text = TokenizerRuntime(tokenizer_runtime).Decode(run.generated_token_ids);
    return run;
}

InferenceRunResult RunFullCpuInference(const QwenCausalLM& cpu_model,
                                       const RuntimeGenerationOptions& generation,
                                       const std::string& prepared_prompt,
                                       const TokenizerRuntimeData& tokenizer_runtime) {
    InferenceRunResult run;
    const std::size_t sequence_capacity = std::max<std::size_t>(generation.max_sequence_length,
                                                                TokenizerRuntime(tokenizer_runtime).Encode(prepared_prompt).size() + generation.max_new_tokens + 8);
    ::GenerationSession session(cpu_model,
                                TokenizerRuntime(tokenizer_runtime),
                                ::Sampler(generation.sampler),
                                sequence_capacity);
    const Clock::time_point start_time = Clock::now();
    const GenerationResult result = session.Generate(prepared_prompt, generation.max_new_tokens, generation.eos_token_id);
    run.wall_ms = DurationMilliseconds(start_time, Clock::now());
    run.generated_token_ids = result.generated_token_ids;
    run.generated_text = result.generated_text;
    return run;
}

InferenceRunResult RunHybridInference(const soc::gpu::MetalContext& context,
                                      soc::gpu::PipelineCache* pipeline_cache,
                                      soc::gpu::BufferArena* temporary_arena,
                                      const soc::gpu::QwenCausalLM& gpu_model,
                                      const QwenCausalLM& cpu_model,
                                      const ExecutionPlan& plan,
                                      const RuntimeGenerationOptions& generation,
                                      const std::vector<int>& prompt_token_ids,
                                      const TokenizerRuntimeData& tokenizer_runtime,
                                      std::string* error_message) {
    InferenceRunResult run;
    const std::size_t sequence_capacity = std::max<std::size_t>(generation.max_sequence_length,
                                                                prompt_token_ids.size() + generation.max_new_tokens + 8);
    auto gpu_cache = soc::gpu::KVCache::CreateShared(context,
                                                     gpu_model.num_layers(),
                                                     gpu_model.num_key_value_heads(),
                                                     gpu_model.head_dim(),
                                                     sequence_capacity,
                                                     "hybrid_kv_cache_gpu",
                                                     error_message);
    if (gpu_cache == nullptr) {
        return run;
    }
    TensorKVCache cpu_cache(cpu_model.num_layers(), 1, cpu_model.num_key_value_heads(), cpu_model.head_dim(), sequence_capacity);
    ::Sampler cpu_sampler(generation.sampler);

    const Clock::time_point start_time = Clock::now();
    context.ResetProfiling();
    if (generation.max_new_tokens == 0) {
        run.wall_ms = DurationMilliseconds(start_time, Clock::now());
        run.gpu_ms = 0.0;
        run.profile = {};
        return run;
    }

    const soc::gpu::DeviceTensor prompt_tensor = UploadGpuTokenIds(context, prompt_token_ids, "hybrid_prompt_tokens", error_message);
    if (!prompt_tensor.IsValid()) {
        return run;
    }
    const soc::gpu::DeviceTensor prompt_hidden = CreateGpuFloatTensor(context,
                                                                      {prompt_token_ids.size(), gpu_model.params().hidden_size},
                                                                      "hybrid_prompt_hidden",
                                                                      error_message);
    if (!prompt_hidden.IsValid()) {
        return run;
    }

    temporary_arena->Reset();
    if (!gpu_model.ForwardHiddenCachedRange(context,
                                            pipeline_cache,
                                            prompt_tensor,
                                            gpu_cache.get(),
                                            prompt_hidden,
                                            temporary_arena,
                                            0,
                                            0,
                                            plan.resolved_gpu_layers,
                                            false,
                                            soc::gpu::RangeCommandStreamMode::kDefault,
                                            error_message)) {
        return run;
    }

    Tensor cpu_hidden = ReadGpuHiddenToCpuTensor(prompt_hidden, error_message);
    Tensor cpu_final_hidden = cpu_model.ForwardHiddenFromStatesCachedRange(cpu_hidden,
                                                                           cpu_cache,
                                                                           plan.resolved_gpu_layers,
                                                                           cpu_model.num_layers(),
                                                                           0,
                                                                           true);
    Tensor logits = cpu_model.ForwardLogitsFromHidden(cpu_final_hidden);
    int next_token_id = cpu_sampler.SampleFromLogits(logits, 0, prompt_token_ids.size() - 1);
    run.generated_token_ids.push_back(next_token_id);

    std::vector<int> running_token_ids = prompt_token_ids;
    running_token_ids.push_back(next_token_id);
    for (std::size_t step = 1; step < generation.max_new_tokens; ++step) {
        if (generation.eos_token_id >= 0 && running_token_ids.back() == generation.eos_token_id) {
            break;
        }

        const soc::gpu::DeviceTensor decode_tensor = UploadGpuTokenIds(context,
                                                                       {running_token_ids.back()},
                                                                       "hybrid_decode_token",
                                                                       error_message);
        if (!decode_tensor.IsValid()) {
            return run;
        }
        const soc::gpu::DeviceTensor decode_hidden = CreateGpuFloatTensor(context,
                                                                          {1, gpu_model.params().hidden_size},
                                                                          "hybrid_decode_hidden",
                                                                          error_message);
        if (!decode_hidden.IsValid()) {
            return run;
        }

        temporary_arena->Reset();
        if (!gpu_model.ForwardHiddenCachedRange(context,
                                                pipeline_cache,
                                                decode_tensor,
                                                gpu_cache.get(),
                                                decode_hidden,
                                                temporary_arena,
                                                running_token_ids.size() - 1,
                                                0,
                                                plan.resolved_gpu_layers,
                                                false,
                                                soc::gpu::RangeCommandStreamMode::kDefault,
                                                error_message)) {
            return run;
        }

        cpu_hidden = ReadGpuHiddenToCpuTensor(decode_hidden, error_message);
        cpu_final_hidden = cpu_model.ForwardHiddenFromStatesCachedRange(cpu_hidden,
                                                                        cpu_cache,
                                                                        plan.resolved_gpu_layers,
                                                                        cpu_model.num_layers(),
                                                                        running_token_ids.size() - 1,
                                                                        true);
        logits = cpu_model.ForwardLogitsFromHidden(cpu_final_hidden);
        next_token_id = cpu_sampler.SampleFromLogits(logits, 0, 0);
        run.generated_token_ids.push_back(next_token_id);
        running_token_ids.push_back(next_token_id);
    }

    run.wall_ms = DurationMilliseconds(start_time, Clock::now());
    run.profile = context.GetProfilingSnapshot();
    run.gpu_ms = run.profile.gpu_ms;
    run.generated_text = TokenizerRuntime(tokenizer_runtime).Decode(run.generated_token_ids);
    return run;
}

std::string BuildPrimaryOutput(const CliOptions& options,
                               const ExecutionPlan& plan,
                               const std::string& prompt_text,
                               const std::string& prepared_prompt,
                               const std::vector<int>& prompt_token_ids,
                               const std::vector<int>& generated_token_ids,
                               const std::string& generated_text,
                               const std::string& prompt_cache_mode,
                               double prefill_ms,
                               double decode_ms,
                               double prompt_cache_load_ms,
                               const soc::gpu::MetalDeviceInfo& device_info,
                               double wall_ms,
                               double gpu_ms,
                               const soc::gpu::MetalProfilingSnapshot& profile,
                               const soc::gpu::DecodePlanRunStats& decode_plan_stats,
                               const soc::gpu::RuntimePolicy& runtime_policy) {
    if (!options.json_output) {
        return generated_text + "\n";
    }

    const std::vector<std::string> cpu_stages = BuildCpuStageDescriptions(plan);
    const std::vector<std::string> gpu_stages = BuildGpuStageDescriptions(plan);

    std::ostringstream stream;
    stream << "{\n";
    stream << "  \"manifest\": \"" << JsonEscape(options.manifest_path) << "\",\n";
    stream << "  \"prompt\": \"" << JsonEscape(prompt_text) << "\",\n";
    stream << "  \"prompt_source\": \"" << (options.prompt_file.empty() ? "argument" : (options.prompt_file == "-" ? "stdin" : "file")) << "\",\n";
    if (!options.prompt_file.empty() && options.prompt_file != "-") {
        stream << "  \"prompt_file\": \"" << JsonEscape(options.prompt_file) << "\",\n";
    }
    if (prepared_prompt != prompt_text) {
        stream << "  \"serialized_prompt\": \"" << JsonEscape(prepared_prompt) << "\",\n";
    }
    stream << "  \"generated_text\": \"" << JsonEscape(generated_text) << "\",\n";
    stream << "  \"prompt_token_ids\": " << JsonArray(prompt_token_ids) << ",\n";
    stream << "  \"generated_token_ids\": " << JsonArray(generated_token_ids) << ",\n";
    stream << "  \"prompt_cache\": {\n";
    stream << "    \"mode\": \"" << JsonEscape(prompt_cache_mode) << "\",\n";
    stream << "    \"artifact_load_ms\": " << prompt_cache_load_ms << "\n";
    stream << "  },\n";
    stream << "  \"execution\": {\n";
    stream << "    \"mode\": \"" << JsonEscape(plan.mode) << "\",\n";
    stream << "    \"requested_layer\": " << plan.requested_layer << ",\n";
    stream << "    \"resolved_gpu_layers\": " << plan.resolved_gpu_layers << ",\n";
    stream << "    \"model_layer_count\": " << plan.model_layer_count << "\n";
    stream << "  },\n";
    stream << "  \"stages\": {\n";
    stream << "    \"cpu\": " << JsonStringArray(cpu_stages) << ",\n";
    stream << "    \"gpu\": " << JsonStringArray(gpu_stages) << "\n";
    stream << "  },\n";
    stream << "  \"device\": {\n";
    stream << "    \"name\": \"" << JsonEscape(device_info.name) << "\",\n";
    stream << "    \"is_apple_silicon_gpu\": " << (device_info.is_apple_silicon_gpu ? "true" : "false") << ",\n";
    stream << "    \"has_unified_memory\": " << (device_info.has_unified_memory ? "true" : "false") << ",\n";
    stream << "    \"recommended_max_working_set_size\": " << device_info.recommended_max_working_set_size << ",\n";
    stream << "    \"thread_execution_width\": " << device_info.thread_execution_width << "\n";
    stream << "  },\n";
    stream << "  \"runtime_policy\": {\n";
    stream << "    \"prefill_step_size\": " << runtime_policy.prefill_step_size << ",\n";
    stream << "    \"command_stream_encoder_budget\": " << runtime_policy.command_stream_encoder_budget << ",\n";
    stream << "    \"working_set_budget_bytes\": " << runtime_policy.working_set_budget_bytes << ",\n";
    stream << "    \"recommended_max_working_set_size\": " << runtime_policy.recommended_max_working_set_size << "\n";
    stream << "  },\n";
    stream << "  \"timing\": {\n";
    stream << "    \"prefill_ms\": " << prefill_ms << ",\n";
    stream << "    \"decode_ms\": " << decode_ms << ",\n";
    stream << "    \"prefill_tok_per_s\": " << (prefill_ms > 0.0 ? (prompt_token_ids.size() * 1000.0 / prefill_ms) : 0.0) << ",\n";
    stream << "    \"decode_tok_per_s\": " << (decode_ms > 0.0 ? (generated_token_ids.size() * 1000.0 / decode_ms) : 0.0) << ",\n";
    stream << "    \"wall_ms\": " << wall_ms << ",\n";
    stream << "    \"gpu_ms\": " << gpu_ms << ",\n";
    stream << "    \"wait_ms\": " << profile.wait_ms << ",\n";
    stream << "    \"command_buffer_count\": " << profile.command_buffer_count << ",\n";
    stream << "    \"encoder_count\": " << profile.encoder_count << ",\n";
    stream << "    \"entries\": " << JsonProfilingEntries(profile) << "\n";
    stream << "  },\n";
    stream << "  \"decode_plan\": {\n";
    stream << "    \"used_prebuilt_plan\": " << (decode_plan_stats.used_prebuilt_plan ? "true" : "false") << ",\n";
    stream << "    \"stage_count\": " << decode_plan_stats.stage_count << ",\n";
    stream << "    \"layer_stage_count\": " << decode_plan_stats.layer_stage_count << ",\n";
    stream << "    \"execution_group_count\": " << decode_plan_stats.execution_group_count << ",\n";
    stream << "    \"merged_range_count\": " << decode_plan_stats.merged_range_count << ",\n";
    stream << "    \"merged_stage_count\": " << decode_plan_stats.merged_stage_count << ",\n";
    stream << "    \"max_group_size\": " << decode_plan_stats.max_group_size << ",\n";
    stream << "    \"group_sizes\": " << JsonSizeArray(decode_plan_stats.group_sizes) << ",\n";
    stream << "    \"hidden_slot0_blocker_count\": " << decode_plan_stats.hidden_slot0_blocker_count << ",\n";
    stream << "    \"hidden_slot1_blocker_count\": " << decode_plan_stats.hidden_slot1_blocker_count << ",\n";
    stream << "    \"logits_blocker_count\": " << decode_plan_stats.logits_blocker_count << ",\n";
    stream << "    \"kv_keys_blocker_count\": " << decode_plan_stats.kv_keys_blocker_count << ",\n";
    stream << "    \"kv_values_blocker_count\": " << decode_plan_stats.kv_values_blocker_count << ",\n";
    stream << "    \"read_after_write_blocker_count\": " << decode_plan_stats.read_after_write_blocker_count << ",\n";
    stream << "    \"write_after_read_blocker_count\": " << decode_plan_stats.write_after_read_blocker_count << ",\n";
    stream << "    \"write_after_write_blocker_count\": " << decode_plan_stats.write_after_write_blocker_count << ",\n";
    stream << "    \"stage_blockers\": " << JsonDecodePlanStageBlockers(decode_plan_stats.stage_blockers) << "\n";
    stream << "  }\n";
    stream << "}\n";
    return stream.str();
}

void WritePrimaryOutput(const CliOptions& options, const std::string& payload) {
    if (options.output_file.empty()) {
        std::cout << payload;
        return;
    }

    std::filesystem::path output_path(options.output_file);
    std::error_code create_error;
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path(), create_error);
    }
    if (create_error) {
        throw std::runtime_error("failed to create output directory: " + create_error.message());
    }

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + options.output_file);
    }
    output << payload;
}

void PrintVerboseSummary(const CliOptions& options,
                         const ExecutionPlan& plan,
                         const std::string& prepared_prompt,
                         const std::string& prompt_text,
                         const std::vector<int>& prompt_token_ids,
                         const std::vector<int>& generated_token_ids,
                         const std::string& prompt_cache_mode,
                         double prefill_ms,
                         double decode_ms,
                         double prompt_cache_load_ms,
                         const soc::gpu::MetalDeviceInfo& device_info,
                         double wall_ms,
                         double gpu_ms,
                            const soc::gpu::MetalProfilingSnapshot& profile,
                            const soc::gpu::DecodePlanRunStats& decode_plan_stats,
                            const soc::gpu::RuntimePolicy& runtime_policy) {
    std::cerr << "manifest=" << options.manifest_path << "\n";
    if (!options.prompt_file.empty()) {
        std::cerr << "prompt_file=" << options.prompt_file << "\n";
    }
    if (!options.output_file.empty()) {
        std::cerr << "output_file=" << options.output_file << "\n";
    }
    std::cerr << "output_format=" << (options.json_output ? "json" : "plain-text") << "\n";
    std::cerr << "execution_mode=" << plan.mode << "\n";
    std::cerr << "requested_layer=" << plan.requested_layer << " resolved_gpu_layers=" << plan.resolved_gpu_layers << " model_layer_count=" << plan.model_layer_count << "\n";
    std::cerr << "device=" << device_info.name << "\n";
    std::cerr << "recommended_max_working_set_size=" << device_info.recommended_max_working_set_size << "\n";
    std::cerr << "prompt_cache_mode=" << prompt_cache_mode << "\n";
    std::cerr << "runtime_policy_prefill_step_size=" << runtime_policy.prefill_step_size << "\n";
    std::cerr << "runtime_policy_command_stream_encoder_budget=" << runtime_policy.command_stream_encoder_budget << "\n";
    std::cerr << "runtime_policy_working_set_budget_bytes=" << runtime_policy.working_set_budget_bytes << "\n";
    std::cerr << "max_new_tokens=" << generated_token_ids.size() << "\n";
    if (prepared_prompt != prompt_text) {
        std::cerr << "serialized_prompt=" << prepared_prompt << "\n";
    }
    PrintTokenIds(std::cerr, "prompt_token_ids", prompt_token_ids);
    PrintTokenIds(std::cerr, "generated_token_ids", generated_token_ids);
    std::cerr << "wall_ms=" << wall_ms << "\n";
    std::cerr << "gpu_ms=" << gpu_ms << "\n";
    std::cerr << "prefill_ms=" << prefill_ms << "\n";
    std::cerr << "decode_ms=" << decode_ms << "\n";
    std::cerr << "prompt_cache_load_ms=" << prompt_cache_load_ms << "\n";
    std::cerr << "wait_ms=" << profile.wait_ms << "\n";
    std::cerr << "command_buffer_count=" << profile.command_buffer_count << "\n";
    std::cerr << "encoder_count=" << profile.encoder_count << "\n";
    std::cerr << "decode_plan_used_prebuilt_plan=" << (decode_plan_stats.used_prebuilt_plan ? 1 : 0) << "\n";
    std::cerr << "decode_plan_execution_group_count=" << decode_plan_stats.execution_group_count << "\n";
    std::cerr << "decode_plan_merged_range_count=" << decode_plan_stats.merged_range_count << "\n";
    std::cerr << "decode_plan_merged_stage_count=" << decode_plan_stats.merged_stage_count << "\n";
    std::cerr << "decode_plan_max_group_size=" << decode_plan_stats.max_group_size << "\n";
    std::cerr << "decode_plan_hidden_slot0_blocker_count=" << decode_plan_stats.hidden_slot0_blocker_count << "\n";
    std::cerr << "decode_plan_hidden_slot1_blocker_count=" << decode_plan_stats.hidden_slot1_blocker_count << "\n";
    std::cerr << "decode_plan_logits_blocker_count=" << decode_plan_stats.logits_blocker_count << "\n";
    std::cerr << "decode_plan_kv_keys_blocker_count=" << decode_plan_stats.kv_keys_blocker_count << "\n";
    std::cerr << "decode_plan_kv_values_blocker_count=" << decode_plan_stats.kv_values_blocker_count << "\n";
    std::cerr << "decode_plan_read_after_write_blocker_count=" << decode_plan_stats.read_after_write_blocker_count << "\n";
    std::cerr << "decode_plan_write_after_read_blocker_count=" << decode_plan_stats.write_after_read_blocker_count << "\n";
    std::cerr << "decode_plan_write_after_write_blocker_count=" << decode_plan_stats.write_after_write_blocker_count << "\n";
    for (const auto& blocker : decode_plan_stats.stage_blockers) {
        std::cerr << "decode_plan_blocker[" << blocker.stage_index << "] stage=" << blocker.stage_label
                  << " prior_stage_index=" << blocker.prior_stage_index
                  << " prior_stage=" << blocker.prior_stage_label
                  << " buffer_kind=" << DecodePlanBufferKindName(blocker.buffer_kind)
                  << " hazard_kind=" << DecodePlanHazardKindName(blocker.prior_write, blocker.current_write) << "\n";
    }
    for (const auto& entry : profile.entries) {
        std::cerr << "profile[" << entry.label << "] gpu_ms=" << entry.gpu_ms
                  << " wait_ms=" << entry.wait_ms
                  << " command_buffers=" << entry.command_buffer_count
                  << " encoders=" << entry.encoder_count << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = ParseArgs(argc, argv);
        const std::string prompt_text = ResolvePromptText(options);

        const ManifestData manifest = ManifestLoader::LoadFromFile(options.manifest_path);
        const ExecutionPlan plan = ResolveExecutionPlan(options, manifest);
        const TokenizerRuntimeData tokenizer_runtime = TokenizerRuntimeLoader::LoadFromFile(manifest.tokenizer_runtime_file);
        const RuntimeGenerationOptions generation = ResolveGenerationOptions(options, manifest);
        const std::string prepared_prompt = RuntimePipeline::PreparePrompt(tokenizer_runtime, prompt_text, options.prompt_options);
        const TokenizerRuntime tokenizer(tokenizer_runtime);
        const std::vector<int> prompt_token_ids = tokenizer.Encode(prepared_prompt);
        if ((!options.prompt_cache_artifact_save.empty() || !options.prompt_cache_artifact_load.empty()) && plan.mode != "full-gpu") {
            throw std::runtime_error("prompt cache artifacts are only supported in full-gpu mode");
        }

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

        InferenceRunResult run;
        if (plan.mode == "full-cpu") {
            const QwenCausalLM cpu_model = QwenModelLoader::LoadModel(manifest);
            run = RunFullCpuInference(cpu_model, generation, prepared_prompt, tokenizer_runtime);
        } else if (plan.mode == "hybrid") {
            soc::gpu::QwenCausalLMWeights gpu_weights;
            gpu_weights.tie_word_embeddings = true;
            soc::gpu::QwenCausalLM gpu_model(std::move(gpu_weights), {});
            if (!soc::gpu::QwenModelLoader::LoadModelFromFile(*context, options.manifest_path, &gpu_model, &error_message)) {
                std::cerr << "failed to load GPU model: " << error_message << '\n';
                return 1;
            }
            const QwenCausalLM cpu_model = QwenModelLoader::LoadModel(manifest);
            run = RunHybridInference(*context,
                                     &pipeline_cache,
                                     temporary_arena.get(),
                                     gpu_model,
                                     cpu_model,
                                     plan,
                                     generation,
                                     prompt_token_ids,
                                     tokenizer_runtime,
                                     &error_message);
        } else {
            soc::gpu::QwenCausalLMWeights gpu_weights;
            gpu_weights.tie_word_embeddings = true;
            soc::gpu::QwenCausalLM gpu_model(std::move(gpu_weights), {});
            if (!soc::gpu::QwenModelLoader::LoadModelFromFile(*context, options.manifest_path, &gpu_model, &error_message)) {
                std::cerr << "failed to load GPU model: " << error_message << '\n';
                return 1;
            }
            run = RunFullGpuInference(*context,
                                      &pipeline_cache,
                                      temporary_arena.get(),
                                      std::move(gpu_model),
                                      generation,
                                      prompt_token_ids,
                                      options.prompt_cache_artifact_save,
                                      options.prompt_cache_artifact_load,
                                      tokenizer_runtime,
                                      &error_message);
        }
        if (!error_message.empty() && run.generated_token_ids.empty() && generation.max_new_tokens != 0) {
            std::cerr << "generation failed: " << error_message << '\n';
            return 1;
        }

        const std::string payload = BuildPrimaryOutput(options,
                                                       plan,
                                                       prompt_text,
                                                       prepared_prompt,
                                                       prompt_token_ids,
                                                       run.generated_token_ids,
                                                       run.generated_text,
                                                       run.prompt_cache_mode,
                                                       run.prefill_ms,
                                                       run.decode_ms,
                                                       run.prompt_cache_load_ms,
                                                       context->GetDeviceInfo(),
                                                       run.wall_ms,
                                                       run.gpu_ms,
                                                       run.profile,
                                                      run.decode_plan_stats,
                                                      run.runtime_policy);
        WritePrimaryOutput(options, payload);

        if (options.verbose) {
            PrintVerboseSummary(options,
                                plan,
                                prepared_prompt,
                                prompt_text,
                                prompt_token_ids,
                                run.generated_token_ids,
                                run.prompt_cache_mode,
                                run.prefill_ms,
                                run.decode_ms,
                                run.prompt_cache_load_ms,
                                context->GetDeviceInfo(),
                                run.wall_ms,
                                run.gpu_ms,
                                run.profile,
                                run.decode_plan_stats,
                                run.runtime_policy);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "runtime error: " << error.what() << "\n";
        PrintUsageAndExit(argv[0], 1);
    }
}
