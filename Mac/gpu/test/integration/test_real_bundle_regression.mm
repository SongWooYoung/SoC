#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unordered_set>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "model/qwen_causal_lm.h"
#include "model/qwen_model_loader.h"
#include "runtime/command_scheduler.h"
#include "runtime/generation_context.h"
#include "runtime/kv_cache.h"
#include "runtime/sampler.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

#include "header/generation_session.h"
#include "header/runtime_pipeline.h"
#include "header/storage.h"
#include "header/tensor.h"
#include "header/tokenizer_runtime.h"

namespace {

using Clock = std::chrono::steady_clock;

struct TimingStats {
    double wall_ms = 0.0;
    double cpu_ms = 0.0;
    double gpu_ms = 0.0;
    double cpu_active_ratio = 0.0;
    double gpu_active_ratio = 0.0;
};

struct GenerationRun {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string generated_text;
    int first_token_id = -1;
    std::vector<int> first_top_token_ids;
    std::vector<float> first_top_logits;
    TimingStats prefill_timing;
    TimingStats decode_timing;
    TimingStats total_timing;
    std::size_t peak_temp_bytes = 0;
    double working_set_ratio = 0.0;
};

struct RegressionCase {
    std::string name;
    std::string prepared_prompt;
};

double ReadProcessCpuMilliseconds() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }

    const double user_ms = static_cast<double>(usage.ru_utime.tv_sec) * 1000.0 +
                           static_cast<double>(usage.ru_utime.tv_usec) / 1000.0;
    const double system_ms = static_cast<double>(usage.ru_stime.tv_sec) * 1000.0 +
                             static_cast<double>(usage.ru_stime.tv_usec) / 1000.0;
    return user_ms + system_ms;
}

TimingStats FinishTiming(const Clock::time_point& wall_start, double cpu_start_ms, double gpu_ms = 0.0) {
    TimingStats stats;
    stats.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
    stats.cpu_ms = ReadProcessCpuMilliseconds() - cpu_start_ms;
    stats.gpu_ms = gpu_ms;
    stats.cpu_active_ratio = stats.wall_ms <= 0.0 ? 0.0 : stats.cpu_ms / stats.wall_ms;
    stats.gpu_active_ratio = stats.wall_ms <= 0.0 ? 0.0 : stats.gpu_ms / stats.wall_ms;
    return stats;
}

std::string ResolveManifestPath() {
    const char* manifest_env = std::getenv("SOC_QWEN3_MANIFEST");
    if (manifest_env != nullptr && manifest_env[0] != '\0') {
        return manifest_env;
    }
    return "../../models/cpp/qwen3-0.6b/manifest.json";
}

std::string ResolvePrompt() {
    const char* prompt_env = std::getenv("SOC_QWEN_PROMPT");
    if (prompt_env != nullptr && prompt_env[0] != '\0') {
        return prompt_env;
    }
    return "Summarize how CPU and GPU scheduling differ in one short paragraph.";
}

std::size_t ResolveMaxNewTokens() {
    const char* max_new_tokens_env = std::getenv("SOC_QWEN_MAX_NEW_TOKENS");
    if (max_new_tokens_env != nullptr && max_new_tokens_env[0] != '\0') {
        return static_cast<std::size_t>(std::strtoull(max_new_tokens_env, nullptr, 10));
    }
    return 8;
}

std::filesystem::path ResolveReportPath() {
    const char* report_env = std::getenv("SOC_GPU_REPORT_PATH");
    if (report_env != nullptr && report_env[0] != '\0') {
        return report_env;
    }
    return "build/reports/test_real_bundle_regression_report.md";
}

void LogProgress(const std::string& message) {
    std::cout << "[test_real_bundle_regression] " << message << std::endl;
}

soc::gpu::DeviceTensor UploadTokenIds(const soc::gpu::MetalContext& context,
                                      const std::vector<int>& token_ids,
                                      const std::string& label,
                                      std::string* error_message) {
    auto token_buffer = soc::gpu::MetalBuffer::CreateShared(context,
                                                            token_ids.size() * sizeof(std::int32_t),
                                                            label,
                                                            error_message);
    if (token_buffer == nullptr) {
        return {};
    }
    if (!token_ids.empty() && !token_buffer->Write(token_ids.data(), token_ids.size() * sizeof(std::int32_t), 0, error_message)) {
        return {};
    }
    return soc::gpu::DeviceTensor(token_buffer,
                                  0,
                                  soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kInt32, {token_ids.size()}));
}

soc::gpu::DeviceTensor MakeLogitsTensor(const std::shared_ptr<soc::gpu::MetalBuffer>& buffer,
                                        std::size_t row_count,
                                        std::size_t vocab_size) {
    return soc::gpu::DeviceTensor(buffer,
                                  0,
                                  soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {row_count, vocab_size}));
}

bool ReadLogitRow(const soc::gpu::DeviceTensor& logits,
                  std::size_t row_index,
                  std::vector<float>* row,
                  std::string* error_message) {
    if (row == nullptr) {
        if (error_message != nullptr) {
            *error_message = "read logit row requires a non-null output";
        }
        return false;
    }
    const std::size_t vocab_size = logits.GetDesc().GetShape()[1];
    row->assign(vocab_size, 0.0f);
    return logits.GetBuffer()->Read(row->data(),
                                    vocab_size * sizeof(float),
                                    logits.GetByteOffset() + row_index * vocab_size * sizeof(float),
                                    error_message);
}

std::vector<int> TopKFromRow(const std::vector<float>& row, std::size_t top_k) {
    std::vector<std::size_t> ranked(row.size(), 0);
    for (std::size_t index = 0; index < row.size(); ++index) {
        ranked[index] = index;
    }

    const std::size_t bounded_top_k = std::min(top_k, row.size());
    std::partial_sort(ranked.begin(),
                      ranked.begin() + static_cast<std::ptrdiff_t>(bounded_top_k),
                      ranked.end(),
                      [&row](std::size_t lhs, std::size_t rhs) {
                          return row[lhs] > row[rhs];
                      });

    std::vector<int> top_token_ids;
    top_token_ids.reserve(bounded_top_k);
    for (std::size_t rank = 0; rank < bounded_top_k; ++rank) {
        top_token_ids.push_back(static_cast<int>(ranked[rank]));
    }
    return top_token_ids;
}

RuntimeGenerationOptions BuildGenerationOptions(const ManifestData& manifest) {
    RuntimeGenerationOptions options = RuntimePipeline::GenerationOptionsFromManifest(manifest);
    options.max_new_tokens = ResolveMaxNewTokens();
    options.sampler.temperature = 1.0f;
    options.sampler.top_k = std::max<std::size_t>(8, options.sampler.top_k);
    options.max_sequence_length = std::max<std::size_t>(options.max_sequence_length, 512);
    return options;
}

GenerationRun RunCpuGeneration(const RuntimeBundle& bundle,
                               const RuntimeGenerationOptions& options,
                               const std::string& prepared_prompt) {
    GenerationSession session = RuntimePipeline::CreateSession(bundle, options);
    const TokenizerRuntime tokenizer(bundle.tokenizer_runtime);

    GenerationRun run;
    const Clock::time_point total_wall_start = Clock::now();
    const double total_cpu_start_ms = ReadProcessCpuMilliseconds();

    const Clock::time_point prefill_wall_start = Clock::now();
    const double prefill_cpu_start_ms = ReadProcessCpuMilliseconds();
    run.prompt_token_ids = session.Prefill(prepared_prompt);
    run.prefill_timing = FinishTiming(prefill_wall_start, prefill_cpu_start_ms);

    std::vector<int> running_token_ids = run.prompt_token_ids;
    const Clock::time_point decode_wall_start = Clock::now();
    const double decode_cpu_start_ms = ReadProcessCpuMilliseconds();
    for (std::size_t step = 0; step < options.max_new_tokens; ++step) {
        const int token_id = session.DecodeNextToken(running_token_ids);
        if (step == 0) {
            run.first_token_id = token_id;
        }
        run.generated_token_ids.push_back(token_id);
        running_token_ids.push_back(token_id);
        if (options.eos_token_id >= 0 && token_id == options.eos_token_id) {
            break;
        }
    }
    run.decode_timing = FinishTiming(decode_wall_start, decode_cpu_start_ms);
    run.total_timing = FinishTiming(total_wall_start, total_cpu_start_ms);
    run.generated_text = tokenizer.Decode(run.generated_token_ids);
    return run;
}

soc::gpu::SamplerConfig BuildGpuSamplerConfig(const RuntimeGenerationOptions& options) {
    soc::gpu::SamplerConfig config;
    config.temperature = options.sampler.temperature;
    config.top_k = options.sampler.top_k;
    return config;
}

std::size_t ResolveSequenceCapacity(const RuntimeGenerationOptions& options, std::size_t prompt_token_count) {
    return std::max<std::size_t>(options.max_sequence_length, prompt_token_count + options.max_new_tokens + 8);
}

bool CompareCpuAndGpuFirstToken(const RuntimeBundle& cpu_bundle,
                                const RuntimeGenerationOptions& options,
                                const std::vector<int>& prompt_token_ids,
                                const soc::gpu::MetalContext& context,
                                soc::gpu::PipelineCache* pipeline_cache,
                                const soc::gpu::QwenCausalLM& gpu_model,
                                const soc::gpu::Sampler& gpu_sampler,
                                const soc::gpu::CommandScheduler& scheduler,
                                soc::gpu::BufferArena* temporary_arena,
                                int* cpu_token_id,
                                std::vector<int>* cpu_top_token_ids,
                                int* gpu_token_id,
                                std::vector<int>* gpu_top_token_ids,
                                float* max_abs_diff,
                                std::string* error_message) {
    const Tensor cpu_token_tensor(Storage::FromOwnedCopy(prompt_token_ids.data(), prompt_token_ids.size() * sizeof(int)),
                                  DType::Int32,
                                  {1, prompt_token_ids.size()});
    const Tensor cpu_logits = cpu_bundle.model.ForwardLogits(cpu_token_tensor);
    const std::size_t cpu_position_index = prompt_token_ids.size() - 1;

    const Sampler cpu_sampler(options.sampler);
    *cpu_token_id = cpu_sampler.SampleFromLogits(cpu_logits, 0, cpu_position_index);

    const std::size_t cpu_vocab_size = cpu_logits.shape()[2];
    const float* cpu_row = cpu_logits.data<float>() + cpu_position_index * cpu_vocab_size;
    const std::vector<float> cpu_row_vector(cpu_row, cpu_row + cpu_vocab_size);
    *cpu_top_token_ids = TopKFromRow(cpu_row_vector, options.sampler.top_k);

    auto kv_cache = soc::gpu::KVCache::CreateShared(context,
                                                    gpu_model.num_layers(),
                                                    gpu_model.num_key_value_heads(),
                                                    gpu_model.head_dim(),
                                                    std::max<std::size_t>(prompt_token_ids.size() + options.max_new_tokens + 8,
                                                                          options.max_sequence_length),
                                                    "first_token_regression_kv",
                                                    error_message);
    if (kv_cache == nullptr) {
        return false;
    }

    const soc::gpu::DeviceTensor token_tensor = UploadTokenIds(context, prompt_token_ids, "first_token_prompt", error_message);
    if (!token_tensor.IsValid()) {
        return false;
    }

    auto logits_buffer = soc::gpu::MetalBuffer::CreateShared(context,
                                                             prompt_token_ids.size() * gpu_model.vocab_size() * sizeof(float),
                                                             "first_token_logits",
                                                             error_message);
    if (logits_buffer == nullptr) {
        return false;
    }
    const soc::gpu::DeviceTensor logits_tensor = MakeLogitsTensor(logits_buffer, prompt_token_ids.size(), gpu_model.vocab_size());

    temporary_arena->Reset();
    if (!scheduler.RunPrefill(context,
                              pipeline_cache,
                              gpu_model,
                              token_tensor,
                              kv_cache.get(),
                              logits_tensor,
                              temporary_arena,
                              0,
                              error_message)) {
        return false;
    }

    std::vector<float> gpu_top_logits;
    if (!gpu_sampler.SampleFromLogits(logits_tensor,
                                      prompt_token_ids.size() - 1,
                                      gpu_token_id,
                                      &gpu_top_logits,
                                      gpu_top_token_ids,
                                      error_message)) {
        return false;
    }

    std::vector<float> gpu_row;
    if (!ReadLogitRow(logits_tensor, prompt_token_ids.size() - 1, &gpu_row, error_message)) {
        return false;
    }

    std::unordered_set<int> candidate_ids(cpu_top_token_ids->begin(), cpu_top_token_ids->end());
    candidate_ids.insert(gpu_top_token_ids->begin(), gpu_top_token_ids->end());
    *max_abs_diff = 0.0f;
    for (int token_id : candidate_ids) {
        *max_abs_diff = std::max(*max_abs_diff,
                                 std::fabs(cpu_row_vector[static_cast<std::size_t>(token_id)] -
                                           gpu_row[static_cast<std::size_t>(token_id)]));
    }
    return true;
}

bool RunGpuManualGeneration(const TokenizerRuntime& tokenizer,
                            const RuntimeGenerationOptions& options,
                            const std::vector<int>& prompt_token_ids,
                            const soc::gpu::MetalContext& context,
                            soc::gpu::PipelineCache* pipeline_cache,
                            const soc::gpu::QwenCausalLM& model,
                            const soc::gpu::Sampler& sampler,
                            const soc::gpu::CommandScheduler& scheduler,
                            soc::gpu::BufferArena* temporary_arena,
                            std::uint64_t recommended_working_set_size,
                            GenerationRun* run,
                            std::string* error_message) {
    if (run == nullptr) {
        if (error_message != nullptr) {
            *error_message = "RunGpuManualGeneration requires a non-null run output";
        }
        return false;
    }

    run->prompt_token_ids = prompt_token_ids;
    run->generated_token_ids.clear();
    run->generated_text.clear();
    run->first_token_id = -1;
    run->first_top_token_ids.clear();
    run->first_top_logits.clear();
    run->prefill_timing = {};
    run->decode_timing = {};
    run->total_timing = {};
    run->peak_temp_bytes = 0;
    run->working_set_ratio = 0.0;

    temporary_arena->Reset();
    temporary_arena->ClearTelemetry();

    auto kv_cache = soc::gpu::KVCache::CreateShared(context,
                                                    model.num_layers(),
                                                    model.num_key_value_heads(),
                                                    model.head_dim(),
                                                    ResolveSequenceCapacity(options, prompt_token_ids.size()),
                                                    "manual_generation_kv",
                                                    error_message);
    if (kv_cache == nullptr) {
        return false;
    }

    const Clock::time_point total_wall_start = Clock::now();
    const double total_cpu_start_ms = ReadProcessCpuMilliseconds();

    const soc::gpu::DeviceTensor prompt_tensor = UploadTokenIds(context, prompt_token_ids, "manual_prompt", error_message);
    if (!prompt_tensor.IsValid()) {
        return false;
    }

    auto prefill_logits_buffer = soc::gpu::MetalBuffer::CreateShared(context,
                                                                     prompt_token_ids.size() * model.vocab_size() * sizeof(float),
                                                                     "manual_prefill_logits",
                                                                     error_message);
    if (prefill_logits_buffer == nullptr) {
        return false;
    }
    const soc::gpu::DeviceTensor prefill_logits = MakeLogitsTensor(prefill_logits_buffer,
                                                                   prompt_token_ids.size(),
                                                                   model.vocab_size());

    const Clock::time_point prefill_wall_start = Clock::now();
    const double prefill_cpu_start_ms = ReadProcessCpuMilliseconds();
    context.ResetProfiling();
    if (!scheduler.RunPrefill(context,
                              pipeline_cache,
                              model,
                              prompt_tensor,
                              kv_cache.get(),
                              prefill_logits,
                              temporary_arena,
                              0,
                              error_message)) {
        return false;
    }
    const soc::gpu::MetalProfilingSnapshot prefill_profile = context.GetProfilingSnapshot();
    if (!sampler.SampleFromLogits(prefill_logits,
                                  prompt_token_ids.size() - 1,
                                  &run->first_token_id,
                                  &run->first_top_logits,
                                  &run->first_top_token_ids,
                                  error_message)) {
        return false;
    }
    run->generated_token_ids.push_back(run->first_token_id);
    run->prefill_timing = FinishTiming(prefill_wall_start, prefill_cpu_start_ms, prefill_profile.gpu_ms);

    std::vector<int> running_token_ids = prompt_token_ids;
    running_token_ids.push_back(run->first_token_id);
    if (!(options.eos_token_id >= 0 && run->first_token_id == options.eos_token_id)) {
        const Clock::time_point decode_wall_start = Clock::now();
        const double decode_cpu_start_ms = ReadProcessCpuMilliseconds();
        context.ResetProfiling();
        for (std::size_t step = 1; step < options.max_new_tokens; ++step) {
            const int last_token_id = running_token_ids.back();
            const soc::gpu::DeviceTensor decode_input = UploadTokenIds(context, {last_token_id}, "manual_decode_token", error_message);
            if (!decode_input.IsValid()) {
                return false;
            }

            auto decode_logits_buffer = soc::gpu::MetalBuffer::CreateShared(context,
                                                                            model.vocab_size() * sizeof(float),
                                                                            "manual_decode_logits",
                                                                            error_message);
            if (decode_logits_buffer == nullptr) {
                return false;
            }
            const soc::gpu::DeviceTensor decode_logits = MakeLogitsTensor(decode_logits_buffer, 1, model.vocab_size());

            temporary_arena->Reset();
            if (!scheduler.RunDecode(context,
                                     pipeline_cache,
                                     model,
                                     decode_input,
                                     kv_cache.get(),
                                     decode_logits,
                                     temporary_arena,
                                     running_token_ids.size() - 1,
                                     error_message)) {
                return false;
            }

            int next_token_id = -1;
            std::vector<float> top_logits;
            std::vector<int> top_token_ids;
            if (!sampler.SampleFromLogits(decode_logits,
                                          0,
                                          &next_token_id,
                                          &top_logits,
                                          &top_token_ids,
                                          error_message)) {
                return false;
            }
            run->generated_token_ids.push_back(next_token_id);
            running_token_ids.push_back(next_token_id);
            if (options.eos_token_id >= 0 && next_token_id == options.eos_token_id) {
                break;
            }
        }
        const soc::gpu::MetalProfilingSnapshot decode_profile = context.GetProfilingSnapshot();
        run->decode_timing = FinishTiming(decode_wall_start, decode_cpu_start_ms, decode_profile.gpu_ms);
    }

    run->total_timing = FinishTiming(total_wall_start,
                                     total_cpu_start_ms,
                                     run->prefill_timing.gpu_ms + run->decode_timing.gpu_ms);
    run->generated_text = tokenizer.Decode(run->generated_token_ids);
    run->peak_temp_bytes = temporary_arena->GetPeakUsedBytes();
    if (recommended_working_set_size > 0) {
        run->working_set_ratio = static_cast<double>(run->peak_temp_bytes) /
                                 static_cast<double>(recommended_working_set_size);
    }
    return true;
}

bool RunGpuGenerationContext(const TokenizerRuntime& tokenizer,
                             const RuntimeGenerationOptions& options,
                             const std::vector<int>& prompt_token_ids,
                             const soc::gpu::MetalContext& context,
                             soc::gpu::PipelineCache* pipeline_cache,
                             const soc::gpu::QwenCausalLM& model,
                             const soc::gpu::Sampler& sampler,
                             const soc::gpu::CommandScheduler& scheduler,
                             soc::gpu::BufferArena* temporary_arena,
                             std::uint64_t recommended_working_set_size,
                             GenerationRun* run,
                             std::string* error_message) {
    if (run == nullptr) {
        if (error_message != nullptr) {
            *error_message = "RunGpuGenerationContext requires a non-null run output";
        }
        return false;
    }

    run->prompt_token_ids = prompt_token_ids;
    run->generated_token_ids.clear();
    run->generated_text.clear();
    run->first_top_token_ids.clear();
    run->first_top_logits.clear();
    run->prefill_timing = {};
    run->decode_timing = {};
    run->total_timing = {};
    run->peak_temp_bytes = 0;
    run->working_set_ratio = 0.0;

    temporary_arena->Reset();
    temporary_arena->ClearTelemetry();

    soc::gpu::GenerationContext generation_context(model,
                                                   sampler,
                                                   scheduler,
                                                   ResolveSequenceCapacity(options, prompt_token_ids.size()));
    const Clock::time_point total_wall_start = Clock::now();
    const double total_cpu_start_ms = ReadProcessCpuMilliseconds();
    context.ResetProfiling();
    if (!generation_context.Generate(context,
                                     pipeline_cache,
                                     prompt_token_ids,
                                     options.max_new_tokens,
                                     options.eos_token_id,
                                     temporary_arena,
                                     &run->generated_token_ids,
                                     error_message)) {
        return false;
    }
    const soc::gpu::MetalProfilingSnapshot total_profile = context.GetProfilingSnapshot();
    run->total_timing = FinishTiming(total_wall_start, total_cpu_start_ms, total_profile.gpu_ms);
    run->generated_text = tokenizer.Decode(run->generated_token_ids);
    run->first_token_id = run->generated_token_ids.empty() ? -1 : run->generated_token_ids.front();
    run->peak_temp_bytes = temporary_arena->GetPeakUsedBytes();
    if (recommended_working_set_size > 0) {
        run->working_set_ratio = static_cast<double>(run->peak_temp_bytes) /
                                 static_cast<double>(recommended_working_set_size);
    }
    return true;
}

std::string JoinTokenIds(const std::vector<int>& token_ids) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << token_ids[index];
    }
    return stream.str();
}

std::string FormatDouble(double value, int precision = 3) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

double SafeDivide(double numerator, double denominator) {
    return denominator <= 0.0 ? 0.0 : numerator / denominator;
}

void AppendCaseReport(std::ostringstream* report,
                      const RegressionCase& regression_case,
                      const GenerationRun& cpu_run,
                      const GenerationRun& gpu_manual_run,
                      const GenerationRun& gpu_context_run,
                      float max_abs_logit_diff,
                      const soc::gpu::MetalDeviceInfo& device_info) {
    *report << "## " << regression_case.name << "\n\n";
    *report << "- Prompt tokens: " << cpu_run.prompt_token_ids.size() << "\n";
    *report << "- Generated tokens: " << cpu_run.generated_token_ids.size() << "\n";
    *report << "- First token parity: CPU=" << cpu_run.first_token_id
            << ", GPU=" << gpu_manual_run.first_token_id << "\n";
    *report << "- First top-k ids: [" << JoinTokenIds(gpu_manual_run.first_top_token_ids) << "]\n";
    *report << "- Max abs logit diff on compared candidates: " << FormatDouble(max_abs_logit_diff, 6) << "\n";
    *report << "- CPU text: " << cpu_run.generated_text << "\n";
    *report << "- GPU text: " << gpu_context_run.generated_text << "\n\n";

    *report << "| Metric | CPU | GPU manual | GPU context |\n";
    *report << "| --- | ---: | ---: | ---: |\n";
    *report << "| Total wall ms | " << FormatDouble(cpu_run.total_timing.wall_ms)
            << " | " << FormatDouble(gpu_manual_run.total_timing.wall_ms)
            << " | " << FormatDouble(gpu_context_run.total_timing.wall_ms) << " |\n";
    *report << "| Total CPU ms | " << FormatDouble(cpu_run.total_timing.cpu_ms)
            << " | " << FormatDouble(gpu_manual_run.total_timing.cpu_ms)
            << " | " << FormatDouble(gpu_context_run.total_timing.cpu_ms) << " |\n";
        *report << "| Total GPU ms | n/a"
            << " | " << FormatDouble(gpu_manual_run.total_timing.gpu_ms)
            << " | " << FormatDouble(gpu_context_run.total_timing.gpu_ms) << " |\n";
    *report << "| CPU active ratio | " << FormatDouble(cpu_run.total_timing.cpu_active_ratio)
            << " | " << FormatDouble(gpu_manual_run.total_timing.cpu_active_ratio)
            << " | " << FormatDouble(gpu_context_run.total_timing.cpu_active_ratio) << " |\n";
        *report << "| GPU active ratio | n/a"
            << " | " << FormatDouble(gpu_manual_run.total_timing.gpu_active_ratio, 6)
            << " | " << FormatDouble(gpu_context_run.total_timing.gpu_active_ratio, 6) << " |\n";
    *report << "| Prefill wall ms | " << FormatDouble(cpu_run.prefill_timing.wall_ms)
            << " | " << FormatDouble(gpu_manual_run.prefill_timing.wall_ms)
            << " | n/a |\n";
    *report << "| Decode wall ms | " << FormatDouble(cpu_run.decode_timing.wall_ms)
            << " | " << FormatDouble(gpu_manual_run.decode_timing.wall_ms)
            << " | n/a |\n";
        *report << "| Prefill GPU ms | n/a"
            << " | " << FormatDouble(gpu_manual_run.prefill_timing.gpu_ms)
            << " | n/a |\n";
        *report << "| Decode GPU ms | n/a"
            << " | " << FormatDouble(gpu_manual_run.decode_timing.gpu_ms)
            << " | n/a |\n";
        *report << "| Prefill ms / prompt token | "
            << FormatDouble(SafeDivide(cpu_run.prefill_timing.wall_ms, static_cast<double>(cpu_run.prompt_token_ids.size())), 6)
            << " | "
            << FormatDouble(SafeDivide(gpu_manual_run.prefill_timing.wall_ms, static_cast<double>(gpu_manual_run.prompt_token_ids.size())), 6)
            << " | n/a |\n";
        *report << "| Decode ms / generated token | "
            << FormatDouble(SafeDivide(cpu_run.decode_timing.wall_ms, static_cast<double>(std::max<std::size_t>(cpu_run.generated_token_ids.size(), 1))), 6)
            << " | "
            << FormatDouble(SafeDivide(gpu_manual_run.decode_timing.wall_ms, static_cast<double>(std::max<std::size_t>(gpu_manual_run.generated_token_ids.size(), 1))), 6)
            << " | "
            << FormatDouble(SafeDivide(gpu_context_run.total_timing.wall_ms, static_cast<double>(std::max<std::size_t>(gpu_context_run.generated_token_ids.size(), 1))), 6)
            << " |\n";
        *report << "| Tokens / sec | "
            << FormatDouble(SafeDivide(static_cast<double>(cpu_run.generated_token_ids.size()) * 1000.0, cpu_run.total_timing.wall_ms), 6)
            << " | "
            << FormatDouble(SafeDivide(static_cast<double>(gpu_manual_run.generated_token_ids.size()) * 1000.0, gpu_manual_run.total_timing.wall_ms), 6)
            << " | "
            << FormatDouble(SafeDivide(static_cast<double>(gpu_context_run.generated_token_ids.size()) * 1000.0, gpu_context_run.total_timing.wall_ms), 6)
            << " |\n";
    *report << "| Temp peak bytes | n/a | " << gpu_manual_run.peak_temp_bytes
            << " | " << gpu_context_run.peak_temp_bytes << " |\n";
    *report << "| Temp working-set ratio | n/a | " << FormatDouble(gpu_manual_run.working_set_ratio, 6)
            << " | " << FormatDouble(gpu_context_run.working_set_ratio, 6) << " |\n\n";

    *report << "- CPU:GPU context wall ratio = "
            << FormatDouble(cpu_run.total_timing.wall_ms / std::max(gpu_context_run.total_timing.wall_ms, 1.0e-9)) << "\n";
    *report << "- CPU:GPU context host-CPU ratio = "
            << FormatDouble(cpu_run.total_timing.cpu_ms / std::max(gpu_context_run.total_timing.cpu_ms, 1.0e-9)) << "\n";
        *report << "- CPU decode share of total wall = "
            << FormatDouble(SafeDivide(cpu_run.decode_timing.wall_ms, cpu_run.total_timing.wall_ms), 6) << "\n";
        *report << "- GPU manual prefill share of total wall = "
            << FormatDouble(SafeDivide(gpu_manual_run.prefill_timing.wall_ms, gpu_manual_run.total_timing.wall_ms), 6) << "\n";
        *report << "- GPU manual decode share of total wall = "
            << FormatDouble(SafeDivide(gpu_manual_run.decode_timing.wall_ms, gpu_manual_run.total_timing.wall_ms), 6) << "\n";
        *report << "- GPU context command-buffer busy share of total wall = "
            << FormatDouble(gpu_context_run.total_timing.gpu_active_ratio, 6) << "\n";
        *report << "- GPU memory pressure proxy uses temporary arena peak / recommended working set ("
            << device_info.recommended_max_working_set_size << " bytes).\n\n";
}

bool WriteReport(const std::filesystem::path& report_path,
                 const std::string& manifest_path,
                 const soc::gpu::MetalDeviceInfo& device_info,
                 const std::ostringstream& report,
                 std::string* error_message) {
    std::error_code create_error;
    std::filesystem::create_directories(report_path.parent_path(), create_error);
    if (create_error) {
        if (error_message != nullptr) {
            *error_message = "failed to create report directory: " + create_error.message();
        }
        return false;
    }

    std::ofstream output(report_path);
    if (!output) {
        if (error_message != nullptr) {
            *error_message = "failed to open report path for writing";
        }
        return false;
    }

    output << "# GPU Real-Bundle Regression Report\n\n";
    output << "- Manifest: " << manifest_path << "\n";
    output << "- Device: " << device_info.name << "\n";
    output << "- Unified memory: " << (device_info.has_unified_memory ? "yes" : "no") << "\n";
    output << "- Thread execution width: " << device_info.thread_execution_width << "\n";
    output << "- Recommended max working set size: " << device_info.recommended_max_working_set_size << "\n";
    output << "- Utilization note: CPU usage is measured as process CPU time / wall time. GPU execution time is measured from Metal command-buffer GPU timestamps and reported as GPU ms and GPU active ratio. Temporary arena working-set ratio remains a separate memory-pressure proxy.\n\n";
    output << report.str();
    return true;
}

}  // namespace

int main() {
    try {
        const std::string manifest_path = ResolveManifestPath();
        const RuntimeBundle cpu_bundle = RuntimePipeline::LoadBundle(manifest_path);
        const RuntimeGenerationOptions options = BuildGenerationOptions(cpu_bundle.manifest);
        const TokenizerRuntime tokenizer(cpu_bundle.tokenizer_runtime);
        const std::string raw_prompt = ResolvePrompt();

        LogProgress("starting with manifest=" + manifest_path +
                    ", max_new_tokens=" + std::to_string(options.max_new_tokens));

        RuntimePromptOptions chat_options;
        chat_options.apply_chat_template = true;
        chat_options.add_generation_prompt = true;
        chat_options.enable_thinking = true;

        const std::vector<RegressionCase> cases = {
            {"Raw Prompt", raw_prompt},
            {"Chat Template Prompt", RuntimePipeline::PreparePrompt(cpu_bundle.tokenizer_runtime, raw_prompt, chat_options)},
        };

        std::string error_message;
        auto context = soc::gpu::MetalContext::CreateDefault("build/shaders/gpu.metallib",
                                                             "shaders/gpu_kernels.metal",
                                                             &error_message);
        if (context == nullptr) {
            std::cerr << "failed to create Metal context: " << error_message << '\n';
            return 1;
        }

        soc::gpu::PipelineCache pipeline_cache(*context);
        auto temporary_arena = soc::gpu::BufferArena::CreateShared(*context, 1ull << 26, "integration_temp", &error_message);
        if (temporary_arena == nullptr) {
            std::cerr << "failed to create temporary arena: " << error_message << '\n';
            return 1;
        }

        soc::gpu::QwenCausalLM gpu_model({{}, {}, {}, {}, true}, {});
        if (!soc::gpu::QwenModelLoader::LoadModelFromFile(*context, manifest_path, &gpu_model, &error_message)) {
            std::cerr << "failed to load GPU model: " << error_message << '\n';
            return 1;
        }

        const soc::gpu::Sampler gpu_sampler(BuildGpuSamplerConfig(options));
        const soc::gpu::CommandScheduler scheduler;
        const soc::gpu::MetalDeviceInfo device_info = context->GetDeviceInfo();

        std::ostringstream report;
        for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
            const RegressionCase& regression_case = cases[case_index];
            LogProgress("case " + std::to_string(case_index + 1) + "/" + std::to_string(cases.size()) +
                        ": " + regression_case.name + " - CPU reference generation");
            const GenerationRun cpu_run = RunCpuGeneration(cpu_bundle, options, regression_case.prepared_prompt);
            LogProgress("case " + std::to_string(case_index + 1) + "/" + std::to_string(cases.size()) +
                        ": " + regression_case.name + " - prompt_tokens=" +
                        std::to_string(cpu_run.prompt_token_ids.size()) +
                        ", generated_tokens=" + std::to_string(cpu_run.generated_token_ids.size()));

            int cpu_first_token_id = -1;
            int gpu_first_token_id = -1;
            std::vector<int> cpu_top_token_ids;
            std::vector<int> gpu_top_token_ids;
            float max_abs_logit_diff = 0.0f;
            LogProgress("case " + std::to_string(case_index + 1) + "/" + std::to_string(cases.size()) +
                        ": " + regression_case.name + " - first-token parity check");
            if (!CompareCpuAndGpuFirstToken(cpu_bundle,
                                            options,
                                            cpu_run.prompt_token_ids,
                                            *context,
                                            &pipeline_cache,
                                            gpu_model,
                                            gpu_sampler,
                                            scheduler,
                                            temporary_arena.get(),
                                            &cpu_first_token_id,
                                            &cpu_top_token_ids,
                                            &gpu_first_token_id,
                                            &gpu_top_token_ids,
                                            &max_abs_logit_diff,
                                            &error_message)) {
                std::cerr << "first-token regression failed for " << regression_case.name << ": " << error_message << '\n';
                return 1;
            }

            if (cpu_first_token_id != gpu_first_token_id) {
                std::cerr << "first-token mismatch for " << regression_case.name
                          << ": CPU=" << cpu_first_token_id
                          << " GPU=" << gpu_first_token_id
                          << " prompt_tokens=" << cpu_run.prompt_token_ids.size()
                          << " cpu_topk=[" << JoinTokenIds(cpu_top_token_ids) << "]"
                          << " gpu_topk=[" << JoinTokenIds(gpu_top_token_ids) << "]"
                          << " max_abs_diff=" << max_abs_logit_diff << '\n';
                return 1;
            }
            if (cpu_top_token_ids != gpu_top_token_ids) {
                std::cerr << "top-k token mismatch for " << regression_case.name << '\n';
                return 1;
            }
            if (max_abs_logit_diff > 1.0e-3f) {
                std::cerr << "logit drift too large for " << regression_case.name
                          << ": " << max_abs_logit_diff << '\n';
                return 1;
            }

            GenerationRun gpu_manual_run;
            LogProgress("case " + std::to_string(case_index + 1) + "/" + std::to_string(cases.size()) +
                        ": " + regression_case.name + " - manual GPU generation");
            if (!RunGpuManualGeneration(tokenizer,
                                        options,
                                        cpu_run.prompt_token_ids,
                                        *context,
                                        &pipeline_cache,
                                        gpu_model,
                                        gpu_sampler,
                                        scheduler,
                                        temporary_arena.get(),
                                        device_info.recommended_max_working_set_size,
                                        &gpu_manual_run,
                                        &error_message)) {
                std::cerr << "manual GPU generation failed for " << regression_case.name << ": " << error_message << '\n';
                return 1;
            }

            GenerationRun gpu_context_run;
            LogProgress("case " + std::to_string(case_index + 1) + "/" + std::to_string(cases.size()) +
                        ": " + regression_case.name + " - GenerationContext GPU generation");
            if (!RunGpuGenerationContext(tokenizer,
                                         options,
                                         cpu_run.prompt_token_ids,
                                         *context,
                                         &pipeline_cache,
                                         gpu_model,
                                         gpu_sampler,
                                         scheduler,
                                         temporary_arena.get(),
                                         device_info.recommended_max_working_set_size,
                                         &gpu_context_run,
                                         &error_message)) {
                std::cerr << "GenerationContext GPU generation failed for " << regression_case.name << ": " << error_message << '\n';
                return 1;
            }

            if (cpu_run.generated_token_ids != gpu_manual_run.generated_token_ids) {
                std::cerr << "manual GPU sequence mismatch for " << regression_case.name << "\n"
                          << "cpu: [" << JoinTokenIds(cpu_run.generated_token_ids) << "]\n"
                          << "gpu: [" << JoinTokenIds(gpu_manual_run.generated_token_ids) << "]\n";
                return 1;
            }
            if (cpu_run.generated_token_ids != gpu_context_run.generated_token_ids) {
                std::cerr << "GenerationContext sequence mismatch for " << regression_case.name << "\n"
                          << "cpu: [" << JoinTokenIds(cpu_run.generated_token_ids) << "]\n"
                          << "gpu: [" << JoinTokenIds(gpu_context_run.generated_token_ids) << "]\n";
                return 1;
            }
            if (gpu_manual_run.generated_token_ids != gpu_context_run.generated_token_ids) {
                std::cerr << "manual/context GPU mismatch for " << regression_case.name << '\n';
                return 1;
            }

            AppendCaseReport(&report,
                             regression_case,
                             cpu_run,
                             gpu_manual_run,
                             gpu_context_run,
                             max_abs_logit_diff,
                             device_info);
            LogProgress("case " + std::to_string(case_index + 1) + "/" + std::to_string(cases.size()) +
                        ": " + regression_case.name + " - completed");
        }

        const std::filesystem::path report_path = ResolveReportPath();
        if (!WriteReport(report_path, manifest_path, device_info, report, &error_message)) {
            std::cerr << "failed to write report: " << error_message << '\n';
            return 1;
        }

        LogProgress("report written to " + report_path.string());
        std::cout << "test_real_bundle_regression passed\n";
        std::cout << "report written to " << report_path.string() << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "integration regression failed: " << exception.what() << '\n';
        return 1;
    }
}