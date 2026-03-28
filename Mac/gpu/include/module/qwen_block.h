#pragma once

#include <cstddef>
#include <string>

#include "module/qwen_attention.h"
#include "module/qwen_mlp.h"
#include "runtime/kv_cache.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct QwenBlockWeights {
    DeviceTensor input_layernorm_weight;
    QwenAttentionWeights attention;
    DeviceTensor post_attention_layernorm_weight;
    QwenMlpWeights mlp;
};

struct QwenBlockParams {
    QwenAttentionParams attention;
    QwenMlpParams mlp;
    float rms_epsilon = 1.0e-6f;
};

class QwenBlock {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const QwenBlockWeights& weights,
                    const DeviceTensor& output,
                    const QwenBlockParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);

    static bool RunPrefill(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           const DeviceTensor& input,
                           const QwenBlockWeights& weights,
                           KVCache* kv_cache,
                           std::size_t layer_index,
                           const DeviceTensor& output,
                           const QwenBlockParams& params,
                           BufferArena* temporary_arena,
                           std::string* error_message);

    static bool RunDecode(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          const DeviceTensor& input,
                          const QwenBlockWeights& weights,
                          KVCache* kv_cache,
                          std::size_t layer_index,
                          const DeviceTensor& output,
                          const QwenBlockParams& params,
                          BufferArena* temporary_arena,
                          std::string* error_message);
};

}  // namespace soc::gpu