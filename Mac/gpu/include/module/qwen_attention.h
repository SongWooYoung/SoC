#pragma once

#include <cstddef>
#include <string>

#include "kernel/pipeline_cache.h"
#include "runtime/kv_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct QwenAttentionWeights {
    DeviceTensor q_proj_weight;
    DeviceTensor k_proj_weight;
    DeviceTensor v_proj_weight;
    DeviceTensor o_proj_weight;
    DeviceTensor q_norm_weight;
    DeviceTensor k_norm_weight;
};

struct QwenAttentionParams {
    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;
    std::size_t rotary_dim = 0;
    std::size_t position_offset = 0;
    float rope_theta = 1000000.0f;
    float rms_epsilon = 1.0e-6f;
    bool add_residual = false;
};

class QwenAttention {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const DeviceTensor* residual,
                    const QwenAttentionWeights& weights,
                    const DeviceTensor& output,
                    const QwenAttentionParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);

    static bool RunPrefill(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           const DeviceTensor& input,
                           const DeviceTensor* residual,
                           const QwenAttentionWeights& weights,
                           KVCache* kv_cache,
                           std::size_t layer_index,
                           const DeviceTensor& output,
                           const QwenAttentionParams& params,
                           BufferArena* temporary_arena,
                           std::string* error_message);

    static bool RunDecode(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          const DeviceTensor& input,
                          const DeviceTensor* residual,
                          const QwenAttentionWeights& weights,
                          KVCache* kv_cache,
                          std::size_t layer_index,
                          const DeviceTensor& output,
                          const QwenAttentionParams& params,
                          BufferArena* temporary_arena,
                          std::string* error_message);
};

}  // namespace soc::gpu