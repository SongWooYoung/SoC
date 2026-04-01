#pragma once

#include <cstddef>
#include <string>

namespace soc::gpu {

class BufferArena;
class DeviceTensor;
class KVCache;
class MetalContext;
class PipelineCache;

struct LayerRangeExecutionOptions {
    std::size_t position_offset = 0;
    std::size_t start_layer = 0;
    std::size_t end_layer = 0;
    bool apply_final_norm = false;
};

class LayerRangeModelRunner {
public:
    virtual ~LayerRangeModelRunner() = default;

    virtual bool ForwardHiddenCachedRange(const MetalContext& context,
                                          PipelineCache* pipeline_cache,
                                          const DeviceTensor& token_ids,
                                          KVCache* kv_cache,
                                          const DeviceTensor& output,
                                          BufferArena* temporary_arena,
                                          const LayerRangeExecutionOptions& options,
                                          std::string* error_message) const = 0;

    virtual bool ForwardHiddenFromStatesCachedRange(const MetalContext& context,
                                                    PipelineCache* pipeline_cache,
                                                    const DeviceTensor& hidden_states,
                                                    KVCache* kv_cache,
                                                    const DeviceTensor& output,
                                                    BufferArena* temporary_arena,
                                                    const LayerRangeExecutionOptions& options,
                                                    std::string* error_message) const = 0;
};

}  // namespace soc::gpu
