#pragma once

#include <cstddef>
#include <string>

namespace soc::gpu {

class BufferArena;
class DecodePlanner;
class DeviceTensor;
class KVCache;
class MetalContext;
class PipelineCache;

class ModelRunner {
public:
    virtual ~ModelRunner() = default;

    virtual std::size_t num_layers() const = 0;
    virtual std::size_t hidden_size() const = 0;
    virtual std::size_t num_key_value_heads() const = 0;
    virtual std::size_t head_dim() const = 0;
    virtual std::size_t vocab_size() const = 0;
    virtual std::size_t max_position_embeddings() const = 0;

    virtual bool ForwardLogitsCached(const MetalContext& context,
                                     PipelineCache* pipeline_cache,
                                     const DeviceTensor& token_ids,
                                     KVCache* kv_cache,
                                     const DeviceTensor& logits_output,
                                     BufferArena* temporary_arena,
                                     std::size_t position_offset,
                                     std::string* error_message) const = 0;

    virtual bool ForwardLogitsFromHidden(const MetalContext& context,
                                         PipelineCache* pipeline_cache,
                                         const DeviceTensor& hidden_states,
                                         const DeviceTensor& logits_output,
                                         BufferArena* temporary_arena,
                                         std::string* error_message) const = 0;

    virtual DecodePlanner* GetDecodePlanner() = 0;
    virtual const DecodePlanner* GetDecodePlanner() const = 0;
};

}  // namespace soc::gpu
