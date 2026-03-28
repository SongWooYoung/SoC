#pragma once

#include <cstdint>
#include <string>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct EmbeddingParams {
    std::uint32_t token_count = 0;
    std::uint32_t hidden_size = 0;
    std::uint32_t vocab_size = 0;
};

class EmbeddingOp {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& token_ids,
                    const DeviceTensor& embedding_table,
                    const DeviceTensor& output,
                    const EmbeddingParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);
};

}  // namespace soc::gpu