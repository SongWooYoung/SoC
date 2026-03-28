#pragma once

#include <cstdint>
#include <string>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct ElementwiseMulParams {
    std::uint32_t row_count = 0;
    std::uint32_t row_size = 0;
};

class ElementwiseMulOp {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& lhs,
                    const DeviceTensor& rhs,
                    const DeviceTensor& output,
                    const ElementwiseMulParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);
};

}  // namespace soc::gpu