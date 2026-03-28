#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "op/linear_op.h"

#include <algorithm>

#include "buffer/buffer_arena.h"
#include "metal/metal_context.h"

namespace soc::gpu {

bool LinearOp::Run(const MetalContext& context,
                   PipelineCache* pipeline_cache,
                   const DeviceTensor& input,
                   const DeviceTensor& weight,
                   const DeviceTensor* bias,
                   const DeviceTensor* residual,
                   const DeviceTensor& output,
                   const LinearParams& params,
                   BufferArena* temporary_arena,
                   std::string* error_message) {
    MatMulParams fused_params = params.matmul;
    fused_params.enable_silu = params.activation == LinearActivation::kSiLU;
    fused_params.add_residual = params.add_residual;
    return MatMulOp::Run(context,
                         pipeline_cache,
                         input,
                         weight,
                         bias,
                         residual,
                         output,
                         fused_params,
                         temporary_arena,
                         error_message);
}

}  // namespace soc::gpu