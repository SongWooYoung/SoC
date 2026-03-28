#include "module/qwen_block.h"

#include "buffer/buffer_arena.h"
#include "op/rms_norm_op.h"

namespace soc::gpu {
namespace {

bool AllocateTemporaryTensor(BufferArena* arena,
                             const TensorDesc& desc,
                             DeviceTensor* tensor,
                             std::string* error_message) {
    if (arena == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Temporary arena is required for QwenBlock";
        }
        return false;
    }
    BufferArenaSlice slice;
    if (!arena->Allocate(desc.ByteSize(), 256, &slice, error_message)) {
        return false;
    }
    *tensor = DeviceTensor(slice.buffer, slice.offset_bytes, desc);
    return true;
}

}  // namespace

namespace {

bool RunBlockInternal(const soc::gpu::MetalContext& context,
                      soc::gpu::PipelineCache* pipeline_cache,
                      const soc::gpu::DeviceTensor& input,
                      const soc::gpu::QwenBlockWeights& weights,
                      soc::gpu::KVCache* kv_cache,
                      std::size_t layer_index,
                      bool decode_mode,
                      const soc::gpu::DeviceTensor& output,
                      const soc::gpu::QwenBlockParams& params,
                      soc::gpu::BufferArena* temporary_arena,
                      std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }
    if (!input.IsValid() || !output.IsValid() || !weights.input_layernorm_weight.IsValid() ||
        !weights.post_attention_layernorm_weight.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenBlock tensors must be valid";
        }
        return false;
    }
    if (input.GetDesc().GetDataType() != soc::gpu::DataType::kFloat32 ||
        output.GetDesc().GetDataType() != soc::gpu::DataType::kFloat32 ||
        input.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2 || input.GetDesc().GetShape() != output.GetDesc().GetShape()) {
        if (error_message != nullptr) {
            *error_message = "QwenBlock expects matching rank-2 float32 input and output tensors";
        }
        return false;
    }

    soc::gpu::BufferArenaMarkGuard arena_mark(temporary_arena, decode_mode ? "QwenBlockDecode" : "QwenBlock");

    const auto& input_shape = input.GetDesc().GetShape();
    const soc::gpu::TensorDesc hidden_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, input_shape);
    soc::gpu::DeviceTensor input_norm;
    soc::gpu::DeviceTensor attention_output;
    soc::gpu::DeviceTensor post_attention_norm;
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &input_norm, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, hidden_desc, &attention_output, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, hidden_desc, &post_attention_norm, error_message)) {
        return false;
    }

    soc::gpu::RmsNormParams input_norm_params;
    input_norm_params.epsilon = params.rms_epsilon;
    input_norm_params.row_count = static_cast<std::uint32_t>(input_shape[0]);
    input_norm_params.row_size = static_cast<std::uint32_t>(input_shape[1]);
    if (!soc::gpu::RmsNormOp::Run(context,
                                  pipeline_cache,
                                  input,
                                  weights.input_layernorm_weight,
                                  input_norm,
                                  input_norm_params,
                                  temporary_arena,
                                  error_message)) {
        return false;
    }

    soc::gpu::QwenAttentionParams attention_params = params.attention;
    attention_params.add_residual = true;
    attention_params.rms_epsilon = params.rms_epsilon;
    const bool attention_ok = decode_mode
        ? soc::gpu::QwenAttention::RunDecode(context,
                                             pipeline_cache,
                                             input_norm,
                                             &input,
                                             weights.attention,
                                             kv_cache,
                                             layer_index,
                                             attention_output,
                                             attention_params,
                                             temporary_arena,
                                             error_message)
        : soc::gpu::QwenAttention::RunPrefill(context,
                                              pipeline_cache,
                                              input_norm,
                                              &input,
                                              weights.attention,
                                              kv_cache,
                                              layer_index,
                                              attention_output,
                                              attention_params,
                                              temporary_arena,
                                              error_message);
    if (!attention_ok) {
        return false;
    }

    soc::gpu::RmsNormParams post_norm_params;
    post_norm_params.epsilon = params.rms_epsilon;
    post_norm_params.row_count = static_cast<std::uint32_t>(input_shape[0]);
    post_norm_params.row_size = static_cast<std::uint32_t>(input_shape[1]);
    if (!soc::gpu::RmsNormOp::Run(context,
                                  pipeline_cache,
                                  attention_output,
                                  weights.post_attention_layernorm_weight,
                                  post_attention_norm,
                                  post_norm_params,
                                  temporary_arena,
                                  error_message)) {
        return false;
    }

    soc::gpu::QwenMlpParams mlp_params = params.mlp;
    mlp_params.add_residual = true;
    if (!soc::gpu::QwenMLP::Run(context,
                                pipeline_cache,
                                post_attention_norm,
                                &attention_output,
                                weights.mlp,
                                output,
                                mlp_params,
                                temporary_arena,
                                error_message)) {
        return false;
    }

    return true;
}

}  // namespace

bool QwenBlock::Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const QwenBlockWeights& weights,
                    const DeviceTensor& output,
                    const QwenBlockParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message) {
    return RunBlockInternal(context,
                            pipeline_cache,
                            input,
                            weights,
                            nullptr,
                            0,
                            false,
                            output,
                            params,
                            temporary_arena,
                            error_message);
}

bool QwenBlock::RunPrefill(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           const DeviceTensor& input,
                           const QwenBlockWeights& weights,
                           KVCache* kv_cache,
                           std::size_t layer_index,
                           const DeviceTensor& output,
                           const QwenBlockParams& params,
                           BufferArena* temporary_arena,
                           std::string* error_message) {
    return RunBlockInternal(context,
                            pipeline_cache,
                            input,
                            weights,
                            kv_cache,
                            layer_index,
                            false,
                            output,
                            params,
                            temporary_arena,
                            error_message);
}

bool QwenBlock::RunDecode(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          const DeviceTensor& input,
                          const QwenBlockWeights& weights,
                          KVCache* kv_cache,
                          std::size_t layer_index,
                          const DeviceTensor& output,
                          const QwenBlockParams& params,
                          BufferArena* temporary_arena,
                          std::string* error_message) {
    return RunBlockInternal(context,
                            pipeline_cache,
                            input,
                            weights,
                            kv_cache,
                            layer_index,
                            true,
                            output,
                            params,
                            temporary_arena,
                            error_message);
}

}  // namespace soc::gpu