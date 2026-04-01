#include "module/qwen_block.h"

#include <cstdlib>
#include <mutex>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "metal/command_stream.h"
#include "op/rms_norm_op.h"
#include "runtime/runtime_policy.h"

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

bool UseExperimentalSafeDecodeBatch() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalBlockPrepBatch() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_BLOCK_PREP_BATCH");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalAttentionFullBatch() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_ATTENTION_FULL_BATCH");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalBlockAttentionBatch() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_BLOCK_ATTENTION_BATCH");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalDeferredMlpWait() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_DEFERRED_MLP_WAIT");
    return value != nullptr && std::string(value) == "1";
}

bool DisablePostNormMlpBatch() {
    const char* value = std::getenv("SOC_GPU_DISABLE_POSTNORM_MLP_BATCH");
    return value != nullptr && std::string(value) == "1";
}

struct DecodeBlockScratch {
    std::shared_ptr<MetalBuffer> attention_output_buffer;
    std::shared_ptr<MetalBuffer> post_attention_norm_buffer;
    std::shared_ptr<MetalBuffer> gate_buffer;
    std::shared_ptr<MetalBuffer> up_buffer;
    std::shared_ptr<MetalBuffer> fused_buffer;
};

bool EnsureScratchBuffer(const MetalContext& context,
                         std::shared_ptr<MetalBuffer>* buffer,
                         std::size_t size_bytes,
                         const std::string& label,
                         std::string* error_message) {
    if (*buffer != nullptr && (*buffer)->GetSizeBytes() >= size_bytes) {
        return true;
    }
    *buffer = MetalBuffer::CreateForTensorClass(context,
                                                size_bytes,
                                                label,
                                                TensorClass::kTemporary,
                                                error_message);
    return *buffer != nullptr;
}

bool AcquireDecodeBlockScratch(const MetalContext& context,
                               std::size_t layer_index,
                               std::size_t hidden_size,
                               std::size_t intermediate_size,
                               DecodeBlockScratch* scratch,
                               std::string* error_message) {
    static std::mutex scratch_mutex;
    static std::vector<DecodeBlockScratch> scratch_by_layer;

    if (scratch == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Decode block scratch output must not be null";
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(scratch_mutex);
    if (scratch_by_layer.size() <= layer_index) {
        scratch_by_layer.resize(layer_index + 1);
    }
    DecodeBlockScratch& slot = scratch_by_layer[layer_index];
    const std::size_t hidden_bytes = hidden_size * sizeof(float);
    const std::size_t intermediate_bytes = intermediate_size * sizeof(float);
    if (!EnsureScratchBuffer(context, &slot.attention_output_buffer, hidden_bytes, "decode_attention_output_" + std::to_string(layer_index), error_message) ||
        !EnsureScratchBuffer(context, &slot.post_attention_norm_buffer, hidden_bytes, "decode_post_attention_norm_" + std::to_string(layer_index), error_message) ||
        !EnsureScratchBuffer(context, &slot.gate_buffer, intermediate_bytes, "decode_gate_" + std::to_string(layer_index), error_message) ||
        !EnsureScratchBuffer(context, &slot.up_buffer, intermediate_bytes, "decode_up_" + std::to_string(layer_index), error_message) ||
        !EnsureScratchBuffer(context, &slot.fused_buffer, intermediate_bytes, "decode_fused_" + std::to_string(layer_index), error_message)) {
        return false;
    }
    *scratch = slot;
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
                      soc::gpu::CommandStream* stream,
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
    soc::gpu::CommandStream prep_stream;
    soc::gpu::CommandStream local_stream;
    soc::gpu::CommandStream* attention_stream = stream;
    soc::gpu::CommandStream* active_stream = stream;
    const std::size_t encoder_budget = decode_mode ? ResolveCommandStreamEncoderBudget(context) : 0;
    const bool use_deferred_mlp_wait =
        decode_mode && stream == nullptr && UseExperimentalSafeDecodeBatch() && UseExperimentalDeferredMlpWait();
    const bool use_block_attention_batch =
        decode_mode && stream == nullptr && UseExperimentalSafeDecodeBatch() &&
        UseExperimentalAttentionFullBatch() && UseExperimentalBlockAttentionBatch();
    const bool use_block_prep_batch =
        decode_mode && stream == nullptr && UseExperimentalSafeDecodeBatch() &&
        UseExperimentalBlockPrepBatch() && !UseExperimentalAttentionFullBatch() && !use_block_attention_batch;
    const bool use_local_decode_batch = decode_mode && stream == nullptr && UseExperimentalSafeDecodeBatch() &&
        !use_block_prep_batch && !use_block_attention_batch;
    const bool use_postnorm_mlp_batch =
        decode_mode && stream == nullptr && !UseExperimentalSafeDecodeBatch() && !DisablePostNormMlpBatch();
    if (use_block_prep_batch) {
        if (!prep_stream.Begin(context, error_message)) {
            return false;
        }
        attention_stream = &prep_stream;
    } else if (use_block_attention_batch) {
        if (!prep_stream.Begin(context, error_message)) {
            return false;
        }
        attention_stream = &prep_stream;
    } else if (use_local_decode_batch) {
        if (!local_stream.Begin(context, error_message)) {
            return false;
        }
        attention_stream = &local_stream;
        active_stream = &local_stream;
    }

    const auto& input_shape = input.GetDesc().GetShape();
    const soc::gpu::TensorDesc hidden_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, input_shape);
    soc::gpu::DeviceTensor input_norm;
    soc::gpu::DeviceTensor attention_output;
    soc::gpu::DeviceTensor post_attention_norm;
    DecodeBlockScratch decode_scratch;
    const bool use_persistent_decode_scratch =
        use_deferred_mlp_wait && AcquireDecodeBlockScratch(context,
                                                           layer_index,
                                                           input_shape[1],
                                                           params.mlp.intermediate_size,
                                                           &decode_scratch,
                                                           error_message);
    if (use_deferred_mlp_wait && !use_persistent_decode_scratch) {
        return false;
    }
    if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &input_norm, error_message)) {
        return false;
    }
    if (use_persistent_decode_scratch) {
        attention_output = soc::gpu::DeviceTensor(decode_scratch.attention_output_buffer, 0, hidden_desc);
        post_attention_norm = soc::gpu::DeviceTensor(decode_scratch.post_attention_norm_buffer, 0, hidden_desc);
    } else {
        if (!AllocateTemporaryTensor(temporary_arena, hidden_desc, &attention_output, error_message) ||
            !AllocateTemporaryTensor(temporary_arena, hidden_desc, &post_attention_norm, error_message)) {
            return false;
        }
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
                                  attention_stream,
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
                                             attention_stream,
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
                                              attention_stream,
                                              error_message);
    if (!attention_ok) {
        return false;
    }

    if (use_block_attention_batch) {
        if (!prep_stream.Flush(context, "DecodeBlockAttentionBatch", error_message)) {
            return false;
        }
        attention_stream = nullptr;
    }

    if (use_local_decode_batch) {
        if (!local_stream.Flush(context, "DecodeBlockAttentionBatch", error_message)) {
            return false;
        }
        if (!local_stream.Begin(context, error_message)) {
            return false;
        }
        active_stream = &local_stream;
    } else if (use_postnorm_mlp_batch) {
        if (!local_stream.Begin(context, error_message)) {
            return false;
        }
        active_stream = &local_stream;
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
                                  use_persistent_decode_scratch ? nullptr : temporary_arena,
                                  active_stream,
                                  error_message)) {
        return false;
    }

    bool split_postnorm_mlp_batch = false;
    if (use_local_decode_batch && encoder_budget > 0 && local_stream.GetEncoderCount() >= encoder_budget) {
        if (!local_stream.Flush(context, "DecodePostNormBatch", error_message)) {
            return false;
        }
        if (!local_stream.Begin(context, error_message)) {
            return false;
        }
        split_postnorm_mlp_batch = true;
    }

    soc::gpu::QwenMlpParams mlp_params = params.mlp;
    mlp_params.add_residual = true;
    soc::gpu::QwenMlpScratch mlp_scratch;
    const soc::gpu::QwenMlpScratch* mlp_scratch_ptr = nullptr;
    if (use_persistent_decode_scratch) {
        const soc::gpu::TensorDesc intermediate_desc =
            soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {input_shape[0], params.mlp.intermediate_size});
        mlp_scratch.gate_tensor = soc::gpu::DeviceTensor(decode_scratch.gate_buffer, 0, intermediate_desc);
        mlp_scratch.up_tensor = soc::gpu::DeviceTensor(decode_scratch.up_buffer, 0, intermediate_desc);
        mlp_scratch.fused_tensor = soc::gpu::DeviceTensor(decode_scratch.fused_buffer, 0, intermediate_desc);
        mlp_scratch_ptr = &mlp_scratch;
    }
    if (!soc::gpu::QwenMLP::Run(context,
                                pipeline_cache,
                                post_attention_norm,
                                &attention_output,
                                weights.mlp,
                                output,
                                mlp_params,
                                use_persistent_decode_scratch ? nullptr : temporary_arena,
                                mlp_scratch_ptr,
                                active_stream,
                                error_message)) {
        return false;
    }

    if (use_local_decode_batch || use_postnorm_mlp_batch) {
        if (use_persistent_decode_scratch) {
            if (!local_stream.FlushDeferred(context,
                                           split_postnorm_mlp_batch ? "DecodeMlpBatch" : "DecodePostNormMlpBatch",
                                           error_message)) {
                return false;
            }
        } else if (!local_stream.Flush(context,
                                       split_postnorm_mlp_batch ? "DecodeMlpBatch" : "DecodePostNormMlpBatch",
                                       error_message)) {
            return false;
        }
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
                    CommandStream* stream,
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
                            stream,
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
                           CommandStream* stream,
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
                            stream,
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
                          CommandStream* stream,
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
                            stream,
                            error_message);
}

}  // namespace soc::gpu
