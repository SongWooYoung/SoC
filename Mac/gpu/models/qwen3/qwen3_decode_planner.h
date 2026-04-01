#pragma once

#include <memory>
#include <string>
#include <vector>

#include "buffer/metal_buffer.h"
#include "runtime/decode_planner.h"

namespace soc::gpu::models::qwen3 {

enum class DecodePlanBufferKind {
    kHiddenSlot0,
    kHiddenSlot1,
    kLogits,
    kKvKeys,
    kKvValues,
};

struct DecodePlanAccessRange {
    DecodePlanBufferKind buffer_kind = DecodePlanBufferKind::kHiddenSlot0;
    std::size_t byte_offset = 0;
    std::size_t byte_size = 0;
    bool write = false;
};

struct DecodePlanStage {
    enum class Kind {
        kEmbedAndFirstLayer,
        kLayer,
        kLogits,
    };

    Kind kind = Kind::kLayer;
    std::size_t start_layer = 0;
    std::size_t end_layer = 0;
    bool apply_final_norm = false;
    std::size_t input_slot = 0;
    std::size_t output_slot = 0;
    std::size_t batch_id = 0;
    const char* label = nullptr;
    std::vector<DecodePlanAccessRange> accesses;
};

struct DecodeExecutionPlan {
    std::size_t layer_count = 0;
    std::size_t hidden_size = 0;
    std::size_t vocab_size = 0;
    std::size_t max_sequence_length = 0;
    bool q4_decode_enabled = false;
    bool safe_decode_batch_enabled = false;
    std::vector<DecodePlanStage> stages;
};

class Qwen3DecodePlanner final : public DecodePlanner {
public:
    Qwen3DecodePlanner() = default;

    bool RunDecode(const MetalContext& context,
                   PipelineCache* pipeline_cache,
                   const ModelRunner& model,
                   const DeviceTensor& token_ids,
                   KVCache* kv_cache,
                   const DeviceTensor& logits_output,
                   BufferArena* temporary_arena,
                   std::size_t position_offset,
                   std::string* error_message) const override;

    const DecodeExecutionPlan* decode_plan() const;
    const DecodePlanRunStats& last_run_stats() const override;

private:
    bool EnsureHiddenBuffer(const MetalContext& context,
                            std::size_t slot,
                            std::size_t hidden_size,
                            std::string* error_message) const;

    mutable std::unique_ptr<DecodeExecutionPlan> decode_plan_;
    mutable DecodePlanRunStats last_run_stats_;
    mutable std::shared_ptr<MetalBuffer> decode_hidden_buffers_[2];
};

}  // namespace soc::gpu::models::qwen3
