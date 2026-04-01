#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "buffer/metal_buffer.h"
#include "model/qwen_causal_lm.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

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

struct DecodePlanRunStats {
    bool used_prebuilt_plan = false;
    std::size_t stage_count = 0;
    std::size_t layer_stage_count = 0;
    std::size_t execution_group_count = 0;
    std::size_t merged_range_count = 0;
    std::size_t merged_stage_count = 0;
    std::size_t max_group_size = 0;
    std::vector<std::size_t> group_sizes;
    std::size_t hidden_slot0_blocker_count = 0;
    std::size_t hidden_slot1_blocker_count = 0;
    std::size_t logits_blocker_count = 0;
    std::size_t kv_keys_blocker_count = 0;
    std::size_t kv_values_blocker_count = 0;
    std::size_t read_after_write_blocker_count = 0;
    std::size_t write_after_read_blocker_count = 0;
    std::size_t write_after_write_blocker_count = 0;
    struct StageBlocker {
        std::size_t stage_index = 0;
        std::string stage_label;
        std::size_t prior_stage_index = 0;
        std::string prior_stage_label;
        DecodePlanBufferKind buffer_kind = DecodePlanBufferKind::kHiddenSlot0;
        bool prior_write = false;
        bool current_write = false;
    };
    std::vector<StageBlocker> stage_blockers;
};

class CommandScheduler {
public:
    CommandScheduler() = default;
    CommandScheduler(const CommandScheduler&);
    CommandScheduler& operator=(const CommandScheduler&);
    CommandScheduler(CommandScheduler&&) noexcept = default;
    CommandScheduler& operator=(CommandScheduler&&) noexcept = default;

    bool RunPrefill(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const QwenCausalLM& model,
                    const DeviceTensor& token_ids,
                    KVCache* kv_cache,
                    const DeviceTensor& logits_output,
                    BufferArena* temporary_arena,
                    std::size_t position_offset,
                    std::string* error_message) const;

    bool RunDecode(const MetalContext& context,
                   PipelineCache* pipeline_cache,
                   const QwenCausalLM& model,
                   const DeviceTensor& token_ids,
                   KVCache* kv_cache,
                   const DeviceTensor& logits_output,
                   BufferArena* temporary_arena,
                   std::size_t position_offset,
                   std::string* error_message) const;

    const DecodeExecutionPlan* decode_plan() const;
    const DecodePlanRunStats& last_decode_plan_run_stats() const;

private:
    bool RunDecodeWithPlan(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           const QwenCausalLM& model,
                           const DeviceTensor& token_ids,
                           KVCache* kv_cache,
                           const DeviceTensor& logits_output,
                           BufferArena* temporary_arena,
                           std::size_t position_offset,
                           std::string* error_message) const;
    bool EnsureDecodePlan(const QwenCausalLM& model, const KVCache& kv_cache) const;
    bool EnsureHiddenBuffer(const MetalContext& context,
                            std::size_t slot,
                            std::size_t hidden_size,
                            std::string* error_message) const;

    mutable std::unique_ptr<DecodeExecutionPlan> decode_plan_;
    mutable DecodePlanRunStats last_decode_plan_run_stats_;
    mutable std::shared_ptr<MetalBuffer> decode_hidden_buffers_[2];
};

}  // namespace soc::gpu
