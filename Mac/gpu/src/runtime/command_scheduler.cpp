#include "runtime/command_scheduler.h"

#include <cstdlib>

#include "tensor/tensor_desc.h"

namespace soc::gpu {
namespace {

class HazardTracker {
public:
    bool CanMerge(const std::vector<DecodePlanAccessRange>& accesses) const {
        for (const DecodePlanAccessRange& access : accesses) {
            for (const DecodePlanAccessRange& prior : active_accesses_) {
                if (prior.buffer_kind != access.buffer_kind) {
                    continue;
                }
                const std::size_t prior_end = prior.byte_offset + prior.byte_size;
                const std::size_t access_end = access.byte_offset + access.byte_size;
                const bool overlaps = prior.byte_offset < access_end && access.byte_offset < prior_end;
                if (overlaps && (prior.write || access.write)) {
                    return false;
                }
            }
        }
        return true;
    }

    void Record(const std::vector<DecodePlanAccessRange>& accesses) {
        active_accesses_.insert(active_accesses_.end(), accesses.begin(), accesses.end());
    }

    void Reset() { active_accesses_.clear(); }

private:
    std::vector<DecodePlanAccessRange> active_accesses_;
};

bool UseExperimentalPrebuiltDecodePlan() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQ4Decode() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalSafeDecodeBatch() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH");
    return value != nullptr && std::string(value) == "1";
}

std::vector<DecodePlanAccessRange> StageAccesses(const DecodePlanStage& stage,
                                                 const QwenCausalLM& model,
                                                 const KVCache& kv_cache) {
    const std::size_t hidden_bytes = model.params().hidden_size * sizeof(float);
    const std::size_t logits_bytes = model.vocab_size() * sizeof(float);
    switch (stage.kind) {
        case DecodePlanStage::Kind::kEmbedAndFirstLayer:
        case DecodePlanStage::Kind::kLayer: {
            const KVCacheByteRange kv_range = kv_cache.DescribeLayerAppendByteRange(stage.start_layer, 1);
            return {
                {stage.input_slot == 0 ? DecodePlanBufferKind::kHiddenSlot0 : DecodePlanBufferKind::kHiddenSlot1,
                 0,
                 hidden_bytes,
                 false},
                {stage.output_slot == 0 ? DecodePlanBufferKind::kHiddenSlot0 : DecodePlanBufferKind::kHiddenSlot1,
                 0,
                 hidden_bytes,
                 true},
                {DecodePlanBufferKind::kKvKeys, kv_range.byte_offset, kv_range.byte_size, true},
                {DecodePlanBufferKind::kKvValues, kv_range.byte_offset, kv_range.byte_size, true},
            };
        }
        case DecodePlanStage::Kind::kLogits:
            return {
                {stage.input_slot == 0 ? DecodePlanBufferKind::kHiddenSlot0 : DecodePlanBufferKind::kHiddenSlot1,
                 0,
                 hidden_bytes,
                 false},
                {DecodePlanBufferKind::kLogits, 0, logits_bytes, true},
            };
    }
    return {};
}

void RefreshDecodePlanAccesses(DecodeExecutionPlan* plan, const QwenCausalLM& model, const KVCache& kv_cache) {
    if (plan == nullptr) {
        return;
    }

    HazardTracker tracker;
    std::size_t batch_id = 0;
    for (DecodePlanStage& stage : plan->stages) {
        stage.accesses = StageAccesses(stage, model, kv_cache);
        if (!tracker.CanMerge(stage.accesses)) {
            tracker.Reset();
            ++batch_id;
        }
        stage.batch_id = batch_id;
        tracker.Record(stage.accesses);
    }
}

std::unique_ptr<DecodeExecutionPlan> BuildDecodePlan(const QwenCausalLM& model, const KVCache& kv_cache) {
    auto plan = std::make_unique<DecodeExecutionPlan>();
    plan->layer_count = model.num_layers();
    plan->hidden_size = model.params().hidden_size;
    plan->vocab_size = model.vocab_size();
    plan->max_sequence_length = kv_cache.GetMaxSequenceLength();
    plan->q4_decode_enabled = UseExperimentalQ4Decode();
    plan->safe_decode_batch_enabled = UseExperimentalSafeDecodeBatch();

    std::size_t current_slot = 0;
    std::size_t next_slot = 1;
    HazardTracker tracker;
    std::size_t batch_id = 0;

    for (std::size_t layer_index = 0; layer_index < plan->layer_count; ++layer_index) {
        DecodePlanStage stage;
        stage.kind = layer_index == 0 ? DecodePlanStage::Kind::kEmbedAndFirstLayer : DecodePlanStage::Kind::kLayer;
        stage.start_layer = layer_index;
        stage.end_layer = layer_index + 1;
        stage.apply_final_norm = layer_index + 1 == plan->layer_count;
        stage.input_slot = current_slot;
        stage.output_slot = next_slot;
        stage.label = stage.kind == DecodePlanStage::Kind::kEmbedAndFirstLayer ? "DecodePlanEmbedLayer0" : "DecodePlanLayer";

        stage.accesses = StageAccesses(stage, model, kv_cache);
        const std::vector<DecodePlanAccessRange>& accesses = stage.accesses;
        if (!tracker.CanMerge(accesses)) {
            tracker.Reset();
            ++batch_id;
        }
        stage.batch_id = batch_id;
        tracker.Record(accesses);
        plan->stages.push_back(stage);
        std::swap(current_slot, next_slot);
    }

    DecodePlanStage logits_stage;
    logits_stage.kind = DecodePlanStage::Kind::kLogits;
    logits_stage.start_layer = plan->layer_count;
    logits_stage.end_layer = plan->layer_count;
    logits_stage.apply_final_norm = false;
    logits_stage.input_slot = current_slot;
    logits_stage.output_slot = current_slot;
    logits_stage.label = "DecodePlanLogits";
    logits_stage.accesses = StageAccesses(logits_stage, model, kv_cache);
    const std::vector<DecodePlanAccessRange>& logits_accesses = logits_stage.accesses;
    if (!tracker.CanMerge(logits_accesses)) {
        tracker.Reset();
        ++batch_id;
    }
    logits_stage.batch_id = batch_id;
    tracker.Record(logits_accesses);
    plan->stages.push_back(logits_stage);
    RefreshDecodePlanAccesses(plan.get(), model, kv_cache);
    return plan;
}

}  // namespace

CommandScheduler::CommandScheduler(const CommandScheduler&) {}

CommandScheduler& CommandScheduler::operator=(const CommandScheduler&) {
    decode_plan_.reset();
    decode_hidden_buffers_[0].reset();
    decode_hidden_buffers_[1].reset();
    return *this;
}

bool CommandScheduler::RunPrefill(const MetalContext& context,
                                  PipelineCache* pipeline_cache,
                                  const QwenCausalLM& model,
                                  const DeviceTensor& token_ids,
                                  KVCache* kv_cache,
                                  const DeviceTensor& logits_output,
                                  BufferArena* temporary_arena,
                                  std::size_t position_offset,
                                  std::string* error_message) const {
    return model.ForwardLogitsCached(context,
                                     pipeline_cache,
                                     token_ids,
                                     kv_cache,
                                     logits_output,
                                     temporary_arena,
                                     position_offset,
                                     error_message);
}

bool CommandScheduler::RunDecode(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const QwenCausalLM& model,
                                 const DeviceTensor& token_ids,
                                 KVCache* kv_cache,
                                 const DeviceTensor& logits_output,
                                 BufferArena* temporary_arena,
                                 std::size_t position_offset,
                                 std::string* error_message) const {
    const bool use_prebuilt_plan =
        UseExperimentalPrebuiltDecodePlan() &&
        token_ids.IsValid() &&
        token_ids.GetDesc().Rank() == 1 &&
        token_ids.GetDesc().GetShape() == std::vector<std::size_t>{1};
    if (use_prebuilt_plan) {
        return RunDecodeWithPlan(context,
                                 pipeline_cache,
                                 model,
                                 token_ids,
                                 kv_cache,
                                 logits_output,
                                 temporary_arena,
                                 position_offset,
                                 error_message);
    }

    return model.ForwardLogitsCached(context,
                                     pipeline_cache,
                                     token_ids,
                                     kv_cache,
                                     logits_output,
                                     temporary_arena,
                                     position_offset,
                                     error_message);
}

const DecodeExecutionPlan* CommandScheduler::decode_plan() const {
    return decode_plan_.get();
}

bool CommandScheduler::RunDecodeWithPlan(const MetalContext& context,
                                         PipelineCache* pipeline_cache,
                                         const QwenCausalLM& model,
                                         const DeviceTensor& token_ids,
                                         KVCache* kv_cache,
                                         const DeviceTensor& logits_output,
                                         BufferArena* temporary_arena,
                                         std::size_t position_offset,
                                         std::string* error_message) const {
    if (!EnsureDecodePlan(model, *kv_cache)) {
        if (error_message != nullptr) {
            *error_message = "Failed to build decode execution plan";
        }
        return false;
    }
    RefreshDecodePlanAccesses(decode_plan_.get(), model, *kv_cache);
    if (!EnsureHiddenBuffer(context, 0, model.params().hidden_size, error_message) ||
        !EnsureHiddenBuffer(context, 1, model.params().hidden_size, error_message)) {
        return false;
    }
    const TensorDesc hidden_desc = TensorDesc::CreateContiguous(DataType::kFloat32, {1, model.params().hidden_size});
    DeviceTensor hidden_slots[2] = {
        DeviceTensor(decode_hidden_buffers_[0], 0, hidden_desc),
        DeviceTensor(decode_hidden_buffers_[1], 0, hidden_desc),
    };

    for (std::size_t stage_index = 0; stage_index < decode_plan_->stages.size();) {
        const DecodePlanStage& stage = decode_plan_->stages[stage_index];
        switch (stage.kind) {
            case DecodePlanStage::Kind::kEmbedAndFirstLayer:
            case DecodePlanStage::Kind::kLayer: {
                std::size_t group_end_index = stage_index;
                while (group_end_index + 1 < decode_plan_->stages.size()) {
                    const DecodePlanStage& next_stage = decode_plan_->stages[group_end_index + 1];
                    if (next_stage.kind == DecodePlanStage::Kind::kLogits || next_stage.batch_id != stage.batch_id) {
                        break;
                    }
                    group_end_index += 1;
                }

                const DecodePlanStage& group_last_stage = decode_plan_->stages[group_end_index];
                const RangeCommandStreamMode stream_mode = group_end_index > stage_index
                    ? RangeCommandStreamMode::kFullRange
                    : RangeCommandStreamMode::kOff;

                const bool ok = stage.kind == DecodePlanStage::Kind::kEmbedAndFirstLayer
                    ? model.ForwardHiddenCachedRange(context,
                                                     pipeline_cache,
                                                     token_ids,
                                                     kv_cache,
                                                     hidden_slots[group_last_stage.output_slot],
                                                     temporary_arena,
                                                     position_offset,
                                                     stage.start_layer,
                                                     group_last_stage.end_layer,
                                                     group_last_stage.apply_final_norm,
                                                     stream_mode,
                                                     error_message)
                    : model.ForwardHiddenFromStatesCachedRange(context,
                                                               pipeline_cache,
                                                               hidden_slots[stage.input_slot],
                                                               kv_cache,
                                                               hidden_slots[group_last_stage.output_slot],
                                                               temporary_arena,
                                                               position_offset,
                                                               stage.start_layer,
                                                               group_last_stage.end_layer,
                                                               group_last_stage.apply_final_norm,
                                                               stream_mode,
                                                               error_message);
                if (!ok) {
                    return false;
                }
                stage_index = group_end_index + 1;
                break;
            }
            case DecodePlanStage::Kind::kLogits:
                if (!model.ForwardLogitsFromHidden(context,
                                                   pipeline_cache,
                                                   hidden_slots[stage.input_slot],
                                                   logits_output,
                                                   temporary_arena,
                                                   error_message)) {
                    return false;
                }
                stage_index += 1;
                break;
        }
    }

    return true;
}

bool CommandScheduler::EnsureDecodePlan(const QwenCausalLM& model, const KVCache& kv_cache) const {
    if (decode_plan_ != nullptr &&
        decode_plan_->layer_count == model.num_layers() &&
        decode_plan_->hidden_size == model.params().hidden_size &&
        decode_plan_->vocab_size == model.vocab_size() &&
        decode_plan_->max_sequence_length == kv_cache.GetMaxSequenceLength() &&
        decode_plan_->q4_decode_enabled == UseExperimentalQ4Decode() &&
        decode_plan_->safe_decode_batch_enabled == UseExperimentalSafeDecodeBatch()) {
        return true;
    }
    decode_plan_ = BuildDecodePlan(model, kv_cache);
    return decode_plan_ != nullptr;
}

bool CommandScheduler::EnsureHiddenBuffer(const MetalContext& context,
                                          std::size_t slot,
                                          std::size_t hidden_size,
                                          std::string* error_message) const {
    if (slot >= 2) {
        if (error_message != nullptr) {
            *error_message = "Decode hidden buffer slot is out of range";
        }
        return false;
    }
    const std::size_t required_size = hidden_size * sizeof(float);
    if (decode_hidden_buffers_[slot] != nullptr && decode_hidden_buffers_[slot]->GetSizeBytes() >= required_size) {
        return true;
    }
    decode_hidden_buffers_[slot] =
        MetalBuffer::CreatePrivate(context, required_size, "decode_hidden_plan_slot", error_message);
    return decode_hidden_buffers_[slot] != nullptr;
}

}  // namespace soc::gpu
