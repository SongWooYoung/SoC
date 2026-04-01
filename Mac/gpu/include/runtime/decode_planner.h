#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace soc::gpu {

class BufferArena;
class DeviceTensor;
class KVCache;
class MetalContext;
class MetalBuffer;
class ModelRunner;
class PipelineCache;

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
        std::string resource_label;
        bool prior_write = false;
        bool current_write = false;
    };
    std::vector<StageBlocker> stage_blockers;
};

class DecodePlanner {
public:
    virtual ~DecodePlanner() = default;

    virtual bool RunDecode(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           const ModelRunner& model,
                           const DeviceTensor& token_ids,
                           KVCache* kv_cache,
                           const DeviceTensor& logits_output,
                           BufferArena* temporary_arena,
                           std::size_t position_offset,
                           std::string* error_message) const = 0;

    virtual const DecodePlanRunStats& last_run_stats() const = 0;
};

}  // namespace soc::gpu
