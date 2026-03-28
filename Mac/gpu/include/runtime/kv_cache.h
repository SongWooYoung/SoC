#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensor/device_tensor.h"

namespace soc::gpu {

class MetalContext;

struct KVCacheLayerView {
    DeviceTensor keys;
    DeviceTensor values;
    std::size_t sequence_length = 0;
};

class KVCache {
public:
    static std::unique_ptr<KVCache> CreateShared(const MetalContext& context,
                                                 std::size_t layer_count,
                                                 std::size_t key_value_head_count,
                                                 std::size_t head_dim,
                                                 std::size_t max_sequence_length,
                                                 const std::string& label,
                                                 std::string* error_message);

    ~KVCache();

    bool Reserve(std::size_t sequence_length, std::string* error_message);
    bool AppendPrefill(const MetalContext& context,
                       std::size_t layer_index,
                       const DeviceTensor& keys,
                       const DeviceTensor& values,
                       std::string* error_message);
    bool AppendDecodeToken(const MetalContext& context,
                           std::size_t layer_index,
                           const DeviceTensor& key,
                           const DeviceTensor& value,
                           std::string* error_message);
    KVCacheLayerView ViewForLayer(std::size_t layer_index) const;
    std::size_t GetLayerCount() const;
    std::size_t GetKeyValueHeadCount() const;
    std::size_t GetHeadDim() const;
    std::size_t GetMaxSequenceLength() const;

private:
    KVCache(std::shared_ptr<class MetalBuffer> key_buffer,
            std::shared_ptr<class MetalBuffer> value_buffer,
            std::size_t layer_count,
            std::size_t key_value_head_count,
            std::size_t head_dim,
            std::size_t max_sequence_length);

    std::size_t LayerByteOffset(std::size_t layer_index) const;
    bool CopyIntoLayer(const MetalContext& context,
                       std::shared_ptr<class MetalBuffer> destination,
                       std::size_t layer_index,
                       std::size_t sequence_offset,
                       const DeviceTensor& source,
                       std::string* error_message);

    std::shared_ptr<class MetalBuffer> key_buffer_;
    std::shared_ptr<class MetalBuffer> value_buffer_;
    std::size_t layer_count_ = 0;
    std::size_t key_value_head_count_ = 0;
    std::size_t head_dim_ = 0;
    std::size_t max_sequence_length_ = 0;
    std::vector<std::size_t> sequence_lengths_;
};

}  // namespace soc::gpu