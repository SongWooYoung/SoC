#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runtime/kv_cache.h"

#include "buffer/metal_buffer.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

bool ValidateAppendTensor(const DeviceTensor& tensor,
                          std::size_t expected_hidden_size,
                          std::string* error_message) {
    if (!tensor.IsValid() || tensor.GetDesc().GetDataType() != DataType::kFloat32 || tensor.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "KVCache append expects a valid rank-2 float32 tensor";
        }
        return false;
    }
    if (tensor.GetDesc().GetShape()[1] != expected_hidden_size) {
        if (error_message != nullptr) {
            *error_message = "KVCache append tensor hidden size does not match cache configuration";
        }
        return false;
    }
    return true;
}

}  // namespace

std::unique_ptr<KVCache> KVCache::CreateShared(const MetalContext& context,
                                               std::size_t layer_count,
                                               std::size_t key_value_head_count,
                                               std::size_t head_dim,
                                               std::size_t max_sequence_length,
                                               const std::string& label,
                                               std::string* error_message) {
    if (layer_count == 0 || key_value_head_count == 0 || head_dim == 0 || max_sequence_length == 0) {
        if (error_message != nullptr) {
            *error_message = "KVCache dimensions must be non-zero";
        }
        return nullptr;
    }
    const std::size_t hidden_size = key_value_head_count * head_dim;
    const std::size_t byte_size = layer_count * max_sequence_length * hidden_size * sizeof(float);
    auto key_buffer = MetalBuffer::CreateShared(context, byte_size, label + "_keys", error_message);
    auto value_buffer = MetalBuffer::CreateShared(context, byte_size, label + "_values", error_message);
    if (key_buffer == nullptr || value_buffer == nullptr) {
        return nullptr;
    }
    return std::unique_ptr<KVCache>(
        new KVCache(std::move(key_buffer), std::move(value_buffer), layer_count, key_value_head_count, head_dim, max_sequence_length));
}

KVCache::KVCache(std::shared_ptr<MetalBuffer> key_buffer,
                 std::shared_ptr<MetalBuffer> value_buffer,
                 std::size_t layer_count,
                 std::size_t key_value_head_count,
                 std::size_t head_dim,
                 std::size_t max_sequence_length)
    : key_buffer_(std::move(key_buffer)),
      value_buffer_(std::move(value_buffer)),
      layer_count_(layer_count),
      key_value_head_count_(key_value_head_count),
      head_dim_(head_dim),
      max_sequence_length_(max_sequence_length),
      sequence_lengths_(layer_count, 0) {}

KVCache::~KVCache() = default;

bool KVCache::Reserve(std::size_t sequence_length, std::string* error_message) {
    if (sequence_length > max_sequence_length_) {
        if (error_message != nullptr) {
            *error_message = "KVCache reserve length exceeds max_sequence_length";
        }
        return false;
    }
    return true;
}

bool KVCache::AppendPrefill(const MetalContext& context,
                            std::size_t layer_index,
                            const DeviceTensor& keys,
                            const DeviceTensor& values,
                            std::string* error_message) {
    if (layer_index >= layer_count_) {
        if (error_message != nullptr) {
            *error_message = "KVCache layer index is out of range";
        }
        return false;
    }
    const std::size_t hidden_size = key_value_head_count_ * head_dim_;
    if (!ValidateAppendTensor(keys, hidden_size, error_message) ||
        !ValidateAppendTensor(values, hidden_size, error_message) ||
        keys.GetDesc().GetShape() != values.GetDesc().GetShape()) {
        if (error_message != nullptr && error_message->empty()) {
            *error_message = "KVCache key/value append tensors must have matching shapes";
        }
        return false;
    }
    const std::size_t row_count = keys.GetDesc().GetShape()[0];
    if (sequence_lengths_[layer_index] + row_count > max_sequence_length_) {
        if (error_message != nullptr) {
            *error_message = "KVCache append would exceed max_sequence_length";
        }
        return false;
    }
    const std::size_t sequence_offset = sequence_lengths_[layer_index];
    if (!CopyIntoLayer(context, key_buffer_, layer_index, sequence_offset, keys, error_message) ||
        !CopyIntoLayer(context, value_buffer_, layer_index, sequence_offset, values, error_message)) {
        return false;
    }
    sequence_lengths_[layer_index] += row_count;
    return true;
}

bool KVCache::AppendDecodeToken(const MetalContext& context,
                                std::size_t layer_index,
                                const DeviceTensor& key,
                                const DeviceTensor& value,
                                std::string* error_message) {
    if (key.GetDesc().Rank() != 2 || value.GetDesc().Rank() != 2 || key.GetDesc().GetShape()[0] != 1 ||
        value.GetDesc().GetShape()[0] != 1) {
        if (error_message != nullptr) {
            *error_message = "KVCache decode append expects single-row key/value tensors";
        }
        return false;
    }
    return AppendPrefill(context, layer_index, key, value, error_message);
}

KVCacheLayerView KVCache::ViewForLayer(std::size_t layer_index) const {
    if (layer_index >= layer_count_) {
        return {};
    }
    const std::size_t hidden_size = key_value_head_count_ * head_dim_;
    const std::size_t sequence_length = sequence_lengths_[layer_index];
    const TensorDesc desc = TensorDesc::CreateContiguous(DataType::kFloat32, {sequence_length, hidden_size});
    return {DeviceTensor(key_buffer_, LayerByteOffset(layer_index), desc),
            DeviceTensor(value_buffer_, LayerByteOffset(layer_index), desc),
            sequence_length};
}

std::size_t KVCache::GetLayerCount() const {
    return layer_count_;
}

std::size_t KVCache::GetKeyValueHeadCount() const {
    return key_value_head_count_;
}

std::size_t KVCache::GetHeadDim() const {
    return head_dim_;
}

std::size_t KVCache::GetMaxSequenceLength() const {
    return max_sequence_length_;
}

std::size_t KVCache::LayerByteOffset(std::size_t layer_index) const {
    const std::size_t hidden_size = key_value_head_count_ * head_dim_;
    return layer_index * max_sequence_length_ * hidden_size * sizeof(float);
}

bool KVCache::CopyIntoLayer(const MetalContext& context,
                            std::shared_ptr<MetalBuffer> destination,
                            std::size_t layer_index,
                            std::size_t sequence_offset,
                            const DeviceTensor& source,
                            std::string* error_message) {
    const std::size_t hidden_size = key_value_head_count_ * head_dim_;
    const std::size_t row_count = source.GetDesc().GetShape()[0];
    const std::size_t row_bytes = hidden_size * sizeof(float);
    const std::size_t destination_offset = LayerByteOffset(layer_index) + sequence_offset * row_bytes;

    @autoreleasepool {
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLBuffer> source_buffer = (__bridge id<MTLBuffer>)source.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> destination_buffer = (__bridge id<MTLBuffer>)destination->GetNativeHandle();
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create KVCache blit command objects";
            }
            return false;
        }

        [encoder copyFromBuffer:source_buffer
                   sourceOffset:source.GetByteOffset()
                       toBuffer:destination_buffer
              destinationOffset:destination_offset
                           size:row_count * row_bytes];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "KVCache blit command failed",
                                           error_message)) {
            return false;
        }
    }

    return true;
}

}  // namespace soc::gpu