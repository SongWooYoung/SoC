#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runtime/kv_cache.h"

#include <cstdint>

#include "buffer/metal_buffer.h"
#include "metal/command_stream.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

bool DisableKvCacheBatch() {
    const char* value = std::getenv("SOC_GPU_DISABLE_KV_CACHE_BATCH");
    return value != nullptr && std::string(value) == "1";
}

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

bool CopyBufferToShared(const MetalContext& context,
                       id<MTLBuffer> source,
                       id<MTLBuffer> destination,
                       std::size_t size_bytes,
                       const char* profile_label,
                       std::string* error_message) {
    @autoreleasepool {
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create KVCache serialization blit command objects";
            }
            return false;
        }

        [encoder copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:size_bytes];
        [encoder endEncoding];
        return context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                             "KVCache serialization blit failed",
                                             profile_label,
                                             1,
                                             error_message);
    }
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
    auto key_buffer = MetalBuffer::CreateForTensorClass(context,
                                                        byte_size,
                                                        label + "_keys",
                                                        TensorClass::kKvCache,
                                                        error_message);
    auto value_buffer = MetalBuffer::CreateForTensorClass(context,
                                                          byte_size,
                                                          label + "_values",
                                                          TensorClass::kKvCache,
                                                          error_message);
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
                            CommandStream* stream,
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
    CommandStream local_stream;
    CommandStream* active_stream = stream;
    const bool use_local_decode_batch = row_count == 1 && stream == nullptr && !DisableKvCacheBatch();
    if (use_local_decode_batch) {
        if (!local_stream.Begin(context, error_message)) {
            return false;
        }
        active_stream = &local_stream;
    }

    if (!CopyIntoLayer(context, key_buffer_, layer_index, sequence_offset, keys, active_stream, error_message) ||
        !CopyIntoLayer(context, value_buffer_, layer_index, sequence_offset, values, active_stream, error_message)) {
        return false;
    }

    if (use_local_decode_batch) {
        if (!local_stream.Flush(context, "KVCacheBlitBatch", error_message)) {
            return false;
        }
    }

    sequence_lengths_[layer_index] += row_count;
    return true;
}

bool KVCache::AppendDecodeToken(const MetalContext& context,
                                std::size_t layer_index,
                                const DeviceTensor& key,
                                const DeviceTensor& value,
                                CommandStream* stream,
                                std::string* error_message) {
    if (key.GetDesc().Rank() != 2 || value.GetDesc().Rank() != 2 || key.GetDesc().GetShape()[0] != 1 ||
        value.GetDesc().GetShape()[0] != 1) {
        if (error_message != nullptr) {
            *error_message = "KVCache decode append expects single-row key/value tensors";
        }
        return false;
    }
    return AppendPrefill(context, layer_index, key, value, stream, error_message);
}

bool KVCache::Serialize(const MetalContext& context,
                        KVCacheSerializedState* state,
                        std::string* error_message) const {
    if (state == nullptr) {
        if (error_message != nullptr) {
            *error_message = "KVCache serialize requires a non-null state output";
        }
        return false;
    }

    state->layer_count = layer_count_;
    state->key_value_head_count = key_value_head_count_;
    state->head_dim = head_dim_;
    state->max_sequence_length = max_sequence_length_;
    state->sequence_lengths = sequence_lengths_;
    state->key_bytes.assign(key_buffer_->GetSizeBytes(), 0);
    state->value_bytes.assign(value_buffer_->GetSizeBytes(), 0);

    auto key_staging = MetalBuffer::CreateShared(context, key_buffer_->GetSizeBytes(), "kv_cache_serialize_keys", error_message);
    auto value_staging = MetalBuffer::CreateShared(context, value_buffer_->GetSizeBytes(), "kv_cache_serialize_values", error_message);
    if (key_staging == nullptr || value_staging == nullptr) {
        return false;
    }

    if (!CopyBufferToShared(context,
                            (__bridge id<MTLBuffer>)key_buffer_->GetNativeHandle(),
                            (__bridge id<MTLBuffer>)key_staging->GetNativeHandle(),
                            key_buffer_->GetSizeBytes(),
                            "KVCacheSerializeKeys",
                            error_message) ||
        !CopyBufferToShared(context,
                            (__bridge id<MTLBuffer>)value_buffer_->GetNativeHandle(),
                            (__bridge id<MTLBuffer>)value_staging->GetNativeHandle(),
                            value_buffer_->GetSizeBytes(),
                            "KVCacheSerializeValues",
                            error_message)) {
        return false;
    }

    return key_staging->Read(state->key_bytes.data(), state->key_bytes.size(), 0, error_message) &&
           value_staging->Read(state->value_bytes.data(), state->value_bytes.size(), 0, error_message);
}

std::unique_ptr<KVCache> KVCache::Deserialize(const MetalContext& context,
                                              const KVCacheSerializedState& state,
                                              const std::string& label,
                                              std::string* error_message) {
    if (state.layer_count == 0 || state.key_value_head_count == 0 || state.head_dim == 0 ||
        state.max_sequence_length == 0) {
        if (error_message != nullptr) {
            *error_message = "KVCache deserialize requires non-zero dimensions";
        }
        return nullptr;
    }

    const std::size_t hidden_size = state.key_value_head_count * state.head_dim;
    const std::size_t expected_bytes = state.layer_count * state.max_sequence_length * hidden_size * sizeof(float);
    if (state.sequence_lengths.size() != state.layer_count || state.key_bytes.size() != expected_bytes ||
        state.value_bytes.size() != expected_bytes) {
        if (error_message != nullptr) {
            *error_message = "KVCache deserialize state dimensions do not match payload sizes";
        }
        return nullptr;
    }
    for (std::size_t sequence_length : state.sequence_lengths) {
        if (sequence_length > state.max_sequence_length) {
            if (error_message != nullptr) {
                *error_message = "KVCache deserialize sequence length exceeds max_sequence_length";
            }
            return nullptr;
        }
    }

    auto key_buffer = MetalBuffer::CreatePrivateInitialized(context,
                                                            state.key_bytes.data(),
                                                            state.key_bytes.size(),
                                                            label + "_keys",
                                                            error_message);
    auto value_buffer = MetalBuffer::CreatePrivateInitialized(context,
                                                              state.value_bytes.data(),
                                                              state.value_bytes.size(),
                                                              label + "_values",
                                                              error_message);
    if (key_buffer == nullptr || value_buffer == nullptr) {
        return nullptr;
    }

    auto cache = std::unique_ptr<KVCache>(new KVCache(std::move(key_buffer),
                                                      std::move(value_buffer),
                                                      state.layer_count,
                                                      state.key_value_head_count,
                                                      state.head_dim,
                                                      state.max_sequence_length));
    cache->sequence_lengths_ = state.sequence_lengths;
    return cache;
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

KVCacheByteRange KVCache::DescribeLayerByteRange(std::size_t layer_index) const {
    if (layer_index >= layer_count_) {
        return {};
    }
    return {LayerByteOffset(layer_index), GetLayerSpanBytes()};
}

KVCacheByteRange KVCache::DescribeLayerAppendByteRange(std::size_t layer_index, std::size_t row_count) const {
    if (layer_index >= layer_count_ || row_count == 0) {
        return {};
    }
    const std::size_t hidden_size = key_value_head_count_ * head_dim_;
    const std::size_t row_bytes = hidden_size * sizeof(float);
    return {LayerByteOffset(layer_index) + sequence_lengths_[layer_index] * row_bytes, row_count * row_bytes};
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

std::size_t KVCache::GetLayerSpanBytes() const {
    const std::size_t hidden_size = key_value_head_count_ * head_dim_;
    return max_sequence_length_ * hidden_size * sizeof(float);
}

std::size_t KVCache::GetSequenceLengthForLayer(std::size_t layer_index) const {
    if (layer_index >= layer_count_) {
        return 0;
    }
    return sequence_lengths_[layer_index];
}

std::size_t KVCache::LayerByteOffset(std::size_t layer_index) const {
    return layer_index * GetLayerSpanBytes();
}

bool KVCache::CopyIntoLayer(const MetalContext& context,
                            std::shared_ptr<MetalBuffer> destination,
                            std::size_t layer_index,
                            std::size_t sequence_offset,
                            const DeviceTensor& source,
                            CommandStream* stream,
                            std::string* error_message) {
    const std::size_t hidden_size = key_value_head_count_ * head_dim_;
    const std::size_t row_count = source.GetDesc().GetShape()[0];
    const std::size_t row_bytes = hidden_size * sizeof(float);
    const std::size_t destination_offset = LayerByteOffset(layer_index) + sequence_offset * row_bytes;

    @autoreleasepool {
        id<MTLBuffer> source_buffer = (__bridge id<MTLBuffer>)source.GetBuffer()->GetNativeHandle();
        id<MTLBuffer> destination_buffer = (__bridge id<MTLBuffer>)destination->GetNativeHandle();

        id<MTLBlitCommandEncoder> encoder = nil;
        id<MTLCommandBuffer> command_buffer = nil;
        if (stream != nullptr) {
            encoder = (__bridge id<MTLBlitCommandEncoder>)stream->BeginBlitEncoder();
        } else {
            id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
            command_buffer = [command_queue commandBuffer];
            encoder = [command_buffer blitCommandEncoder];
        }
        if (encoder == nil) {
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

        if (stream != nullptr) {
            stream->EndEncoder();
        } else {
            [encoder endEncoding];
            if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                               "KVCache blit command failed",
                                               "KVCacheBlit",
                                               1,
                                               error_message)) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace soc::gpu
