#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "buffer/metal_buffer.h"

#include <cstring>

#include "metal/metal_context.h"

namespace soc::gpu {

namespace {

std::string BuildBufferError(const std::string& prefix) {
    return prefix;
}

}  // namespace

struct MetalBuffer::Impl {
    id<MTLBuffer> buffer = nil;
    std::size_t size_bytes = 0;
    bool host_visible = false;
    TensorClass tensor_class = TensorClass::kUnknown;
    MetalBufferStorageMode storage_mode = MetalBufferStorageMode::kShared;
};

namespace {

MetalBufferStorageMode ResolveStorageModeForTensorClass(const TensorClass tensor_class) {
    switch (tensor_class) {
        case TensorClass::kStaticWeight:
        case TensorClass::kKvCache:
            return MetalBufferStorageMode::kPrivate;
        case TensorClass::kTemporary:
        case TensorClass::kTokenMetadata:
        case TensorClass::kStaging:
        case TensorClass::kUnknown:
            return MetalBufferStorageMode::kShared;
    }
    return MetalBufferStorageMode::kShared;
}

}  // namespace

std::shared_ptr<MetalBuffer> MetalBuffer::CreateWithMode(const MetalContext& context,
                                                         const std::size_t size_bytes,
                                                         const std::string& label,
                                                         const MetalBufferStorageMode storage_mode,
                                                         const TensorClass tensor_class,
                                                         std::string* error_message) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
        const MTLResourceOptions options = storage_mode == MetalBufferStorageMode::kPrivate
            ? MTLResourceStorageModePrivate
            : MTLResourceStorageModeShared;
        id<MTLBuffer> buffer = [device newBufferWithLength:size_bytes options:options];
        if (buffer == nil) {
            if (error_message != nullptr) {
                *error_message = BuildBufferError(storage_mode == MetalBufferStorageMode::kPrivate
                    ? "Failed to allocate private Metal buffer"
                    : "Failed to allocate shared Metal buffer");
            }
            return nullptr;
        }

        if (!label.empty()) {
            buffer.label = [NSString stringWithUTF8String:label.c_str()];
        }

        auto impl = std::make_unique<MetalBuffer::Impl>();
        impl->buffer = buffer;
        impl->size_bytes = size_bytes;
        impl->host_visible = storage_mode == MetalBufferStorageMode::kShared;
        impl->tensor_class = tensor_class;
        impl->storage_mode = storage_mode;
        return std::shared_ptr<MetalBuffer>(new MetalBuffer(std::move(impl)));
    }
}

std::shared_ptr<MetalBuffer> MetalBuffer::CreateShared(const MetalContext& context,
                                                       std::size_t size_bytes,
                                                       const std::string& label,
                                                       std::string* error_message) {
    return CreateWithMode(context,
                          size_bytes,
                          label,
                          MetalBufferStorageMode::kShared,
                          TensorClass::kUnknown,
                          error_message);
}

std::shared_ptr<MetalBuffer> MetalBuffer::CreatePrivate(const MetalContext& context,
                                                        std::size_t size_bytes,
                                                        const std::string& label,
                                                        std::string* error_message) {
    return CreateWithMode(context,
                          size_bytes,
                          label,
                          MetalBufferStorageMode::kPrivate,
                          TensorClass::kUnknown,
                          error_message);
}

std::shared_ptr<MetalBuffer> MetalBuffer::CreatePrivateInitialized(const MetalContext& context,
                                                                   const void* source,
                                                                   std::size_t size_bytes,
                                                                   const std::string& label,
                                                                   std::string* error_message) {
    return CreateInitializedForTensorClass(context,
                                           source,
                                           size_bytes,
                                           label,
                                           TensorClass::kUnknown,
                                           error_message);
}

std::shared_ptr<MetalBuffer> MetalBuffer::CreateForTensorClass(const MetalContext& context,
                                                               const std::size_t size_bytes,
                                                               const std::string& label,
                                                               const TensorClass tensor_class,
                                                               std::string* error_message) {
    return CreateWithMode(context,
                          size_bytes,
                          label,
                          ResolveStorageModeForTensorClass(tensor_class),
                          tensor_class,
                          error_message);
}

std::shared_ptr<MetalBuffer> MetalBuffer::CreateInitializedForTensorClass(const MetalContext& context,
                                                                          const void* source,
                                                                          const std::size_t size_bytes,
                                                                          const std::string& label,
                                                                          const TensorClass tensor_class,
                                                                          std::string* error_message) {
    if (source == nullptr) {
        if (error_message != nullptr) {
            *error_message = "CreatePrivateInitialized source must not be null";
        }
        return nullptr;
    }

    const MetalBufferStorageMode storage_mode = ResolveStorageModeForTensorClass(tensor_class);
    if (storage_mode == MetalBufferStorageMode::kShared) {
        auto shared_buffer = CreateWithMode(context,
                            size_bytes,
                            label,
                            MetalBufferStorageMode::kShared,
                            tensor_class,
                            error_message);
        if (shared_buffer == nullptr) {
            return nullptr;
        }
        if (!shared_buffer->Write(source, size_bytes, 0, error_message)) {
            return nullptr;
        }
        return shared_buffer;
    }

    auto private_buffer = CreateWithMode(context,
                                         size_bytes,
                                         label,
                                         MetalBufferStorageMode::kPrivate,
                                         tensor_class,
                                         error_message);
    if (private_buffer == nullptr) {
        return nullptr;
    }

    @autoreleasepool {
        auto staging_buffer = CreateWithMode(context,
                             size_bytes,
                             label + "_staging",
                             MetalBufferStorageMode::kShared,
                             TensorClass::kStaging,
                             error_message);
        if (staging_buffer == nullptr ||
            !staging_buffer->Write(source, size_bytes, 0, error_message)) {
            return nullptr;
        }

        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = BuildBufferError("Failed to create Metal upload command objects");
            }
            return nullptr;
        }

        id<MTLBuffer> destination = (__bridge id<MTLBuffer>)private_buffer->GetNativeHandle();
        [encoder copyFromBuffer:(__bridge id<MTLBuffer>)staging_buffer->GetNativeHandle()
                   sourceOffset:0
                       toBuffer:destination
              destinationOffset:0
                           size:size_bytes];
        [encoder endEncoding];
        if (!context.FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                           "Metal private buffer upload failed",
                                           "BufferUpload",
                                           1,
                                           error_message)) {
            return nullptr;
        }
    }

    return private_buffer;
}

MetalBuffer::MetalBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

MetalBuffer::~MetalBuffer() = default;

std::size_t MetalBuffer::GetSizeBytes() const {
    return impl_->size_bytes;
}

bool MetalBuffer::IsHostVisible() const {
    return impl_->host_visible;
}

TensorClass MetalBuffer::GetTensorClass() const {
    return impl_->tensor_class;
}

MetalBufferStorageMode MetalBuffer::GetStorageMode() const {
    return impl_->storage_mode;
}

bool MetalBuffer::Write(const void* source,
                        std::size_t size_bytes,
                        std::size_t offset_bytes,
                        std::string* error_message) {
    if (source == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Write source must not be null";
        }
        return false;
    }
    if (offset_bytes + size_bytes > impl_->size_bytes) {
        if (error_message != nullptr) {
            *error_message = "Write range exceeds Metal buffer size";
        }
        return false;
    }
    if (!impl_->host_visible) {
        if (error_message != nullptr) {
            *error_message = "Write requires a host-visible Metal buffer";
        }
        return false;
    }
    std::memcpy(static_cast<char*>([impl_->buffer contents]) + offset_bytes, source, size_bytes);
    return true;
}

bool MetalBuffer::Read(void* destination,
                       std::size_t size_bytes,
                       std::size_t offset_bytes,
                       std::string* error_message) const {
    if (destination == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Read destination must not be null";
        }
        return false;
    }
    if (offset_bytes + size_bytes > impl_->size_bytes) {
        if (error_message != nullptr) {
            *error_message = "Read range exceeds Metal buffer size";
        }
        return false;
    }
    if (!impl_->host_visible) {
        if (error_message != nullptr) {
            *error_message = "Read requires a host-visible Metal buffer";
        }
        return false;
    }
    std::memcpy(destination, static_cast<const char*>([impl_->buffer contents]) + offset_bytes, size_bytes);
    return true;
}

const void* MetalBuffer::GetNativeHandle() const {
    return (__bridge const void*)impl_->buffer;
}

}  // namespace soc::gpu
