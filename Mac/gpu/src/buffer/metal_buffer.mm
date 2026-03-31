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
};

std::shared_ptr<MetalBuffer> MetalBuffer::CreateShared(const MetalContext& context,
                                                       std::size_t size_bytes,
                                                       const std::string& label,
                                                       std::string* error_message) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
        id<MTLBuffer> buffer = [device newBufferWithLength:size_bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            if (error_message != nullptr) {
                *error_message = BuildBufferError("Failed to allocate shared Metal buffer");
            }
            return nullptr;
        }

        if (!label.empty()) {
            buffer.label = [NSString stringWithUTF8String:label.c_str()];
        }

        auto impl = std::make_unique<Impl>();
        impl->buffer = buffer;
        impl->size_bytes = size_bytes;
        impl->host_visible = true;
        return std::shared_ptr<MetalBuffer>(new MetalBuffer(std::move(impl)));
    }
}

std::shared_ptr<MetalBuffer> MetalBuffer::CreatePrivate(const MetalContext& context,
                                                        std::size_t size_bytes,
                                                        const std::string& label,
                                                        std::string* error_message) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
        id<MTLBuffer> buffer = [device newBufferWithLength:size_bytes options:MTLResourceStorageModePrivate];
        if (buffer == nil) {
            if (error_message != nullptr) {
                *error_message = BuildBufferError("Failed to allocate private Metal buffer");
            }
            return nullptr;
        }

        if (!label.empty()) {
            buffer.label = [NSString stringWithUTF8String:label.c_str()];
        }

        auto impl = std::make_unique<Impl>();
        impl->buffer = buffer;
        impl->size_bytes = size_bytes;
        impl->host_visible = false;
        return std::shared_ptr<MetalBuffer>(new MetalBuffer(std::move(impl)));
    }
}

std::shared_ptr<MetalBuffer> MetalBuffer::CreatePrivateInitialized(const MetalContext& context,
                                                                   const void* source,
                                                                   std::size_t size_bytes,
                                                                   const std::string& label,
                                                                   std::string* error_message) {
    if (source == nullptr) {
        if (error_message != nullptr) {
            *error_message = "CreatePrivateInitialized source must not be null";
        }
        return nullptr;
    }

    auto private_buffer = CreatePrivate(context, size_bytes, label, error_message);
    if (private_buffer == nullptr) {
        return nullptr;
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context.GetNativeDevice();
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLBuffer> staging_buffer = [device newBufferWithLength:size_bytes options:MTLResourceStorageModeShared];
        if (staging_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = BuildBufferError("Failed to allocate staging Metal buffer");
            }
            return nullptr;
        }
        std::memcpy([staging_buffer contents], source, size_bytes);

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            if (error_message != nullptr) {
                *error_message = BuildBufferError("Failed to create Metal upload command objects");
            }
            return nullptr;
        }

        id<MTLBuffer> destination = (__bridge id<MTLBuffer>)private_buffer->GetNativeHandle();
        [encoder copyFromBuffer:staging_buffer sourceOffset:0 toBuffer:destination destinationOffset:0 size:size_bytes];
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
