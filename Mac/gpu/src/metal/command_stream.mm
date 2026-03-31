#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/command_stream.h"
#include "metal/metal_context.h"

namespace soc::gpu {

CommandStream::~CommandStream() {
    // If active, the command buffer was never flushed — just release it.
    command_buffer_ = nullptr;
    current_encoder_ = nullptr;
    active_ = false;
}

bool CommandStream::Begin(const MetalContext& context, std::string* error_message) {
    if (active_) {
        if (error_message != nullptr) {
            *error_message = "CommandStream::Begin called while already active";
        }
        return false;
    }

    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context.GetNativeCommandQueue();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (cb == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create command buffer for CommandStream";
            }
            return false;
        }
        // Retain the command buffer so it survives the autorelease pool
        command_buffer_ = (__bridge_retained void*)cb;
        current_encoder_ = nullptr;
        encoder_count_ = 0;
        active_ = true;
        return true;
    }
}

const void* CommandStream::GetCommandBuffer() const {
    if (!active_) {
        return nullptr;
    }
    return command_buffer_;
}

const void* CommandStream::BeginEncoder() {
    if (!active_ || command_buffer_ == nullptr) {
        return nullptr;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer_;
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        if (encoder == nil) {
            return nullptr;
        }
        current_encoder_ = (__bridge_retained void*)encoder;
        encoder_count_++;
        return current_encoder_;
    }
}

const void* CommandStream::BeginBlitEncoder() {
    if (!active_ || command_buffer_ == nullptr) {
        return nullptr;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer_;
        id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
        if (encoder == nil) {
            return nullptr;
        }
        current_encoder_ = (__bridge_retained void*)encoder;
        is_blit_encoder_ = true;
        encoder_count_++;
        return current_encoder_;
    }
}

void CommandStream::EndEncoder() {
    if (current_encoder_ != nullptr) {
        @autoreleasepool {
            if (is_blit_encoder_) {
                id<MTLBlitCommandEncoder> encoder = (__bridge_transfer id<MTLBlitCommandEncoder>)current_encoder_;
                [encoder endEncoding];
            } else {
                id<MTLComputeCommandEncoder> encoder = (__bridge_transfer id<MTLComputeCommandEncoder>)current_encoder_;
                [encoder endEncoding];
            }
            current_encoder_ = nullptr;
            is_blit_encoder_ = false;
        }
    }
}

bool CommandStream::Flush(const MetalContext& context, const char* profile_label, std::string* error_message) {
    if (!active_) {
        if (error_message != nullptr) {
            *error_message = "CommandStream::Flush called without active stream";
        }
        return false;
    }
    // End any dangling encoder
    if (current_encoder_ != nullptr) {
        EndEncoder();
    }

    bool result = true;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)command_buffer_;
        command_buffer_ = nullptr;
        active_ = false;

        if (cb != nil) {
            result = context.FinalizeCommandBuffer((__bridge const void*)cb,
                                                    "CommandStream flush failed",
                                                    profile_label,
                                                    encoder_count_,
                                                    error_message);
        }
    }
    return result;
}

bool CommandStream::FlushDeferred(const MetalContext& context, const char* profile_label, std::string* error_message) {
    if (!active_) {
        if (error_message != nullptr) {
            *error_message = "CommandStream::FlushDeferred called without active stream";
        }
        return false;
    }
    if (current_encoder_ != nullptr) {
        EndEncoder();
    }

    bool result = true;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)command_buffer_;
        command_buffer_ = nullptr;
        active_ = false;

        if (cb != nil) {
            result = context.CommitCommandBufferDeferred((__bridge const void*)cb,
                                                         "CommandStream deferred flush failed",
                                                         profile_label,
                                                         encoder_count_,
                                                         error_message);
        }
    }
    return result;
}

bool CommandStream::IsActive() const {
    return active_;
}

std::size_t CommandStream::GetEncoderCount() const {
    return encoder_count_;
}

}  // namespace soc::gpu
