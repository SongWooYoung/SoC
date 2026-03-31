#pragma once

#include <string>

namespace soc::gpu {

class MetalContext;

/// CommandStream allows batching multiple GPU compute dispatches into a single
/// MTLCommandBuffer. Instead of each Op creating, committing, and waiting on its
/// own command buffer, Ops encode into the stream's shared command buffer.
/// Only when Flush() is called does the command buffer commit and wait.
///
/// Usage:
///   CommandStream stream;
///   stream.Begin(context);
///   // ... pass &stream to multiple Ops ...
///   stream.Flush(context, error_message);  // single commit+wait
///
/// When a non-null CommandStream* is passed to an Op, the Op must use
/// stream->GetEncoder() instead of creating its own command buffer.
class CommandStream {
public:
    CommandStream() = default;
    ~CommandStream();

    CommandStream(const CommandStream&) = delete;
    CommandStream& operator=(const CommandStream&) = delete;

    /// Begin a new command buffer on the given context's command queue.
    /// Must be called before any Op encodes into this stream.
    bool Begin(const MetalContext& context, std::string* error_message);

    /// Get the native MTLComputeCommandEncoder handle (as void*).
    /// Each call returns the current encoder. Ops should use this to set
    /// pipeline state, buffers, and dispatch. The encoder is valid until
    /// the next Flush() or the stream is destroyed.
    ///
    /// After dispatching, the Op should NOT end the encoder. The stream
    /// manages encoder lifecycle.
    ///
    /// Returns nullptr if Begin() was not called or the stream is in error state.
    const void* GetCommandBuffer() const;

    /// Create a new compute command encoder from the current command buffer.
    /// The caller must call EndEncoder() after dispatching.
    /// This allows each Op to have its own encoder (pipeline state, buffer bindings).
    const void* BeginEncoder();

    /// Create a new blit command encoder from the current command buffer.
    /// Used for KV cache copy operations. Must be paired with EndEncoder().
    const void* BeginBlitEncoder();

    /// End the current encoder (compute or blit). Must be paired with
    /// BeginEncoder() or BeginBlitEncoder().
    void EndEncoder();

    /// Commit the command buffer, wait for GPU completion, and accumulate profiling.
    /// After Flush(), the stream can be reused by calling Begin() again.
    bool Flush(const MetalContext& context, std::string* error_message);

    /// Returns true if Begin() has been called and Flush() has not yet been called.
    bool IsActive() const;

    /// Get the number of dispatches encoded since the last Begin().
    std::size_t GetEncoderCount() const;

private:
    void* command_buffer_ = nullptr;  // id<MTLCommandBuffer>
    void* current_encoder_ = nullptr; // id<MTLComputeCommandEncoder> or id<MTLBlitCommandEncoder>
    std::size_t encoder_count_ = 0;
    bool active_ = false;
    bool is_blit_encoder_ = false;
};

}  // namespace soc::gpu
