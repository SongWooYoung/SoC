#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "buffer/metal_buffer.h"

namespace soc::gpu {

using BufferArenaMark = std::size_t;

class MetalContext;

struct BufferArenaSlice {
    std::shared_ptr<MetalBuffer> buffer;
    std::size_t offset_bytes = 0;
    std::size_t size_bytes = 0;

    bool IsValid() const;
};

struct BufferArenaScopeTelemetry {
    std::string name;
    std::size_t invocation_count = 0;
    std::size_t peak_delta_bytes = 0;
    std::size_t peak_used_bytes = 0;
    std::size_t peak_allocation_count = 0;
};

class BufferArena {
public:
    static std::unique_ptr<BufferArena> CreateShared(const MetalContext& context,
                                                     std::size_t capacity_bytes,
                                                     const std::string& label,
                                                     std::string* error_message);

    ~BufferArena();

    bool Allocate(std::size_t size_bytes,
                  std::size_t alignment_bytes,
                  BufferArenaSlice* slice,
                  std::string* error_message);
    BufferArenaMark GetMark() const;
    bool ResetToMark(BufferArenaMark mark, std::string* error_message);
    void Reset();
    void ClearTelemetry();
    std::size_t GetCapacityBytes() const;
    std::size_t GetUsedBytes() const;
    std::size_t GetPeakUsedBytes() const;
    const std::vector<BufferArenaScopeTelemetry>& GetTelemetrySnapshot() const;

private:
    struct ActiveScope;

    BufferArena(std::shared_ptr<MetalBuffer> buffer, std::size_t capacity_bytes);
    void BeginScope(const std::string& scope_name);
    void EndScope();

    std::shared_ptr<MetalBuffer> buffer_;
    std::size_t capacity_bytes_ = 0;
    std::size_t used_bytes_ = 0;
    std::size_t peak_used_bytes_ = 0;
    std::vector<ActiveScope> active_scopes_;
    std::vector<BufferArenaScopeTelemetry> telemetry_;

    friend class BufferArenaMarkGuard;
};

class BufferArenaMarkGuard {
public:
    explicit BufferArenaMarkGuard(BufferArena* arena, const std::string& scope_name = {});
    ~BufferArenaMarkGuard();

private:
    BufferArena* arena_ = nullptr;
    BufferArenaMark mark_ = 0;
    bool has_scope_ = false;
};

}  // namespace soc::gpu