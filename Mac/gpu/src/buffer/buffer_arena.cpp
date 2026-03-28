#include "buffer/buffer_arena.h"

#include <algorithm>

#include "metal/metal_context.h"

namespace soc::gpu {

namespace {

std::size_t AlignUp(std::size_t value, std::size_t alignment_bytes) {
    if (alignment_bytes == 0) {
        return value;
    }
    const std::size_t remainder = value % alignment_bytes;
    return remainder == 0 ? value : value + alignment_bytes - remainder;
}

}  // namespace

struct BufferArena::ActiveScope {
    std::string name;
    std::size_t entry_mark = 0;
    std::size_t peak_used_bytes = 0;
    std::size_t allocation_count = 0;
};

bool BufferArenaSlice::IsValid() const {
    return buffer != nullptr && size_bytes > 0;
}

std::unique_ptr<BufferArena> BufferArena::CreateShared(const MetalContext& context,
                                                       std::size_t capacity_bytes,
                                                       const std::string& label,
                                                       std::string* error_message) {
    auto buffer = MetalBuffer::CreateShared(context, capacity_bytes, label, error_message);
    if (buffer == nullptr) {
        return nullptr;
    }
    return std::unique_ptr<BufferArena>(new BufferArena(std::move(buffer), capacity_bytes));
}

BufferArena::BufferArena(std::shared_ptr<MetalBuffer> buffer, std::size_t capacity_bytes)
    : buffer_(std::move(buffer)), capacity_bytes_(capacity_bytes) {}

BufferArena::~BufferArena() = default;

bool BufferArena::Allocate(std::size_t size_bytes,
                           std::size_t alignment_bytes,
                           BufferArenaSlice* slice,
                           std::string* error_message) {
    if (slice == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Arena allocation slice must not be null";
        }
        return false;
    }

    const std::size_t aligned_offset = AlignUp(used_bytes_, std::max<std::size_t>(alignment_bytes, 1));
    if (aligned_offset + size_bytes > capacity_bytes_) {
        if (error_message != nullptr) {
            *error_message = "BufferArena capacity exceeded";
        }
        return false;
    }

    slice->buffer = buffer_;
    slice->offset_bytes = aligned_offset;
    slice->size_bytes = size_bytes;
    used_bytes_ = aligned_offset + size_bytes;
    peak_used_bytes_ = std::max(peak_used_bytes_, used_bytes_);
    for (ActiveScope& scope : active_scopes_) {
        scope.peak_used_bytes = std::max(scope.peak_used_bytes, used_bytes_);
        scope.allocation_count += 1;
    }
    return true;
}

BufferArenaMark BufferArena::GetMark() const {
    return used_bytes_;
}

bool BufferArena::ResetToMark(BufferArenaMark mark, std::string* error_message) {
    if (mark > used_bytes_) {
        if (error_message != nullptr) {
            *error_message = "BufferArena mark is ahead of current usage";
        }
        return false;
    }
    used_bytes_ = mark;
    return true;
}

void BufferArena::Reset() {
    used_bytes_ = 0;
}

void BufferArena::ClearTelemetry() {
    telemetry_.clear();
}

std::size_t BufferArena::GetCapacityBytes() const {
    return capacity_bytes_;
}

std::size_t BufferArena::GetUsedBytes() const {
    return used_bytes_;
}

std::size_t BufferArena::GetPeakUsedBytes() const {
    return peak_used_bytes_;
}

const std::vector<BufferArenaScopeTelemetry>& BufferArena::GetTelemetrySnapshot() const {
    return telemetry_;
}

void BufferArena::BeginScope(const std::string& scope_name) {
    if (scope_name.empty()) {
        return;
    }
    active_scopes_.push_back(ActiveScope{scope_name, used_bytes_, used_bytes_, 0});
}

void BufferArena::EndScope() {
    if (active_scopes_.empty()) {
        return;
    }

    const ActiveScope finished_scope = active_scopes_.back();
    active_scopes_.pop_back();
    const std::size_t peak_delta = finished_scope.peak_used_bytes >= finished_scope.entry_mark
                                       ? finished_scope.peak_used_bytes - finished_scope.entry_mark
                                       : 0;

    for (BufferArenaScopeTelemetry& entry : telemetry_) {
        if (entry.name == finished_scope.name) {
            entry.invocation_count += 1;
            entry.peak_delta_bytes = std::max(entry.peak_delta_bytes, peak_delta);
            entry.peak_used_bytes = std::max(entry.peak_used_bytes, finished_scope.peak_used_bytes);
            entry.peak_allocation_count = std::max(entry.peak_allocation_count, finished_scope.allocation_count);
            return;
        }
    }

    telemetry_.push_back(BufferArenaScopeTelemetry{finished_scope.name,
                                                   1,
                                                   peak_delta,
                                                   finished_scope.peak_used_bytes,
                                                   finished_scope.allocation_count});
}

BufferArenaMarkGuard::BufferArenaMarkGuard(BufferArena* arena, const std::string& scope_name)
    : arena_(arena), has_scope_(!scope_name.empty()) {
    if (arena_ != nullptr) {
        mark_ = arena_->GetMark();
        if (has_scope_) {
            arena_->BeginScope(scope_name);
        }
    }
}

BufferArenaMarkGuard::~BufferArenaMarkGuard() {
    if (arena_ != nullptr) {
        if (has_scope_) {
            arena_->EndScope();
        }
        std::string ignored_error;
        arena_->ResetToMark(mark_, &ignored_error);
    }
}

}  // namespace soc::gpu