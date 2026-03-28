#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace soc::gpu {

class MetalContext;

class MetalBuffer {
public:
    static std::shared_ptr<MetalBuffer> CreateShared(const MetalContext& context,
                                                     std::size_t size_bytes,
                                                     const std::string& label,
                                                     std::string* error_message);

    ~MetalBuffer();

    std::size_t GetSizeBytes() const;
    bool IsHostVisible() const;
    bool Write(const void* source, std::size_t size_bytes, std::size_t offset_bytes, std::string* error_message);
    bool Read(void* destination,
              std::size_t size_bytes,
              std::size_t offset_bytes,
              std::string* error_message) const;
    const void* GetNativeHandle() const;

private:
    struct Impl;

    explicit MetalBuffer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace soc::gpu