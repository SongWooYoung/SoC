#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace soc::gpu {

class MetalContext;

enum class MetalBufferStorageMode {
    kShared,
    kPrivate,
};

enum class TensorClass {
    kUnknown,
    kStaticWeight,
    kKvCache,
    kTemporary,
    kTokenMetadata,
    kStaging,
};

class MetalBuffer {
public:
    static std::shared_ptr<MetalBuffer> CreateShared(const MetalContext& context,
                                                     std::size_t size_bytes,
                                                     const std::string& label,
                                                     std::string* error_message);
    static std::shared_ptr<MetalBuffer> CreatePrivate(const MetalContext& context,
                                                      std::size_t size_bytes,
                                                      const std::string& label,
                                                      std::string* error_message);
    static std::shared_ptr<MetalBuffer> CreatePrivateInitialized(const MetalContext& context,
                                                                 const void* source,
                                                                 std::size_t size_bytes,
                                                                 const std::string& label,
                                                                 std::string* error_message);
    static std::shared_ptr<MetalBuffer> CreateForTensorClass(const MetalContext& context,
                                                             std::size_t size_bytes,
                                                             const std::string& label,
                                                             TensorClass tensor_class,
                                                             std::string* error_message);
    static std::shared_ptr<MetalBuffer> CreateInitializedForTensorClass(const MetalContext& context,
                                                                        const void* source,
                                                                        std::size_t size_bytes,
                                                                        const std::string& label,
                                                                        TensorClass tensor_class,
                                                                        std::string* error_message);

    ~MetalBuffer();

    std::size_t GetSizeBytes() const;
    bool IsHostVisible() const;
    TensorClass GetTensorClass() const;
    MetalBufferStorageMode GetStorageMode() const;
    bool Write(const void* source, std::size_t size_bytes, std::size_t offset_bytes, std::string* error_message);
    bool Read(void* destination,
              std::size_t size_bytes,
              std::size_t offset_bytes,
              std::string* error_message) const;
    const void* GetNativeHandle() const;

private:
    struct Impl;
    static std::shared_ptr<MetalBuffer> CreateWithMode(const MetalContext& context,
                                                       std::size_t size_bytes,
                                                       const std::string& label,
                                                       MetalBufferStorageMode storage_mode,
                                                       TensorClass tensor_class,
                                                       std::string* error_message);

    explicit MetalBuffer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace soc::gpu
