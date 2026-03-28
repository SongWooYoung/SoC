#pragma once

#include <cstddef>
#include <memory>

#include "buffer/metal_buffer.h"
#include "tensor/tensor_desc.h"

namespace soc::gpu {

class DeviceTensor {
public:
    DeviceTensor() = default;
    DeviceTensor(std::shared_ptr<MetalBuffer> buffer, std::size_t byte_offset, TensorDesc desc);

    bool IsValid() const;
    const std::shared_ptr<MetalBuffer>& GetBuffer() const;
    std::size_t GetByteOffset() const;
    const TensorDesc& GetDesc() const;

private:
    std::shared_ptr<MetalBuffer> buffer_;
    std::size_t byte_offset_ = 0;
    TensorDesc desc_;
};

}  // namespace soc::gpu