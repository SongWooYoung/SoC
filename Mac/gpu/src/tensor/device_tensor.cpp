#include "tensor/device_tensor.h"

namespace soc::gpu {

DeviceTensor::DeviceTensor(std::shared_ptr<MetalBuffer> buffer, std::size_t byte_offset, TensorDesc desc)
    : buffer_(std::move(buffer)), byte_offset_(byte_offset), desc_(std::move(desc)) {}

bool DeviceTensor::IsValid() const {
    return buffer_ != nullptr && byte_offset_ + desc_.ByteSize() <= buffer_->GetSizeBytes();
}

const std::shared_ptr<MetalBuffer>& DeviceTensor::GetBuffer() const {
    return buffer_;
}

std::size_t DeviceTensor::GetByteOffset() const {
    return byte_offset_;
}

const TensorDesc& DeviceTensor::GetDesc() const {
    return desc_;
}

}  // namespace soc::gpu