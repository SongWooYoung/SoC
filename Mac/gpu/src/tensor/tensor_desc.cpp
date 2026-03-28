#include "tensor/tensor_desc.h"

#include <numeric>
#include <stdexcept>

namespace soc::gpu {

TensorDesc TensorDesc::CreateContiguous(DataType data_type, const std::vector<std::size_t>& shape) {
    TensorDesc desc;
    desc.data_type_ = data_type;
    desc.shape_ = shape;
    desc.strides_.resize(shape.size(), 1);
    std::size_t running_stride = 1;
    for (std::size_t index = shape.size(); index > 0; --index) {
        desc.strides_[index - 1] = running_stride;
        running_stride *= shape[index - 1];
    }
    desc.contiguous_ = true;
    return desc;
}

DataType TensorDesc::GetDataType() const {
    return data_type_;
}

const std::vector<std::size_t>& TensorDesc::GetShape() const {
    return shape_;
}

const std::vector<std::size_t>& TensorDesc::GetStrides() const {
    return strides_;
}

std::size_t TensorDesc::Rank() const {
    return shape_.size();
}

std::size_t TensorDesc::ElementCount() const {
    if (shape_.empty()) {
        return 1;
    }
    return std::accumulate(shape_.begin(), shape_.end(), static_cast<std::size_t>(1), std::multiplies<>());
}

std::size_t TensorDesc::ByteSize() const {
    return ElementCount() * GetDataTypeSize(data_type_);
}

bool TensorDesc::IsContiguous() const {
    return contiguous_;
}

std::size_t GetDataTypeSize(DataType data_type) {
    switch (data_type) {
        case DataType::kFloat16:
            return 2;
        case DataType::kFloat32:
            return sizeof(float);
        case DataType::kInt32:
            return sizeof(std::int32_t);
    }
    throw std::runtime_error("Unsupported data type");
}

}  // namespace soc::gpu