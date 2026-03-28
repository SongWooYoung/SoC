#pragma once

#include <cstddef>
#include <vector>

namespace soc::gpu {

enum class DataType {
    kFloat16,
    kFloat32,
    kInt32,
};

class TensorDesc {
public:
    static TensorDesc CreateContiguous(DataType data_type, const std::vector<std::size_t>& shape);

    DataType GetDataType() const;
    const std::vector<std::size_t>& GetShape() const;
    const std::vector<std::size_t>& GetStrides() const;
    std::size_t Rank() const;
    std::size_t ElementCount() const;
    std::size_t ByteSize() const;
    bool IsContiguous() const;

private:
    DataType data_type_ = DataType::kFloat32;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    bool contiguous_ = false;
};

std::size_t GetDataTypeSize(DataType data_type);

}  // namespace soc::gpu