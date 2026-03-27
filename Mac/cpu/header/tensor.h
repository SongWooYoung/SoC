#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "header/dtype.h"
#include "header/storage.h"

class Tensor {
public:
    Tensor();
    Tensor(Storage storage, DType dtype, std::vector<std::size_t> shape);
    Tensor(Storage storage, DType dtype, std::vector<std::size_t> shape, std::vector<std::size_t> strides, std::size_t offset_bytes);

    const Storage& storage() const;
    DType dtype() const;
    const std::vector<std::size_t>& shape() const;
    const std::vector<std::size_t>& strides() const;
    std::size_t offset_bytes() const;

    std::size_t dim() const;
    std::size_t numel() const;
    std::size_t nbytes() const;
    bool is_contiguous() const;

    Tensor Reshape(const std::vector<std::size_t>& new_shape) const;
    Tensor View(const std::vector<std::size_t>& new_shape) const;

    template<typename T>
    T* data() {
        static_assert(!std::is_const<T>::value, "Use const overload for const data access");
        return reinterpret_cast<T*>(storage_.data() + offset_bytes_);
    }

    template<typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(storage_.data() + offset_bytes_);
    }

    static std::vector<std::size_t> ComputeContiguousStrides(const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> strides(shape.size(), 1);
        if (shape.empty()) {
            return strides;
        }

        for (std::size_t index = shape.size(); index-- > 1;) {
            strides[index - 1] = strides[index] * shape[index];
        }
        return strides;
    }

private:
    void Validate() const;

    Storage storage_;
    DType dtype_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t offset_bytes_;
};

inline Tensor::Tensor() : dtype_(DType::Float32), offset_bytes_(0) {}

inline Tensor::Tensor(Storage storage, DType dtype, std::vector<std::size_t> shape)
    : storage_(std::move(storage)), dtype_(dtype), shape_(std::move(shape)), strides_(ComputeContiguousStrides(shape_)), offset_bytes_(0) {
    Validate();
}

inline Tensor::Tensor(Storage storage, DType dtype, std::vector<std::size_t> shape, std::vector<std::size_t> strides, std::size_t offset_bytes)
    : storage_(std::move(storage)), dtype_(dtype), shape_(std::move(shape)), strides_(std::move(strides)), offset_bytes_(offset_bytes) {
    Validate();
}

inline const Storage& Tensor::storage() const {
    return storage_;
}

inline DType Tensor::dtype() const {
    return dtype_;
}

inline const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

inline const std::vector<std::size_t>& Tensor::strides() const {
    return strides_;
}

inline std::size_t Tensor::offset_bytes() const {
    return offset_bytes_;
}

inline std::size_t Tensor::dim() const {
    return shape_.size();
}

inline std::size_t Tensor::numel() const {
    if (shape_.empty()) {
        return 1;
    }
    return std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

inline std::size_t Tensor::nbytes() const {
    return numel() * DTypeByteWidth(dtype_);
}

inline bool Tensor::is_contiguous() const {
    return strides_ == ComputeContiguousStrides(shape_);
}

inline Tensor Tensor::Reshape(const std::vector<std::size_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("reshape requires a contiguous tensor");
    }
    Tensor reshaped(storage_, dtype_, new_shape, ComputeContiguousStrides(new_shape), offset_bytes_);
    if (reshaped.numel() != numel()) {
        throw std::runtime_error("reshape must preserve tensor element count");
    }
    return reshaped;
}

inline Tensor Tensor::View(const std::vector<std::size_t>& new_shape) const {
    return Reshape(new_shape);
}

inline void Tensor::Validate() const {
    if (!storage_.valid()) {
        throw std::runtime_error("tensor requires valid storage");
    }
    if (shape_.size() != strides_.size()) {
        throw std::runtime_error("tensor shape and strides must have the same rank");
    }
    if (offset_bytes_ + nbytes() > storage_.byte_size()) {
        throw std::runtime_error("tensor view exceeds underlying storage size");
    }
}

#endif