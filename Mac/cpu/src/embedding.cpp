#include "header/embedding.h"

#include "header/dtype.h"

#include <cstring>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireContiguous(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

std::size_t ReadTokenId(const Tensor& token_ids, std::size_t index) {
    switch (token_ids.dtype()) {
        case DType::Int32:
            return static_cast<std::size_t>(token_ids.data<const int32_t>()[index]);
        case DType::Int64:
            return static_cast<std::size_t>(token_ids.data<const int64_t>()[index]);
        default:
            throw std::runtime_error("token ids must be int32 or int64");
    }
}
}

Embedding::Embedding(Tensor weight) : weight_(std::move(weight)) {
    Require(DTypeIsFloatLike(weight_.dtype()), "embedding weight must be float32, float16, or bfloat16");
    Require(weight_.dim() == 2, "embedding weight must have shape [vocab_size, embedding_dim]");
    RequireContiguous(weight_, "embedding weight");
}

const Tensor& Embedding::weight() const {
    return weight_;
}

std::size_t Embedding::vocab_size() const {
    return weight_.shape()[0];
}

std::size_t Embedding::embedding_dim() const {
    return weight_.shape()[1];
}

Tensor Embedding::Forward(const Tensor& token_ids) const {
    Require(token_ids.dim() >= 1, "embedding input must have at least one dimension");
    RequireContiguous(token_ids, "embedding input");

    std::vector<std::size_t> output_shape = token_ids.shape();
    output_shape.push_back(embedding_dim());

    Storage output_storage = Storage::AllocateOwned(token_ids.numel() * embedding_dim() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, output_shape);

    float* output_data = output.data<float>();
    const void* weight_data = static_cast<const void*>(weight_.data<const std::byte>());

    for (std::size_t token_index = 0; token_index < token_ids.numel(); ++token_index) {
        const std::size_t vocab_index = ReadTokenId(token_ids, token_index);
        if (vocab_index >= vocab_size()) {
            throw std::runtime_error("embedding input token id out of range");
        }
        float* destination = output_data + token_index * embedding_dim();
        const std::size_t base_index = vocab_index * embedding_dim();
        for (std::size_t embedding_index = 0; embedding_index < embedding_dim(); ++embedding_index) {
            destination[embedding_index] = DTypeReadFloat(weight_data, weight_.dtype(), base_index + embedding_index);
        }
    }

    return output;
}