#ifndef LINEAR_H
#define LINEAR_H

#include <optional>

#include "header/tensor.h"

class Linear {
public:
    explicit Linear(Tensor weight);
    Linear(Tensor weight, Tensor bias);

    const Tensor& weight() const;
    const std::optional<Tensor>& bias() const;
    std::size_t input_dim() const;
    std::size_t output_dim() const;

    Tensor Forward(const Tensor& input) const;

private:
    Tensor weight_;
    std::optional<Tensor> bias_;
};

#endif