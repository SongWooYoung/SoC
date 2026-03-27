#ifndef ROPE_H
#define ROPE_H

#include "header/tensor.h"

class RoPE {
public:
    RoPE(std::size_t head_dim, double rope_theta);

    std::size_t head_dim() const;
    double rope_theta() const;

    Tensor Forward(const Tensor& input, std::size_t position_offset = 0) const;
    void ApplyInPlace(Tensor& input, std::size_t position_offset = 0) const;

private:
    std::size_t head_dim_;
    double rope_theta_;
};

#endif