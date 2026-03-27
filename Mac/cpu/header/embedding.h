#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "header/tensor.h"

class Embedding {
public:
    explicit Embedding(Tensor weight);

    const Tensor& weight() const;
    std::size_t vocab_size() const;
    std::size_t embedding_dim() const;

    Tensor Forward(const Tensor& token_ids) const;

private:
    Tensor weight_;
};

#endif