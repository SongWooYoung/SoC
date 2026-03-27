#ifndef QWEN_MLP_H
#define QWEN_MLP_H

#include "header/linear.h"

class QwenMLP {
public:
    QwenMLP(Linear gate_proj, Linear up_proj, Linear down_proj);

    Tensor Forward(const Tensor& hidden_states) const;

private:
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;
};

#endif