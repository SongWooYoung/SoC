#ifndef RMS_NORM_MODULE_H
#define RMS_NORM_MODULE_H

#include "header/tensor.h"

class RMSNormModule {
public:
    explicit RMSNormModule(Tensor weight, float epsilon = 1e-6f);

    const Tensor& weight() const;
    float epsilon() const;

    Tensor Forward(const Tensor& input) const;

private:
    Tensor weight_;
    float epsilon_;
};

#endif