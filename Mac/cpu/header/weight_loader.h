#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <string>

#include "header/runtime_assets.h"
#include "header/tensor.h"

class WeightLoader {
public:
    static Tensor LoadTensor(const TensorRecord& record);
    static Tensor LoadByName(const ManifestData& manifest, const std::string& name);
};

#endif