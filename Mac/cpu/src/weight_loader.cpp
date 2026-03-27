#include "header/weight_loader.h"

#include <numeric>
#include <stdexcept>

#include "header/dtype.h"

namespace {
std::size_t Product(const std::vector<std::size_t>& values) {
    if (values.empty()) {
        return 1;
    }
    return std::accumulate(values.begin(), values.end(), std::size_t{1}, std::multiplies<std::size_t>());
}
}

Tensor WeightLoader::LoadTensor(const TensorRecord& record) {
    const DType dtype = DTypeFromString(record.dtype);
    const std::size_t expected_bytes = Product(record.shape) * DTypeByteWidth(dtype);
    if (expected_bytes != record.byte_size) {
        throw std::runtime_error("tensor record byte_size does not match dtype and shape for: " + record.name);
    }

    Storage storage = Storage::MapReadOnly(record.file);
    if (storage.byte_size() < record.byte_size) {
        throw std::runtime_error("tensor file is smaller than expected byte_size for: " + record.name);
    }

    return Tensor(storage, dtype, record.shape);
}

Tensor WeightLoader::LoadByName(const ManifestData& manifest, const std::string& name) {
    return LoadTensor(manifest.FindTensor(name));
}