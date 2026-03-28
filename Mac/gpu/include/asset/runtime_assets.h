#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "asset/json_value.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class MetalContext;

struct TensorRecord {
    std::string name;
    std::string file;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t byte_size = 0;
    std::string source_shard;
};

struct ManifestData {
    std::string format;
    int format_version = 0;
    std::string model_id;
    std::string source_dir;
    std::string export_dtype;
    std::string manifest_dir;
    std::string tokenizer_runtime_file;
    JsonValue config;
    JsonValue generation_config;
    std::vector<TensorRecord> tensors;

    const TensorRecord& FindTensor(const std::string& name) const;
};

class ManifestLoader {
public:
    static ManifestData LoadFromFile(const std::string& manifest_path);
};

class TensorFileLoader {
public:
    static bool LoadBytes(const TensorRecord& tensor_record,
                          std::vector<char>* bytes,
                          std::string* error_message);
    static DeviceTensor LoadDeviceTensor(const MetalContext& context,
                                         const TensorRecord& tensor_record,
                                         std::string* error_message);
};

DataType ParseDataTypeString(const std::string& dtype_string);

}  // namespace soc::gpu