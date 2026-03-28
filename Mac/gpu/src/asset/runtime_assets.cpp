#include "asset/runtime_assets.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "asset/json_parser.h"
#include "buffer/metal_buffer.h"
#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

std::string RequireString(const JsonValue& object, const std::string& key) {
    return object.at(key).as_string();
}

int RequireInt(const JsonValue& object, const std::string& key) {
    return object.at(key).as_int();
}

std::size_t RequireSize(const JsonValue& object, const std::string& key) {
    return static_cast<std::size_t>(object.at(key).as_int64());
}

std::vector<std::size_t> ParseShape(const JsonValue& shape_value) {
    std::vector<std::size_t> shape;
    for (const JsonValue& value : shape_value.as_array()) {
        shape.push_back(static_cast<std::size_t>(value.as_int64()));
    }
    return shape;
}

std::string ResolveRelativePath(const std::filesystem::path& base_dir, const std::string& relative_path) {
    return (base_dir / relative_path).lexically_normal().string();
}

}  // namespace

const TensorRecord& ManifestData::FindTensor(const std::string& name) const {
    for (const TensorRecord& tensor : tensors) {
        if (tensor.name == name) {
            return tensor;
        }
    }
    throw std::runtime_error("missing tensor in manifest: " + name);
}

ManifestData ManifestLoader::LoadFromFile(const std::string& manifest_path) {
    const JsonValue root = JsonParser::ParseFile(manifest_path);
    const std::filesystem::path manifest_file = std::filesystem::path(manifest_path).lexically_normal();
    const std::filesystem::path manifest_dir = manifest_file.parent_path();

    ManifestData manifest;
    manifest.format = RequireString(root, "format");
    manifest.format_version = RequireInt(root, "format_version");
    manifest.model_id = root.contains("model_id") && !root.at("model_id").is_null() ? root.at("model_id").as_string() : "";
    manifest.source_dir = root.contains("source_dir") && !root.at("source_dir").is_null() ? root.at("source_dir").as_string() : "";
    manifest.export_dtype = root.contains("export_dtype") && !root.at("export_dtype").is_null() ? root.at("export_dtype").as_string() : "";
    manifest.manifest_dir = manifest_dir.string();
    manifest.tokenizer_runtime_file = root.contains("tokenizer_runtime_file") && !root.at("tokenizer_runtime_file").is_null()
        ? ResolveRelativePath(manifest_dir, root.at("tokenizer_runtime_file").as_string())
        : "";
    manifest.config = root.contains("config") ? root.at("config") : JsonValue();
    manifest.generation_config = root.contains("generation_config") ? root.at("generation_config") : JsonValue();

    for (const JsonValue& tensor_value : root.at("tensors").as_array()) {
        TensorRecord tensor;
        tensor.name = RequireString(tensor_value, "name");
        tensor.file = ResolveRelativePath(manifest_dir, RequireString(tensor_value, "file"));
        tensor.dtype = RequireString(tensor_value, "dtype");
        tensor.shape = ParseShape(tensor_value.at("shape"));
        tensor.byte_size = RequireSize(tensor_value, "byte_size");
        tensor.source_shard = RequireString(tensor_value, "source_shard");
        manifest.tensors.push_back(std::move(tensor));
    }

    return manifest;
}

DataType ParseDataTypeString(const std::string& dtype_string) {
    if (dtype_string == "float16") {
        return DataType::kFloat16;
    }
    if (dtype_string == "float32") {
        return DataType::kFloat32;
    }
    if (dtype_string == "int32") {
        return DataType::kInt32;
    }
    throw std::runtime_error("unsupported tensor dtype: " + dtype_string);
}

bool TensorFileLoader::LoadBytes(const TensorRecord& tensor_record,
                                 std::vector<char>* bytes,
                                 std::string* error_message) {
    if (bytes == nullptr) {
        if (error_message != nullptr) {
            *error_message = "output byte vector must not be null";
        }
        return false;
    }

    std::ifstream stream(tensor_record.file, std::ios::binary);
    if (!stream) {
        if (error_message != nullptr) {
            *error_message = "failed to open tensor file: " + tensor_record.file;
        }
        return false;
    }

    stream.seekg(0, std::ios::end);
    const std::size_t file_size = static_cast<std::size_t>(stream.tellg());
    stream.seekg(0, std::ios::beg);
    const std::size_t expected_size = tensor_record.byte_size == 0 ? file_size : tensor_record.byte_size;
    if (file_size != expected_size) {
        if (error_message != nullptr) {
            *error_message = "tensor file size mismatch for " + tensor_record.name;
        }
        return false;
    }

    bytes->assign(expected_size, 0);
    stream.read(bytes->data(), static_cast<std::streamsize>(expected_size));
    if (!stream) {
        if (error_message != nullptr) {
            *error_message = "failed to read tensor file: " + tensor_record.file;
        }
        return false;
    }
    return true;
}

DeviceTensor TensorFileLoader::LoadDeviceTensor(const MetalContext& context,
                                                const TensorRecord& tensor_record,
                                                std::string* error_message) {
    std::vector<char> bytes;
    if (!LoadBytes(tensor_record, &bytes, error_message)) {
        return DeviceTensor();
    }

    auto buffer = MetalBuffer::CreateShared(context, bytes.size(), tensor_record.name, error_message);
    if (buffer == nullptr) {
        return DeviceTensor();
    }
    if (!buffer->Write(bytes.data(), bytes.size(), 0, error_message)) {
        return DeviceTensor();
    }

    return DeviceTensor(buffer, 0, TensorDesc::CreateContiguous(ParseDataTypeString(tensor_record.dtype), tensor_record.shape));
}

}  // namespace soc::gpu