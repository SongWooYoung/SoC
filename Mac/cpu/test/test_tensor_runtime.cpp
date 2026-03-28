#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "header/dtype.h"
#include "header/runtime_assets.h"
#include "header/storage.h"
#include "header/tensor.h"
#include "header/weight_loader.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_binary_file(const std::filesystem::path& path, const void* data, std::size_t byte_size) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to create binary test file");
    }
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(byte_size));
}

void test_dtype_helpers() {
    std::cout << "=== Testing DType Helpers ===" << std::endl;
    require(DTypeFromString("float32") == DType::Float32, "dtype parser must handle float32");
    require(DTypeByteWidth(DType::BFloat16) == 2, "dtype byte width must match bfloat16");
    require(DTypeName(DType::Int64) == "int64", "dtype name must round-trip int64");
}

void test_storage_and_tensor() {
    std::cout << "=== Testing Storage And Tensor ===" << std::endl;
    const float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    const Storage storage = Storage::FromOwnedCopy(values, sizeof(values));
    require(storage.kind() == StorageKind::Owned, "owned storage must report owned kind");
    require(storage.byte_size() == sizeof(values), "owned storage must report copied size");

    const Tensor tensor(storage, DType::Float32, {2, 3});
    require(tensor.numel() == 6, "tensor numel must match shape product");
    require(tensor.nbytes() == sizeof(values), "tensor nbytes must match storage footprint");
    require(tensor.is_contiguous(), "new tensor with shape-only constructor must be contiguous");
    require(tensor.strides()[0] == 3 && tensor.strides()[1] == 1, "tensor strides must be row-major contiguous");
    require(tensor.data<float>()[4] == 5.0f, "tensor data access must expose stored elements");

    const Tensor reshaped = tensor.Reshape({3, 2});
    require(reshaped.shape()[0] == 3 && reshaped.shape()[1] == 2, "reshape must update shape metadata");
    require(reshaped.data<float>()[4] == 5.0f, "reshape must preserve storage aliasing");

    float external_values[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    const Storage external = Storage::External(external_values, sizeof(external_values));
    const Tensor external_tensor(external, DType::Float32, {2, 2});
    require(external.kind() == StorageKind::External, "external storage must report external kind");
    require(external_tensor.data<float>()[0] == 9.0f, "external tensor must access caller-owned memory");
}

void test_weight_loader() {
    std::cout << "=== Testing Weight Loader ===" << std::endl;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "soc_weight_loader_test";
    std::filesystem::create_directories(temp_dir / "weights");

    const float weight_values[4] = {0.25f, -1.5f, 2.0f, 4.25f};
    const std::filesystem::path weight_file = temp_dir / "weights" / "linear.bin";
    write_binary_file(weight_file, weight_values, sizeof(weight_values));

    const TensorRecord record{
        "model.layers.0.self_attn.q_proj.weight",
        weight_file.string(),
        "float32",
        {2, 2},
        sizeof(weight_values),
        "model-00001-of-00001.safetensors",
    };

    const Tensor tensor = WeightLoader::LoadTensor(record);
    require(tensor.storage().kind() == StorageKind::Mapped, "weight loader must mmap weight files by default");
    require(tensor.shape()[0] == 2 && tensor.shape()[1] == 2, "weight loader must preserve tensor shape metadata");
    require(tensor.data<float>()[0] == 0.25f && tensor.data<float>()[3] == 4.25f,
            "weight loader must expose mapped tensor bytes as typed data");

    ManifestData manifest;
    manifest.format = "soc.cpp.llm_export";
    manifest.format_version = 2;
    manifest.export_dtype = "float32";
    manifest.tensors.push_back(record);

    const Tensor by_name = WeightLoader::LoadByName(manifest, record.name);
    require(by_name.data<float>()[1] == -1.5f, "weight loader must load tensors by manifest name");
    require(&manifest.FindTensor(record.name) == &manifest.tensors[0], "manifest lookup must return matching tensor record");

    bool threw_for_bad_size = false;
    try {
        const TensorRecord bad_record{
            "bad.weight",
            weight_file.string(),
            "float32",
            {2, 2},
            sizeof(weight_values) - 4,
            "model-00001-of-00001.safetensors",
        };
        static_cast<void>(WeightLoader::LoadTensor(bad_record));
    } catch (const std::runtime_error&) {
        threw_for_bad_size = true;
    }
    require(threw_for_bad_size, "weight loader must reject byte_size mismatches");

    std::filesystem::remove_all(temp_dir);
}
}

int main() {
    test_dtype_helpers();
    test_storage_and_tensor();
    test_weight_loader();

    std::cout << "=== Tensor Runtime Tests Completed ===" << std::endl;
    return 0;
}