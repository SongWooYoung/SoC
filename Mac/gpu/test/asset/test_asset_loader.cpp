#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "asset/runtime_assets.h"
#include "metal/metal_context.h"

int main() {
    const std::filesystem::path manifest_path = std::filesystem::path("../../models/cpp/qwen3-0.6b/manifest.json");
    if (!std::filesystem::exists(manifest_path)) {
        std::cerr << "missing manifest: " << manifest_path << '\n';
        return 1;
    }

    soc::gpu::ManifestData manifest = soc::gpu::ManifestLoader::LoadFromFile(manifest_path.string());
    if (manifest.format != "soc.cpp.llm_export" || manifest.tensors.empty()) {
        std::cerr << "manifest parse failed\n";
        return 1;
    }

    const soc::gpu::TensorRecord& tensor_record = manifest.FindTensor("model.layers.0.input_layernorm.weight");
    if (tensor_record.dtype != "float16" || tensor_record.shape.size() != 1 || tensor_record.shape[0] != 1024) {
        std::cerr << "unexpected tensor metadata\n";
        return 1;
    }

    std::string error_message;
    auto context = soc::gpu::MetalContext::CreateDefault("build/shaders/gpu.metallib", "shaders/gpu_kernels.metal", &error_message);
    if (context == nullptr) {
        std::cerr << "failed to create Metal context: " << error_message << '\n';
        return 1;
    }

    soc::gpu::DeviceTensor tensor = soc::gpu::TensorFileLoader::LoadDeviceTensor(*context, tensor_record, &error_message);
    if (!tensor.IsValid()) {
        std::cerr << "failed to upload tensor: " << error_message << '\n';
        return 1;
    }
    if (tensor.GetDesc().GetDataType() != soc::gpu::DataType::kFloat16 || tensor.GetDesc().ByteSize() != tensor_record.byte_size) {
        std::cerr << "uploaded tensor metadata mismatch\n";
        return 1;
    }

    std::vector<char> file_prefix(16, 0);
    std::ifstream stream(tensor_record.file, std::ios::binary);
    stream.read(file_prefix.data(), static_cast<std::streamsize>(file_prefix.size()));
    std::vector<char> buffer_prefix(file_prefix.size(), 0);
    if (!tensor.GetBuffer()->Read(buffer_prefix.data(), buffer_prefix.size(), 0, &error_message)) {
        std::cerr << "failed to read uploaded tensor prefix: " << error_message << '\n';
        return 1;
    }
    if (buffer_prefix != file_prefix) {
        std::cerr << "uploaded tensor bytes do not match source file\n";
        return 1;
    }

    std::cout << "test_asset_loader passed\n";
    return 0;
}