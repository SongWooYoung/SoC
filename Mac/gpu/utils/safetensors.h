#pragma once

#include "utils/json.h"

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// ── Data types ──────────────────────────────────────────────────────────────

enum class DType { BF16, F16, F32, F64, I32, I64, BOOL, UNKNOWN };

inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::BF16: return 2;
        case DType::F16:  return 2;
        case DType::F32:  return 4;
        case DType::F64:  return 8;
        case DType::I32:  return 4;
        case DType::I64:  return 8;
        case DType::BOOL: return 1;
        default:          return 0;
    }
}

inline DType parse_dtype(const std::string& s) {
    if (s == "BF16") return DType::BF16;
    if (s == "F16")  return DType::F16;
    if (s == "F32")  return DType::F32;
    if (s == "F64")  return DType::F64;
    if (s == "I32")  return DType::I32;
    if (s == "I64")  return DType::I64;
    if (s == "BOOL") return DType::BOOL;
    return DType::UNKNOWN;
}

// ── bf16 ↔ f32 conversion ──────────────────────────────────────────────────

inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

inline void bf16_to_f32_array(float* dst, const uint16_t* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &bits, sizeof(float));
    }
}

// ── Tensor metadata ─────────────────────────────────────────────────────────

struct TensorMeta {
    DType dtype;
    std::vector<int64_t> shape;
    const uint8_t* data;
    size_t nbytes;

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    const uint16_t* as_bf16() const {
        return reinterpret_cast<const uint16_t*>(data);
    }
};

// ── Single safetensors file ─────────────────────────────────────────────────

class SafetensorsFile {
    int fd_ = -1;
    uint8_t* mapped_ = nullptr;
    size_t file_size_ = 0;
    uint64_t header_size_ = 0;
    const uint8_t* data_start_ = nullptr;
    std::unordered_map<std::string, TensorMeta> tensors_;

public:
    SafetensorsFile() = default;

    SafetensorsFile(SafetensorsFile&& o) noexcept
        : fd_(o.fd_), mapped_(o.mapped_), file_size_(o.file_size_),
          header_size_(o.header_size_), data_start_(o.data_start_),
          tensors_(std::move(o.tensors_)) {
        o.fd_ = -1;
        o.mapped_ = nullptr;
    }

    SafetensorsFile& operator=(SafetensorsFile&& o) noexcept {
        if (this != &o) {
            close();
            fd_ = o.fd_; o.fd_ = -1;
            mapped_ = o.mapped_; o.mapped_ = nullptr;
            file_size_ = o.file_size_;
            header_size_ = o.header_size_;
            data_start_ = o.data_start_;
            tensors_ = std::move(o.tensors_);
        }
        return *this;
    }

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    ~SafetensorsFile() { close(); }

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("Cannot open safetensors: " + path);

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close();
            throw std::runtime_error("Cannot stat: " + path);
        }
        file_size_ = static_cast<size_t>(st.st_size);

        mapped_ = static_cast<uint8_t*>(
            mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapped_ == MAP_FAILED) {
            mapped_ = nullptr;
            close();
            throw std::runtime_error("mmap failed: " + path);
        }

        // Header: 8-byte LE u64 header_size, then JSON header, then raw data
        std::memcpy(&header_size_, mapped_, 8);
        data_start_ = mapped_ + 8 + header_size_;

        // Parse JSON header
        std::string header_str(
            reinterpret_cast<const char*>(mapped_ + 8), header_size_);
        JsonValue header = JsonParser::parse(header_str);

        for (auto& [name, info] : header.as_object()) {
            if (name == "__metadata__") continue;

            TensorMeta meta;
            meta.dtype = ::parse_dtype(info.find("dtype")->as_string());

            for (auto& s : info.find("shape")->as_array())
                meta.shape.push_back(s.as_int64());

            auto& offsets = info.find("data_offsets")->as_array();
            auto start = static_cast<size_t>(offsets[0].as_int64());
            auto end   = static_cast<size_t>(offsets[1].as_int64());
            meta.data   = data_start_ + start;
            meta.nbytes = end - start;

            tensors_[name] = std::move(meta);
        }
    }

    void close() {
        if (mapped_) { munmap(mapped_, file_size_); mapped_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        tensors_.clear();
    }

    bool has(const std::string& name) const {
        return tensors_.count(name) > 0;
    }

    const TensorMeta& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("Tensor not found: " + name);
        return it->second;
    }

    const auto& tensors() const { return tensors_; }
};

// ── Bundle of sharded safetensors files ─────────────────────────────────────

class SafetensorsBundle {
    std::vector<SafetensorsFile> files_;
    std::unordered_map<std::string, size_t> idx_;   // name → file index

public:
    void add_file(const std::string& path) {
        files_.emplace_back();
        files_.back().open(path);
        size_t fi = files_.size() - 1;
        for (auto& [name, _] : files_.back().tensors())
            idx_[name] = fi;
    }

    bool has(const std::string& name) const {
        return idx_.count(name) > 0;
    }

    const TensorMeta& get(const std::string& name) const {
        auto it = idx_.find(name);
        if (it == idx_.end())
            throw std::runtime_error("Tensor not found in bundle: " + name);
        return files_[it->second].get(name);
    }

    const uint16_t* bf16(const std::string& name) const {
        auto& meta = get(name);
        if (meta.dtype != DType::BF16)
            throw std::runtime_error("Expected BF16: " + name);
        return meta.as_bf16();
    }

    std::vector<float> load_f32(const std::string& name) const {
        auto& meta = get(name);
        int64_t n = meta.numel();
        std::vector<float> out(static_cast<size_t>(n));
        if (meta.dtype == DType::BF16) {
            bf16_to_f32_array(out.data(), meta.as_bf16(), static_cast<size_t>(n));
        } else if (meta.dtype == DType::F32) {
            std::memcpy(out.data(), meta.data, static_cast<size_t>(n) * 4);
        } else {
            throw std::runtime_error("Unsupported dtype for f32 load: " + name);
        }
        return out;
    }
};
