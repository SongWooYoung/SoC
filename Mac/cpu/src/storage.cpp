#include "header/storage.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
class OwnedStorageBuffer : public StorageBuffer {
public:
    explicit OwnedStorageBuffer(std::size_t byte_size)
        : data_(byte_size == 0 ? nullptr : new std::byte[byte_size]), byte_size_(byte_size), alignment_(alignof(std::max_align_t)) {}

    std::byte* data() override { return data_.get(); }
    const std::byte* data() const override { return data_.get(); }
    std::size_t byte_size() const override { return byte_size_; }
    StorageKind kind() const override { return StorageKind::Owned; }
    std::size_t alignment() const override { return alignment_; }
    const std::string& source_path() const override { return source_path_; }

private:
    std::unique_ptr<std::byte[]> data_;
    std::size_t byte_size_;
    std::size_t alignment_;
    std::string source_path_;
};

class MappedStorageBuffer : public StorageBuffer {
public:
    explicit MappedStorageBuffer(const std::string& path) : data_(nullptr), byte_size_(0), source_path_(path) {
        const int file_descriptor = open(path.c_str(), O_RDONLY);
        if (file_descriptor < 0) {
            throw std::runtime_error("failed to open mapped file: " + path);
        }

        struct stat file_stats;
        if (fstat(file_descriptor, &file_stats) != 0) {
            close(file_descriptor);
            throw std::runtime_error("failed to stat mapped file: " + path);
        }

        byte_size_ = static_cast<std::size_t>(file_stats.st_size);
        if (byte_size_ != 0) {
            void* mapped = mmap(nullptr, byte_size_, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
            if (mapped == MAP_FAILED) {
                close(file_descriptor);
                throw std::runtime_error("failed to mmap file: " + path);
            }
            data_ = static_cast<std::byte*>(mapped);
        }

        close(file_descriptor);
    }

    ~MappedStorageBuffer() override {
        if (data_ != nullptr && byte_size_ != 0) {
            munmap(data_, byte_size_);
        }
    }

    std::byte* data() override { return data_; }
    const std::byte* data() const override { return data_; }
    std::size_t byte_size() const override { return byte_size_; }
    StorageKind kind() const override { return StorageKind::Mapped; }
    std::size_t alignment() const override { return alignof(std::max_align_t); }
    const std::string& source_path() const override { return source_path_; }

private:
    std::byte* data_;
    std::size_t byte_size_;
    std::string source_path_;
};

class ExternalStorageBuffer : public StorageBuffer {
public:
    ExternalStorageBuffer(void* data, std::size_t byte_size) : data_(static_cast<std::byte*>(data)), byte_size_(byte_size) {}

    std::byte* data() override { return data_; }
    const std::byte* data() const override { return data_; }
    std::size_t byte_size() const override { return byte_size_; }
    StorageKind kind() const override { return StorageKind::External; }
    std::size_t alignment() const override { return alignof(std::max_align_t); }
    const std::string& source_path() const override { return source_path_; }

private:
    std::byte* data_;
    std::size_t byte_size_;
    std::string source_path_;
};

StorageBuffer& RequireMutableBuffer(const std::shared_ptr<StorageBuffer>& buffer) {
    if (!buffer) {
        throw std::runtime_error("storage is not initialized");
    }
    return *buffer;
}

const StorageBuffer& RequireConstBuffer(const std::shared_ptr<StorageBuffer>& buffer) {
    if (!buffer) {
        throw std::runtime_error("storage is not initialized");
    }
    return *buffer;
}
}

Storage::Storage() = default;

Storage::Storage(std::shared_ptr<StorageBuffer> buffer) : buffer_(std::move(buffer)) {}

Storage Storage::AllocateOwned(std::size_t byte_size) {
    return Storage(std::make_shared<OwnedStorageBuffer>(byte_size));
}

Storage Storage::FromOwnedCopy(const void* source, std::size_t byte_size) {
    Storage storage = AllocateOwned(byte_size);
    if (byte_size != 0) {
        std::memcpy(storage.data(), source, byte_size);
    }
    return storage;
}

Storage Storage::MapReadOnly(const std::string& path) {
    return Storage(std::make_shared<MappedStorageBuffer>(path));
}

Storage Storage::External(void* data, std::size_t byte_size) {
    return Storage(std::make_shared<ExternalStorageBuffer>(data, byte_size));
}

bool Storage::valid() const {
    return static_cast<bool>(buffer_);
}

std::byte* Storage::data() {
    return RequireMutableBuffer(buffer_).data();
}

const std::byte* Storage::data() const {
    return RequireConstBuffer(buffer_).data();
}

std::size_t Storage::byte_size() const {
    return RequireConstBuffer(buffer_).byte_size();
}

StorageKind Storage::kind() const {
    return RequireConstBuffer(buffer_).kind();
}

std::size_t Storage::alignment() const {
    return RequireConstBuffer(buffer_).alignment();
}

const std::string& Storage::source_path() const {
    return RequireConstBuffer(buffer_).source_path();
}