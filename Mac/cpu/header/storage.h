#ifndef STORAGE_H
#define STORAGE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

enum class StorageKind {
    Owned,
    Mapped,
    External,
};

class StorageBuffer {
public:
    virtual ~StorageBuffer() = default;

    virtual std::byte* data() = 0;
    virtual const std::byte* data() const = 0;
    virtual std::size_t byte_size() const = 0;
    virtual StorageKind kind() const = 0;
    virtual std::size_t alignment() const = 0;
    virtual const std::string& source_path() const = 0;
};

class Storage {
public:
    Storage();
    explicit Storage(std::shared_ptr<StorageBuffer> buffer);

    static Storage AllocateOwned(std::size_t byte_size);
    static Storage FromOwnedCopy(const void* source, std::size_t byte_size);
    static Storage MapReadOnly(const std::string& path);
    static Storage External(void* data, std::size_t byte_size);

    bool valid() const;
    std::byte* data();
    const std::byte* data() const;
    std::size_t byte_size() const;
    StorageKind kind() const;
    std::size_t alignment() const;
    const std::string& source_path() const;

private:
    std::shared_ptr<StorageBuffer> buffer_;
};

#endif