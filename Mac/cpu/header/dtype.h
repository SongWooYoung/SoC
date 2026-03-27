#ifndef DTYPE_H
#define DTYPE_H

#include <cstddef>
#include <cstdint>
#include <string>

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    UInt16,
    Int8,
    UInt8,
    Bool,
};

std::size_t DTypeByteWidth(DType dtype);
std::string DTypeName(DType dtype);
DType DTypeFromString(const std::string& dtype_name);
bool DTypeIsFloatLike(DType dtype);
float DTypeReadFloat(const void* data, DType dtype, std::size_t index);
float Float16ToFloat32(std::uint16_t value);
float BFloat16ToFloat32(std::uint16_t value);
std::uint16_t Float32ToFloat16(float value);
std::uint16_t Float32ToBFloat16(float value);

#endif