#include "header/dtype.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

std::size_t DTypeByteWidth(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
        case DType::UInt16: return 2;
        case DType::Int8: return 1;
        case DType::UInt8: return 1;
        case DType::Bool: return 1;
    }

    throw std::runtime_error("unsupported dtype enum");
}

std::string DTypeName(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt16: return "uint16";
        case DType::Int8: return "int8";
        case DType::UInt8: return "uint8";
        case DType::Bool: return "bool";
    }

    throw std::runtime_error("unsupported dtype enum");
}

DType DTypeFromString(const std::string& dtype_name) {
    if (dtype_name == "float32") {
        return DType::Float32;
    }
    if (dtype_name == "float16") {
        return DType::Float16;
    }
    if (dtype_name == "bfloat16") {
        return DType::BFloat16;
    }
    if (dtype_name == "int32") {
        return DType::Int32;
    }
    if (dtype_name == "int64") {
        return DType::Int64;
    }
    if (dtype_name == "uint16") {
        return DType::UInt16;
    }
    if (dtype_name == "int8") {
        return DType::Int8;
    }
    if (dtype_name == "uint8") {
        return DType::UInt8;
    }
    if (dtype_name == "bool") {
        return DType::Bool;
    }

    throw std::runtime_error("unsupported dtype string: " + dtype_name);
}

bool DTypeIsFloatLike(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
}

float Float16ToFloat32(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    const std::uint32_t exponent = (value >> 10) & 0x1fu;
    const std::uint32_t mantissa = value & 0x03ffu;

    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            std::uint32_t normalized_mantissa = mantissa;
            std::int32_t adjusted_exponent = -14;
            while ((normalized_mantissa & 0x0400u) == 0) {
                normalized_mantissa <<= 1;
                --adjusted_exponent;
            }
            normalized_mantissa &= 0x03ffu;
            bits = sign
                | static_cast<std::uint32_t>(adjusted_exponent + 127) << 23
                | (normalized_mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float BFloat16ToFloat32(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t Float32ToFloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;

    if (((bits >> 23) & 0xffu) == 0xffu) {
        if (mantissa == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa >> 13) | 1u);
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const std::uint32_t shifted = mantissa >> static_cast<std::uint32_t>(1 - exponent);
        const std::uint32_t rounded = (shifted + 0x1000u) >> 13;
        return static_cast<std::uint16_t>(sign | rounded);
    }

    if (exponent >= 0x1f) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    const std::uint32_t rounded_mantissa = mantissa + 0x1000u;
    if (rounded_mantissa & 0x800000u) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 0x1f) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    } else {
        mantissa = rounded_mantissa;
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13));
}

std::uint16_t Float32ToBFloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

float DTypeReadFloat(const void* data, DType dtype, std::size_t index) {
    switch (dtype) {
        case DType::Float32:
            return static_cast<const float*>(data)[index];
        case DType::Float16:
            return Float16ToFloat32(static_cast<const std::uint16_t*>(data)[index]);
        case DType::BFloat16:
            return BFloat16ToFloat32(static_cast<const std::uint16_t*>(data)[index]);
        default:
            throw std::runtime_error("dtype is not float-like: " + DTypeName(dtype));
    }
}