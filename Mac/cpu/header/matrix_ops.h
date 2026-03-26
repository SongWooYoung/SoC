#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "header/matrix.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int matmul_block_size = 64;
}

template<typename T>
struct PackedMatMulRhs {
    matrix<T> transposed;
    int shared_dim;
    int output_dim;

    explicit PackedMatMulRhs(const matrix<T>& rhs)
        : transposed(rhs), shared_dim(rhs.row_count()), output_dim(rhs.col_count()) {
        transposed.transpose();
    }
};

template<typename T>
PackedMatMulRhs<T> PackMatMulRhs(const matrix<T>& rhs) {
    return PackedMatMulRhs<T>(rhs);
}

struct QuantizedPackedMatMulRhs8 {
    int shared_dim;
    int output_dim;
    std::vector<int8_t> values;
    std::vector<float> scales;

    QuantizedPackedMatMulRhs8(int shared_dim, int output_dim)
        : shared_dim(shared_dim), output_dim(output_dim), values(shared_dim * output_dim), scales(output_dim, 0.0f) {}
};

inline QuantizedPackedMatMulRhs8 QuantizeMatMulRhs8(const matrix<float>& rhs) {
    const int shared_dim = rhs.row_count();
    const int output_dim = rhs.col_count();
    QuantizedPackedMatMulRhs8 packed(shared_dim, output_dim);

    const float* rhs_data = rhs.raw_data();

    for (int out_col = 0; out_col < output_dim; ++out_col) {
        float max_abs = 0.0f;
        for (int inner = 0; inner < shared_dim; ++inner) {
            float value = rhs_data[inner * output_dim + out_col];
            max_abs = std::max(max_abs, std::fabs(value));
        }

        if (max_abs == 0.0f) {
            continue;
        }

        const float scale = max_abs / 127.0f;
        const float inverse_scale = 1.0f / scale;
        packed.scales[out_col] = scale;

        int8_t* destination_row = packed.values.data() + out_col * shared_dim;
        for (int inner = 0; inner < shared_dim; ++inner) {
            float value = rhs_data[inner * output_dim + out_col] * inverse_scale;
            int quantized = static_cast<int>(std::round(value));
            quantized = std::clamp(quantized, -127, 127);
            destination_row[inner] = static_cast<int8_t>(quantized);
        }
    }

    return packed;
}

// ============================================================================
// Basic Matrix Operations
// ============================================================================

// Matrix Addition: C = A + B
template<typename T>
matrix<T> MatAdd(const matrix<T>& a, const matrix<T>& b) {
    if (a.row_count() != b.row_count() || a.col_count() != b.col_count()) {
        throw std::invalid_argument("matrix dimensions do not match for addition");
    }

    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    const T* b_data = b.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = a_data[i] + b_data[i];
        result_data[i + 1] = a_data[i + 1] + b_data[i + 1];
        result_data[i + 2] = a_data[i + 2] + b_data[i + 2];
        result_data[i + 3] = a_data[i + 3] + b_data[i + 3];
        result_data[i + 4] = a_data[i + 4] + b_data[i + 4];
        result_data[i + 5] = a_data[i + 5] + b_data[i + 5];
        result_data[i + 6] = a_data[i + 6] + b_data[i + 6];
        result_data[i + 7] = a_data[i + 7] + b_data[i + 7];
    }
    for (; i < total_size; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

// Matrix Subtraction: C = A - B
template<typename T>
matrix<T> MatSub(const matrix<T>& a, const matrix<T>& b) {
    if (a.row_count() != b.row_count() || a.col_count() != b.col_count()) {
        throw std::invalid_argument("matrix dimensions do not match for subtraction");
    }

    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    const T* b_data = b.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = a_data[i] - b_data[i];
        result_data[i + 1] = a_data[i + 1] - b_data[i + 1];
        result_data[i + 2] = a_data[i + 2] - b_data[i + 2];
        result_data[i + 3] = a_data[i + 3] - b_data[i + 3];
        result_data[i + 4] = a_data[i + 4] - b_data[i + 4];
        result_data[i + 5] = a_data[i + 5] - b_data[i + 5];
        result_data[i + 6] = a_data[i + 6] - b_data[i + 6];
        result_data[i + 7] = a_data[i + 7] - b_data[i + 7];
    }
    for (; i < total_size; ++i) {
        result_data[i] = a_data[i] - b_data[i];
    }

    return result;
}

// Scalar Multiplication: C = scalar * A
template<typename T>
matrix<T> MatScale(const matrix<T>& a, T scalar) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = scalar * a_data[i];
        result_data[i + 1] = scalar * a_data[i + 1];
        result_data[i + 2] = scalar * a_data[i + 2];
        result_data[i + 3] = scalar * a_data[i + 3];
        result_data[i + 4] = scalar * a_data[i + 4];
        result_data[i + 5] = scalar * a_data[i + 5];
        result_data[i + 6] = scalar * a_data[i + 6];
        result_data[i + 7] = scalar * a_data[i + 7];
    }
    for (; i < total_size; ++i) {
        result_data[i] = scalar * a_data[i];
    }

    return result;
}

// Hadamard Product (Element-wise multiplication): C = A ⊙ B
template<typename T>
matrix<T> MatHadamard(const matrix<T>& a, const matrix<T>& b) {
    if (a.row_count() != b.row_count() || a.col_count() != b.col_count()) {
        throw std::invalid_argument("matrix dimensions do not match for Hadamard product");
    }

    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    const T* b_data = b.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = a_data[i] * b_data[i];
        result_data[i + 1] = a_data[i + 1] * b_data[i + 1];
        result_data[i + 2] = a_data[i + 2] * b_data[i + 2];
        result_data[i + 3] = a_data[i + 3] * b_data[i + 3];
        result_data[i + 4] = a_data[i + 4] * b_data[i + 4];
        result_data[i + 5] = a_data[i + 5] * b_data[i + 5];
        result_data[i + 6] = a_data[i + 6] * b_data[i + 6];
        result_data[i + 7] = a_data[i + 7] * b_data[i + 7];
    }
    for (; i < total_size; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }

    return result;
}

// Element-wise Division: C = A / B
template<typename T>
matrix<T> MatDivide(const matrix<T>& a, const matrix<T>& b) {
    if (a.row_count() != b.row_count() || a.col_count() != b.col_count()) {
        throw std::invalid_argument("matrix dimensions do not match for division");
    }

    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    const T* b_data = b.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = a_data[i] / b_data[i];
        result_data[i + 1] = a_data[i + 1] / b_data[i + 1];
        result_data[i + 2] = a_data[i + 2] / b_data[i + 2];
        result_data[i + 3] = a_data[i + 3] / b_data[i + 3];
    }
    for (; i < total_size; ++i) {
        result_data[i] = a_data[i] / b_data[i];
    }

    return result;
}

// Matrix Multiplication: C = A @ B
// Optimized with blocking and transpose
template<typename T>
matrix<T> MatMulTransposedB(const matrix<T>& a, const matrix<T>& b_transposed) {
    if (a.col_count() != b_transposed.col_count()) {
        throw std::invalid_argument("matrix dimensions do not match for multiplication with transposed rhs");
    }

    const int rows = a.row_count();
    const int shared = a.col_count();
    const int cols = b_transposed.row_count();

    matrix<T> result(rows, cols);
    const T* a_data = a.raw_data();
    const T* bt_data = b_transposed.raw_data();
    T* result_data = result.raw_data();

    for (int row_block = 0; row_block < rows; row_block += matmul_block_size) {
        int row_limit = std::min(row_block + matmul_block_size, rows);

        for (int shared_block = 0; shared_block < shared; shared_block += matmul_block_size) {
            int shared_limit = std::min(shared_block + matmul_block_size, shared);

            for (int col_block = 0; col_block < cols; col_block += matmul_block_size) {
                int col_limit = std::min(col_block + matmul_block_size, cols);

                for (int row = row_block; row < row_limit; ++row) {
                    const T* a_row = a_data + row * shared;
                    T* result_row = result_data + row * cols;

                    for (int col = col_block; col < col_limit; ++col) {
                        const T* bt_row = bt_data + col * shared;
                        T sum = result_row[col];

                        int k = shared_block;
                        for (; k + 3 < shared_limit; k += 4) {
                            sum += a_row[k] * bt_row[k];
                            sum += a_row[k + 1] * bt_row[k + 1];
                            sum += a_row[k + 2] * bt_row[k + 2];
                            sum += a_row[k + 3] * bt_row[k + 3];
                        }

                        for (; k < shared_limit; ++k) {
                            sum += a_row[k] * bt_row[k];
                        }

                        result_row[col] = sum;
                    }
                }
            }
        }
    }

    return result;
}

template<typename T>
matrix<T> MatMul(const matrix<T>& a, const matrix<T>& b) {
    if (a.col_count() != b.row_count()) {
        throw std::invalid_argument("matrix dimensions do not match for multiplication");
    }

    matrix<T> transposed_b(b);
    transposed_b.transpose();

    return MatMulTransposedB(a, transposed_b);
}

template<typename T>
matrix<T> MatMulPacked(const matrix<T>& a, const PackedMatMulRhs<T>& packed_b) {
    if (a.col_count() != packed_b.shared_dim) {
        throw std::invalid_argument("matrix dimensions do not match for packed multiplication");
    }

    return MatMulTransposedB(a, packed_b.transposed);
}

inline matrix<float> MatMulQuantized(const matrix<float>& a, const QuantizedPackedMatMulRhs8& packed_b) {
    if (a.col_count() != packed_b.shared_dim) {
        throw std::invalid_argument("matrix dimensions do not match for quantized multiplication");
    }

    const int rows = a.row_count();
    const int shared_dim = a.col_count();
    const int output_dim = packed_b.output_dim;

    matrix<float> result(rows, output_dim);
    const float* a_data = a.raw_data();
    float* result_data = result.raw_data();

    for (int row = 0; row < rows; ++row) {
        const float* a_row = a_data + row * shared_dim;
        float* result_row = result_data + row * output_dim;

        for (int out_col = 0; out_col < output_dim; ++out_col) {
            const float scale = packed_b.scales[out_col];
            if (scale == 0.0f) {
                result_row[out_col] = 0.0f;
                continue;
            }

            const int8_t* packed_row = packed_b.values.data() + out_col * shared_dim;
            float sum = 0.0f;
            int inner = 0;
            for (; inner + 3 < shared_dim; inner += 4) {
                sum += a_row[inner] * static_cast<float>(packed_row[inner]);
                sum += a_row[inner + 1] * static_cast<float>(packed_row[inner + 1]);
                sum += a_row[inner + 2] * static_cast<float>(packed_row[inner + 2]);
                sum += a_row[inner + 3] * static_cast<float>(packed_row[inner + 3]);
            }
            for (; inner < shared_dim; ++inner) {
                sum += a_row[inner] * static_cast<float>(packed_row[inner]);
            }

            result_row[out_col] = sum * scale;
        }
    }

    return result;
}

#endif // MATRIX_OPS_H
