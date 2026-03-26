#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "header/matrix.h"
#include <stdexcept>
#include <algorithm>

namespace {
constexpr int matmul_block_size = 64;
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
matrix<T> MatMul(const matrix<T>& a, const matrix<T>& b) {
    if (a.col_count() != b.row_count()) {
        throw std::invalid_argument("matrix dimensions do not match for multiplication");
    }

    const int rows = a.row_count();
    const int shared = a.col_count();
    const int cols = b.col_count();

    // Transpose B for better cache locality
    matrix<T> transposed_b(b);
    transposed_b.transpose();

    matrix<T> result(rows, cols);
    const T* a_data = a.raw_data();
    const T* bt_data = transposed_b.raw_data();
    T* result_data = result.raw_data();

    // Blocked matrix multiplication for cache efficiency
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

                        // Inner product with loop unrolling
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

#endif // MATRIX_OPS_H
