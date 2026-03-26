#include "header/custom_math.h"

namespace {
constexpr int multiply_block_size = 64;
}


matrix MatAdd(const matrix& a, const matrix& b) {
    if (a.row_count() != b.row_count() || a.col_count() != b.col_count()) {
        throw std::invalid_argument("matrix dimensions do not match for addition");
    }

    matrix result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const int* a_data = a.raw_data();
    const int* b_data = b.raw_data();
    int* result_data = result.raw_data();

    for (int index = 0; index < total_size; ++index) {
        result_data[index] = a_data[index] + b_data[index];
    }

    return result;
}


matrix MatMul(const matrix& a, const matrix& b) {
    if (a.col_count() != b.row_count()) {
        throw std::invalid_argument("matrix dimensions do not match for multiplication");
    }

    const int rows = a.row_count();
    const int shared = a.col_count();
    const int cols = b.col_count();

    matrix transposed_b(b);
    transposed_b.transpose();

    matrix result(rows, cols);
    const int* a_data = a.raw_data();
    const int* bt_data = transposed_b.raw_data();
    int* result_data = result.raw_data();

    for (int row_block = 0; row_block < rows; row_block += multiply_block_size) {
        int row_limit = row_block + multiply_block_size;
        if (row_limit > rows) {
            row_limit = rows;
        }

        for (int shared_block = 0; shared_block < shared; shared_block += multiply_block_size) {
            int shared_limit = shared_block + multiply_block_size;
            if (shared_limit > shared) {
                shared_limit = shared;
            }

            for (int col_block = 0; col_block < cols; col_block += multiply_block_size) {
                int col_limit = col_block + multiply_block_size;
                if (col_limit > cols) {
                    col_limit = cols;
                }

                for (int row = row_block; row < row_limit; ++row) {
                    const int* a_row = a_data + row * shared;
                    int* result_row = result_data + row * cols;

                    for (int col = col_block; col < col_limit; ++col) {
                        const int* bt_row = bt_data + col * shared;
                        long long sum = result_row[col];

                        int k = shared_block;
                        for (; k + 3 < shared_limit; k += 4) {
                            sum += static_cast<long long>(a_row[k]) * bt_row[k];
                            sum += static_cast<long long>(a_row[k + 1]) * bt_row[k + 1];
                            sum += static_cast<long long>(a_row[k + 2]) * bt_row[k + 2];
                            sum += static_cast<long long>(a_row[k + 3]) * bt_row[k + 3];
                        }

                        for (; k < shared_limit; ++k) {
                            sum += static_cast<long long>(a_row[k]) * bt_row[k];
                        }

                        result_row[col] = static_cast<int>(sum);
                    }
                }
            }
        }
    }

    return result;
}