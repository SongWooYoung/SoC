#include <chrono>
#include <iostream>
#include "header/custom_math.h"
#include "header/matrix.h"

namespace {
template<typename T>
void print_matrix_contents(const matrix<T>& value) {
    for (int row = 0; row < value.row_count(); ++row) {
        for (int col = 0; col < value.col_count(); ++col) {
            std::cout << value[row][col] << ' ';
        }
        std::cout << '\n';
    }
}

long long benchmark_transpose(int rows, int cols, int iterations) {
    using clock = std::chrono::steady_clock;

    matrix<int> sample(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            sample[row][col] = row * cols + col;
        }
    }

    long long checksum = 0;
    auto start = clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        matrix<int> working(sample);
        working.transpose();
        checksum += working[0][0];
        checksum += working[cols - 1][rows - 1];
    }
    auto end = clock::now();

    std::cout << "benchmark checksum: " << checksum << '\n';

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}


long long benchmark_multiply(int rows, int shared, int cols, int iterations) {
    using clock = std::chrono::steady_clock;

    matrix<int> left(rows, shared);
    matrix<int> right(shared, cols);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < shared; ++col) {
            left[row][col] = (row + col) % 17;
        }
    }

    for (int row = 0; row < shared; ++row) {
        for (int col = 0; col < cols; ++col) {
            right[row][col] = (row * 3 + col) % 19;
        }
    }

    long long checksum = 0;
    auto start = clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        matrix<int> product = MatMul(left, right);
        checksum += product[0][0];
        checksum += product[rows - 1][cols - 1];
    }
    auto end = clock::now();

    std::cout << "multiply checksum: " << checksum << '\n';

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
}

int main() {
    matrix<int> m(3, 4);
    std::cout << "initial m[2][3]: " << m[2][3] << '\n';

    m = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    std::cout << "after assignment m[2][3]: " << m[2][3] << '\n';

    m[2][3] = 12;
    std::cout << "after write m[2][3]: " << m[2][3] << '\n';

    std::cout << "before transpose" << '\n';
    print_matrix_contents(m);
    m.transpose();
    std::cout << "after transpose" << '\n';
    print_matrix_contents(m);

    matrix<int> left = {
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix<int> right = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    matrix<int> product = MatMul(left, right);

    std::cout << "matrix multiply result" << '\n';
    print_matrix_contents(product);

    const int square_time = static_cast<int>(benchmark_transpose(512, 512, 40));
    const int rectangular_time = static_cast<int>(benchmark_transpose(384, 768, 25));
    const int multiply_time = static_cast<int>(benchmark_multiply(256, 256, 256, 20));

    std::cout << "512x512 transpose total time: " << square_time << " us" << '\n';
    std::cout << "384x768 transpose total time: " << rectangular_time << " us" << '\n';
    std::cout << "256x256 multiply total time: " << multiply_time << " us" << '\n';

    m.display();

    // Test with double type
    std::cout << "\n=== Testing with double type ===" << '\n';
    matrix<double> dm = {
        {1.5, 2.5, 3.5},
        {4.5, 5.5, 6.5}
    };
    std::cout << "double matrix:" << '\n';
    print_matrix_contents(dm);

    matrix<double> dleft = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    matrix<double> dright = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    matrix<double> dproduct = MatMul(dleft, dright);
    std::cout << "double matrix multiply result:" << '\n';
    print_matrix_contents(dproduct);

    return 0;
}