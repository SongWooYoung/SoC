#ifndef ELEMENTWISE_OPS_H
#define ELEMENTWISE_OPS_H

#include "header/matrix.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Element-wise Operations
// ============================================================================

// Exponential: e^x
template<typename T>
matrix<T> Exp(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = std::exp(a_data[i]);
        result_data[i + 1] = std::exp(a_data[i + 1]);
        result_data[i + 2] = std::exp(a_data[i + 2]);
        result_data[i + 3] = std::exp(a_data[i + 3]);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::exp(a_data[i]);
    }

    return result;
}

// Natural Logarithm: ln(x)
template<typename T>
matrix<T> Log(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = std::log(a_data[i]);
        result_data[i + 1] = std::log(a_data[i + 1]);
        result_data[i + 2] = std::log(a_data[i + 2]);
        result_data[i + 3] = std::log(a_data[i + 3]);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::log(a_data[i]);
    }

    return result;
}

// Square Root: √x
template<typename T>
matrix<T> Sqrt(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = std::sqrt(a_data[i]);
        result_data[i + 1] = std::sqrt(a_data[i + 1]);
        result_data[i + 2] = std::sqrt(a_data[i + 2]);
        result_data[i + 3] = std::sqrt(a_data[i + 3]);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::sqrt(a_data[i]);
    }

    return result;
}

// Power: x^n
template<typename T>
matrix<T> Power(const matrix<T>& a, T exponent) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = std::pow(a_data[i], exponent);
        result_data[i + 1] = std::pow(a_data[i + 1], exponent);
        result_data[i + 2] = std::pow(a_data[i + 2], exponent);
        result_data[i + 3] = std::pow(a_data[i + 3], exponent);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::pow(a_data[i], exponent);
    }

    return result;
}

// Absolute Value: |x|
template<typename T>
matrix<T> Abs(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = std::abs(a_data[i]);
        result_data[i + 1] = std::abs(a_data[i + 1]);
        result_data[i + 2] = std::abs(a_data[i + 2]);
        result_data[i + 3] = std::abs(a_data[i + 3]);
        result_data[i + 4] = std::abs(a_data[i + 4]);
        result_data[i + 5] = std::abs(a_data[i + 5]);
        result_data[i + 6] = std::abs(a_data[i + 6]);
        result_data[i + 7] = std::abs(a_data[i + 7]);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::abs(a_data[i]);
    }

    return result;
}

// Clip: clamp values to [min_val, max_val]
template<typename T>
matrix<T> Clip(const matrix<T>& a, T min_val, T max_val) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = std::clamp(a_data[i], min_val, max_val);
        result_data[i + 1] = std::clamp(a_data[i + 1], min_val, max_val);
        result_data[i + 2] = std::clamp(a_data[i + 2], min_val, max_val);
        result_data[i + 3] = std::clamp(a_data[i + 3], min_val, max_val);
        result_data[i + 4] = std::clamp(a_data[i + 4], min_val, max_val);
        result_data[i + 5] = std::clamp(a_data[i + 5], min_val, max_val);
        result_data[i + 6] = std::clamp(a_data[i + 6], min_val, max_val);
        result_data[i + 7] = std::clamp(a_data[i + 7], min_val, max_val);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::clamp(a_data[i], min_val, max_val);
    }

    return result;
}

// Negative: -x
template<typename T>
matrix<T> Negative(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();

    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = -a_data[i];
        result_data[i + 1] = -a_data[i + 1];
        result_data[i + 2] = -a_data[i + 2];
        result_data[i + 3] = -a_data[i + 3];
        result_data[i + 4] = -a_data[i + 4];
        result_data[i + 5] = -a_data[i + 5];
        result_data[i + 6] = -a_data[i + 6];
        result_data[i + 7] = -a_data[i + 7];
    }
    for (; i < total_size; ++i) {
        result_data[i] = -a_data[i];
    }

    return result;
}

#endif // ELEMENTWISE_OPS_H
