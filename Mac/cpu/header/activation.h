#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "header/matrix.h"
#include "header/elementwise_ops.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Activation Functions
// ============================================================================

// ReLU: max(0, x)
template<typename T>
matrix<T> ReLU(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();
    
    const T zero = T(0);
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        result_data[i] = std::max(zero, a_data[i]);
        result_data[i + 1] = std::max(zero, a_data[i + 1]);
        result_data[i + 2] = std::max(zero, a_data[i + 2]);
        result_data[i + 3] = std::max(zero, a_data[i + 3]);
        result_data[i + 4] = std::max(zero, a_data[i + 4]);
        result_data[i + 5] = std::max(zero, a_data[i + 5]);
        result_data[i + 6] = std::max(zero, a_data[i + 6]);
        result_data[i + 7] = std::max(zero, a_data[i + 7]);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::max(zero, a_data[i]);
    }
    
    return result;
}

// Leaky ReLU: max(alpha * x, x)
template<typename T>
matrix<T> LeakyReLU(const matrix<T>& a, T alpha = T(0.01)) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = (a_data[i] > T(0)) ? a_data[i] : alpha * a_data[i];
        result_data[i + 1] = (a_data[i + 1] > T(0)) ? a_data[i + 1] : alpha * a_data[i + 1];
        result_data[i + 2] = (a_data[i + 2] > T(0)) ? a_data[i + 2] : alpha * a_data[i + 2];
        result_data[i + 3] = (a_data[i + 3] > T(0)) ? a_data[i + 3] : alpha * a_data[i + 3];
    }
    for (; i < total_size; ++i) {
        result_data[i] = (a_data[i] > T(0)) ? a_data[i] : alpha * a_data[i];
    }
    
    return result;
}

// Sigmoid: 1 / (1 + e^(-x))
template<typename T>
matrix<T> Sigmoid(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = T(1) / (T(1) + std::exp(-a_data[i]));
        result_data[i + 1] = T(1) / (T(1) + std::exp(-a_data[i + 1]));
        result_data[i + 2] = T(1) / (T(1) + std::exp(-a_data[i + 2]));
        result_data[i + 3] = T(1) / (T(1) + std::exp(-a_data[i + 3]));
    }
    for (; i < total_size; ++i) {
        result_data[i] = T(1) / (T(1) + std::exp(-a_data[i]));
    }
    
    return result;
}

// Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
template<typename T>
matrix<T> Tanh(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        result_data[i] = std::tanh(a_data[i]);
        result_data[i + 1] = std::tanh(a_data[i + 1]);
        result_data[i + 2] = std::tanh(a_data[i + 2]);
        result_data[i + 3] = std::tanh(a_data[i + 3]);
    }
    for (; i < total_size; ++i) {
        result_data[i] = std::tanh(a_data[i]);
    }
    
    return result;
}

// GELU: x * Φ(x) where Φ is the CDF of standard normal distribution
// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
template<typename T>
matrix<T> GELU(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();
    
    const T sqrt_2_over_pi = std::sqrt(T(2) / T(3.14159265358979323846));
    const T coeff = T(0.044715);
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        T x0 = a_data[i];
        T x0_cubed = x0 * x0 * x0;
        result_data[i] = T(0.5) * x0 * (T(1) + std::tanh(sqrt_2_over_pi * (x0 + coeff * x0_cubed)));
        
        T x1 = a_data[i + 1];
        T x1_cubed = x1 * x1 * x1;
        result_data[i + 1] = T(0.5) * x1 * (T(1) + std::tanh(sqrt_2_over_pi * (x1 + coeff * x1_cubed)));
        
        T x2 = a_data[i + 2];
        T x2_cubed = x2 * x2 * x2;
        result_data[i + 2] = T(0.5) * x2 * (T(1) + std::tanh(sqrt_2_over_pi * (x2 + coeff * x2_cubed)));
        
        T x3 = a_data[i + 3];
        T x3_cubed = x3 * x3 * x3;
        result_data[i + 3] = T(0.5) * x3 * (T(1) + std::tanh(sqrt_2_over_pi * (x3 + coeff * x3_cubed)));
    }
    for (; i < total_size; ++i) {
        T x = a_data[i];
        T x_cubed = x * x * x;
        result_data[i] = T(0.5) * x * (T(1) + std::tanh(sqrt_2_over_pi * (x + coeff * x_cubed)));
    }
    
    return result;
}

// SiLU/Swish: x * sigmoid(x)
template<typename T>
matrix<T> SiLU(const matrix<T>& a) {
    matrix<T> result(a.row_count(), a.col_count());
    const int total_size = a.row_count() * a.col_count();
    const T* a_data = a.raw_data();
    T* result_data = result.raw_data();
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 3 < total_size; i += 4) {
        T x0 = a_data[i];
        result_data[i] = x0 / (T(1) + std::exp(-x0));
        
        T x1 = a_data[i + 1];
        result_data[i + 1] = x1 / (T(1) + std::exp(-x1));
        
        T x2 = a_data[i + 2];
        result_data[i + 2] = x2 / (T(1) + std::exp(-x2));
        
        T x3 = a_data[i + 3];
        result_data[i + 3] = x3 / (T(1) + std::exp(-x3));
    }
    for (; i < total_size; ++i) {
        T x = a_data[i];
        result_data[i] = x / (T(1) + std::exp(-x));
    }
    
    return result;
}

// Softmax: exp(x_i) / sum(exp(x_j)) along axis
// Numerically stable implementation with max subtraction
template<typename T>
matrix<T> Softmax(const matrix<T>& a, int axis = 1) {
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();
    
    matrix<T> result(rows, cols);
    T* result_data = result.raw_data();

    if (rows == 0 || cols == 0) {
        return result;
    }
    
    if (axis == 1) {
        // Softmax over rows (most common for neural networks)
        for (int row = 0; row < rows; ++row) {
            const T* row_data = a_data + row * cols;
            T* result_row = result_data + row * cols;
            
            // Find max for numerical stability
            T max_val = row_data[0];
            for (int col = 1; col < cols; ++col) {
                if (row_data[col] > max_val) {
                    max_val = row_data[col];
                }
            }

            if (!std::isfinite(max_val)) {
                std::fill(result_row, result_row + cols, T(0));
                continue;
            }
            
            // Compute exp(x - max) and sum
            T sum = T(0);
            int col = 0;
            for (; col + 3 < cols; col += 4) {
                result_row[col] = std::exp(row_data[col] - max_val);
                sum += result_row[col];
                
                result_row[col + 1] = std::exp(row_data[col + 1] - max_val);
                sum += result_row[col + 1];
                
                result_row[col + 2] = std::exp(row_data[col + 2] - max_val);
                sum += result_row[col + 2];
                
                result_row[col + 3] = std::exp(row_data[col + 3] - max_val);
                sum += result_row[col + 3];
            }
            for (; col < cols; ++col) {
                result_row[col] = std::exp(row_data[col] - max_val);
                sum += result_row[col];
            }
            
            // Normalize
            T inv_sum = T(1) / sum;
            col = 0;
            for (; col + 7 < cols; col += 8) {
                result_row[col] *= inv_sum;
                result_row[col + 1] *= inv_sum;
                result_row[col + 2] *= inv_sum;
                result_row[col + 3] *= inv_sum;
                result_row[col + 4] *= inv_sum;
                result_row[col + 5] *= inv_sum;
                result_row[col + 6] *= inv_sum;
                result_row[col + 7] *= inv_sum;
            }
            for (; col < cols; ++col) {
                result_row[col] *= inv_sum;
            }
        }
    } else if (axis == 0) {
        // Softmax over columns
        for (int col = 0; col < cols; ++col) {
            // Find max
            T max_val = a_data[col];
            for (int row = 1; row < rows; ++row) {
                T val = a_data[row * cols + col];
                if (val > max_val) {
                    max_val = val;
                }
            }

            if (!std::isfinite(max_val)) {
                for (int row = 0; row < rows; ++row) {
                    result_data[row * cols + col] = T(0);
                }
                continue;
            }
            
            // Compute exp and sum
            T sum = T(0);
            for (int row = 0; row < rows; ++row) {
                T val = std::exp(a_data[row * cols + col] - max_val);
                result_data[row * cols + col] = val;
                sum += val;
            }
            
            // Normalize
            T inv_sum = T(1) / sum;
            for (int row = 0; row < rows; ++row) {
                result_data[row * cols + col] *= inv_sum;
            }
        }
    } else {
        throw std::invalid_argument("axis must be 0 or 1");
    }
    
    return result;
}

// Log Softmax: log(softmax(x)) - more numerically stable than log(softmax(x))
template<typename T>
matrix<T> LogSoftmax(const matrix<T>& a, int axis = 1) {
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();
    
    matrix<T> result(rows, cols);
    T* result_data = result.raw_data();

    if (rows == 0 || cols == 0) {
        return result;
    }
    
    if (axis == 1) {
        for (int row = 0; row < rows; ++row) {
            const T* row_data = a_data + row * cols;
            T* result_row = result_data + row * cols;
            
            // Find max
            T max_val = row_data[0];
            for (int col = 1; col < cols; ++col) {
                if (row_data[col] > max_val) {
                    max_val = row_data[col];
                }
            }
            
            // Compute log-sum-exp
            T sum_exp = T(0);
            for (int col = 0; col < cols; ++col) {
                sum_exp += std::exp(row_data[col] - max_val);
            }
            T log_sum_exp = std::log(sum_exp) + max_val;
            
            // Compute log softmax
            for (int col = 0; col < cols; ++col) {
                result_row[col] = row_data[col] - log_sum_exp;
            }
        }
    } else if (axis == 0) {
        for (int col = 0; col < cols; ++col) {
            // Find max
            T max_val = a_data[col];
            for (int row = 1; row < rows; ++row) {
                T val = a_data[row * cols + col];
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // Compute log-sum-exp
            T sum_exp = T(0);
            for (int row = 0; row < rows; ++row) {
                sum_exp += std::exp(a_data[row * cols + col] - max_val);
            }
            T log_sum_exp = std::log(sum_exp) + max_val;
            
            // Compute log softmax
            for (int row = 0; row < rows; ++row) {
                result_data[row * cols + col] = a_data[row * cols + col] - log_sum_exp;
            }
        }
    } else {
        throw std::invalid_argument("axis must be 0 or 1");
    }
    
    return result;
}

#endif // ACTIVATION_H
