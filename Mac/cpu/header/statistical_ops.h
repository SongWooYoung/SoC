#ifndef STATISTICAL_OPS_H
#define STATISTICAL_OPS_H

#include "header/matrix.h"
#include <cmath>
#include <limits>
#include <utility>

// ============================================================================
// Statistical Operations
// ============================================================================

// Mean: average over all elements or along an axis
// axis = -1: all elements, axis = 0: column-wise, axis = 1: row-wise
template<typename T>
matrix<T> Mean(const matrix<T>& a, int axis = -1) {
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();

    if (axis == -1) {
        // Mean over all elements
        T sum = T(0);
        const int total_size = rows * cols;
        
        // Optimized: Kahan summation for numerical stability + unrolling
        T compensation = T(0);
        int i = 0;
        for (; i + 3 < total_size; i += 4) {
            T y0 = a_data[i] - compensation;
            T t0 = sum + y0;
            compensation = (t0 - sum) - y0;
            sum = t0;
            
            T y1 = a_data[i + 1] - compensation;
            T t1 = sum + y1;
            compensation = (t1 - sum) - y1;
            sum = t1;
            
            T y2 = a_data[i + 2] - compensation;
            T t2 = sum + y2;
            compensation = (t2 - sum) - y2;
            sum = t2;
            
            T y3 = a_data[i + 3] - compensation;
            T t3 = sum + y3;
            compensation = (t3 - sum) - y3;
            sum = t3;
        }
        for (; i < total_size; ++i) {
            T y = a_data[i] - compensation;
            T t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        
        matrix<T> result(1, 1);
        result[0][0] = sum / static_cast<T>(total_size);
        return result;
    } else if (axis == 0) {
        // Mean over columns (result: 1 x cols)
        matrix<T> result(1, cols);
        T* result_data = result.raw_data();
        
        for (int col = 0; col < cols; ++col) {
            T sum = T(0);
            for (int row = 0; row < rows; ++row) {
                sum += a_data[row * cols + col];
            }
            result_data[col] = sum / static_cast<T>(rows);
        }
        return result;
    } else if (axis == 1) {
        // Mean over rows (result: rows x 1)
        matrix<T> result(rows, 1);
        T* result_data = result.raw_data();
        
        for (int row = 0; row < rows; ++row) {
            T sum = T(0);
            const T* row_data = a_data + row * cols;
            
            // Optimized: loop unrolling
            int col = 0;
            for (; col + 3 < cols; col += 4) {
                sum += row_data[col];
                sum += row_data[col + 1];
                sum += row_data[col + 2];
                sum += row_data[col + 3];
            }
            for (; col < cols; ++col) {
                sum += row_data[col];
            }
            
            result_data[row] = sum / static_cast<T>(cols);
        }
        return result;
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

// Variance: variance over all elements or along an axis
template<typename T>
matrix<T> Variance(const matrix<T>& a, int axis = -1, bool unbiased = true) {
    matrix<T> mean = Mean(a, axis);
    
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();
    
    if (axis == -1) {
        T mean_val = mean[0][0];
        T sum_sq_diff = T(0);
        const int total_size = rows * cols;
        
        // Optimized: loop unrolling
        int i = 0;
        for (; i + 3 < total_size; i += 4) {
            T diff0 = a_data[i] - mean_val;
            sum_sq_diff += diff0 * diff0;
            
            T diff1 = a_data[i + 1] - mean_val;
            sum_sq_diff += diff1 * diff1;
            
            T diff2 = a_data[i + 2] - mean_val;
            sum_sq_diff += diff2 * diff2;
            
            T diff3 = a_data[i + 3] - mean_val;
            sum_sq_diff += diff3 * diff3;
        }
        for (; i < total_size; ++i) {
            T diff = a_data[i] - mean_val;
            sum_sq_diff += diff * diff;
        }
        
        int divisor = unbiased ? (total_size - 1) : total_size;
        matrix<T> result(1, 1);
        result[0][0] = sum_sq_diff / static_cast<T>(divisor);
        return result;
    } else if (axis == 0) {
        matrix<T> result(1, cols);
        const T* mean_data = mean.raw_data();
        T* result_data = result.raw_data();
        
        for (int col = 0; col < cols; ++col) {
            T sum_sq_diff = T(0);
            T mean_val = mean_data[col];
            for (int row = 0; row < rows; ++row) {
                T diff = a_data[row * cols + col] - mean_val;
                sum_sq_diff += diff * diff;
            }
            int divisor = unbiased ? (rows - 1) : rows;
            result_data[col] = sum_sq_diff / static_cast<T>(divisor);
        }
        return result;
    } else if (axis == 1) {
        matrix<T> result(rows, 1);
        const T* mean_data = mean.raw_data();
        T* result_data = result.raw_data();
        
        for (int row = 0; row < rows; ++row) {
            T sum_sq_diff = T(0);
            T mean_val = mean_data[row];
            const T* row_data = a_data + row * cols;
            
            for (int col = 0; col < cols; ++col) {
                T diff = row_data[col] - mean_val;
                sum_sq_diff += diff * diff;
            }
            int divisor = unbiased ? (cols - 1) : cols;
            result_data[row] = sum_sq_diff / static_cast<T>(divisor);
        }
        return result;
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

// Standard Deviation
template<typename T>
matrix<T> StandardDeviation(const matrix<T>& a, int axis = -1, bool unbiased = true) {
    matrix<T> var = Variance(a, axis, unbiased);
    const int total_size = var.row_count() * var.col_count();
    T* var_data = var.raw_data();
    
    for (int i = 0; i < total_size; ++i) {
        var_data[i] = std::sqrt(var_data[i]);
    }
    
    return var;
}

// Sum: sum over all elements or along an axis
template<typename T>
matrix<T> Sum(const matrix<T>& a, int axis = -1) {
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();

    if (axis == -1) {
        T sum = T(0);
        const int total_size = rows * cols;
        
        // Optimized: loop unrolling
        int i = 0;
        for (; i + 7 < total_size; i += 8) {
            sum += a_data[i];
            sum += a_data[i + 1];
            sum += a_data[i + 2];
            sum += a_data[i + 3];
            sum += a_data[i + 4];
            sum += a_data[i + 5];
            sum += a_data[i + 6];
            sum += a_data[i + 7];
        }
        for (; i < total_size; ++i) {
            sum += a_data[i];
        }
        
        matrix<T> result(1, 1);
        result[0][0] = sum;
        return result;
    } else if (axis == 0) {
        matrix<T> result(1, cols);
        T* result_data = result.raw_data();
        
        for (int col = 0; col < cols; ++col) {
            T sum = T(0);
            for (int row = 0; row < rows; ++row) {
                sum += a_data[row * cols + col];
            }
            result_data[col] = sum;
        }
        return result;
    } else if (axis == 1) {
        matrix<T> result(rows, 1);
        T* result_data = result.raw_data();
        
        for (int row = 0; row < rows; ++row) {
            T sum = T(0);
            const T* row_data = a_data + row * cols;
            
            int col = 0;
            for (; col + 7 < cols; col += 8) {
                sum += row_data[col];
                sum += row_data[col + 1];
                sum += row_data[col + 2];
                sum += row_data[col + 3];
                sum += row_data[col + 4];
                sum += row_data[col + 5];
                sum += row_data[col + 6];
                sum += row_data[col + 7];
            }
            for (; col < cols; ++col) {
                sum += row_data[col];
            }
            
            result_data[row] = sum;
        }
        return result;
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

// Max: maximum value over all elements or along an axis
template<typename T>
matrix<T> Max(const matrix<T>& a, int axis = -1) {
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();

    if (axis == -1) {
        T max_val = std::numeric_limits<T>::lowest();
        const int total_size = rows * cols;
        
        for (int i = 0; i < total_size; ++i) {
            if (a_data[i] > max_val) {
                max_val = a_data[i];
            }
        }
        
        matrix<T> result(1, 1);
        result[0][0] = max_val;
        return result;
    } else if (axis == 0) {
        matrix<T> result(1, cols);
        T* result_data = result.raw_data();
        
        for (int col = 0; col < cols; ++col) {
            T max_val = std::numeric_limits<T>::lowest();
            for (int row = 0; row < rows; ++row) {
                T val = a_data[row * cols + col];
                if (val > max_val) {
                    max_val = val;
                }
            }
            result_data[col] = max_val;
        }
        return result;
    } else if (axis == 1) {
        matrix<T> result(rows, 1);
        T* result_data = result.raw_data();
        
        for (int row = 0; row < rows; ++row) {
            T max_val = std::numeric_limits<T>::lowest();
            const T* row_data = a_data + row * cols;
            
            for (int col = 0; col < cols; ++col) {
                if (row_data[col] > max_val) {
                    max_val = row_data[col];
                }
            }
            
            result_data[row] = max_val;
        }
        return result;
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

// Min: minimum value over all elements or along an axis
template<typename T>
matrix<T> Min(const matrix<T>& a, int axis = -1) {
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();

    if (axis == -1) {
        T min_val = std::numeric_limits<T>::max();
        const int total_size = rows * cols;
        
        for (int i = 0; i < total_size; ++i) {
            if (a_data[i] < min_val) {
                min_val = a_data[i];
            }
        }
        
        matrix<T> result(1, 1);
        result[0][0] = min_val;
        return result;
    } else if (axis == 0) {
        matrix<T> result(1, cols);
        T* result_data = result.raw_data();
        
        for (int col = 0; col < cols; ++col) {
            T min_val = std::numeric_limits<T>::max();
            for (int row = 0; row < rows; ++row) {
                T val = a_data[row * cols + col];
                if (val < min_val) {
                    min_val = val;
                }
            }
            result_data[col] = min_val;
        }
        return result;
    } else if (axis == 1) {
        matrix<T> result(rows, 1);
        T* result_data = result.raw_data();
        
        for (int row = 0; row < rows; ++row) {
            T min_val = std::numeric_limits<T>::max();
            const T* row_data = a_data + row * cols;
            
            for (int col = 0; col < cols; ++col) {
                if (row_data[col] < min_val) {
                    min_val = row_data[col];
                }
            }
            
            result_data[row] = min_val;
        }
        return result;
    } else {
        throw std::invalid_argument("axis must be -1, 0, or 1");
    }
}

// ArgMax: index of maximum value along an axis
// Returns matrix of integers containing indices
template<typename T>
matrix<int> ArgMax(const matrix<T>& a, int axis = 1) {
    const int rows = a.row_count();
    const int cols = a.col_count();
    const T* a_data = a.raw_data();

    if (axis == 0) {
        matrix<int> result(1, cols);
        int* result_data = result.raw_data();
        
        for (int col = 0; col < cols; ++col) {
            T max_val = std::numeric_limits<T>::lowest();
            int max_idx = 0;
            for (int row = 0; row < rows; ++row) {
                T val = a_data[row * cols + col];
                if (val > max_val) {
                    max_val = val;
                    max_idx = row;
                }
            }
            result_data[col] = max_idx;
        }
        return result;
    } else if (axis == 1) {
        matrix<int> result(rows, 1);
        int* result_data = result.raw_data();
        
        for (int row = 0; row < rows; ++row) {
            T max_val = std::numeric_limits<T>::lowest();
            int max_idx = 0;
            const T* row_data = a_data + row * cols;
            
            for (int col = 0; col < cols; ++col) {
                if (row_data[col] > max_val) {
                    max_val = row_data[col];
                    max_idx = col;
                }
            }
            
            result_data[row] = max_idx;
        }
        return result;
    } else {
        throw std::invalid_argument("axis must be 0 or 1");
    }
}

#endif // STATISTICAL_OPS_H
