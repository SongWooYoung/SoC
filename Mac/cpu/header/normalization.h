#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "header/matrix.h"
#include "header/statistical_ops.h"
#include <cmath>

// ============================================================================
// Normalization Operations
// ============================================================================

// Layer Normalization
// Normalizes across features: (x - mean) / sqrt(variance + epsilon)
// Then applies affine transformation: gamma * normalized + beta
template<typename T>
matrix<T> LayerNorm(const matrix<T>& x, const matrix<T>& gamma, const matrix<T>& beta, T epsilon = T(1e-5)) {
    const int rows = x.row_count();
    const int cols = x.col_count();
    
    // Check gamma and beta dimensions
    if (gamma.row_count() != 1 || gamma.col_count() != cols ||
        beta.row_count() != 1 || beta.col_count() != cols) {
        throw std::invalid_argument("gamma and beta must have shape (1, features)");
    }
    
    const T* x_data = x.raw_data();
    const T* gamma_data = gamma.raw_data();
    const T* beta_data = beta.raw_data();
    
    matrix<T> result(rows, cols);
    T* result_data = result.raw_data();
    
    // Normalize each row independently
    for (int row = 0; row < rows; ++row) {
        const T* row_data = x_data + row * cols;
        T* result_row = result_data + row * cols;
        
        // Compute mean
        T sum = T(0);
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
        T mean = sum / static_cast<T>(cols);
        
        // Compute variance
        T sum_sq_diff = T(0);
        col = 0;
        for (; col + 3 < cols; col += 4) {
            T diff0 = row_data[col] - mean;
            sum_sq_diff += diff0 * diff0;
            
            T diff1 = row_data[col + 1] - mean;
            sum_sq_diff += diff1 * diff1;
            
            T diff2 = row_data[col + 2] - mean;
            sum_sq_diff += diff2 * diff2;
            
            T diff3 = row_data[col + 3] - mean;
            sum_sq_diff += diff3 * diff3;
        }
        for (; col < cols; ++col) {
            T diff = row_data[col] - mean;
            sum_sq_diff += diff * diff;
        }
        T variance = sum_sq_diff / static_cast<T>(cols);
        
        // Normalize and apply affine transformation
        T inv_std = T(1) / std::sqrt(variance + epsilon);
        col = 0;
        for (; col + 3 < cols; col += 4) {
            result_row[col] = gamma_data[col] * (row_data[col] - mean) * inv_std + beta_data[col];
            result_row[col + 1] = gamma_data[col + 1] * (row_data[col + 1] - mean) * inv_std + beta_data[col + 1];
            result_row[col + 2] = gamma_data[col + 2] * (row_data[col + 2] - mean) * inv_std + beta_data[col + 2];
            result_row[col + 3] = gamma_data[col + 3] * (row_data[col + 3] - mean) * inv_std + beta_data[col + 3];
        }
        for (; col < cols; ++col) {
            result_row[col] = gamma_data[col] * (row_data[col] - mean) * inv_std + beta_data[col];
        }
    }
    
    return result;
}

// RMS Normalization (Root Mean Square Normalization)
// More efficient than LayerNorm, used in Llama
// normalized = x / sqrt(mean(x^2) + epsilon) * gamma
template<typename T>
matrix<T> RMSNorm(const matrix<T>& x, const matrix<T>& gamma, T epsilon = T(1e-6)) {
    const int rows = x.row_count();
    const int cols = x.col_count();
    
    // Check gamma dimensions
    if (gamma.row_count() != 1 || gamma.col_count() != cols) {
        throw std::invalid_argument("gamma must have shape (1, features)");
    }
    
    const T* x_data = x.raw_data();
    const T* gamma_data = gamma.raw_data();
    
    matrix<T> result(rows, cols);
    T* result_data = result.raw_data();
    
    // Normalize each row independently
    for (int row = 0; row < rows; ++row) {
        const T* row_data = x_data + row * cols;
        T* result_row = result_data + row * cols;
        
        // Compute RMS (root mean square)
        T sum_sq = T(0);
        int col = 0;
        for (; col + 3 < cols; col += 4) {
            sum_sq += row_data[col] * row_data[col];
            sum_sq += row_data[col + 1] * row_data[col + 1];
            sum_sq += row_data[col + 2] * row_data[col + 2];
            sum_sq += row_data[col + 3] * row_data[col + 3];
        }
        for (; col < cols; ++col) {
            sum_sq += row_data[col] * row_data[col];
        }
        
        T rms = std::sqrt(sum_sq / static_cast<T>(cols) + epsilon);
        T inv_rms = T(1) / rms;
        
        // Normalize and scale
        col = 0;
        for (; col + 3 < cols; col += 4) {
            result_row[col] = gamma_data[col] * row_data[col] * inv_rms;
            result_row[col + 1] = gamma_data[col + 1] * row_data[col + 1] * inv_rms;
            result_row[col + 2] = gamma_data[col + 2] * row_data[col + 2] * inv_rms;
            result_row[col + 3] = gamma_data[col + 3] * row_data[col + 3] * inv_rms;
        }
        for (; col < cols; ++col) {
            result_row[col] = gamma_data[col] * row_data[col] * inv_rms;
        }
    }
    
    return result;
}

// Batch Normalization (for completeness, though less common in Transformers)
// Normalizes across batch: (x - mean) / sqrt(variance + epsilon) * gamma + beta
template<typename T>
matrix<T> BatchNorm(const matrix<T>& x, const matrix<T>& gamma, const matrix<T>& beta, T epsilon = T(1e-5)) {
    const int rows = x.row_count();
    const int cols = x.col_count();
    
    // Check gamma and beta dimensions
    if (gamma.row_count() != 1 || gamma.col_count() != cols ||
        beta.row_count() != 1 || beta.col_count() != cols) {
        throw std::invalid_argument("gamma and beta must have shape (1, features)");
    }
    
    const T* x_data = x.raw_data();
    const T* gamma_data = gamma.raw_data();
    const T* beta_data = beta.raw_data();
    
    matrix<T> result(rows, cols);
    T* result_data = result.raw_data();
    
    // Normalize each feature (column) independently
    for (int col = 0; col < cols; ++col) {
        // Compute mean
        T sum = T(0);
        for (int row = 0; row < rows; ++row) {
            sum += x_data[row * cols + col];
        }
        T mean = sum / static_cast<T>(rows);
        
        // Compute variance
        T sum_sq_diff = T(0);
        for (int row = 0; row < rows; ++row) {
            T diff = x_data[row * cols + col] - mean;
            sum_sq_diff += diff * diff;
        }
        T variance = sum_sq_diff / static_cast<T>(rows);
        
        // Normalize and apply affine transformation
        T inv_std = T(1) / std::sqrt(variance + epsilon);
        for (int row = 0; row < rows; ++row) {
            T normalized = (x_data[row * cols + col] - mean) * inv_std;
            result_data[row * cols + col] = gamma_data[col] * normalized + beta_data[col];
        }
    }
    
    return result;
}

// Group Normalization
// Divides channels into groups and normalizes within each group
template<typename T>
matrix<T> GroupNorm(const matrix<T>& x, int num_groups, const matrix<T>& gamma, 
                    const matrix<T>& beta, T epsilon = T(1e-5)) {
    const int rows = x.row_count();
    const int cols = x.col_count();
    
    if (cols % num_groups != 0) {
        throw std::invalid_argument("number of features must be divisible by num_groups");
    }
    
    const int group_size = cols / num_groups;
    const T* x_data = x.raw_data();
    const T* gamma_data = gamma.raw_data();
    const T* beta_data = beta.raw_data();
    
    matrix<T> result(rows, cols);
    T* result_data = result.raw_data();
    
    // Normalize each row and group independently
    for (int row = 0; row < rows; ++row) {
        const T* row_data = x_data + row * cols;
        T* result_row = result_data + row * cols;
        
        for (int group = 0; group < num_groups; ++group) {
            int group_start = group * group_size;
            int group_end = group_start + group_size;
            
            // Compute mean for this group
            T sum = T(0);
            for (int col = group_start; col < group_end; ++col) {
                sum += row_data[col];
            }
            T mean = sum / static_cast<T>(group_size);
            
            // Compute variance for this group
            T sum_sq_diff = T(0);
            for (int col = group_start; col < group_end; ++col) {
                T diff = row_data[col] - mean;
                sum_sq_diff += diff * diff;
            }
            T variance = sum_sq_diff / static_cast<T>(group_size);
            
            // Normalize and apply affine transformation
            T inv_std = T(1) / std::sqrt(variance + epsilon);
            for (int col = group_start; col < group_end; ++col) {
                result_row[col] = gamma_data[col] * (row_data[col] - mean) * inv_std + beta_data[col];
            }
        }
    }
    
    return result;
}

#endif // NORMALIZATION_H
