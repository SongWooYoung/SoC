#ifndef LOSS_H
#define LOSS_H

#include "header/matrix.h"
#include "header/activation.h"
#include <cmath>
#include <stdexcept>

// ============================================================================
// Loss Functions
// ============================================================================

// Cross Entropy Loss
// For multi-class classification
// loss = -sum(target * log(pred))
// pred: (batch_size, num_classes) - predicted probabilities (after softmax)
// target: (batch_size, num_classes) - one-hot encoded targets
template<typename T>
T CrossEntropyLoss(const matrix<T>& pred, const matrix<T>& target) {
    if (pred.row_count() != target.row_count() || pred.col_count() != target.col_count()) {
        throw std::invalid_argument("pred and target must have same dimensions");
    }
    
    const int batch_size = pred.row_count();
    const int num_classes = pred.col_count();
    const T* pred_data = pred.raw_data();
    const T* target_data = target.raw_data();
    
    T total_loss = T(0);
    const T epsilon = T(1e-10); // For numerical stability
    
    for (int batch = 0; batch < batch_size; ++batch) {
        T sample_loss = T(0);
        const T* pred_row = pred_data + batch * num_classes;
        const T* target_row = target_data + batch * num_classes;
        
        // Optimized: loop unrolling
        int cls = 0;
        for (; cls + 3 < num_classes; cls += 4) {
            sample_loss -= target_row[cls] * std::log(pred_row[cls] + epsilon);
            sample_loss -= target_row[cls + 1] * std::log(pred_row[cls + 1] + epsilon);
            sample_loss -= target_row[cls + 2] * std::log(pred_row[cls + 2] + epsilon);
            sample_loss -= target_row[cls + 3] * std::log(pred_row[cls + 3] + epsilon);
        }
        for (; cls < num_classes; ++cls) {
            sample_loss -= target_row[cls] * std::log(pred_row[cls] + epsilon);
        }
        
        total_loss += sample_loss;
    }
    
    return total_loss / static_cast<T>(batch_size);
}

// Cross Entropy Loss with logits (more numerically stable)
// Combines log_softmax and cross entropy
// logits: (batch_size, num_classes) - raw scores before softmax
// target: (batch_size, num_classes) - one-hot encoded targets
template<typename T>
T CrossEntropyLossWithLogits(const matrix<T>& logits, const matrix<T>& target) {
    if (logits.row_count() != target.row_count() || logits.col_count() != target.col_count()) {
        throw std::invalid_argument("logits and target must have same dimensions");
    }
    
    // Apply log_softmax for numerical stability
    matrix<T> log_probs = LogSoftmax(logits, 1);
    
    const int batch_size = logits.row_count();
    const int num_classes = logits.col_count();
    const T* log_probs_data = log_probs.raw_data();
    const T* target_data = target.raw_data();
    
    T total_loss = T(0);
    
    for (int batch = 0; batch < batch_size; ++batch) {
        T sample_loss = T(0);
        const T* log_probs_row = log_probs_data + batch * num_classes;
        const T* target_row = target_data + batch * num_classes;
        
        for (int cls = 0; cls < num_classes; ++cls) {
            sample_loss -= target_row[cls] * log_probs_row[cls];
        }
        
        total_loss += sample_loss;
    }
    
    return total_loss / static_cast<T>(batch_size);
}

// Sparse Cross Entropy Loss (when targets are class indices, not one-hot)
// logits: (batch_size, num_classes)
// target_indices: (batch_size, 1) - class indices
template<typename T>
T SparseCrossEntropyLoss(const matrix<T>& logits, const matrix<int>& target_indices) {
    if (logits.row_count() != target_indices.row_count()) {
        throw std::invalid_argument("logits and target_indices must have same batch size");
    }
    if (target_indices.col_count() != 1) {
        throw std::invalid_argument("target_indices must have shape (batch_size, 1)");
    }
    
    // Apply log_softmax
    matrix<T> log_probs = LogSoftmax(logits, 1);
    
    const int batch_size = logits.row_count();
    const int num_classes = logits.col_count();
    const T* log_probs_data = log_probs.raw_data();
    const int* target_data = target_indices.raw_data();
    
    T total_loss = T(0);
    
    for (int batch = 0; batch < batch_size; ++batch) {
        int target_class = target_data[batch];
        if (target_class < 0 || target_class >= num_classes) {
            throw std::out_of_range("target class index out of range");
        }
        
        // Get log probability of correct class
        T log_prob = log_probs_data[batch * num_classes + target_class];
        total_loss -= log_prob;
    }
    
    return total_loss / static_cast<T>(batch_size);
}

// Mean Squared Error Loss
// loss = mean((pred - target)^2)
template<typename T>
T MSELoss(const matrix<T>& pred, const matrix<T>& target) {
    if (pred.row_count() != target.row_count() || pred.col_count() != target.col_count()) {
        throw std::invalid_argument("pred and target must have same dimensions");
    }
    
    const int total_size = pred.row_count() * pred.col_count();
    const T* pred_data = pred.raw_data();
    const T* target_data = target.raw_data();
    
    T sum_squared_error = T(0);
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        T diff0 = pred_data[i] - target_data[i];
        sum_squared_error += diff0 * diff0;
        
        T diff1 = pred_data[i + 1] - target_data[i + 1];
        sum_squared_error += diff1 * diff1;
        
        T diff2 = pred_data[i + 2] - target_data[i + 2];
        sum_squared_error += diff2 * diff2;
        
        T diff3 = pred_data[i + 3] - target_data[i + 3];
        sum_squared_error += diff3 * diff3;
        
        T diff4 = pred_data[i + 4] - target_data[i + 4];
        sum_squared_error += diff4 * diff4;
        
        T diff5 = pred_data[i + 5] - target_data[i + 5];
        sum_squared_error += diff5 * diff5;
        
        T diff6 = pred_data[i + 6] - target_data[i + 6];
        sum_squared_error += diff6 * diff6;
        
        T diff7 = pred_data[i + 7] - target_data[i + 7];
        sum_squared_error += diff7 * diff7;
    }
    for (; i < total_size; ++i) {
        T diff = pred_data[i] - target_data[i];
        sum_squared_error += diff * diff;
    }
    
    return sum_squared_error / static_cast<T>(total_size);
}

// Mean Absolute Error Loss (L1 Loss)
// loss = mean(|pred - target|)
template<typename T>
T MAELoss(const matrix<T>& pred, const matrix<T>& target) {
    if (pred.row_count() != target.row_count() || pred.col_count() != target.col_count()) {
        throw std::invalid_argument("pred and target must have same dimensions");
    }
    
    const int total_size = pred.row_count() * pred.col_count();
    const T* pred_data = pred.raw_data();
    const T* target_data = target.raw_data();
    
    T sum_abs_error = T(0);
    
    // Optimized: loop unrolling
    int i = 0;
    for (; i + 7 < total_size; i += 8) {
        sum_abs_error += std::abs(pred_data[i] - target_data[i]);
        sum_abs_error += std::abs(pred_data[i + 1] - target_data[i + 1]);
        sum_abs_error += std::abs(pred_data[i + 2] - target_data[i + 2]);
        sum_abs_error += std::abs(pred_data[i + 3] - target_data[i + 3]);
        sum_abs_error += std::abs(pred_data[i + 4] - target_data[i + 4]);
        sum_abs_error += std::abs(pred_data[i + 5] - target_data[i + 5]);
        sum_abs_error += std::abs(pred_data[i + 6] - target_data[i + 6]);
        sum_abs_error += std::abs(pred_data[i + 7] - target_data[i + 7]);
    }
    for (; i < total_size; ++i) {
        sum_abs_error += std::abs(pred_data[i] - target_data[i]);
    }
    
    return sum_abs_error / static_cast<T>(total_size);
}

// KL Divergence Loss
// KL(P || Q) = sum(P * log(P / Q))
// P: target distribution
// Q: predicted distribution
template<typename T>
T KLDivergenceLoss(const matrix<T>& pred, const matrix<T>& target) {
    if (pred.row_count() != target.row_count() || pred.col_count() != target.col_count()) {
        throw std::invalid_argument("pred and target must have same dimensions");
    }
    
    const int batch_size = pred.row_count();
    const int num_features = pred.col_count();
    const T* pred_data = pred.raw_data();
    const T* target_data = target.raw_data();
    const T epsilon = T(1e-10);
    
    T total_kl = T(0);
    
    for (int batch = 0; batch < batch_size; ++batch) {
        T sample_kl = T(0);
        const T* pred_row = pred_data + batch * num_features;
        const T* target_row = target_data + batch * num_features;
        
        for (int feat = 0; feat < num_features; ++feat) {
            T p = target_row[feat];
            T q = pred_row[feat] + epsilon;
            
            if (p > epsilon) {
                sample_kl += p * std::log(p / q);
            }
        }
        
        total_kl += sample_kl;
    }
    
    return total_kl / static_cast<T>(batch_size);
}

// Binary Cross Entropy Loss
// For binary classification
// loss = -mean(target * log(pred) + (1 - target) * log(1 - pred))
template<typename T>
T BinaryCrossEntropyLoss(const matrix<T>& pred, const matrix<T>& target) {
    if (pred.row_count() != target.row_count() || pred.col_count() != target.col_count()) {
        throw std::invalid_argument("pred and target must have same dimensions");
    }
    
    const int total_size = pred.row_count() * pred.col_count();
    const T* pred_data = pred.raw_data();
    const T* target_data = target.raw_data();
    const T epsilon = T(1e-10);
    
    T total_loss = T(0);
    
    for (int i = 0; i < total_size; ++i) {
        T p = pred_data[i];
        T t = target_data[i];
        
        // Clamp predictions for numerical stability
        p = std::clamp(p, epsilon, T(1) - epsilon);
        
        total_loss -= t * std::log(p) + (T(1) - t) * std::log(T(1) - p);
    }
    
    return total_loss / static_cast<T>(total_size);
}

#endif // LOSS_H
