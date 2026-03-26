#ifndef NN_OPS_H
#define NN_OPS_H

// ============================================================================
// Neural Network Operations - Convenience Header
// Include this file to get all LLM/Neural Network operations
// ============================================================================

// Core matrix class
#include "header/matrix.h"

// Computation modules
#include "header/matrix_ops.h"        // MatMul, MatAdd, MatSub, MatScale, MatHadamard, MatDivide
#include "header/elementwise_ops.h"   // Exp, Log, Sqrt, Power, Abs, Clip, Negative
#include "header/statistical_ops.h"   // Mean, Variance, Sum, Max, Min, ArgMax

// Neural network modules
#include "header/activation.h"        // ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax
#include "header/normalization.h"     // LayerNorm, RMSNorm, BatchNorm, GroupNorm
#include "header/attention.h"         // ScaledDotProductAttention, RoPE, MultiHeadAttention
#include "header/loss.h"              // CrossEntropy, MSE, MAE, KLDivergence

// Backward compatibility
#include "header/custom_math.h"       // Legacy computation module

/*
Usage example:

#include "header/nn_ops.h"

int main() {
    // Create matrices
    matrix<float> x(batch_size, features);
    
    // Apply operations
    auto activated = ReLU(x);
    auto normalized = LayerNorm(activated, gamma, beta);
    auto output = Softmax(normalized);
    
    // Compute loss
    float loss = CrossEntropyLoss(output, targets);
    
    return 0;
}
*/

#endif // NN_OPS_H
