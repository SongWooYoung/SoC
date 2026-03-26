#ifndef CUSTOM_MATH_H
#define CUSTOM_MATH_H  

// Core computation module - includes all matrix operations
#include "header/matrix_ops.h"
#include "header/elementwise_ops.h"
#include "header/statistical_ops.h"

// All basic matrix computation functions are now in matrix_ops.h:
// - MatAdd: Matrix addition
// - MatSub: Matrix subtraction  
// - MatMul: Matrix multiplication (optimized with blocking)
// - MatScale: Scalar multiplication
// - MatHadamard: Element-wise multiplication
// - MatDivide: Element-wise division

// All element-wise operations are in elementwise_ops.h:
// - Exp, Log, Sqrt, Power, Abs, Clip, Negative

// All statistical operations are in statistical_ops.h:
// - Mean, Variance, StandardDeviation, Sum, Max, Min, ArgMax

#endif // CUSTOM_MATH_H
