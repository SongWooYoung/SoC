#include <iostream>
#include <iomanip>
#include "header/nn_ops.h"

void test_basic_ops() {
    std::cout << "=== Testing Basic Matrix Operations ===" << std::endl;
    
    matrix<float> a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    matrix<float> b = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    
    auto sum = MatAdd(a, b);
    std::cout << "MatAdd:" << std::endl;
    sum.display();
    
    auto diff = MatSub(a, b);
    std::cout << "MatSub:" << std::endl;
    diff.display();
    
    auto prod = MatMul(a, b);
    std::cout << "MatMul:" << std::endl;
    prod.display();
    
    auto scaled = MatScale(a, 2.0f);
    std::cout << "MatScale(2.0):" << std::endl;
    scaled.display();
    
    auto hadamard = MatHadamard(a, b);
    std::cout << "MatHadamard:" << std::endl;
    hadamard.display();
}

void test_activation() {
    std::cout << "\n=== Testing Activation Functions ===" << std::endl;
    
    matrix<float> x = {{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}};
    
    std::cout << "Input:" << std::endl;
    x.display();
    
    auto relu = ReLU(x);
    std::cout << "ReLU:" << std::endl;
    relu.display();
    
    auto sigmoid = Sigmoid(x);
    std::cout << "Sigmoid:" << std::endl;
    sigmoid.display();
    
    auto tanh_out = Tanh(x);
    std::cout << "Tanh:" << std::endl;
    tanh_out.display();
    
    auto gelu = GELU(x);
    std::cout << "GELU:" << std::endl;
    gelu.display();
    
    auto silu = SiLU(x);
    std::cout << "SiLU:" << std::endl;
    silu.display();
}

void test_softmax() {
    std::cout << "\n=== Testing Softmax ===" << std::endl;
    
    matrix<float> logits = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    
    std::cout << "Input logits:" << std::endl;
    logits.display();
    
    auto probs = Softmax(logits, 1);
    std::cout << "Softmax (axis=1):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    probs.display();
}

void test_normalization() {
    std::cout << "\n=== Testing Normalization ===" << std::endl;
    
    matrix<float> x = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f}
    };
    
    matrix<float> gamma = {{1.0f, 1.0f, 1.0f, 1.0f}};
    matrix<float> beta = {{0.0f, 0.0f, 0.0f, 0.0f}};
    
    std::cout << "Input:" << std::endl;
    x.display();
    
    auto layer_norm = LayerNorm(x, gamma, beta);
    std::cout << "LayerNorm:" << std::endl;
    layer_norm.display();
    
    auto rms_norm = RMSNorm(x, gamma);
    std::cout << "RMSNorm:" << std::endl;
    rms_norm.display();
}

void test_statistical() {
    std::cout << "\n=== Testing Statistical Operations ===" << std::endl;
    
    matrix<float> x = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    
    std::cout << "Input:" << std::endl;
    x.display();
    
    auto mean_all = Mean(x, -1);
    std::cout << "Mean (all): " << mean_all[0][0] << std::endl;
    
    auto mean_row = Mean(x, 1);
    std::cout << "Mean (axis=1):" << std::endl;
    mean_row.display();
    
    auto sum_all = Sum(x, -1);
    std::cout << "Sum (all): " << sum_all[0][0] << std::endl;
    
    auto max_val = Max(x, -1);
    std::cout << "Max: " << max_val[0][0] << std::endl;
    
    auto argmax = ArgMax(x, 1);
    std::cout << "ArgMax (axis=1):" << std::endl;
    argmax.display();
}

void test_attention() {
    std::cout << "\n=== Testing Attention ===" << std::endl;
    
    // Small example: seq_len=2, d_k=3, d_v=4
    matrix<float> Q = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };
    
    matrix<float> K = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };
    
    matrix<float> V = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f}
    };
    
    std::cout << "Q:" << std::endl;
    Q.display();
    std::cout << "K:" << std::endl;
    K.display();
    std::cout << "V:" << std::endl;
    V.display();
    
    auto output = ScaledDotProductAttention(Q, K, V);
    std::cout << "Attention output:" << std::endl;
    output.display();
}

void test_loss() {
    std::cout << "\n=== Testing Loss Functions ===" << std::endl;
    
    matrix<float> pred = {
        {0.7f, 0.2f, 0.1f},
        {0.1f, 0.8f, 0.1f}
    };
    
    matrix<float> target = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };
    
    std::cout << "Predictions:" << std::endl;
    pred.display();
    std::cout << "Targets:" << std::endl;
    target.display();
    
    float ce_loss = CrossEntropyLoss(pred, target);
    std::cout << "Cross Entropy Loss: " << ce_loss << std::endl;
    
    float mse = MSELoss(pred, target);
    std::cout << "MSE Loss: " << mse << std::endl;
}

void test_elementwise() {
    std::cout << "\n=== Testing Element-wise Operations ===" << std::endl;
    
    matrix<float> x = {{1.0f, 2.0f, 3.0f, 4.0f}};
    
    std::cout << "Input:" << std::endl;
    x.display();
    
    auto exp_x = Exp(x);
    std::cout << "Exp:" << std::endl;
    exp_x.display();
    
    auto log_x = Log(x);
    std::cout << "Log:" << std::endl;
    log_x.display();
    
    auto sqrt_x = Sqrt(x);
    std::cout << "Sqrt:" << std::endl;
    sqrt_x.display();
    
    auto abs_x = Abs(matrix<float>{{-1.0f, -2.0f, 3.0f, -4.0f}});
    std::cout << "Abs([-1, -2, 3, -4]):" << std::endl;
    abs_x.display();
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    test_basic_ops();
    test_elementwise();
    test_statistical();
    test_activation();
    test_softmax();
    test_normalization();
    test_attention();
    test_loss();
    
    std::cout << "\n=== All Tests Completed ===" << std::endl;
    
    return 0;
}
