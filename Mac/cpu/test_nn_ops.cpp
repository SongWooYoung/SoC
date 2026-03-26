#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "header/nn_ops.h"

namespace {
bool approx_equal(float lhs, float rhs, float epsilon = 1e-4f) {
    return std::fabs(lhs - rhs) <= epsilon;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_matrix_close(const matrix<float>& actual,
                          const matrix<float>& expected,
                          float epsilon,
                          const char* message) {
    require(actual.row_count() == expected.row_count() && actual.col_count() == expected.col_count(), message);

    for (int row = 0; row < actual.row_count(); ++row) {
        for (int col = 0; col < actual.col_count(); ++col) {
            if (!approx_equal(actual[row][col], expected[row][col], epsilon)) {
                throw std::runtime_error(message);
            }
        }
    }
}

template<typename T>
matrix<T> SliceColumns(const matrix<T>& source, int start_col, int width) {
    matrix<T> result(source.row_count(), width);

    for (int row = 0; row < source.row_count(); ++row) {
        for (int col = 0; col < width; ++col) {
            result[row][col] = source[row][start_col + col];
        }
    }

    return result;
}
}

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

void test_inference_matmul_runtime() {
    std::cout << "\n=== Testing Packed/Quantized MatMul ===" << std::endl;

    matrix<float> lhs = {
        {1.0f, -2.0f, 0.5f},
        {3.0f, 1.0f, -4.0f}
    };
    matrix<float> rhs = {
        {0.25f, -1.0f},
        {1.5f, 0.75f},
        {-0.5f, 2.0f}
    };

    auto dense = MatMul(lhs, rhs);
    auto packed = PackMatMulRhs(rhs);
    auto packed_output = MatMulPacked(lhs, packed);
    require_matrix_close(packed_output, dense, 1e-5f, "packed matmul must match dense matmul");

    auto quantized = QuantizeMatMulRhs8(rhs);
    auto quantized_output = MatMulQuantized(lhs, quantized);
    require_matrix_close(quantized_output, dense, 5e-2f, "quantized matmul must stay close to dense matmul");
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

    require(approx_equal(probs[0][0] + probs[0][1] + probs[0][2], 1.0f), "softmax row 0 must sum to 1");
    require(approx_equal(probs[1][0] + probs[1][1] + probs[1][2], 1.0f), "softmax row 1 must sum to 1");

    matrix<float> masked_logits = {
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}
    };
    auto masked_probs = Softmax(masked_logits, 1);
    require(masked_probs[0][0] == 0.0f && masked_probs[0][1] == 0.0f && masked_probs[0][2] == 0.0f,
            "softmax should return zeros when an entire row is masked");
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

    matrix<float> causal_q = {
        {10.0f, 0.0f},
        {0.0f, 10.0f}
    };
    matrix<float> causal_k = causal_q;
    matrix<float> causal_v = {
        {10.0f, 1.0f},
        {1.0f, 10.0f}
    };
    auto masked_output = MaskedScaledDotProductAttention(causal_q, causal_k, causal_v, CreateCausalMask(2));
    require(approx_equal(masked_output[0][0], 10.0f) && approx_equal(masked_output[0][1], 1.0f),
            "causal attention should preserve the only visible token on the first row");
    require(masked_output[1][1] > masked_output[1][0],
            "second row should attend more strongly to the second token");

    matrix<float> multi_q = {
        {4.0f, 0.0f, 0.0f, 4.0f},
        {0.0f, 4.0f, 4.0f, 0.0f}
    };
    matrix<float> multi_k = multi_q;
    matrix<float> multi_v = {
        {10.0f, 1.0f, 3.0f, 7.0f},
        {1.0f, 10.0f, 8.0f, 2.0f}
    };

    auto multi_output = MultiHeadAttention(multi_q, multi_k, multi_v, 2);

    auto q_head0 = SliceColumns(multi_q, 0, 2);
    auto q_head1 = SliceColumns(multi_q, 2, 2);
    auto k_head0 = SliceColumns(multi_k, 0, 2);
    auto k_head1 = SliceColumns(multi_k, 2, 2);
    auto v_head0 = SliceColumns(multi_v, 0, 2);
    auto v_head1 = SliceColumns(multi_v, 2, 2);

    auto expected_head0 = ScaledDotProductAttention(q_head0, k_head0, v_head0);
    auto expected_head1 = ScaledDotProductAttention(q_head1, k_head1, v_head1);

    require(approx_equal(multi_output[0][0], expected_head0[0][0]) && approx_equal(multi_output[0][1], expected_head0[0][1]),
            "multi-head attention head 0 output mismatch on row 0");
    require(approx_equal(multi_output[0][2], expected_head1[0][0]) && approx_equal(multi_output[0][3], expected_head1[0][1]),
            "multi-head attention head 1 output mismatch on row 0");
    require(approx_equal(multi_output[1][0], expected_head0[1][0]) && approx_equal(multi_output[1][1], expected_head0[1][1]),
            "multi-head attention head 0 output mismatch on row 1");
    require(approx_equal(multi_output[1][2], expected_head1[1][0]) && approx_equal(multi_output[1][3], expected_head1[1][1]),
            "multi-head attention head 1 output mismatch on row 1");
}

void test_batched_attention_and_kv_cache() {
    std::cout << "\n=== Testing Batched Attention And KV Cache ===" << std::endl;

    matrix<float> q0 = {
        {1.0f, 0.0f, 0.5f, -0.5f},
        {0.0f, 1.0f, -0.5f, 0.5f}
    };
    matrix<float> k0 = q0;
    matrix<float> v0 = {
        {2.0f, 1.0f, 0.0f, -1.0f},
        {0.0f, 3.0f, 1.0f, 2.0f}
    };

    matrix<float> q1 = {
        {0.5f, -0.5f, 1.0f, 0.0f},
        {-0.5f, 0.5f, 0.0f, 1.0f}
    };
    matrix<float> k1 = q1;
    matrix<float> v1 = {
        {1.0f, 0.0f, 4.0f, 2.0f},
        {2.0f, 1.0f, 0.0f, 3.0f}
    };

    matrix<float> output_projection = {
        {1.0f, 0.5f, -0.25f},
        {0.0f, 1.0f, 0.75f},
        {0.5f, -0.5f, 1.0f},
        {-1.0f, 0.25f, 0.5f}
    };
    matrix<float> output_bias = {{0.1f, -0.2f, 0.3f}};

    batched_matrix<float> batched_q(2, 2, 4);
    batched_matrix<float> batched_k(2, 2, 4);
    batched_matrix<float> batched_v(2, 2, 4);
    batched_q.set_batch_slice(0, q0);
    batched_q.set_batch_slice(1, q1);
    batched_k.set_batch_slice(0, k0);
    batched_k.set_batch_slice(1, k1);
    batched_v.set_batch_slice(0, v0);
    batched_v.set_batch_slice(1, v1);

    auto batched_output = MultiHeadAttention(batched_q, batched_k, batched_v, 2, output_projection, &output_bias);
    auto expected_batch0 = MultiHeadAttention(q0, k0, v0, 2, output_projection, &output_bias);
    auto expected_batch1 = MultiHeadAttention(q1, k1, v1, 2, output_projection, &output_bias);

    require_matrix_close(batched_output.batch_slice(0), expected_batch0, 1e-5f, "batched attention batch 0 mismatch");
    require_matrix_close(batched_output.batch_slice(1), expected_batch1, 1e-5f, "batched attention batch 1 mismatch");

    KVCache<float> cache(2, 4, 4, 4);

    matrix<float> q0_step1 = {{1.0f, 0.0f, 0.5f, -0.5f}};
    matrix<float> k0_step1 = q0_step1;
    matrix<float> v0_step1 = {{2.0f, 1.0f, 0.0f, -1.0f}};
    matrix<float> q1_step1 = {{0.5f, -0.5f, 1.0f, 0.0f}};
    matrix<float> k1_step1 = q1_step1;
    matrix<float> v1_step1 = {{1.0f, 0.0f, 4.0f, 2.0f}};

    batched_matrix<float> q_step1(2, 1, 4);
    batched_matrix<float> k_step1(2, 1, 4);
    batched_matrix<float> v_step1(2, 1, 4);
    q_step1.set_batch_slice(0, q0_step1);
    q_step1.set_batch_slice(1, q1_step1);
    k_step1.set_batch_slice(0, k0_step1);
    k_step1.set_batch_slice(1, k1_step1);
    v_step1.set_batch_slice(0, v0_step1);
    v_step1.set_batch_slice(1, v1_step1);

    auto step1_output = IncrementalMultiHeadAttention(q_step1, k_step1, v_step1, cache, 2, output_projection, &output_bias);
    require(cache.length() == 1, "KV cache length must advance after first incremental step");
    require_matrix_close(step1_output.batch_slice(0), MultiHeadAttention(q0_step1, k0_step1, v0_step1, 2, output_projection, &output_bias), 1e-5f,
                         "incremental attention step 1 batch 0 mismatch");
    require_matrix_close(step1_output.batch_slice(1), MultiHeadAttention(q1_step1, k1_step1, v1_step1, 2, output_projection, &output_bias), 1e-5f,
                         "incremental attention step 1 batch 1 mismatch");

    matrix<float> q0_step2 = {{0.0f, 1.0f, -0.5f, 0.5f}};
    matrix<float> k0_step2 = q0_step2;
    matrix<float> v0_step2 = {{0.0f, 3.0f, 1.0f, 2.0f}};
    matrix<float> q1_step2 = {{-0.5f, 0.5f, 0.0f, 1.0f}};
    matrix<float> k1_step2 = q1_step2;
    matrix<float> v1_step2 = {{2.0f, 1.0f, 0.0f, 3.0f}};

    batched_matrix<float> q_step2(2, 1, 4);
    batched_matrix<float> k_step2(2, 1, 4);
    batched_matrix<float> v_step2(2, 1, 4);
    q_step2.set_batch_slice(0, q0_step2);
    q_step2.set_batch_slice(1, q1_step2);
    k_step2.set_batch_slice(0, k0_step2);
    k_step2.set_batch_slice(1, k1_step2);
    v_step2.set_batch_slice(0, v0_step2);
    v_step2.set_batch_slice(1, v1_step2);

    auto step2_output = IncrementalMultiHeadAttention(q_step2, k_step2, v_step2, cache, 2, output_projection, &output_bias);
    require(cache.length() == 2, "KV cache length must advance after second incremental step");

    matrix<float> k0_full = {
        {1.0f, 0.0f, 0.5f, -0.5f},
        {0.0f, 1.0f, -0.5f, 0.5f}
    };
    matrix<float> v0_full = {
        {2.0f, 1.0f, 0.0f, -1.0f},
        {0.0f, 3.0f, 1.0f, 2.0f}
    };
    matrix<float> k1_full = {
        {0.5f, -0.5f, 1.0f, 0.0f},
        {-0.5f, 0.5f, 0.0f, 1.0f}
    };
    matrix<float> v1_full = {
        {1.0f, 0.0f, 4.0f, 2.0f},
        {2.0f, 1.0f, 0.0f, 3.0f}
    };

    require_matrix_close(step2_output.batch_slice(0), MultiHeadAttention(q0_step2, k0_full, v0_full, 2, output_projection, &output_bias), 1e-5f,
                         "incremental attention step 2 batch 0 mismatch");
    require_matrix_close(step2_output.batch_slice(1), MultiHeadAttention(q1_step2, k1_full, v1_full, 2, output_projection, &output_bias), 1e-5f,
                         "incremental attention step 2 batch 1 mismatch");
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
    test_inference_matmul_runtime();
    test_elementwise();
    test_statistical();
    test_activation();
    test_softmax();
    test_normalization();
    test_attention();
    test_batched_attention_and_kv_cache();
    test_loss();
    
    std::cout << "\n=== All Tests Completed ===" << std::endl;
    
    return 0;
}
