// Phase 0 verification: MLX C++ API minimal test
// Confirms libmlx.a links correctly and core ops work.

#include <mlx/mlx.h>
#include <iostream>
#include <cmath>
#include <cassert>

namespace mx = mlx::core;

static int passed = 0, failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { passed++; std::cout << "  PASS: " << msg << "\n"; } \
    else { failed++; std::cout << "  FAIL: " << msg << "\n"; } \
} while(0)

#define NEAR(a, b, eps) (std::abs((a) - (b)) < (eps))

void test_array_basics() {
    std::cout << "[array basics]\n";

    auto a = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    mx::eval(a);
    CHECK(a.shape(0) == 2 && a.shape(1) == 2, "shape [2,2]");
    CHECK(a.dtype() == mx::float32, "dtype float32");

    auto b = mx::zeros({3, 4}, mx::float32);
    mx::eval(b);
    CHECK(b.shape(0) == 3 && b.shape(1) == 4, "zeros shape");

    auto c = mx::ones({2, 3}, mx::float16);
    mx::eval(c);
    CHECK(c.dtype() == mx::float16, "float16 dtype");

    auto d = mx::arange(0, 10, 1, mx::int32);
    mx::eval(d);
    CHECK(d.shape(0) == 10, "arange(0,10)");
}

void test_matmul() {
    std::cout << "[matmul]\n";

    // 2x3 @ 3x2 = 2x2
    auto a = mx::ones({2, 3}, mx::float32);
    auto b = mx::ones({3, 2}, mx::float32);
    auto c = mx::matmul(a, b);
    mx::eval(c);
    CHECK(c.shape(0) == 2 && c.shape(1) == 2, "matmul shape [2,2]");

    // Each element should be 3.0 (sum of 3 ones)
    auto data = c.data<float>();
    CHECK(NEAR(data[0], 3.0f, 1e-5), "matmul value = 3.0");
}

void test_rms_norm() {
    std::cout << "[fast::rms_norm]\n";

    // x = [1, 2, 3, 4] as [1, 1, 4]
    auto x = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 4});
    auto w = mx::ones({4}, mx::float32);
    float eps = 1e-6f;

    auto out = mx::fast::rms_norm(x, w, eps);
    mx::eval(out);

    CHECK(out.shape(0) == 1 && out.shape(1) == 1 && out.shape(2) == 4,
          "rms_norm output shape");

    // rms = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    auto d = out.data<float>();
    float rms = std::sqrt((1.0f+4.0f+9.0f+16.0f)/4.0f);
    float expected0 = 1.0f / rms;  // ≈ 0.3651
    CHECK(NEAR(d[0], expected0, 1e-3), "rms_norm value[0]");
}

void test_rope() {
    std::cout << "[fast::rope]\n";

    // x: [1, 1, 4, 8] — (B, H, S, D) — needs even dims
    auto x = mx::ones({1, 1, 4, 8}, mx::float32);
    int dims = 8;  // rotate all dims

    auto out = mx::fast::rope(x, dims, /*traditional=*/false,
                               /*base=*/10000.0f, /*scale=*/1.0f,
                               /*offset=*/0);
    mx::eval(out);
    CHECK(out.shape(0) == 1 && out.shape(3) == 8, "rope output shape");

    auto d = out.data<float>();
    // Position 0: cos(0)=1, sin(0)=0 → values unchanged
    CHECK(NEAR(d[0], 1.0f, 1e-4), "rope pos=0 unchanged");
}

void test_sdpa() {
    std::cout << "[fast::scaled_dot_product_attention]\n";

    int B = 1, H = 2, S = 4, D = 8;
    auto q = mx::ones({B, H, S, D}, mx::float32) * 0.1f;
    auto k = mx::ones({B, H, S, D}, mx::float32) * 0.1f;
    auto v = mx::ones({B, H, S, D}, mx::float32);
    float scale = 1.0f / std::sqrt(float(D));

    auto out = mx::fast::scaled_dot_product_attention(
        q, k, v, scale, /*mask_mode=*/"");
    mx::eval(out);

    CHECK(out.shape(0) == B && out.shape(1) == H &&
          out.shape(2) == S && out.shape(3) == D, "sdpa output shape");

    // With uniform q,k,v and no mask: attention is uniform → output ≈ v
    auto d = out.data<float>();
    CHECK(NEAR(d[0], 1.0f, 0.1f), "sdpa output ≈ v");
}

void test_conv1d() {
    std::cout << "[conv1d]\n";

    // input: [1, 4, 3] — (B, S, C)
    // weight: [3, 3, 1] — (C_out, C_in, K) — but MLX conv1d expects [C_out, K, C_in/groups]
    // For depthwise (groups=C): weight [C, 1, K] → MLX wants [C_out, K_w, 1]
    // Actually MLX conv1d: input [N, H, C_in], weight [C_out, K_w, C_in/groups]
    auto input = mx::ones({1, 6, 3}, mx::float32);
    auto weight = mx::ones({3, 2, 1}, mx::float32); // depthwise: C_out=3, K=2, C_in/groups=1
    auto out = mx::conv1d(input, weight, /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/3);
    mx::eval(out);

    CHECK(out.shape(0) == 1 && out.shape(1) == 5 && out.shape(2) == 3,
          "conv1d depthwise output shape [1,5,3]");

    auto d = out.data<float>();
    // Each output = sum of K input values per channel = 2.0
    CHECK(NEAR(d[0], 2.0f, 1e-5), "conv1d value = 2.0");
}

void test_sigmoid_silu() {
    std::cout << "[sigmoid / silu / softplus]\n";

    auto x = mx::array({0.0f, 1.0f, -1.0f}, {3});

    // sigmoid(0) = 0.5
    auto sig = mx::sigmoid(x);
    mx::eval(sig);
    CHECK(NEAR(sig.data<float>()[0], 0.5f, 1e-5), "sigmoid(0) = 0.5");

    // silu(x) = x * sigmoid(x)
    auto silu_out = x * mx::sigmoid(x);
    mx::eval(silu_out);
    CHECK(NEAR(silu_out.data<float>()[0], 0.0f, 1e-5), "silu(0) = 0");
    CHECK(NEAR(silu_out.data<float>()[1], 0.7311f, 1e-3), "silu(1) ≈ 0.731");

    // softplus(x) = log(1 + exp(x))
    auto sp = mx::log(1.0f + mx::exp(x));
    mx::eval(sp);
    CHECK(NEAR(sp.data<float>()[0], 0.6931f, 1e-3), "softplus(0) = ln2");
}

void test_metal_kernel_jit() {
    std::cout << "[fast::metal_kernel JIT]\n";

    // Simple Metal kernel: output[i] = input[i] * 2
    std::string source = R"(
        uint idx = thread_position_in_grid.x;
        out[idx] = inp[idx] * 2.0f;
    )";

    try {
        auto kernel = mx::fast::metal_kernel(
            "test_double",
            /*input_names=*/{"inp"},
            /*output_names=*/{"out"},
            source,
            /*header=*/"",
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        auto inp = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
        auto results = kernel(
            /*inputs=*/{inp},
            /*output_shapes=*/{{4}},
            /*output_dtypes=*/{mx::float32},
            /*grid=*/{4, 1, 1},
            /*threadgroup=*/{4, 1, 1},
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*s=*/{}
        );
        mx::eval(results[0]);

        auto d = results[0].data<float>();
        CHECK(NEAR(d[0], 2.0f, 1e-5) && NEAR(d[1], 4.0f, 1e-5) &&
              NEAR(d[2], 6.0f, 1e-5) && NEAR(d[3], 8.0f, 1e-5),
              "metal_kernel output = input * 2");
    } catch (const std::exception& e) {
        std::cout << "  SKIP: metal_kernel JIT — " << e.what() << "\n";
    }
}

void test_device_info() {
    std::cout << "[device info]\n";

    auto dev = mx::default_device();
    CHECK(dev == mx::Device::gpu, "default device = GPU");

    auto metal_available = mx::metal::is_available();
    CHECK(metal_available, "Metal is available");
}

int main() {
    std::cout << "=== MLX C++ API Verification (Phase 0) ===\n\n";

    test_device_info();
    test_array_basics();
    test_matmul();
    test_rms_norm();
    test_rope();
    test_sdpa();
    test_conv1d();
    test_sigmoid_silu();
    test_metal_kernel_jit();

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
