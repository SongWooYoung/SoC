// Phase 2 — MLX C++ Module Tests
//
// Tests each module from language.h, gated_delta.h, mlx_helpers.h
// with small synthetic tensors.  No model weights needed.
//
// Build:  make test_mlx_phase2
// Run:    MLX_METAL_PATH=… build/test_mlx_phase2

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "models/qwen3_5_mlx/mlx_helpers.h"
#include "models/qwen3_5_mlx/gated_delta.h"
#include "models/qwen3_5_mlx/language.h"

namespace mx = mlx::core;
using namespace qwen3_5_mlx;

// ─── Helpers ────────────────────────────────────────────────────────────────

static int pass_count = 0;
static int fail_count = 0;

#define CHECK(cond, msg)                                               \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::cerr << "FAIL: " << (msg) << "  (" << __FILE__        \
                      << ":" << __LINE__ << ")\n";                     \
            fail_count++;                                              \
        } else {                                                       \
            pass_count++;                                              \
        }                                                              \
    } while (0)

static float scalar(const mx::array& a) {
    return a.item<float>();
}

static bool near(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) < tol;
}

// ─── mlx_helpers tests ──────────────────────────────────────────────────────

void test_silu() {
    auto x = mx::array({0.0f, 1.0f, -1.0f});
    auto y = mlx_helpers::silu(x);
    mx::eval(y);
    CHECK(near(y.data<float>()[0], 0.0f), "silu(0)=0");
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    CHECK(near(y.data<float>()[1], 0.7311f, 1e-3f), "silu(1)≈0.731");
}

void test_softplus() {
    auto x = mx::array({0.0f, 1.0f});
    auto y = mlx_helpers::softplus(x);
    mx::eval(y);
    CHECK(near(y.data<float>()[0], std::log(2.0f)), "softplus(0)=ln2");
}

void test_swiglu() {
    auto gate = mx::array({1.0f, 2.0f});
    auto x = mx::array({3.0f, 4.0f});
    auto y = mlx_helpers::swiglu(gate, x);
    mx::eval(y);
    // swiglu = silu(gate) * x
    float expected0 = (1.0f * (1.0f / (1.0f + std::exp(-1.0f)))) * 3.0f;
    CHECK(near(y.data<float>()[0], expected0, 1e-3f), "swiglu check 0");
}

void test_linear() {
    auto x = mx::ones({1, 3});  // [1, 3]
    auto w = mx::ones({2, 3});  // [2, 3]
    auto y = mlx_helpers::linear(x, w);  // [1, 2]
    mx::eval(y);
    CHECK(y.shape(0) == 1 && y.shape(1) == 2, "linear shape");
    CHECK(near(y.data<float>()[0], 3.0f), "linear value");

    auto b = mx::array({10.0f, 20.0f});
    auto yb = mlx_helpers::linear(x, w, b);
    mx::eval(yb);
    CHECK(near(yb.data<float>()[0], 13.0f), "linear+bias");
}

void test_embedding() {
    auto weight = mx::reshape(mx::arange(12), {4, 3});  // [4, 3]
    auto idx = mx::array({0, 2});
    auto y = mlx_helpers::embedding(weight, idx);
    mx::eval(y);
    CHECK(y.shape(0) == 2 && y.shape(1) == 3, "embedding shape");
    CHECK(y.data<int>()[0] == 0, "embedding row 0");
    CHECK(y.data<int>()[3] == 6, "embedding row 2");
}

void test_rotate_half() {
    // [1, 1, 1, 4] → split into [1,1,1,2] halves
    auto x = mx::reshape(mx::array({1.0f, 2.0f, 3.0f, 4.0f}), {1, 1, 1, 4});
    auto y = mlx_helpers::rotate_half(x);
    mx::eval(y);
    // [-x2, x1] = [-3, -4, 1, 2]
    auto d = y.data<float>();
    CHECK(near(d[0], -3.0f) && near(d[1], -4.0f) && near(d[2], 1.0f) &&
              near(d[3], 2.0f),
          "rotate_half");
}

void test_compute_g() {
    auto A_log = mx::array({0.0f});  // exp(0)=1
    auto a = mx::array({0.0f});
    auto dt_bias = mx::array({0.0f});
    auto g = mlx_helpers::compute_g(A_log, a, dt_bias);
    mx::eval(g);
    // g = exp(-1 * softplus(0)) = exp(-ln2) ≈ 0.5
    CHECK(near(scalar(g), 0.5f, 1e-3f), "compute_g");
}

// ─── RotaryEmbedding test ───────────────────────────────────────────────────

void test_rotary_embedding() {
    // dim=8, base=10000, section=[2,2,0,0]
    RotaryEmbedding rope(8, 10000.0f, {2, 2, 0, 0});
    CHECK(rope.inv_freq.shape(0) == 4, "inv_freq dim");

    // Position ids: [3, 1, 2] (3D MRoPE, batch 1, seq 2)
    auto pos = mx::broadcast_to(
        mx::expand_dims(mx::expand_dims(mx::arange(2), 0), 0), {3, 1, 2});
    auto [cos, sin] = rope(mx::float16, pos);
    mx::eval(cos);
    mx::eval(sin);
    CHECK(cos.shape(0) == 1 && cos.shape(1) == 2 && cos.shape(2) == 8,
          "rope cos shape");
    CHECK(sin.shape(2) == 8, "rope sin shape");
    // cos(0) should be 1.0 for position 0
    auto cos32 = mx::astype(cos, mx::float32);
    mx::eval(cos32);
    CHECK(near(cos32.data<float>()[0], 1.0f, 1e-2f), "rope cos pos0");
}

// ─── apply_rotary_pos_emb test ──────────────────────────────────────────────

void test_apply_rotary() {
    int B = 1, H = 2, L = 1, D = 8, rdim = 4;
    auto q = mx::ones({B, H, L, D});
    auto k = mx::ones({B, H, L, D});
    auto cos = mx::ones({B, L, rdim});
    auto sin = mx::zeros({B, L, rdim});  // sin=0 → no rotation

    auto [qe, ke] = apply_rotary_pos_emb(q, k, cos, sin);
    mx::eval(qe);
    CHECK(qe.shape(3) == D, "rotary output dim");
    // With sin=0, cos=1: q_rot * 1 + rotate_half(q_rot) * 0 = q_rot
    auto qe32 = mx::astype(qe, mx::float32);
    mx::eval(qe32);
    CHECK(near(qe32.data<float>()[0], 1.0f), "rotary identity");
}

// ─── KVCache test ───────────────────────────────────────────────────────────

void test_kv_cache() {
    KVCache cache;
    CHECK(cache.empty(), "kvcache initially empty");

    auto k1 = mx::ones({1, 2, 3, 4});   // [B, H, L=3, D=4]
    auto v1 = mx::ones({1, 2, 3, 4});
    auto [ck1, cv1] = cache.update_and_fetch(k1, v1);
    mx::eval(ck1);
    CHECK(cache.offset == 3, "kvcache offset after first");
    CHECK(ck1.shape(2) == 3, "kvcache len after first");

    auto k2 = mx::ones({1, 2, 1, 4});   // single decode step
    auto v2 = mx::ones({1, 2, 1, 4});
    auto [ck2, cv2] = cache.update_and_fetch(k2, v2);
    mx::eval(ck2);
    CHECK(cache.offset == 4, "kvcache offset after second");
    CHECK(ck2.shape(2) == 4, "kvcache len after second");
}

// ─── gated_delta_step_ops test ──────────────────────────────────────────────

void test_gd_step_ops() {
    int B = 1, H = 2, Dk = 4, Dv = 4;
    auto q = mx::ones({B, H, Dk});
    auto k = mx::ones({B, H, Dk});
    auto v = mx::ones({B, H, Dv});
    auto g = mx::ones({B, H}) * 0.9f;   // scalar decay
    auto beta = mx::ones({B, H}) * 0.5f;
    auto state = mx::zeros({B, H, Dv, Dk}, mx::float32);

    auto [y, new_state] = gated_delta_step_ops(q, k, v, g, beta, state);
    mx::eval(y);
    mx::eval(new_state);
    CHECK(y.shape(0) == B && y.shape(1) == H && y.shape(2) == Dv,
          "gd_step output shape");
    CHECK(new_state.shape(2) == Dv && new_state.shape(3) == Dk,
          "gd_step state shape");
    // With zero state: kv_mem=0, delta=v*beta=0.5, state=k*delta
    // output = sum(state * q) per dv = sum(k*delta*q) = sum(1*0.5*1) = Dk*0.5
    auto y32 = mx::astype(y, mx::float32);
    mx::eval(y32);
    CHECK(near(y32.data<float>()[0], Dk * 0.5f, 1e-3f), "gd_step value");
}

// ─── gated_delta_ops (prefill loop) test ────────────────────────────────────

void test_gd_ops() {
    int B = 1, T = 3, Hk = 2, Dk = 4, Hv = 2, Dv = 4;
    auto q = mx::ones({B, T, Hk, Dk});
    auto k = mx::ones({B, T, Hk, Dk});
    auto v = mx::ones({B, T, Hv, Dv});
    auto g = mx::ones({B, T, Hv}) * 0.9f;
    auto beta = mx::ones({B, T, Hv}) * 0.5f;
    auto state = mx::zeros({B, Hv, Dv, Dk}, mx::float32);

    auto [y, new_state] = gated_delta_ops(q, k, v, g, beta, state);
    mx::eval(y);
    mx::eval(new_state);
    CHECK(y.shape(1) == T, "gd_ops output seqlen");
    CHECK(y.shape(2) == Hv && y.shape(3) == Dv, "gd_ops output headdim");
}

// ─── gated_delta_update (full pipeline) test ────────────────────────────────

void test_gd_update() {
    int B = 1, T = 2, Hk = 2, Dk = 32, Hv = 2, Dv = 4;
    auto q = mx::ones({B, T, Hk, Dk}, mx::float16);
    auto k = mx::ones({B, T, Hk, Dk}, mx::float16);
    auto v = mx::ones({B, T, Hv, Dv}, mx::float16);
    auto a_proj = mx::zeros({B, T, Hv}, mx::float16);
    auto b_proj = mx::zeros({B, T, Hv}, mx::float16);
    auto A_log = mx::zeros({Hv});
    auto dt_bias = mx::zeros({Hv});
    auto state = mx::zeros({B, Hv, Dv, Dk}, mx::float32);

    // Test ops path (use_kernel=false)
    auto [y, ns] = gated_delta_update(
        q, k, v, a_proj, b_proj, A_log, dt_bias, state, std::nullopt, false);
    mx::eval(y);
    mx::eval(ns);
    CHECK(y.shape(0) == B && y.shape(1) == T, "gd_update output batch/seq");
    CHECK(ns.shape(0) == B && ns.shape(1) == Hv, "gd_update state shape");
}

// ─── MLP test ───────────────────────────────────────────────────────────────

void test_mlp() {
    int D = 8, I = 16;
    MLP mlp;
    mlp.gate_proj_w = mx::zeros({I, D});  // all-zero weights
    mlp.up_proj_w = mx::zeros({I, D});
    mlp.down_proj_w = mx::zeros({D, I});

    auto x = mx::ones({1, 3, D});
    auto y = mlp.forward(x);
    mx::eval(y);
    CHECK(y.shape(0) == 1 && y.shape(1) == 3 && y.shape(2) == D,
          "mlp output shape");
    // All zero weights → output is zero
    auto y32 = mx::astype(y, mx::float32);
    mx::eval(y32);
    CHECK(near(y32.data<float>()[0], 0.0f), "mlp zero-weight output");
}

// ─── Attention test (smoke) ─────────────────────────────────────────────────

void test_attention_smoke() {
    int D = 32, H = 2, Hkv = 2, Hdim = 16;
    Attention attn;
    attn.num_heads = H;
    attn.num_kv_heads = Hkv;
    attn.head_dim = Hdim;
    attn.scale = std::pow(static_cast<float>(Hdim), -0.5f);
    attn.norm_eps = 1e-6f;
    attn.rope = RotaryEmbedding(8, 10000.0f, {2, 2, 0, 0});

    // Identity-like weights for smoke test
    attn.q_proj_w = mx::zeros({H * Hdim * 2, D});  // q_proj is 2× for gate
    attn.k_proj_w = mx::zeros({Hkv * Hdim, D});
    attn.v_proj_w = mx::zeros({Hkv * Hdim, D});
    attn.o_proj_w = mx::zeros({D, H * Hdim});
    attn.q_norm_w = mx::ones({Hdim});
    attn.k_norm_w = mx::ones({Hdim});

    KVCache cache;
    auto x = mx::ones({1, 4, D}, mx::float16);
    auto y = attn.forward(x, cache, std::nullopt);
    mx::eval(y);
    CHECK(y.shape(0) == 1 && y.shape(1) == 4 && y.shape(2) == D,
          "attention output shape");
    CHECK(cache.offset == 4, "attention cache offset");

    // Decode step
    auto x2 = mx::ones({1, 1, D}, mx::float16);
    auto y2 = attn.forward(x2, cache, std::nullopt);
    mx::eval(y2);
    CHECK(y2.shape(1) == 1, "attention decode shape");
    CHECK(cache.offset == 5, "attention cache offset after decode");
}

// ─── GatedDeltaNet (module) test ────────────────────────────────────────────

void test_gdn_module() {
    int D = 32, Hk = 2, Hv = 2, Dk = 8, Dv = 8, ks = 4;
    int key_dim = Hk * Dk, val_dim = Hv * Dv;
    int conv_dim = key_dim * 2 + val_dim;

    GatedDeltaNet gdn;
    gdn.num_k_heads = Hk;
    gdn.num_v_heads = Hv;
    gdn.head_k_dim = Dk;
    gdn.head_v_dim = Dv;
    gdn.key_dim = key_dim;
    gdn.value_dim = val_dim;
    gdn.conv_dim = conv_dim;
    gdn.conv_kernel_size = ks;
    gdn.norm_eps = 1e-6f;

    gdn.in_proj_qkv_w = mx::zeros({conv_dim, D});
    gdn.in_proj_z_w = mx::zeros({val_dim, D});
    gdn.in_proj_b_w = mx::zeros({Hv, D});
    gdn.in_proj_a_w = mx::zeros({Hv, D});
    gdn.conv1d_w = mx::ones({conv_dim, ks, 1}) * 0.01f;
    gdn.A_log = mx::zeros({Hv});
    gdn.dt_bias = mx::ones({Hv});
    gdn.norm_w = mx::ones({Dv});
    gdn.out_proj_w = mx::zeros({D, val_dim});

    ArraysCache cache;
    auto x = mx::ones({1, 3, D}, mx::float16);
    auto y = gdn.forward(x, cache);
    mx::eval(y);
    CHECK(y.shape(0) == 1 && y.shape(1) == 3 && y.shape(2) == D,
          "gdn output shape");
    CHECK(cache.conv_state.has_value(), "gdn conv_state populated");
    CHECK(cache.rec_state.has_value(), "gdn rec_state populated");
    CHECK(cache.conv_state->shape(1) == ks - 1, "gdn conv_state len");
}

// ─── DecoderLayer test ──────────────────────────────────────────────────────

void test_decoder_layer() {
    int D = 32, Hk = 2, Hv = 2, Dk = 8, Dv = 8, ks = 4, I = 64;
    int key_dim = Hk * Dk, val_dim = Hv * Dv;
    int conv_dim = key_dim * 2 + val_dim;

    // Build a linear (GDN) decoder layer
    DecoderLayer layer;
    layer.is_linear = true;
    layer.ln_eps = 1e-6f;
    layer.input_ln_w = mx::ones({D});
    layer.post_attn_ln_w = mx::ones({D});

    layer.mlp.gate_proj_w = mx::zeros({I, D});
    layer.mlp.up_proj_w = mx::zeros({I, D});
    layer.mlp.down_proj_w = mx::zeros({D, I});

    GatedDeltaNet gdn;
    gdn.num_k_heads = Hk;
    gdn.num_v_heads = Hv;
    gdn.head_k_dim = Dk;
    gdn.head_v_dim = Dv;
    gdn.key_dim = key_dim;
    gdn.value_dim = val_dim;
    gdn.conv_dim = conv_dim;
    gdn.conv_kernel_size = ks;
    gdn.norm_eps = 1e-6f;
    gdn.in_proj_qkv_w = mx::zeros({conv_dim, D});
    gdn.in_proj_z_w = mx::zeros({val_dim, D});
    gdn.in_proj_b_w = mx::zeros({Hv, D});
    gdn.in_proj_a_w = mx::zeros({Hv, D});
    gdn.conv1d_w = mx::ones({conv_dim, ks, 1}) * 0.01f;
    gdn.A_log = mx::zeros({Hv});
    gdn.dt_bias = mx::ones({Hv});
    gdn.norm_w = mx::ones({Dv});
    gdn.out_proj_w = mx::zeros({D, val_dim});
    layer.linear_attn = std::move(gdn);

    LayerCache cache = ArraysCache{};
    auto x = mx::ones({1, 3, D}, mx::float16);
    auto y = layer.forward(x, cache);
    mx::eval(y);
    CHECK(y.shape(0) == 1 && y.shape(1) == 3 && y.shape(2) == D,
          "decoder_layer output shape");
}

// ─── LanguageModel.make_cache + make_position_ids ───────────────────────────

void test_lm_helpers() {
    auto pos = LanguageModel::make_position_ids(5, 0);
    mx::eval(pos);
    CHECK(pos.shape(0) == 3 && pos.shape(1) == 1 && pos.shape(2) == 5,
          "make_position_ids shape");

    auto pos2 = LanguageModel::make_position_ids(1, 10);
    mx::eval(pos2);
    auto p32 = mx::astype(pos2, mx::int32);
    mx::eval(p32);
    CHECK(p32.data<int>()[0] == 10, "make_position_ids offset");
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Phase 2 MLX Module Tests ===\n\n";

    // mlx_helpers
    test_silu();
    test_softplus();
    test_swiglu();
    test_linear();
    test_embedding();
    test_rotate_half();
    test_compute_g();

    // RotaryEmbedding
    test_rotary_embedding();
    test_apply_rotary();

    // Cache
    test_kv_cache();

    // Gated Delta
    test_gd_step_ops();
    test_gd_ops();
    test_gd_update();

    // Modules
    test_mlp();
    test_attention_smoke();
    test_gdn_module();
    test_decoder_layer();

    // LM helpers
    test_lm_helpers();

    std::cout << "\n=== Results: " << pass_count << " PASS, " << fail_count
              << " FAIL ===\n";
    return fail_count > 0 ? 1 : 0;
}
