# Qwen3.5-9B MLX Library vs qwen3_5_mlx

- Generated at: 2026-04-05T09:15:57
- Prompt suite: test/optimization/plan0/decode_followup/qwen3_5_9b_lib_vs_custom_step_buffer_4prompt_suite.json
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 32
- Custom gated_delta mode: metal_kernel
- Custom linear cache mode: legacy
- Custom stage trace enabled: False

## Summary

| Backend | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|--------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_9b_mlx_library | 4 | 36.250 | 32.000 | 540.827 | 86.033 | 3293.875 | 9.804 | 10.545 |
| qwen3_5_9b_qwen3_5_mlx | 4 | 36.250 | 32.000 | 301.401 | 112.407 | 3898.779 | 8.211 | 0.000 |

## Custom Minus MLX Library

- Avg prefill ms delta: -239.426
- Avg decode ms/tok delta: 26.374
- Avg wall ms delta: 604.904
- Avg throughput delta: -1.593
- Prompt token matches: 4/4
- Generated token matches: 3/4
- Output text matches: 3/4

## Per-Prompt

| Prompt | Kind | MLX Tok/s | Custom Tok/s | Delta Tok/s | MLX Wall ms | Custom Wall ms | Delta Wall ms | Match |
|-------|------|----------:|-------------:|------------:|------------:|---------------:|--------------:|:------|
| short_03 | short | 8.307 | 8.111 | -0.196 | 3852.400 | 3945.123 | 92.723 | yes |
| short_10 | short | 10.638 | 8.458 | -2.180 | 3007.994 | 3783.289 | 775.295 | yes |
| long_03 | long | 10.070 | 8.036 | -2.034 | 3177.735 | 3981.958 | 804.223 | no |
| long_09 | long | 10.200 | 8.237 | -1.963 | 3137.371 | 3884.747 | 747.376 | yes |
