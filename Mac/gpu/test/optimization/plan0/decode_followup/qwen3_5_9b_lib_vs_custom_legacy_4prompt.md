# Qwen3.5-9B MLX Library vs qwen3_5_mlx

- Generated at: 2026-04-05T09:16:58
- Prompt suite: test/optimization/plan0/decode_followup/qwen3_5_9b_lib_vs_custom_legacy_4prompt_suite.json
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 32
- Custom gated_delta mode: metal_kernel
- Custom linear cache mode: legacy
- Custom stage trace enabled: False

## Summary

| Backend | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|--------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_9b_mlx_library | 4 | 36.250 | 32.000 | 345.962 | 88.229 | 3169.286 | 10.104 | 10.545 |
| qwen3_5_9b_qwen3_5_mlx | 4 | 36.250 | 32.000 | 303.430 | 111.221 | 3862.802 | 8.288 | 0.000 |

## Custom Minus MLX Library

- Avg prefill ms delta: -42.532
- Avg decode ms/tok delta: 22.992
- Avg wall ms delta: 693.516
- Avg throughput delta: -1.816
- Prompt token matches: 4/4
- Generated token matches: 3/4
- Output text matches: 3/4

## Per-Prompt

| Prompt | Kind | MLX Tok/s | Custom Tok/s | Delta Tok/s | MLX Wall ms | Custom Wall ms | Delta Wall ms | Match |
|-------|------|----------:|-------------:|------------:|------------:|---------------:|--------------:|:------|
| short_03 | short | 10.094 | 8.114 | -1.980 | 3170.341 | 3943.880 | 773.539 | yes |
| short_10 | short | 10.545 | 8.582 | -1.963 | 3034.551 | 3728.603 | 694.052 | yes |
| long_03 | long | 9.919 | 8.188 | -1.731 | 3226.192 | 3908.371 | 682.179 | no |
| long_09 | long | 9.858 | 8.268 | -1.590 | 3246.059 | 3870.355 | 624.296 | yes |
