# Qwen3.5-9B MLX Library vs qwen3_5_mlx

- Generated at: 2026-04-05T09:51:48
- Prompt suite: test/optimization/plan0/decode_followup/qwen3_5_9b_lib_vs_custom_legacy_4prompt_suite.json
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 32
- Custom gated_delta mode: metal_kernel
- Custom linear cache mode: legacy
- Custom stage trace enabled: False

## Summary

| Backend | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|--------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_9b_mlx_library | 4 | 36.250 | 32.000 | 546.781 | 85.699 | 3289.167 | 9.829 | 10.545 |
| qwen3_5_9b_qwen3_5_mlx | 4 | 36.250 | 32.000 | 303.860 | 112.700 | 3910.595 | 8.185 | 0.000 |

## Custom Minus MLX Library

- Avg prefill ms delta: -242.921
- Avg decode ms/tok delta: 27.001
- Avg wall ms delta: 621.428
- Avg throughput delta: -1.644
- Prompt token matches: 4/4
- Generated token matches: 3/4
- Output text matches: 3/4

## Per-Prompt

| Prompt | Kind | MLX Tok/s | Custom Tok/s | Delta Tok/s | MLX Wall ms | Custom Wall ms | Delta Wall ms | Match |
|-------|------|----------:|-------------:|------------:|------------:|---------------:|--------------:|:------|
| short_03 | short | 8.216 | 8.332 | 0.116 | 3894.995 | 3840.688 | -54.307 | yes |
| short_10 | short | 10.505 | 8.281 | -2.224 | 3046.241 | 3864.329 | 818.088 | yes |
| long_03 | long | 10.182 | 8.009 | -2.173 | 3142.937 | 3995.386 | 852.449 | no |
| long_09 | long | 10.415 | 8.118 | -2.297 | 3072.494 | 3941.977 | 869.483 | yes |
