# Qwen3.5-9B MLX Library vs qwen3_5_mlx

- Generated at: 2026-04-04T22:56:36
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 64

## Summary

| Backend | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|--------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_9b_mlx_library | 20 | 34.850 | 49.750 | 325.039 | 85.193 | 4636.610 | 10.473 | 10.544 |
| qwen3_5_9b_qwen3_5_mlx | 20 | 34.850 | 49.750 | 831.370 | 119.557 | 6687.734 | 7.468 | 0.000 |

## Custom Minus MLX Library

- Avg prefill ms delta: 506.331
- Avg decode ms/tok delta: 34.364
- Avg wall ms delta: 2051.124
- Avg throughput delta: -3.005
- Prompt token matches: 20/20
- Generated token matches: 16/20
- Output text matches: 10/20

## Per-Prompt

| Prompt | Kind | MLX Tok/s | Custom Tok/s | Delta Tok/s | MLX Wall ms | Custom Wall ms | Delta Wall ms | Match |
|-------|------|----------:|-------------:|------------:|------------:|---------------:|--------------:|:------|
| short_01 | short | 9.013 | 1.083 | -7.930 | 1109.547 | 9237.579 | 8128.032 | no |
| short_02 | short | 9.463 | 6.906 | -2.557 | 951.101 | 1303.205 | 352.104 | no |
| short_03 | short | 11.126 | 7.998 | -3.128 | 5752.137 | 8002.303 | 2250.166 | yes |
| short_04 | short | 10.271 | 7.959 | -2.312 | 2044.576 | 2638.508 | 593.932 | no |
| short_05 | short | 11.018 | 8.256 | -2.762 | 5808.638 | 7751.574 | 1942.936 | no |
| short_06 | short | 11.080 | 8.314 | -2.766 | 5776.108 | 7697.608 | 1921.500 | yes |
| short_07 | short | 8.445 | 6.390 | -2.055 | 592.086 | 782.475 | 190.389 | no |
| short_08 | short | 10.571 | 8.068 | -2.503 | 3121.616 | 4090.235 | 968.619 | no |
| short_09 | short | 10.292 | 7.895 | -2.397 | 2040.330 | 2659.806 | 619.476 | no |
| short_10 | short | 11.101 | 8.365 | -2.736 | 5765.013 | 7651.149 | 1886.136 | yes |
| long_01 | long | 10.616 | 7.788 | -2.828 | 6028.853 | 8217.409 | 2188.556 | yes |
| long_02 | long | 10.466 | 7.981 | -2.485 | 6115.086 | 8019.478 | 1904.392 | yes |
| long_03 | long | 10.774 | 7.680 | -3.094 | 5940.440 | 8333.846 | 2393.406 | no |
| long_04 | long | 10.657 | 7.940 | -2.717 | 6005.523 | 8060.520 | 2054.997 | no |
| long_05 | long | 10.792 | 7.863 | -2.929 | 5930.346 | 8139.638 | 2209.292 | no |
| long_06 | long | 10.831 | 7.991 | -2.840 | 5909.079 | 8008.653 | 2099.574 | yes |
| long_07 | long | 10.676 | 7.758 | -2.918 | 5994.917 | 8249.940 | 2255.023 | yes |
| long_08 | long | 10.801 | 7.963 | -2.838 | 5925.404 | 8037.025 | 2111.621 | yes |
| long_09 | long | 10.720 | 7.584 | -3.136 | 5970.020 | 8438.959 | 2468.939 | yes |
| long_10 | long | 10.754 | 7.588 | -3.166 | 5951.385 | 8434.776 | 2483.391 | yes |
