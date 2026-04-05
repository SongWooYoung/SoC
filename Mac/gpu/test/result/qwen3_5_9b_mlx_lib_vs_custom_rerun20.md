# Qwen3.5-9B MLX Library vs qwen3_5_mlx

- Generated at: 2026-04-05T09:59:23
- Prompt suite: test/prompt_suite.json
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 64
- Custom gated_delta mode: metal_kernel
- Custom linear cache mode: legacy
- Custom stage trace enabled: False

## Summary

| Backend | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|--------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_9b_mlx_library | 20 | 34.850 | 49.750 | 343.107 | 83.150 | 4556.370 | 10.590 | 10.544 |
| qwen3_5_9b_qwen3_5_mlx | 20 | 34.850 | 49.750 | 293.700 | 108.505 | 5780.042 | 8.520 | 0.000 |

## Custom Minus MLX Library

- Avg prefill ms delta: -49.407
- Avg decode ms/tok delta: 25.355
- Avg wall ms delta: 1223.672
- Avg throughput delta: -2.070
- Prompt token matches: 20/20
- Generated token matches: 16/20
- Output text matches: 10/20

## Per-Prompt

| Prompt | Kind | MLX Tok/s | Custom Tok/s | Delta Tok/s | MLX Wall ms | Custom Wall ms | Delta Wall ms | Match |
|-------|------|----------:|-------------:|------------:|------------:|---------------:|--------------:|:------|
| short_01 | short | 6.485 | 7.869 | 1.384 | 1542.134 | 1270.754 | -271.380 | no |
| short_02 | short | 9.638 | 7.898 | -1.740 | 933.769 | 1139.507 | 205.738 | no |
| short_03 | short | 11.398 | 8.507 | -2.891 | 5614.845 | 7523.464 | 1908.619 | yes |
| short_04 | short | 10.700 | 8.582 | -2.118 | 1962.565 | 2446.961 | 484.396 | no |
| short_05 | short | 11.343 | 9.004 | -2.339 | 5642.338 | 7107.829 | 1465.491 | no |
| short_06 | short | 11.161 | 8.941 | -2.220 | 5734.482 | 7157.980 | 1423.498 | yes |
| short_07 | short | 8.604 | 7.730 | -0.874 | 581.147 | 646.801 | 65.654 | no |
| short_08 | short | 11.009 | 8.843 | -2.166 | 2997.649 | 3731.787 | 734.138 | no |
| short_09 | short | 10.754 | 8.730 | -2.024 | 1952.674 | 2405.375 | 452.701 | no |
| short_10 | short | 11.194 | 8.883 | -2.311 | 5717.257 | 7204.581 | 1487.324 | yes |
| long_01 | long | 10.720 | 8.717 | -2.003 | 5970.325 | 7342.175 | 1371.850 | yes |
| long_02 | long | 10.986 | 8.677 | -2.309 | 5825.471 | 7375.704 | 1550.233 | yes |
| long_03 | long | 10.945 | 8.442 | -2.503 | 5847.346 | 7580.851 | 1733.505 | no |
| long_04 | long | 11.007 | 8.799 | -2.208 | 5814.491 | 7273.329 | 1458.838 | no |
| long_05 | long | 10.889 | 8.686 | -2.203 | 5877.656 | 7368.038 | 1490.382 | no |
| long_06 | long | 11.029 | 8.591 | -2.438 | 5802.698 | 7449.886 | 1647.188 | yes |
| long_07 | long | 11.001 | 8.416 | -2.585 | 5817.641 | 7604.962 | 1787.321 | yes |
| long_08 | long | 10.842 | 8.472 | -2.370 | 5902.802 | 7554.283 | 1651.481 | yes |
| long_09 | long | 11.058 | 8.483 | -2.575 | 5787.588 | 7544.073 | 1756.485 | yes |
| long_10 | long | 11.030 | 8.130 | -2.900 | 5802.518 | 7872.505 | 2069.987 | yes |
