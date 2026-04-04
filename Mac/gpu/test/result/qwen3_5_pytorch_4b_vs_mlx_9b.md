# Qwen3.5 PyTorch 4B vs MLX 9B Comparison

- Generated at: 2026-04-04T19:31:28
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json
- Base model: qwen3_5_4b_pytorch (transformers) /Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b
- Candidate model: qwen3_5_9b_mlx_8bit (mlx_vlm) /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 64

## Summary

| Model | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_4b_pytorch | 20 | 34.850 | 51.700 | 352.396 | 135.037 | 7445.217 | 6.785 | 7.923 |
| qwen3_5_9b_mlx_8bit | 20 | 34.850 | 49.750 | 376.216 | 86.362 | 4735.676 | 10.159 | 10.544 |

## Candidate Minus Base

- Avg prefill ms delta: 23.820
- Avg decode ms/tok delta: -48.675
- Avg wall ms delta: -2709.541
- Avg throughput delta: 3.374
- Avg peak memory GB delta: 2.621
- Exact output matches: 0/20

## Per-Prompt

| Prompt | Kind | 4B Tok/s | 9B Tok/s | Delta Tok/s | 4B Wall ms | 9B Wall ms | Delta Wall ms | Match |
|-------|------|---------:|---------:|------------:|-----------:|-----------:|--------------:|:------|
| short_01 | short | 4.134 | 4.518 | 0.384 | 2419.242 | 2213.255 | -205.987 | no |
| short_02 | short | 6.567 | 9.221 | 2.654 | 2131.733 | 976.031 | -1155.702 | no |
| short_03 | short | 6.972 | 10.972 | 4.000 | 9180.231 | 5833.163 | -3347.068 | no |
| short_04 | short | 7.189 | 10.186 | 2.997 | 3755.667 | 2061.652 | -1694.015 | no |
| short_05 | short | 7.043 | 10.843 | 3.800 | 9086.707 | 5902.448 | -3184.259 | no |
| short_06 | short | 6.921 | 10.855 | 3.934 | 9246.940 | 5896.085 | -3350.855 | no |
| short_07 | short | 5.893 | 8.248 | 2.355 | 848.485 | 606.212 | -242.273 | no |
| short_08 | short | 7.154 | 10.805 | 3.651 | 8806.036 | 3054.269 | -5751.767 | no |
| short_09 | short | 6.900 | 10.490 | 3.590 | 2753.784 | 2001.970 | -751.814 | no |
| short_10 | short | 7.069 | 10.856 | 3.787 | 9053.213 | 5895.115 | -3158.098 | no |
| long_01 | long | 6.912 | 10.585 | 3.673 | 9259.159 | 6046.388 | -3212.771 | no |
| long_02 | long | 7.057 | 10.589 | 3.532 | 9068.498 | 6044.070 | -3024.428 | no |
| long_03 | long | 6.953 | 10.398 | 3.445 | 9204.201 | 6155.112 | -3049.089 | no |
| long_04 | long | 6.803 | 10.726 | 3.923 | 9407.745 | 5966.623 | -3441.122 | no |
| long_05 | long | 7.071 | 10.566 | 3.495 | 9051.038 | 6057.084 | -2993.954 | no |
| long_06 | long | 6.972 | 10.777 | 3.805 | 9179.932 | 5938.458 | -3241.474 | no |
| long_07 | long | 6.985 | 10.584 | 3.599 | 9161.981 | 6046.924 | -3115.057 | no |
| long_08 | long | 7.025 | 10.693 | 3.668 | 9109.882 | 5985.204 | -3124.678 | no |
| long_09 | long | 7.015 | 10.665 | 3.650 | 9123.804 | 6001.022 | -3122.782 | no |
| long_10 | long | 7.067 | 10.609 | 3.542 | 9056.063 | 6032.445 | -3023.618 | no |
