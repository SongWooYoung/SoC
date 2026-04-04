# Qwen3.5-9B MLX Library vs qwen3_5_mlx

- Generated at: 2026-04-05T00:55:41
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 64
- Custom gated_delta mode: metal_kernel
- Custom linear cache mode: legacy
- Custom stage trace enabled: False

## Summary

| Backend | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|--------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_9b_mlx_library | 20 | 34.850 | 49.750 | 371.929 | 85.665 | 4720.391 | 10.210 | 10.544 |
| qwen3_5_9b_qwen3_5_mlx | 20 | 34.850 | 49.750 | 296.364 | 111.496 | 5938.916 | 8.299 | 0.000 |

## Custom Minus MLX Library

- Avg prefill ms delta: -75.565
- Avg decode ms/tok delta: 25.831
- Avg wall ms delta: 1218.525
- Avg throughput delta: -1.911
- Prompt token matches: 20/20
- Generated token matches: 16/20
- Output text matches: 10/20

## Per-Prompt

| Prompt | Kind | MLX Tok/s | Custom Tok/s | Delta Tok/s | MLX Wall ms | Custom Wall ms | Delta Wall ms | Match |
|-------|------|----------:|-------------:|------------:|------------:|---------------:|--------------:|:------|
| short_01 | short | 4.918 | 7.920 | 3.002 | 2033.425 | 1262.663 | -770.762 | no |
| short_02 | short | 9.466 | 7.991 | -1.475 | 950.773 | 1126.320 | 175.547 | no |
| short_03 | short | 10.635 | 7.640 | -2.995 | 6017.999 | 8376.662 | 2358.663 | yes |
| short_04 | short | 10.470 | 7.633 | -2.837 | 2005.785 | 2751.210 | 745.425 | no |
| short_05 | short | 10.960 | 8.426 | -2.534 | 5839.150 | 7595.760 | 1756.610 | no |
| short_06 | short | 10.897 | 8.548 | -2.349 | 5873.365 | 7486.879 | 1613.514 | yes |
| short_07 | short | 8.330 | 7.585 | -0.745 | 600.229 | 659.197 | 58.968 | no |
| short_08 | short | 10.693 | 8.584 | -2.109 | 3086.245 | 3844.465 | 758.220 | no |
| short_09 | short | 10.450 | 8.598 | -1.852 | 2009.637 | 2442.472 | 432.835 | no |
| short_10 | short | 11.026 | 8.735 | -2.291 | 5804.244 | 7326.887 | 1522.643 | yes |
| long_01 | long | 10.525 | 8.429 | -2.096 | 6080.794 | 7592.916 | 1512.122 | yes |
| long_02 | long | 10.738 | 8.537 | -2.201 | 5960.191 | 7496.486 | 1536.295 | yes |
| long_03 | long | 10.622 | 8.398 | -2.224 | 6025.161 | 7620.483 | 1595.322 | no |
| long_04 | long | 10.662 | 8.543 | -2.119 | 6002.784 | 7491.514 | 1488.730 | no |
| long_05 | long | 10.743 | 8.402 | -2.341 | 5957.094 | 7617.342 | 1660.248 | no |
| long_06 | long | 10.341 | 8.323 | -2.018 | 6188.956 | 7689.465 | 1500.509 | yes |
| long_07 | long | 10.677 | 8.349 | -2.328 | 5994.119 | 7665.566 | 1671.447 | yes |
| long_08 | long | 10.622 | 8.494 | -2.128 | 6025.502 | 7535.067 | 1509.565 | yes |
| long_09 | long | 10.744 | 8.367 | -2.377 | 5956.562 | 7649.536 | 1692.974 | yes |
| long_10 | long | 10.674 | 8.480 | -2.194 | 5995.812 | 7547.421 | 1551.609 | yes |
