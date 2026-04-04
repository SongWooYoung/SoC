# Qwen3.5-4B Custom Backend Comparison

- Generated at: 2026-04-04T22:21:04
- Model dir: /Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json
- Max new tokens: 64

## Summary

| Backend | Rows | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |
|--------|-----:|------------:|---------------:|------------------:|------------:|----------:|
| base | 20 | 51.700 | 364.694 | 135.572 | 7458.752 | 6.769 |
| cpp | 20 | 51.250 | 1950.096 | 561.778 | 31329.043 | 1.597 |
| mlx | 20 | 64.000 | 536.146 | 183.569 | 12284.974 | 5.223 |

## Match Counts Vs Base

- cpp prompt token matches: 20/20
- cpp generated token matches: 12/20
- cpp output text matches: 12/20
- mlx prompt token matches: 20/20
- mlx generated token matches: 0/20
- mlx output text matches: 0/20

## Per-Prompt

| Prompt | Kind | Base Tok/s | Cpp Tok/s | MLX Tok/s | Cpp Match | MLX Match |
|-------|------|-----------:|----------:|----------:|:---------:|:---------:|
| short_01 | short | 3.727 | 0.537 | 4.208 | yes | no |
| short_02 | short | 6.496 | 1.541 | 5.261 | no | no |
| short_03 | short | 6.897 | 1.690 | 5.312 | no | no |
| short_04 | short | 7.070 | 1.663 | 5.348 | yes | no |
| short_05 | short | 7.004 | 1.681 | 5.357 | yes | no |
| short_06 | short | 7.173 | 1.695 | 5.253 | no | no |
| short_07 | short | 5.992 | 1.483 | 5.332 | yes | no |
| short_08 | short | 7.147 | 1.685 | 5.343 | yes | no |
| short_09 | short | 7.057 | 1.639 | 5.357 | no | no |
| short_10 | short | 7.071 | 1.693 | 5.327 | no | no |
| long_01 | long | 6.780 | 1.663 | 5.125 | no | no |
| long_02 | long | 6.992 | 1.660 | 5.248 | no | no |
| long_03 | long | 6.915 | 1.658 | 5.219 | yes | no |
| long_04 | long | 7.067 | 1.668 | 5.272 | yes | no |
| long_05 | long | 7.046 | 1.661 | 5.266 | yes | no |
| long_06 | long | 7.036 | 1.667 | 5.182 | yes | no |
| long_07 | long | 7.030 | 1.670 | 5.239 | yes | no |
| long_08 | long | 6.835 | 1.662 | 5.265 | no | no |
| long_09 | long | 7.074 | 1.671 | 5.283 | yes | no |
| long_10 | long | 6.966 | 1.662 | 5.263 | yes | no |
