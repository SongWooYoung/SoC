# Qwen3.5-4B Custom Backend Comparison

- Generated at: 2026-04-04T22:03:14
- Model dir: /Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_pytorch_cpp_mlx_compare_smoke_suite.json
- Max new tokens: 64

## Summary

| Backend | Rows | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |
|--------|-----:|------------:|---------------:|------------------:|------------:|----------:|
| base | 1 | 10.000 | 1242.838 | 132.029 | 2563.126 | 3.901 |
| cpp | 1 | 10.000 | 17992.696 | 501.739 | 23010.209 | 0.435 |
| mlx | 1 | 64.000 | 2985.936 | 189.191 | 15111.863 | 4.235 |

## Match Counts Vs Base

- cpp prompt token matches: 1/1
- cpp generated token matches: 1/1
- cpp output text matches: 1/1
- mlx prompt token matches: 1/1
- mlx generated token matches: 0/1
- mlx output text matches: 0/1

## Per-Prompt

| Prompt | Kind | Base Tok/s | Cpp Tok/s | MLX Tok/s | Cpp Match | MLX Match |
|-------|------|-----------:|----------:|----------:|:---------:|:---------:|
| short_01 | short | 3.901 | 0.435 | 4.235 | yes | no |
