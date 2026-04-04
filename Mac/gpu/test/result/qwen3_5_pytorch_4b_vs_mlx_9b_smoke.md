# Qwen3.5 PyTorch 4B vs MLX 9B Comparison

- Generated at: 2026-04-04T19:26:45
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json
- Base model: qwen3_5_4b_pytorch (transformers) /Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b
- Candidate model: qwen3_5_9b_mlx_8bit (mlx_vlm) /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Max new tokens: 64

## Summary

| Model | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |
|------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|
| qwen3_5_4b_pytorch | 1 | 15.000 | 10.000 | 1454.390 | 140.341 | 2857.802 | 3.499 | 7.909 |
| qwen3_5_9b_mlx_8bit | 1 | 15.000 | 10.000 | 1225.463 | 82.122 | 2046.683 | 4.886 | 10.495 |

## Candidate Minus Base

- Avg prefill ms delta: -228.927
- Avg decode ms/tok delta: -58.219
- Avg wall ms delta: -811.119
- Avg throughput delta: 1.387
- Avg peak memory GB delta: 2.586
- Exact output matches: 0/1

## Per-Prompt

| Prompt | Kind | 4B Tok/s | 9B Tok/s | Delta Tok/s | 4B Wall ms | 9B Wall ms | Delta Wall ms | Match |
|-------|------|---------:|---------:|------------:|-----------:|-----------:|--------------:|:------|
| short_01 | short | 3.499 | 4.886 | 1.387 | 2857.802 | 2046.683 | -811.119 | no |
