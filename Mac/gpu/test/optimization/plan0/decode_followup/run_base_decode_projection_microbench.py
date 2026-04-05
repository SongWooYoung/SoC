#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import mlx.core as mx
from mlx_vlm import load as mlx_load


DEFAULT_MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
DEFAULT_OUTPUT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_decode_projection_microbench.json")
DEFAULT_TOKEN_ID = 42
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 50


def bench(fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
        mx.synchronize()

    dispatch_times: list[float] = []
    sync_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out = fn()
        dispatch_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        mx.eval(out)
        mx.synchronize()
        sync_times.append((time.perf_counter() - t1) * 1000.0)

    return {
        "dispatch_ms": round(mean(dispatch_times), 3),
        "sync_ms": round(mean(sync_times), 3),
    }


def main() -> None:
    model, _processor = mlx_load(str(DEFAULT_MODEL_DIR))
    language_model = model.language_model

    token_input = mx.array([[DEFAULT_TOKEN_ID]], dtype=mx.int32)
    hidden = language_model.model.embed_tokens(token_input)
    mx.eval(hidden)
    mx.synchronize()

    mlp_layers = []
    for layer_index, layer in enumerate(language_model.model.layers):
        mlp_input = layer.post_attention_layernorm(hidden)
        mx.eval(mlp_input)
        mx.synchronize()
        result = bench(lambda layer=layer, mlp_input=mlp_input: layer.mlp(mlp_input), DEFAULT_WARMUP, DEFAULT_ITERATIONS)
        mlp_layers.append(
            {
                "layer_index": layer_index,
                "layer_type": "linear" if layer.is_linear else "full_attention",
                **result,
            }
        )

    if language_model.args.tie_word_embeddings:
        lm_head_result = bench(
            lambda: language_model.model.embed_tokens.as_linear(hidden),
            DEFAULT_WARMUP,
            DEFAULT_ITERATIONS,
        )
    else:
        lm_head_result = bench(
            lambda: language_model.lm_head(hidden),
            DEFAULT_WARMUP,
            DEFAULT_ITERATIONS,
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "token_id": DEFAULT_TOKEN_ID,
        "warmup": DEFAULT_WARMUP,
        "iterations": DEFAULT_ITERATIONS,
        "hidden_dtype": str(hidden.dtype),
        "mlp_average": {
            "dispatch_ms": round(mean(row["dispatch_ms"] for row in mlp_layers), 3),
            "sync_ms": round(mean(row["sync_ms"] for row in mlp_layers), 3),
        },
        "mlp_layers": mlp_layers,
        "lm_head_decode": lm_head_result,
    }

    DEFAULT_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote base decode projection microbench: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()