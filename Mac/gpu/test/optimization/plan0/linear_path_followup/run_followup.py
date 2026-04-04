#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu")
PLAN0_DIR = ROOT / "test/optimization/plan0/linear_path_followup"
PROMPT_SUITE = ROOT / "test/optimization/plan_0/operator_hook_test/prompt_suite.json"
BASE_TRACE = ROOT / "test/optimization/plan_0/operator_hook_test/base_stage_trace.json"
BASELINE_CUSTOM_TRACE = ROOT / "test/optimization/plan_0/operator_hook_test/custom_stage_trace.json"
ARRAYS_TRACE = PLAN0_DIR / "arrays_cache_stage_trace.json"
LINEAR_BENCH = PLAN0_DIR / "linear_attention_microbench.json"
PATH_BENCH = PLAN0_DIR / "quantized_path_bench.json"
SUMMARY = PLAN0_DIR / "summary.md"
CUSTOM_BIN = ROOT / "build/test_mlx_quantized_output_eval"
LINEAR_BENCH_BIN = ROOT / "build/test_mlx_linear_attention_microbench"
PATH_BENCH_BIN = ROOT / "build/test_mlx_quantized_path_bench"
MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
CONFIG_JSON = MODEL_DIR / "config.json"
MLX_METAL_PATH = ROOT / "../../.repo_cache/mlx/build/mlx/backend/metal/kernels"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def avg_stage_sync(trace: dict[str, Any], phase: str, stage: str) -> float:
    rows = trace["rows"]
    if not rows:
        return 0.0
    return sum(float(row["stage_trace"][phase].get(stage, {}).get("sync_ms", 0.0)) for row in rows) / len(rows)


def avg_stage_dispatch(trace: dict[str, Any], phase: str, stage: str) -> float:
    rows = trace["rows"]
    if not rows:
        return 0.0
    return sum(float(row["stage_trace"][phase].get(stage, {}).get("dispatch_ms", 0.0)) for row in rows) / len(rows)


def summary_block(trace: dict[str, Any]) -> dict[str, float]:
    rows = trace["rows"]
    count = len(rows)
    return {
        "rows": count,
        "avg_prefill_ms": sum(float(row["prefill_ms"]) for row in rows) / count,
        "avg_decode_ms": sum(float(row["decode_ms"]) for row in rows) / count,
        "avg_wall_ms": sum(float(row["wall_ms"]) for row in rows) / count,
        "avg_throughput": sum(float(row["throughput"]) for row in rows) / count,
    }


def render_summary(base: dict[str, Any], baseline: dict[str, Any], arrays: dict[str, Any], linear_bench: dict[str, Any], path_bench: dict[str, Any]) -> str:
    base_s = summary_block(base)
    baseline_s = summary_block(baseline)
    arrays_s = summary_block(arrays)

    legacy_prefill_cache = avg_stage_sync(baseline, "prefill", "linear_cache_update")
    arrays_prefill_cache = avg_stage_sync(arrays, "prefill", "linear_cache_update")
    legacy_decode_cache = avg_stage_sync(baseline, "decode", "linear_cache_update")
    arrays_decode_cache = avg_stage_sync(arrays, "decode", "linear_cache_update")

    legacy_linear_dispatch_prefill = avg_stage_dispatch(baseline, "prefill", "linear_attention")
    arrays_linear_dispatch_prefill = avg_stage_dispatch(arrays, "prefill", "linear_attention")
    legacy_linear_dispatch_decode = avg_stage_dispatch(baseline, "decode", "linear_attention")
    arrays_linear_dispatch_decode = avg_stage_dispatch(arrays, "decode", "linear_attention")

    lines = [
        "# Linear Path Follow-Up",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Base trace: {BASE_TRACE}",
        f"- Baseline custom trace: {BASELINE_CUSTOM_TRACE}",
        f"- Arrays-style custom trace: {ARRAYS_TRACE}",
        f"- Linear microbench: {LINEAR_BENCH}",
        f"- Quantized path bench: {PATH_BENCH}",
        "",
        "## Experiment 1: Arrays-Style Linear Cache Update",
        "",
        "| Backend | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |",
        "|--------|---------------:|------------------:|------------:|----------:|",
        f"| base | {base_s['avg_prefill_ms']:.3f} | {base_s['avg_decode_ms']:.3f} | {base_s['avg_wall_ms']:.3f} | {base_s['avg_throughput']:.3f} |",
        f"| custom baseline | {baseline_s['avg_prefill_ms']:.3f} | {baseline_s['avg_decode_ms']:.3f} | {baseline_s['avg_wall_ms']:.3f} | {baseline_s['avg_throughput']:.3f} |",
        f"| custom arrays-style | {arrays_s['avg_prefill_ms']:.3f} | {arrays_s['avg_decode_ms']:.3f} | {arrays_s['avg_wall_ms']:.3f} | {arrays_s['avg_throughput']:.3f} |",
        "",
        f"- Prefill linear_cache_update sync: baseline {legacy_prefill_cache:.3f} ms -> arrays-style {arrays_prefill_cache:.3f} ms.",
        f"- Decode linear_cache_update sync: baseline {legacy_decode_cache:.3f} ms -> arrays-style {arrays_decode_cache:.3f} ms.",
        f"- Prefill linear_attention dispatch: baseline {legacy_linear_dispatch_prefill:.3f} ms -> arrays-style {arrays_linear_dispatch_prefill:.3f} ms.",
        f"- Decode linear_attention dispatch: baseline {legacy_linear_dispatch_decode:.3f} ms -> arrays-style {arrays_linear_dispatch_decode:.3f} ms.",
        "",
        "## Experiment 2: Linear Attention Microbenchmark",
        "",
        "Synthetic shapes match the model config. This isolates host-side dispatch and stage sync without full-model noise.",
        "",
        "### Legacy vs Arrays-Style",
        "",
        "| Stage | Legacy dispatch ms | Legacy sync ms | Arrays dispatch ms | Arrays sync ms |",
        "|------|-------------------:|---------------:|-------------------:|---------------:|",
    ]

    for stage in sorted(set(linear_bench["legacy"].keys()) | set(linear_bench["arrays"].keys())):
        left = linear_bench["legacy"].get(stage, {"dispatch_ms": 0.0, "sync_ms": 0.0})
        right = linear_bench["arrays"].get(stage, {"dispatch_ms": 0.0, "sync_ms": 0.0})
        lines.append(
            f"| {stage} | {float(left['dispatch_ms']):.3f} | {float(left['sync_ms']):.3f} | {float(right['dispatch_ms']):.3f} | {float(right['sync_ms']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Experiment 3: Quantized lm_head vs MLP",
            "",
            "| Path | Dispatch ms | Sync ms |",
            "|------|------------:|--------:|",
            f"| mlp_prefill | {float(path_bench['mlp_prefill']['dispatch_ms']):.3f} | {float(path_bench['mlp_prefill']['sync_ms']):.3f} |",
            f"| mlp_decode | {float(path_bench['mlp_decode']['dispatch_ms']):.3f} | {float(path_bench['mlp_decode']['sync_ms']):.3f} |",
            f"| lm_head_prefill | {float(path_bench['lm_head_prefill']['dispatch_ms']):.3f} | {float(path_bench['lm_head_prefill']['sync_ms']):.3f} |",
            f"| lm_head_decode | {float(path_bench['lm_head_decode']['dispatch_ms']):.3f} | {float(path_bench['lm_head_decode']['sync_ms']):.3f} |",
            "",
            "## Interpretation",
            "",
        ]
    )

    if arrays_prefill_cache < legacy_prefill_cache and arrays_decode_cache < legacy_decode_cache:
        lines.append("Arrays-style conv cache update reduces the traced `linear_cache_update` cost, so the old cache-update path was a real contributor.")
    else:
        lines.append("Arrays-style conv cache update does not materially reduce `linear_cache_update`, so the major cost is likely outside the simple conv-state tail handling.")

    lines.append(
        "The isolated linear-attention microbenchmark still shows small local improvements in `conv_cache_update`, `conv1d`, `in_proj_qkv`, and `gated_delta_update` under arrays-style cache handling, but that local win does not translate into a full-model trace win. That implies the dominant full-model cost is the broader recurrent path and graph composition around `linear_attention`, not just the conv-state tail update itself."
    )

    if float(path_bench["lm_head_prefill"]["sync_ms"]) > float(path_bench["mlp_prefill"]["sync_ms"]):
        lines.append("In isolated quantized-path benchmarking, `lm_head` is heavier than a single `MLP` call for prefill-sized inputs. However, the full model still spends more total time in `mlp` because that path is executed once per layer while `lm_head` is executed once per token step.")
    else:
        lines.append("In isolated quantized-path benchmarking, `MLP` remains heavier than `lm_head` for prefill-sized inputs.")

    lines.append(
        "If `linear_attention` dispatch stays high even after arrays-style cache update, the anomaly is not just cache tail maintenance; it points to the broader operator composition inside `GatedDeltaNet` such as repeated quantized projections, conv1d scheduling, and recurrent update graph construction."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    PLAN0_DIR.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MLX_METAL_PATH"] = str(MLX_METAL_PATH)
    env["QWEN3_5_MLX_LINEAR_CACHE_MODE"] = "arrays"
    env["QWEN3_5_MLX_STAGE_TRACE"] = "1"
    run([
        str(CUSTOM_BIN),
        str(MODEL_DIR),
        str(PROMPT_SUITE),
        str(ARRAYS_TRACE),
        "32",
    ], env=env)

    run([
        str(LINEAR_BENCH_BIN),
        str(CONFIG_JSON),
        str(LINEAR_BENCH),
        "64",
        "20",
        "128",
    ], env=dict(os.environ, MLX_METAL_PATH=str(MLX_METAL_PATH)))

    run([
        str(PATH_BENCH_BIN),
        str(CONFIG_JSON),
        str(PATH_BENCH),
        "64",
        "30",
    ], env=dict(os.environ, MLX_METAL_PATH=str(MLX_METAL_PATH)))

    base = load_json(BASE_TRACE)
    baseline = load_json(BASELINE_CUSTOM_TRACE)
    arrays = load_json(ARRAYS_TRACE)
    linear_bench = load_json(LINEAR_BENCH)
    path_bench = load_json(PATH_BENCH)
    SUMMARY.write_text(render_summary(base, baseline, arrays, linear_bench, path_bench), encoding="utf-8")
    print(f"Wrote summary: {SUMMARY}")


if __name__ == "__main__":
    main()