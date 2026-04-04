#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu")
PLAN0_DIR = ROOT / "test/optimization/plan0/gated_delta_kernel_followup"
PROMPT_SUITE = ROOT / "test/optimization/plan_0/operator_hook_test/prompt_suite.json"
BASE_TRACE = ROOT / "test/optimization/plan_0/operator_hook_test/base_stage_trace.json"
BASELINE_CUSTOM_TRACE = ROOT / "test/optimization/plan_0/operator_hook_test/custom_stage_trace.json"
COMPILED_TRACE = PLAN0_DIR / "compiled_ops_stage_trace.json"
CUSTOM_BENCH = PLAN0_DIR / "custom_gated_delta_microbench.json"
UPSTREAM_BENCH = PLAN0_DIR / "upstream_gated_delta_kernel_bench.json"
SUMMARY = PLAN0_DIR / "summary.md"

CUSTOM_BIN = ROOT / "build/test_mlx_quantized_output_eval"
CUSTOM_BENCH_BIN = ROOT / "build/test_mlx_gated_delta_microbench"
MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
CONFIG_JSON = MODEL_DIR / "config.json"
MLX_METAL_PATH = ROOT / "../../.repo_cache/mlx/build/mlx/backend/metal/kernels"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def avg_stage(trace: dict[str, Any], phase: str, stage: str, field: str) -> float:
    rows = trace["rows"]
    return sum(float(row["stage_trace"][phase].get(stage, {}).get(field, 0.0)) for row in rows) / len(rows)


def pct_change(old: float, new: float) -> float:
    if old == 0.0:
        return 0.0
    return ((new - old) / old) * 100.0


def bench_upstream_gated_delta(output_path: Path, prefill_tokens: int = 64, iterations: int = 100) -> None:
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_update

    config = load_json(CONFIG_JSON)
    text = config["text_config"]

    batch = 1
    key_heads = int(text["linear_num_key_heads"])
    value_heads = int(text["linear_num_value_heads"])
    key_dim = int(text["linear_key_head_dim"])
    value_dim = int(text["linear_value_head_dim"])

    A_log = mx.random.normal((value_heads,), dtype=mx.float32)
    dt_bias = mx.random.normal((value_heads,), dtype=mx.float32)

    def bench_case(steps: int, use_kernel: bool) -> dict[str, float]:
        q = mx.random.normal((batch, steps, key_heads, key_dim), dtype=mx.float32)
        k = mx.random.normal((batch, steps, key_heads, key_dim), dtype=mx.float32)
        v = mx.random.normal((batch, steps, value_heads, value_dim), dtype=mx.float32)
        a = mx.random.normal((batch, steps, value_heads), dtype=mx.float32)
        b = mx.random.normal((batch, steps, value_heads), dtype=mx.float32)

        for _ in range(3):
            out, state = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=use_kernel)
            mx.eval(out, state)
            mx.synchronize()

        dispatch_ms = 0.0
        sync_ms = 0.0
        for _ in range(iterations):
            t0 = time.perf_counter()
            out, state = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=use_kernel)
            dispatch_ms += (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            mx.eval(out, state)
            mx.synchronize()
            sync_ms += (time.perf_counter() - t1) * 1000.0

        return {
            "dispatch_ms": dispatch_ms / iterations,
            "sync_ms": sync_ms / iterations,
        }

    result = {
        "prefill_tokens": prefill_tokens,
        "iterations": iterations,
        "prefill": {
            "ops": bench_case(prefill_tokens, use_kernel=False),
            "kernel": bench_case(prefill_tokens, use_kernel=True),
        },
        "decode": {
            "ops": bench_case(1, use_kernel=False),
            "kernel": bench_case(1, use_kernel=True),
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def render_summary(
    base: dict[str, Any],
    baseline: dict[str, Any],
    compiled: dict[str, Any],
    custom_bench: dict[str, Any],
    upstream_bench: dict[str, Any],
) -> str:
    base_s = summary_block(base)
    baseline_s = summary_block(baseline)
    compiled_s = summary_block(compiled)

    baseline_linear_dispatch_prefill = avg_stage(baseline, "prefill", "linear_attention", "dispatch_ms")
    compiled_linear_dispatch_prefill = avg_stage(compiled, "prefill", "linear_attention", "dispatch_ms")
    baseline_linear_dispatch_decode = avg_stage(baseline, "decode", "linear_attention", "dispatch_ms")
    compiled_linear_dispatch_decode = avg_stage(compiled, "decode", "linear_attention", "dispatch_ms")
    baseline_linear_sync_prefill = avg_stage(baseline, "prefill", "linear_attention", "sync_ms")
    compiled_linear_sync_prefill = avg_stage(compiled, "prefill", "linear_attention", "sync_ms")
    baseline_linear_sync_decode = avg_stage(baseline, "decode", "linear_attention", "sync_ms")
    compiled_linear_sync_decode = avg_stage(compiled, "decode", "linear_attention", "sync_ms")

    custom_prefill_ops = float(custom_bench["prefill"]["ops"]["sync_ms"])
    custom_prefill_compiled = float(custom_bench["prefill"]["compiled_ops"]["sync_ms"])
    custom_decode_ops = float(custom_bench["decode"]["ops"]["sync_ms"])
    custom_decode_compiled = float(custom_bench["decode"]["compiled_ops"]["sync_ms"])

    upstream_prefill_ops = float(upstream_bench["prefill"]["ops"]["sync_ms"])
    upstream_prefill_kernel = float(upstream_bench["prefill"]["kernel"]["sync_ms"])
    upstream_decode_ops = float(upstream_bench["decode"]["ops"]["sync_ms"])
    upstream_decode_kernel = float(upstream_bench["decode"]["kernel"]["sync_ms"])

    lines = [
        "# Gated Delta Kernel Follow-Up",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Base trace: {BASE_TRACE}",
        f"- Baseline custom trace: {BASELINE_CUSTOM_TRACE}",
        f"- Compiled-ops custom trace: {COMPILED_TRACE}",
        f"- Custom gated delta microbench: {CUSTOM_BENCH}",
        f"- Upstream gated delta kernel bench: {UPSTREAM_BENCH}",
        "",
        "## Experiment 1: Full Model With Compiled Gated Delta Ops",
        "",
        "| Backend | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |",
        "|--------|---------------:|------------------:|------------:|----------:|",
        f"| base | {base_s['avg_prefill_ms']:.3f} | {base_s['avg_decode_ms']:.3f} | {base_s['avg_wall_ms']:.3f} | {base_s['avg_throughput']:.3f} |",
        f"| custom ops | {baseline_s['avg_prefill_ms']:.3f} | {baseline_s['avg_decode_ms']:.3f} | {baseline_s['avg_wall_ms']:.3f} | {baseline_s['avg_throughput']:.3f} |",
        f"| custom compiled_ops | {compiled_s['avg_prefill_ms']:.3f} | {compiled_s['avg_decode_ms']:.3f} | {compiled_s['avg_wall_ms']:.3f} | {compiled_s['avg_throughput']:.3f} |",
        "",
        f"- Prefill linear_attention dispatch: ops {baseline_linear_dispatch_prefill:.3f} ms -> compiled_ops {compiled_linear_dispatch_prefill:.3f} ms ({pct_change(baseline_linear_dispatch_prefill, compiled_linear_dispatch_prefill):.1f}%).",
        f"- Decode linear_attention dispatch: ops {baseline_linear_dispatch_decode:.3f} ms -> compiled_ops {compiled_linear_dispatch_decode:.3f} ms ({pct_change(baseline_linear_dispatch_decode, compiled_linear_dispatch_decode):.1f}%).",
        f"- Prefill linear_attention sync: ops {baseline_linear_sync_prefill:.3f} ms -> compiled_ops {compiled_linear_sync_prefill:.3f} ms ({pct_change(baseline_linear_sync_prefill, compiled_linear_sync_prefill):.1f}%).",
        f"- Decode linear_attention sync: ops {baseline_linear_sync_decode:.3f} ms -> compiled_ops {compiled_linear_sync_decode:.3f} ms ({pct_change(baseline_linear_sync_decode, compiled_linear_sync_decode):.1f}%).",
        "",
        "## Experiment 2: Custom C++ Gated Delta Microbenchmark",
        "",
        "| Shape | Ops sync ms | Compiled ops sync ms | Delta |",
        "|------|------------:|---------------------:|------:|",
        f"| prefill | {custom_prefill_ops:.3f} | {custom_prefill_compiled:.3f} | {pct_change(custom_prefill_ops, custom_prefill_compiled):.1f}% |",
        f"| decode | {custom_decode_ops:.3f} | {custom_decode_compiled:.3f} | {pct_change(custom_decode_ops, custom_decode_compiled):.1f}% |",
        "",
        "## Experiment 3: Upstream Python Gated Delta With And Without Kernel",
        "",
        "| Shape | Upstream ops sync ms | Upstream kernel sync ms | Delta |",
        "|------|---------------------:|------------------------:|------:|",
        f"| prefill | {upstream_prefill_ops:.3f} | {upstream_prefill_kernel:.3f} | {pct_change(upstream_prefill_ops, upstream_prefill_kernel):.1f}% |",
        f"| decode | {upstream_decode_ops:.3f} | {upstream_decode_kernel:.3f} | {pct_change(upstream_decode_ops, upstream_decode_kernel):.1f}% |",
        "",
        "## Interpretation",
        "",
    ]

    lines.append(
        "Custom C++ `compiled_ops` directly tests the part we were missing from upstream `@mx.compile` usage. If this reduces isolated gated-delta cost but does not close the full-model gap, then compile alone is only a partial explanation."
    )
    lines.append(
        "The upstream Python benchmark separates `use_kernel=False` and `use_kernel=True`. If the kernel path is much faster than upstream ops, that means the missing Metal-kernel recurrent update is the stronger root-cause candidate than cache-tail handling."
    )

    if abs(pct_change(upstream_prefill_ops, upstream_prefill_kernel)) > abs(pct_change(custom_prefill_ops, custom_prefill_compiled)):
        lines.append(
            "The upstream kernel delta is larger than the custom compiled-ops delta on prefill-sized shapes. That points to the absent kernelized recurrent update as the bigger remaining gap after adding compile-level optimization."
        )
    else:
        lines.append(
            "Compile-level optimization is already large enough that it may explain most of the recurrent-path gap by itself."
        )

    if compiled_s["avg_prefill_ms"] < baseline_s["avg_prefill_ms"] or compiled_s["avg_decode_ms"] < baseline_s["avg_decode_ms"]:
        lines.append(
            "Full-model trace improvement confirms that `gated_delta` implementation details do affect end-to-end runtime, not just the isolated microbenchmark."
        )
    else:
        lines.append(
            "Full-model trace does not materially improve even after compiled ops, which means the remaining problem is likely the still-missing fused kernel path or surrounding linear-attention graph composition."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    PLAN0_DIR.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MLX_METAL_PATH"] = str(MLX_METAL_PATH)
    env["QWEN3_5_MLX_GATED_DELTA_MODE"] = "compiled_ops"
    env["QWEN3_5_MLX_STAGE_TRACE"] = "1"
    run(
        [
            str(CUSTOM_BIN),
            str(MODEL_DIR),
            str(PROMPT_SUITE),
            str(COMPILED_TRACE),
            "32",
        ],
        env=env,
    )

    run(
        [
            str(CUSTOM_BENCH_BIN),
            str(CONFIG_JSON),
            str(CUSTOM_BENCH),
            "64",
            "100",
        ],
        env=dict(os.environ, MLX_METAL_PATH=str(MLX_METAL_PATH)),
    )

    bench_upstream_gated_delta(UPSTREAM_BENCH, prefill_tokens=64, iterations=100)

    base = load_json(BASE_TRACE)
    baseline = load_json(BASELINE_CUSTOM_TRACE)
    compiled = load_json(COMPILED_TRACE)
    custom_bench = load_json(CUSTOM_BENCH)
    upstream_bench = load_json(UPSTREAM_BENCH)

    SUMMARY.write_text(
        render_summary(base, baseline, compiled, custom_bench, upstream_bench),
        encoding="utf-8",
    )
    print(f"Wrote summary: {SUMMARY}")


if __name__ == "__main__":
    main()