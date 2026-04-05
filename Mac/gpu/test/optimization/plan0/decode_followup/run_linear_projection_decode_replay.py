#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import mlx.core as mx
from mlx_vlm import load as mlx_load
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.models.qwen3_5 import language as qwen_language

WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Mac.gpu.test.gen_reference_qwen3_5_4b_mlx_8bit import build_nothink_chatml_tokens, load_prompt_suite


MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/prompt_suite.json")
OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/linear_projection_decode_replay.json")
OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/linear_projection_decode_replay.md")
MAX_NEW_TOKENS = 32
SELECTED_DECODE_STEPS = (0, 15, 30)
WARMUP = 2
ITERATIONS = 8


@dataclass
class CaptureSample:
    prompt_id: str
    decode_step: int
    layer_index: int
    input_tensor: mx.array


class ReplayCollector:
    def __init__(self, selected_steps: tuple[int, ...]) -> None:
        self.selected_steps = set(selected_steps)
        self.prompt_id = ""
        self.phase = "prefill"
        self.decode_step = -1
        self.samples: list[CaptureSample] = []

    def maybe_capture(self, layer_index: int, tensor: mx.array) -> None:
        if self.phase != "decode" or self.decode_step not in self.selected_steps:
            return
        mx.eval(tensor)
        mx.synchronize()
        self.samples.append(
            CaptureSample(
                prompt_id=self.prompt_id,
                decode_step=self.decode_step,
                layer_index=layer_index,
                input_tensor=tensor,
            )
        )


def bench(fn: Callable[[], mx.array], warmup: int = WARMUP, iterations: int = ITERATIONS) -> dict[str, float]:
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


def install_capture_hook(model: Any, collector: ReplayCollector) -> Callable[[], None]:
    for layer_index, layer in enumerate(model.language_model.model.layers):
        setattr(layer, "_capture_layer_index", layer_index)

    original = qwen_language.Qwen3_5DecoderLayer.__call__

    def wrapped(
        self: Any,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        normed = self.input_layernorm(x)
        if self.is_linear:
            collector.maybe_capture(int(self._capture_layer_index), normed)
            residual = self.linear_attn(normed, mask, cache)
        else:
            residual = self.self_attn(normed, mask, cache, position_ids)
        hidden = x + residual
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    qwen_language.Qwen3_5DecoderLayer.__call__ = wrapped

    def restore() -> None:
        qwen_language.Qwen3_5DecoderLayer.__call__ = original

    return restore


def collect_samples(model: Any, tokenizer: Any) -> list[CaptureSample]:
    collector = ReplayCollector(SELECTED_DECODE_STEPS)
    restore = install_capture_hook(model, collector)
    try:
        for prompt_row in load_prompt_suite(PROMPT_SUITE, None):
            collector.prompt_id = prompt_row.id
            collector.phase = "prefill"
            collector.decode_step = -1

            model.language_model._position_ids = None
            model.language_model._rope_deltas = None

            prompt_tokens = build_nothink_chatml_tokens(tokenizer, prompt_row.prompt_text, None)
            prompt_cache = vlm_cache.make_prompt_cache(model.language_model)
            input_ids = mx.array([prompt_tokens], dtype=mx.int32)

            outputs = model.language_model(input_ids, cache=prompt_cache)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            mx.eval(logits)
            mx.synchronize()
            token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

            generated_tokens = [token]
            for decode_step in range(MAX_NEW_TOKENS - 1):
                collector.phase = "decode"
                collector.decode_step = decode_step
                decode_input = mx.array([[token]], dtype=mx.int32)
                outputs = model.language_model(decode_input, cache=prompt_cache)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                mx.eval(logits)
                mx.synchronize()
                token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                generated_tokens.append(token)
    finally:
        restore()
    return collector.samples


def max_abs_diff(lhs: mx.array, rhs: mx.array) -> float:
    diff = mx.max(mx.abs(lhs - rhs))
    mx.eval(diff)
    mx.synchronize()
    return float(diff.item())


def bench_projection_sample(sample: CaptureSample, model: Any) -> dict[str, Any]:
    layer = model.language_model.model.layers[sample.layer_index].linear_attn
    inputs = sample.input_tensor
    batch, seq_len, _hidden = inputs.shape

    qkv_module = layer.in_proj_qkv
    z_module = layer.in_proj_z

    qkv_dense_weight = mx.dequantize(
        qkv_module.weight,
        qkv_module.scales,
        qkv_module.biases,
        qkv_module.group_size,
        qkv_module.bits,
        qkv_module.mode,
    )
    qkv_dense_weight_t = mx.transpose(qkv_dense_weight)

    z_dense_weight = mx.dequantize(
        z_module.weight,
        z_module.scales,
        z_module.biases,
        z_module.group_size,
        z_module.bits,
        z_module.mode,
    )
    z_dense_weight_t = mx.transpose(z_dense_weight)

    qkv_base = qkv_module(inputs)
    qkv_manual = mx.quantized_matmul(
        inputs,
        qkv_module.weight,
        qkv_module.scales,
        qkv_module.biases,
        True,
        qkv_module.group_size,
        qkv_module.bits,
        qkv_module.mode,
    )
    z_base = z_module(inputs)
    z_manual = mx.quantized_matmul(
        inputs,
        z_module.weight,
        z_module.scales,
        z_module.biases,
        True,
        z_module.group_size,
        z_module.bits,
        z_module.mode,
    )
    z_manual_reshaped = z_manual.reshape(batch, seq_len, -1, layer.head_v_dim)

    mx.eval(qkv_base, qkv_manual, z_base, z_manual_reshaped)
    mx.synchronize()

    return {
        "prompt_id": sample.prompt_id,
        "decode_step": sample.decode_step,
        "layer_index": sample.layer_index,
        "qkv": {
            "base_module": bench(lambda: qkv_module(inputs)),
            "manual_quantized_matmul": bench(
                lambda: mx.quantized_matmul(
                    inputs,
                    qkv_module.weight,
                    qkv_module.scales,
                    qkv_module.biases,
                    True,
                    qkv_module.group_size,
                    qkv_module.bits,
                    qkv_module.mode,
                )
            ),
            "dequantize_only": bench(
                lambda: mx.dequantize(
                    qkv_module.weight,
                    qkv_module.scales,
                    qkv_module.biases,
                    qkv_module.group_size,
                    qkv_module.bits,
                    qkv_module.mode,
                )
            ),
            "dense_matmul_predequantized": bench(lambda: mx.matmul(inputs, qkv_dense_weight_t)),
            "dequantize_plus_dense_matmul": bench(
                lambda: mx.matmul(
                    inputs,
                    mx.transpose(
                        mx.dequantize(
                            qkv_module.weight,
                            qkv_module.scales,
                            qkv_module.biases,
                            qkv_module.group_size,
                            qkv_module.bits,
                            qkv_module.mode,
                        )
                    ),
                )
            ),
            "base_vs_manual_max_abs_diff": round(max_abs_diff(qkv_base, qkv_manual), 6),
        },
        "z": {
            "base_module": bench(lambda: z_module(inputs)),
            "manual_quantized_matmul": bench(
                lambda: mx.quantized_matmul(
                    inputs,
                    z_module.weight,
                    z_module.scales,
                    z_module.biases,
                    True,
                    z_module.group_size,
                    z_module.bits,
                    z_module.mode,
                )
            ),
            "manual_quantized_matmul_plus_reshape": bench(
                lambda: mx.quantized_matmul(
                    inputs,
                    z_module.weight,
                    z_module.scales,
                    z_module.biases,
                    True,
                    z_module.group_size,
                    z_module.bits,
                    z_module.mode,
                ).reshape(batch, seq_len, -1, layer.head_v_dim)
            ),
            "reshape_only": bench(lambda: z_manual.reshape(batch, seq_len, -1, layer.head_v_dim)),
            "dequantize_only": bench(
                lambda: mx.dequantize(
                    z_module.weight,
                    z_module.scales,
                    z_module.biases,
                    z_module.group_size,
                    z_module.bits,
                    z_module.mode,
                )
            ),
            "dense_matmul_predequantized": bench(lambda: mx.matmul(inputs, z_dense_weight_t)),
            "dequantize_plus_dense_matmul": bench(
                lambda: mx.matmul(
                    inputs,
                    mx.transpose(
                        mx.dequantize(
                            z_module.weight,
                            z_module.scales,
                            z_module.biases,
                            z_module.group_size,
                            z_module.bits,
                            z_module.mode,
                        )
                    ),
                )
            ),
            "base_vs_manual_max_abs_diff": round(max_abs_diff(z_base, z_manual), 6),
            "base_vs_manual_reshape_max_abs_diff": round(
                max_abs_diff(z_base.reshape(batch, seq_len, -1, layer.head_v_dim), z_manual_reshaped),
                6,
            ),
        },
    }


def summarize_results(rows: list[dict[str, Any]], op: str, variant: str) -> float:
    return round(mean(float(row[op][variant]["sync_ms"]) for row in rows), 3)


def summarize_by_layer(rows: list[dict[str, Any]], op: str, variant: str) -> list[dict[str, Any]]:
    layer_ids = sorted({int(row["layer_index"]) for row in rows})
    result = []
    for layer_index in layer_ids:
        layer_rows = [row for row in rows if int(row["layer_index"]) == layer_index]
        result.append(
            {
                "layer_index": layer_index,
                "sync_ms": round(mean(float(row[op][variant]["sync_ms"]) for row in layer_rows), 3),
            }
        )
    return result


def main() -> None:
    model, processor = mlx_load(str(MODEL_DIR))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    samples = collect_samples(model, tokenizer)
    print(f"captured {len(samples)} live decode replay samples")

    rows = []
    for sample_index, sample in enumerate(samples, start=1):
        rows.append(bench_projection_sample(sample, model))
        if sample_index % 24 == 0:
            print(f"benchmarked {sample_index}/{len(samples)} samples")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(MODEL_DIR),
        "prompt_suite": str(PROMPT_SUITE),
        "selected_decode_steps": list(SELECTED_DECODE_STEPS),
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "sample_count": len(rows),
        "rows": rows,
        "summary": {
            "qkv": {
                "base_module_sync_ms": summarize_results(rows, "qkv", "base_module"),
                "manual_quantized_matmul_sync_ms": summarize_results(rows, "qkv", "manual_quantized_matmul"),
                "dequantize_only_sync_ms": summarize_results(rows, "qkv", "dequantize_only"),
                "dense_matmul_predequantized_sync_ms": summarize_results(rows, "qkv", "dense_matmul_predequantized"),
                "dequantize_plus_dense_matmul_sync_ms": summarize_results(rows, "qkv", "dequantize_plus_dense_matmul"),
                "avg_base_vs_manual_max_abs_diff": round(mean(float(row["qkv"]["base_vs_manual_max_abs_diff"]) for row in rows), 6),
            },
            "z": {
                "base_module_sync_ms": summarize_results(rows, "z", "base_module"),
                "manual_quantized_matmul_sync_ms": summarize_results(rows, "z", "manual_quantized_matmul"),
                "manual_quantized_matmul_plus_reshape_sync_ms": summarize_results(rows, "z", "manual_quantized_matmul_plus_reshape"),
                "reshape_only_sync_ms": summarize_results(rows, "z", "reshape_only"),
                "dequantize_only_sync_ms": summarize_results(rows, "z", "dequantize_only"),
                "dense_matmul_predequantized_sync_ms": summarize_results(rows, "z", "dense_matmul_predequantized"),
                "dequantize_plus_dense_matmul_sync_ms": summarize_results(rows, "z", "dequantize_plus_dense_matmul"),
                "avg_base_vs_manual_max_abs_diff": round(mean(float(row["z"]["base_vs_manual_max_abs_diff"]) for row in rows), 6),
                "avg_base_vs_manual_reshape_max_abs_diff": round(mean(float(row["z"]["base_vs_manual_reshape_max_abs_diff"]) for row in rows), 6),
            },
            "per_layer": {
                "qkv_base_module": summarize_by_layer(rows, "qkv", "base_module"),
                "qkv_manual_quantized_matmul": summarize_by_layer(rows, "qkv", "manual_quantized_matmul"),
                "z_base_module": summarize_by_layer(rows, "z", "base_module"),
                "z_manual_quantized_matmul": summarize_by_layer(rows, "z", "manual_quantized_matmul"),
            },
        },
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    qkv = report["summary"]["qkv"]
    z = report["summary"]["z"]
    lines = [
        "# Linear Projection Decode Replay",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Model dir: {MODEL_DIR}",
        f"- Prompt suite: {PROMPT_SUITE}",
        f"- Selected decode steps: {list(SELECTED_DECODE_STEPS)}",
        f"- Samples: {len(rows)}",
        "",
        "## in_proj_qkv",
        "",
        f"- base module sync: {qkv['base_module_sync_ms']:.3f} ms",
        f"- manual quantized_matmul sync: {qkv['manual_quantized_matmul_sync_ms']:.3f} ms",
        f"- dequantize only sync: {qkv['dequantize_only_sync_ms']:.3f} ms",
        f"- dense matmul on predequantized weight sync: {qkv['dense_matmul_predequantized_sync_ms']:.3f} ms",
        f"- dequantize + dense matmul sync: {qkv['dequantize_plus_dense_matmul_sync_ms']:.3f} ms",
        f"- avg base vs manual max abs diff: {qkv['avg_base_vs_manual_max_abs_diff']:.6f}",
        "",
        "## in_proj_z",
        "",
        f"- base module sync: {z['base_module_sync_ms']:.3f} ms",
        f"- manual quantized_matmul sync: {z['manual_quantized_matmul_sync_ms']:.3f} ms",
        f"- manual quantized_matmul + reshape sync: {z['manual_quantized_matmul_plus_reshape_sync_ms']:.3f} ms",
        f"- reshape only sync: {z['reshape_only_sync_ms']:.3f} ms",
        f"- dequantize only sync: {z['dequantize_only_sync_ms']:.3f} ms",
        f"- dense matmul on predequantized weight sync: {z['dense_matmul_predequantized_sync_ms']:.3f} ms",
        f"- dequantize + dense matmul sync: {z['dequantize_plus_dense_matmul_sync_ms']:.3f} ms",
        f"- avg base vs manual max abs diff: {z['avg_base_vs_manual_max_abs_diff']:.6f}",
        f"- avg base vs manual reshape max abs diff: {z['avg_base_vs_manual_reshape_max_abs_diff']:.6f}",
        "",
        "## Findings",
        "",
        f"- live decode context replay에서 official `QuantizedLinear`와 manual `mx.quantized_matmul`는 `in_proj_qkv` 기준 {qkv['base_module_sync_ms']:.3f} vs {qkv['manual_quantized_matmul_sync_ms']:.3f} ms, `in_proj_z` 기준 {z['base_module_sync_ms']:.3f} vs {z['manual_quantized_matmul_sync_ms']:.3f} ms였다.",
        f"- `in_proj_z`의 reshape only cost는 {z['reshape_only_sync_ms']:.3f} ms라서 projection 본체에 비해 매우 작다.",
        f"- `dequantize + dense matmul`은 `in_proj_qkv`에서 {qkv['dequantize_plus_dense_matmul_sync_ms']:.3f} ms, `in_proj_z`에서 {z['dequantize_plus_dense_matmul_sync_ms']:.3f} ms로 manual quantized path보다 훨씬 크거나 같다면, delta는 dequantize boundary나 reshape보다 full-model graph context 쪽일 가능성이 높다.",
        "- 이 문서는 live cache context에서 나온 실제 decode activation을 기준으로 official base 모듈과 custom manual quantized path를 1:1 replay하기 위한 산출물이다.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()