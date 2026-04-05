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
from mlx_lm.models.cache import ArraysCache
from mlx_vlm import load as mlx_load
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.models.qwen3_5 import language as qwen_language

WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Mac.gpu.test.gen_reference_qwen3_5_4b_mlx_8bit import build_nothink_chatml_tokens, load_prompt_suite


MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/prompt_suite.json")
OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/linear_subgraph_decode_replay.json")
OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/linear_subgraph_decode_replay.md")
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
    mask: mx.array | None
    conv_state: mx.array | None
    rec_state: mx.array | None


class ReplayCollector:
    def __init__(self, selected_steps: tuple[int, ...]) -> None:
        self.selected_steps = set(selected_steps)
        self.prompt_id = ""
        self.phase = "prefill"
        self.decode_step = -1
        self.samples: list[CaptureSample] = []

    def maybe_capture(
        self,
        layer_index: int,
        tensor: mx.array,
        mask: mx.array | None,
        cache: Any | None,
    ) -> None:
        if self.phase != "decode" or self.decode_step not in self.selected_steps:
            return

        conv_state = cache[0] if cache is not None else None
        rec_state = cache[1] if cache is not None else None

        arrays = [tensor]
        if mask is not None:
            arrays.append(mask)
        if conv_state is not None:
            arrays.append(conv_state)
        if rec_state is not None:
            arrays.append(rec_state)
        mx.eval(arrays)
        mx.synchronize()

        self.samples.append(
            CaptureSample(
                prompt_id=self.prompt_id,
                decode_step=self.decode_step,
                layer_index=layer_index,
                input_tensor=tensor,
                mask=mask,
                conv_state=conv_state,
                rec_state=rec_state,
            )
        )


def collect_arrays(value: Any) -> list[mx.array]:
    arrays: list[mx.array] = []
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, tuple):
        for item in value:
            arrays.extend(collect_arrays(item))
    elif isinstance(value, list):
        for item in value:
            arrays.extend(collect_arrays(item))
    elif isinstance(value, dict):
        for item in value.values():
            arrays.extend(collect_arrays(item))
    return arrays


def bench(fn: Callable[[], Any], warmup: int = WARMUP, iterations: int = ITERATIONS) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        arrays = collect_arrays(out)
        if arrays:
            mx.eval(arrays)
            mx.synchronize()

    dispatch_times: list[float] = []
    sync_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out = fn()
        dispatch_times.append((time.perf_counter() - t0) * 1000.0)

        arrays = collect_arrays(out)
        t1 = time.perf_counter()
        if arrays:
            mx.eval(arrays)
            mx.synchronize()
        sync_times.append((time.perf_counter() - t1) * 1000.0)

    return {
        "dispatch_ms": round(mean(dispatch_times), 3),
        "sync_ms": round(mean(sync_times), 3),
    }


def max_abs_diff(lhs: mx.array, rhs: mx.array) -> float:
    diff = mx.max(mx.abs(lhs - rhs))
    mx.eval(diff)
    mx.synchronize()
    return float(diff.item())


def quantized_linear_manual(module: Any, inputs: mx.array) -> mx.array:
    return mx.quantized_matmul(
        inputs,
        module.weight,
        module.scales,
        module.biases,
        True,
        module.group_size,
        module.bits,
        module.mode,
    )


def make_linear_cache(sample: CaptureSample) -> ArraysCache:
    cache = ArraysCache(size=2)
    cache[0] = mx.array(sample.conv_state) if sample.conv_state is not None else None
    cache[1] = mx.array(sample.rec_state) if sample.rec_state is not None else None
    return cache


def projection_bundle_official(layer: Any, inputs: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    batch, seq_len, _hidden = inputs.shape
    mixed_qkv = layer.in_proj_qkv(inputs)
    z = layer.in_proj_z(inputs).reshape(batch, seq_len, -1, layer.head_v_dim)
    a = layer.in_proj_a(inputs)
    b = layer.in_proj_b(inputs)
    return mixed_qkv, z, a, b


def projection_bundle_manual(layer: Any, inputs: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    batch, seq_len, _hidden = inputs.shape
    mixed_qkv = quantized_linear_manual(layer.in_proj_qkv, inputs)
    z = quantized_linear_manual(layer.in_proj_z, inputs).reshape(batch, seq_len, -1, layer.head_v_dim)
    a = quantized_linear_manual(layer.in_proj_a, inputs)
    b = quantized_linear_manual(layer.in_proj_b, inputs)
    return mixed_qkv, z, a, b


def linear_block_manual_projection(
    layer: Any,
    inputs: mx.array,
    mask: mx.array | None,
    cache: ArraysCache | None,
) -> mx.array:
    batch, seq_len, _hidden = inputs.shape
    mixed_qkv, z, a, b = projection_bundle_manual(layer, inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
        if conv_state.shape[0] != batch:
            conv_state = mx.zeros((batch, layer.conv_kernel_size - 1, layer.conv_dim), dtype=inputs.dtype)
    else:
        conv_state = mx.zeros((batch, layer.conv_kernel_size - 1, layer.conv_dim), dtype=inputs.dtype)

    linear_mask = mask
    if linear_mask is not None:
        if linear_mask.shape[0] != batch:
            linear_mask = None
        else:
            mixed_qkv = mx.where(linear_mask[..., None], mixed_qkv, 0)

    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    if cache is not None:
        cache[0] = conv_input[:, -(layer.conv_kernel_size - 1) :]

    conv_out = qwen_language.nn.silu(layer.conv1d(conv_input))

    q, k, v = [
        t.reshape(batch, seq_len, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [layer.key_dim, 2 * layer.key_dim], -1),
            [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
            [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    if state is not None and state.shape[0] != batch:
        state = None

    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = qwen_language.gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        state,
        linear_mask,
        use_kernel=not layer.training,
    )

    if cache is not None:
        cache[1] = state

    out = layer.norm(out, z)
    return layer.out_proj(out.reshape(batch, seq_len, -1))


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
            collector.maybe_capture(int(self._capture_layer_index), normed, mask, cache)
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

            for decode_step in range(MAX_NEW_TOKENS - 1):
                collector.phase = "decode"
                collector.decode_step = decode_step
                decode_input = mx.array([[token]], dtype=mx.int32)
                outputs = model.language_model(decode_input, cache=prompt_cache)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                mx.eval(logits)
                mx.synchronize()
                token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    finally:
        restore()
    return collector.samples


def bench_sample(sample: CaptureSample, model: Any) -> dict[str, Any]:
    layer = model.language_model.model.layers[sample.layer_index].linear_attn
    inputs = sample.input_tensor
    mask = sample.mask

    official_projection = projection_bundle_official(layer, inputs)
    manual_projection = projection_bundle_manual(layer, inputs)
    official_linear_block = layer(inputs, mask, make_linear_cache(sample))
    manual_linear_block = linear_block_manual_projection(layer, inputs, mask, make_linear_cache(sample))

    mx.eval(
        official_projection[0],
        official_projection[1],
        official_projection[2],
        official_projection[3],
        manual_projection[0],
        manual_projection[1],
        manual_projection[2],
        manual_projection[3],
        official_linear_block,
        manual_linear_block,
    )
    mx.synchronize()

    return {
        "prompt_id": sample.prompt_id,
        "decode_step": sample.decode_step,
        "layer_index": sample.layer_index,
        "projection_bundle": {
            "official": bench(lambda: projection_bundle_official(layer, inputs)),
            "manual_quantized": bench(lambda: projection_bundle_manual(layer, inputs)),
            "qkv_max_abs_diff": round(max_abs_diff(official_projection[0], manual_projection[0]), 6),
            "z_max_abs_diff": round(max_abs_diff(official_projection[1], manual_projection[1]), 6),
            "a_max_abs_diff": round(max_abs_diff(official_projection[2], manual_projection[2]), 6),
            "b_max_abs_diff": round(max_abs_diff(official_projection[3], manual_projection[3]), 6),
        },
        "linear_block": {
            "official": bench(lambda: layer(inputs, mask, make_linear_cache(sample))),
            "manual_projection": bench(
                lambda: linear_block_manual_projection(layer, inputs, mask, make_linear_cache(sample))
            ),
            "output_max_abs_diff": round(max_abs_diff(official_linear_block, manual_linear_block), 6),
        },
    }


def summarize_variant(rows: list[dict[str, Any]], section: str, variant: str) -> float:
    return round(mean(float(row[section][variant]["sync_ms"]) for row in rows), 3)


def summarize_diff(rows: list[dict[str, Any]], section: str, key: str) -> float:
    return round(mean(float(row[section][key]) for row in rows), 6)


def main() -> None:
    model, processor = mlx_load(str(MODEL_DIR))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    samples = collect_samples(model, tokenizer)
    print(f"captured {len(samples)} live subgraph replay samples")

    rows = []
    for sample_index, sample in enumerate(samples, start=1):
        rows.append(bench_sample(sample, model))
        if sample_index % 24 == 0:
            print(f"benchmarked {sample_index}/{len(samples)} samples")

    summary = {
        "projection_bundle": {
            "official_sync_ms": summarize_variant(rows, "projection_bundle", "official"),
            "manual_quantized_sync_ms": summarize_variant(rows, "projection_bundle", "manual_quantized"),
            "avg_qkv_max_abs_diff": summarize_diff(rows, "projection_bundle", "qkv_max_abs_diff"),
            "avg_z_max_abs_diff": summarize_diff(rows, "projection_bundle", "z_max_abs_diff"),
            "avg_a_max_abs_diff": summarize_diff(rows, "projection_bundle", "a_max_abs_diff"),
            "avg_b_max_abs_diff": summarize_diff(rows, "projection_bundle", "b_max_abs_diff"),
        },
        "linear_block": {
            "official_sync_ms": summarize_variant(rows, "linear_block", "official"),
            "manual_projection_sync_ms": summarize_variant(rows, "linear_block", "manual_projection"),
            "avg_output_max_abs_diff": summarize_diff(rows, "linear_block", "output_max_abs_diff"),
        },
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(MODEL_DIR),
        "prompt_suite": str(PROMPT_SUITE),
        "selected_decode_steps": list(SELECTED_DECODE_STEPS),
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "sample_count": len(rows),
        "rows": rows,
        "summary": summary,
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    projection = summary["projection_bundle"]
    linear = summary["linear_block"]
    lines = [
        "# Linear Subgraph Decode Replay",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Model dir: {MODEL_DIR}",
        f"- Prompt suite: {PROMPT_SUITE}",
        f"- Selected decode steps: {list(SELECTED_DECODE_STEPS)}",
        f"- Samples: {len(rows)}",
        "",
        "## Projection Bundle",
        "",
        f"- official sync: {projection['official_sync_ms']:.3f} ms",
        f"- manual quantized sync: {projection['manual_quantized_sync_ms']:.3f} ms",
        f"- avg qkv max abs diff: {projection['avg_qkv_max_abs_diff']:.6f}",
        f"- avg z max abs diff: {projection['avg_z_max_abs_diff']:.6f}",
        f"- avg a max abs diff: {projection['avg_a_max_abs_diff']:.6f}",
        f"- avg b max abs diff: {projection['avg_b_max_abs_diff']:.6f}",
        "",
        "## Full Linear-Attention Block",
        "",
        f"- official sync: {linear['official_sync_ms']:.3f} ms",
        f"- manual projection sync: {linear['manual_projection_sync_ms']:.3f} ms",
        f"- avg output max abs diff: {linear['avg_output_max_abs_diff']:.6f}",
        "",
        "## Findings",
        "",
        f"- projection bundle 단위에서도 official과 manual quantized path는 {projection['official_sync_ms']:.3f} vs {projection['manual_quantized_sync_ms']:.3f} ms로 거의 같다.",
        f"- full linear-attention block에서도 projection만 manual path로 바꾼 replay는 {linear['official_sync_ms']:.3f} vs {linear['manual_projection_sync_ms']:.3f} ms로 거의 같다.",
        f"- full block output diff도 {linear['avg_output_max_abs_diff']:.6f}라서, base 내부에서는 projection primitive를 바꿔도 block-level scheduling cost가 거의 달라지지 않는다.",
        "- 따라서 현재 남은 base-vs-custom decode delta는 projection primitive 자체보다, custom full-model graph 안에서 projection 주변 연산이 어떻게 compose/schedule 되는지에서 확인해야 한다.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()