#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx_vlm import load as mlx_load
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.models.qwen3_5 import language as qwen_language

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Mac.gpu.test.gen_reference_qwen3_5_4b_mlx_8bit import (
    EOS_ENDOFTEXT,
    IM_END,
    build_nothink_chatml_tokens,
    load_prompt_suite,
    summarize_rows,
)


DEFAULT_MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
DEFAULT_PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/prompt_suite.json")
DEFAULT_OUTPUT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/base_stage_trace.json")
DEFAULT_MAX_NEW_TOKENS = 32


@dataclass
class StageStats:
    calls: int = 0
    dispatch_ms: float = 0.0
    sync_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "dispatch_ms": round(self.dispatch_ms, 3),
            "sync_ms": round(self.sync_ms, 3),
        }


class TraceCollector:
    def __init__(self) -> None:
        self.phase = "prefill"
        self.prefill: dict[str, StageStats] = defaultdict(StageStats)
        self.decode: dict[str, StageStats] = defaultdict(StageStats)

    def reset(self) -> None:
        self.phase = "prefill"
        self.prefill = defaultdict(StageStats)
        self.decode = defaultdict(StageStats)

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _target(self) -> dict[str, StageStats]:
        return self.decode if self.phase == "decode" else self.prefill

    def record(self, stage: str, dispatch_ms: float, sync_ms: float) -> None:
        stats = self._target()[stage]
        stats.calls += 1
        stats.dispatch_ms += dispatch_ms
        stats.sync_ms += sync_ms

    def stage(self, stage: str, fn: Callable[[], Any], sync_output: bool = True) -> Any:
        t0 = time.perf_counter()
        result = fn()
        dispatch_ms = (time.perf_counter() - t0) * 1000.0
        sync_ms = 0.0
        if sync_output:
            arrays = collect_arrays(result)
            if arrays:
                t1 = time.perf_counter()
                mx.eval(arrays)
                mx.synchronize()
                sync_ms = (time.perf_counter() - t1) * 1000.0
        self.record(stage, dispatch_ms, sync_ms)
        return result

    def sampler(self, logits: mx.array) -> int:
        t0 = time.perf_counter()
        current = mx.argmax(logits, axis=-1)
        dispatch_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        mx.eval(current)
        mx.synchronize()
        token = int(current.item())
        sync_ms = (time.perf_counter() - t1) * 1000.0
        self.record("sampler_sync(argmax/item)", dispatch_ms, sync_ms)
        return token

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefill": stage_map_to_dict(self.prefill),
            "decode": stage_map_to_dict(self.decode),
        }


def collect_arrays(value: Any) -> list[mx.array]:
    arrays: list[mx.array] = []
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            arrays.extend(collect_arrays(item))
    elif isinstance(value, dict):
        for item in value.values():
            arrays.extend(collect_arrays(item))
    return arrays


def stage_map_to_dict(stage_map: dict[str, StageStats]) -> dict[str, Any]:
    return {name: stage_map[name].to_dict() for name in sorted(stage_map.keys())}


def aggregate_stage_maps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, StageStats]] = {
        "prefill": defaultdict(StageStats),
        "decode": defaultdict(StageStats),
    }
    for row in rows:
        for phase in ("prefill", "decode"):
            for stage, stats in row["stage_trace"][phase].items():
                acc = totals[phase][stage]
                acc.calls += int(stats["calls"])
                acc.dispatch_ms += float(stats["dispatch_ms"])
                acc.sync_ms += float(stats["sync_ms"])
    return {phase: stage_map_to_dict(stage_map) for phase, stage_map in totals.items()}


class PatchManager:
    def __init__(self) -> None:
        self._restorers: list[Callable[[], None]] = []

    def patch_class_method(self, cls: type[Any], method_name: str, wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        original = getattr(cls, method_name)
        setattr(cls, method_name, wrapper_factory(original))
        self._restorers.append(lambda: setattr(cls, method_name, original))

    def patch_instance_method(self, instance: Any, method_name: str, wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        original = getattr(instance, method_name)
        wrapped = wrapper_factory(original)
        setattr(instance, method_name, types.MethodType(wrapped, instance))
        self._restorers.append(lambda: setattr(instance, method_name, original))

    def restore(self) -> None:
        while self._restorers:
            self._restorers.pop()()


def install_trace_hooks(model: Any, tracer: TraceCollector) -> PatchManager:
    manager = PatchManager()

    def wrap_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def factory(original: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                return tracer.stage(stage, lambda: original(self, *args, **kwargs))

            return wrapped

        return factory

    manager.patch_class_method(qwen_language.Qwen3_5RotaryEmbedding, "__call__", wrap_stage("rope"))
    manager.patch_class_method(qwen_language.Qwen3_5Attention, "__call__", wrap_stage("full_attention"))
    manager.patch_class_method(qwen_language.Qwen3_5GatedDeltaNet, "__call__", wrap_stage("linear_attention"))
    manager.patch_class_method(qwen_language.Qwen3_5MLP, "__call__", wrap_stage("mlp"))
    manager.patch_class_method(qwen_language.KVCache, "update_and_fetch", wrap_stage("kv_cache_update"))

    def arrays_cache_setitem(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(self: Any, idx: int, value: Any) -> Any:
            return tracer.stage("linear_cache_update", lambda: original(self, idx, value))

        return wrapped

    manager.patch_class_method(qwen_language.ArraysCache, "__setitem__", arrays_cache_setitem)

    original_model_call = qwen_language.Qwen3_5Model.__call__

    def traced_model_call(
        self: Any,
        inputs: mx.array,
        inputs_embeds: mx.array | None = None,
        mask: mx.array | None = None,
        cache: Any = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = tracer.stage("input_embeddings", lambda: self.embed_tokens(inputs))
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = qwen_language.create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = qwen_language.create_ssm_mask(h, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, layer_mask, c, position_ids)

        return tracer.stage("final_norm", lambda: self.norm(h))

    qwen_language.Qwen3_5Model.__call__ = traced_model_call
    manager._restorers.append(lambda: setattr(qwen_language.Qwen3_5Model, "__call__", original_model_call))

    original_language_call = qwen_language.LanguageModel.__call__

    def traced_language_call(self: Any, inputs: mx.array, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache: Any = None, **kwargs: Any) -> Any:
        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None

        cache_offset = 0
        if cache and cache[self.model.fa_idx] is not None:
            offset = cache[self.model.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        def compute_position_ids() -> mx.array:
            if (
                position_ids is None
                and (rope_mask is None or rope_mask.ndim == 2)
            ):
                if (
                    ((cache is not None and cache[self.model.fa_idx] is not None and (cache_offset == 0))
                     or self._rope_deltas is None
                     or cache is None)
                ):
                    if self._position_ids is not None:
                        seq_length = inputs.shape[1]
                        return self._position_ids[:, :, cache_offset : cache_offset + seq_length]
                    pos, rope_deltas = self.get_rope_index(inputs, image_grid_thw, video_grid_thw, rope_mask)
                    self._rope_deltas = rope_deltas
                    self._position_ids = pos
                    return pos

                batch_size, seq_length = inputs.shape
                delta = mx.array(cache_offset + self._rope_deltas if cache is not None else 0)
                pos = mx.arange(seq_length).reshape(1, -1)
                pos = mx.broadcast_to(pos, (batch_size, seq_length))

                if cache_offset is not None:
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)

                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]

                pos = mx.add(pos, delta)[None, ...]
                return mx.broadcast_to(pos, (3, batch_size, seq_length))

            return position_ids

        traced_position_ids = tracer.stage("position_ids", compute_position_ids)
        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=traced_position_ids,
        )
        if self.args.tie_word_embeddings:
            return tracer.stage("lm_head", lambda: self.model.embed_tokens.as_linear(out))
        return tracer.stage("lm_head", lambda: self.lm_head(out))

    qwen_language.LanguageModel.__call__ = traced_language_call
    manager._restorers.append(lambda: setattr(qwen_language.LanguageModel, "__call__", original_language_call))
    return manager


def run_trace(model_dir: Path, prompt_suite: Path, output_path: Path, max_new_tokens: int) -> None:
    prompt_rows = load_prompt_suite(prompt_suite, None)
    model, processor = mlx_load(str(model_dir))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos_ids = {EOS_ENDOFTEXT, IM_END}

    tracer = TraceCollector()
    manager = install_trace_hooks(model, tracer)
    rows: list[dict[str, Any]] = []

    try:
        for prompt_index, prompt_row in enumerate(prompt_rows, start=1):
            tracer.reset()
            model.language_model._position_ids = None
            model.language_model._rope_deltas = None
            prompt_tokens = build_nothink_chatml_tokens(tokenizer, prompt_row.prompt_text, None)
            prompt_cache = vlm_cache.make_prompt_cache(model.language_model)
            input_ids = mx.array([prompt_tokens], dtype=mx.int32)

            tracer.set_phase("prefill")

            t_wall0 = time.perf_counter()
            t0 = t_wall0
            logits = model.language_model(input_ids, cache=prompt_cache)
            mx.eval(logits)
            mx.synchronize()
            t1 = time.perf_counter()

            token = tracer.sampler(logits[:, -1, :])
            generated_tokens = [token]

            t_decode0 = time.perf_counter()
            while len(generated_tokens) < max_new_tokens and token not in eos_ids:
                tracer.set_phase("decode")
                decode_input = mx.array([[token]], dtype=mx.int32)
                logits = model.language_model(decode_input, cache=prompt_cache)
                token = tracer.sampler(logits[:, -1, :])
                generated_tokens.append(token)
            t_decode1 = time.perf_counter()

            prefill_ms = (t1 - t0) * 1000.0
            decode_total_ms = (t_decode1 - t_decode0) * 1000.0
            wall_ms = (t_decode1 - t_wall0) * 1000.0
            decode_ms = decode_total_ms / len(generated_tokens) if generated_tokens else 0.0
            throughput = (len(generated_tokens) * 1000.0 / wall_ms) if generated_tokens and wall_ms > 0.0 else 0.0

            row = {
                "id": prompt_row.id,
                "kind": prompt_row.kind,
                "prompt_text": prompt_row.prompt_text,
                "prompt_tokens": prompt_tokens,
                "prompt_token_count": len(prompt_tokens),
                "generated_tokens": generated_tokens,
                "generated_token_count": len(generated_tokens),
                "output_text": tokenizer.decode(generated_tokens, skip_special_tokens=False),
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "wall_ms": round(wall_ms, 3),
                "throughput": round(throughput, 3),
                "peak_memory_gb": 0.0,
                "stage_trace": tracer.to_dict(),
            }
            rows.append(row)
            print(
                f"[base_trace {prompt_index:02d}/{len(prompt_rows):02d}] {prompt_row.id}: "
                f"prefill={row['prefill_ms']:.1f}ms, decode={row['decode_ms']:.1f}ms/tok, "
                f"wall={row['wall_ms']:.1f}ms, throughput={row['throughput']:.2f} tok/s"
            )
    finally:
        manager.restore()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "base_stage_trace",
        "model_dir": str(model_dir),
        "prompt_suite": str(prompt_suite),
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "summary": summarize_rows(rows),
        "stage_summary": aggregate_stage_maps(rows),
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote base stage trace: {output_path}")


def main() -> None:
    run_trace(
        model_dir=DEFAULT_MODEL_DIR,
        prompt_suite=DEFAULT_PROMPT_SUITE,
        output_path=DEFAULT_OUTPUT,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    )


if __name__ == "__main__":
    main()