#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx_vlm import load as mlx_load
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.models.qwen3_5 import language as qwen_language

WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Mac.gpu.test.gen_reference_qwen3_5_4b_mlx_8bit import (  # noqa: E402
    EOS_ENDOFTEXT,
    IM_END,
    build_nothink_chatml_tokens,
    load_prompt_suite,
    summarize_rows,
)


DEFAULT_MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
DEFAULT_PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/prompt_suite.json")
DEFAULT_OUTPUT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_stage_trace_split.json")
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

    def record_many(self, stages: list[str], dispatch_ms: float, sync_ms: float) -> None:
        for stage in stages:
            stats = self._target()[stage]
            stats.calls += 1
            stats.dispatch_ms += dispatch_ms
            stats.sync_ms += sync_ms

    def stage(self, stage: str, fn: Callable[[], Any], sync_output: bool = True) -> Any:
        return self.stage_aliases([stage], fn, sync_output=sync_output)

    def stage_aliases(self, stages: list[str], fn: Callable[[], Any], sync_output: bool = True) -> Any:
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
        self.record_many(stages, dispatch_ms, sync_ms)
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
        self.record_many(["sampler_sync(argmax/item)"], dispatch_ms, sync_ms)
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

    manager.patch_class_method(qwen_language.Qwen3_5MLP, "__call__", wrap_stage("mlp"))

    original_linear_call = qwen_language.Qwen3_5GatedDeltaNet.__call__

    def traced_linear_call(
        self: Any,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        def run_linear() -> mx.array:
            batch, seq_len, _hidden = inputs.shape
            linear_mask = mask

            mixed_qkv = tracer.stage(
                "linear_attention_in_proj_qkv",
                lambda: self.in_proj_qkv(inputs),
            )
            z = tracer.stage(
                "linear_attention_in_proj_z",
                lambda: self.in_proj_z(inputs).reshape(batch, seq_len, -1, self.head_v_dim),
            )
            b = tracer.stage("linear_attention_in_proj_b", lambda: self.in_proj_b(inputs))
            a = tracer.stage("linear_attention_in_proj_a", lambda: self.in_proj_a(inputs))

            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
                if conv_state.shape[0] != batch:
                    conv_state = mx.zeros(
                        (batch, self.conv_kernel_size - 1, self.conv_dim),
                        dtype=inputs.dtype,
                    )
            else:
                conv_state = mx.zeros(
                    (batch, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )

            if linear_mask is not None:
                if linear_mask.shape[0] != batch:
                    linear_mask = None
                else:
                    mixed_qkv = mx.where(linear_mask[..., None], mixed_qkv, 0)

            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            if cache is not None:
                cache[0] = tracer.stage_aliases(
                    ["linear_cache_update", "linear_cache_conv_state_update"],
                    lambda: conv_input[:, -(self.conv_kernel_size - 1) :],
                )

            conv_out = tracer.stage(
                "linear_attention_conv1d",
                lambda: qwen_language.nn.silu(self.conv1d(conv_input)),
            )

            q, k, v = [
                t.reshape(batch, seq_len, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]

            state = cache[1] if cache else None
            if state is not None and state.shape[0] != batch:
                state = None

            inv_scale = k.shape[-1] ** -0.5
            q = tracer.stage(
                "linear_attention_q_norm",
                lambda: (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6),
            )
            k = tracer.stage(
                "linear_attention_k_norm",
                lambda: inv_scale * mx.fast.rms_norm(k, None, 1e-6),
            )

            out, state = tracer.stage(
                "linear_attention_gated_delta",
                lambda: qwen_language.gated_delta_update(
                    q,
                    k,
                    v,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                    state,
                    linear_mask,
                    use_kernel=not self.training,
                ),
            )

            if cache is not None:
                cache[1] = tracer.stage_aliases(
                    ["linear_cache_update", "linear_cache_rec_state_update"],
                    lambda: state,
                )

            out = tracer.stage(
                "linear_attention_norm_gated",
                lambda: self.norm(out, z),
            )
            out = out.reshape(batch, seq_len, -1)
            return tracer.stage(
                "linear_attention_out_proj",
                lambda: self.out_proj(out),
            )

        return tracer.stage("linear_attention", run_linear)

    qwen_language.Qwen3_5GatedDeltaNet.__call__ = traced_linear_call
    manager._restorers.append(lambda: setattr(qwen_language.Qwen3_5GatedDeltaNet, "__call__", original_linear_call))

    def arrays_cache_setitem(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(self: Any, idx: int, value: Any) -> Any:
            if idx == 0:
                stage_names = ["linear_cache_update", "linear_cache_conv_state_update"]
            elif idx == 1:
                stage_names = ["linear_cache_update", "linear_cache_rec_state_update"]
            else:
                stage_names = ["linear_cache_update"]

            def assign_and_return() -> Any:
                original(self, idx, value)
                return self.cache[idx]

            return tracer.stage_aliases(stage_names, assign_and_return)

        return wrapped

    manager.patch_class_method(qwen_language.ArraysCache, "__setitem__", arrays_cache_setitem)

    original_attention_call = qwen_language.Qwen3_5Attention.__call__

    def traced_attention_call(
        self: Any,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        def run_attention() -> mx.array:
            batch, seq_len, _hidden = x.shape
            attn_mask = mask

            q_proj_output = tracer.stage("full_attention_q_proj", lambda: self.q_proj(x))
            queries, gate = mx.split(
                q_proj_output.reshape(batch, seq_len, self.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(batch, seq_len, -1)

            keys = tracer.stage("full_attention_k_proj", lambda: self.k_proj(x))
            values = tracer.stage("full_attention_v_proj", lambda: self.v_proj(x))

            queries = tracer.stage(
                "full_attention_q_norm",
                lambda: self.q_norm(queries).transpose(0, 2, 1, 3),
            )
            keys = tracer.stage(
                "full_attention_k_norm",
                lambda: self.k_norm(keys.reshape(batch, seq_len, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3),
            )
            values_t = values.reshape(batch, seq_len, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

            kv_seq_len = keys.shape[-2]
            if position_ids is None:
                kv_seq_len += cache.offset + 1
                current_position_ids = mx.arange(cache.offset, cache.offset + seq_len)
                current_position_ids = mx.expand_dims(current_position_ids, axis=0)
                current_position_ids = mx.tile(current_position_ids, (3, 1, 1))
            else:
                current_position_ids = position_ids
                kv_seq_len += cache.offset + 1 if cache is not None else 0

            cos, sin = tracer.stage_aliases(
                ["rope", "full_attention_rope"],
                lambda: self.rotary_emb(values_t, current_position_ids),
            )

            if attn_mask is not None and isinstance(attn_mask, mx.array):
                if isinstance(kv_seq_len, mx.array):
                    kv_seq_len = kv_seq_len.max().item()
                attn_mask = attn_mask[..., : int(kv_seq_len)]

            queries_r, keys_r = qwen_language.apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

            if cache is not None:
                keys_r, values_t = tracer.stage_aliases(
                    ["kv_cache_update", "full_attention_cache_update"],
                    lambda: cache.update_and_fetch(keys_r, values_t),
                )

            output = tracer.stage(
                "full_attention_sdpa",
                lambda: qwen_language.scaled_dot_product_attention(
                    queries_r,
                    keys_r,
                    values_t,
                    cache=cache,
                    scale=self.scale,
                    mask=attn_mask,
                ),
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
            return tracer.stage("full_attention_o_proj", lambda: self.o_proj(output * mx.sigmoid(gate)))

        return tracer.stage("full_attention", run_attention)

    qwen_language.Qwen3_5Attention.__call__ = traced_attention_call
    manager._restorers.append(lambda: setattr(qwen_language.Qwen3_5Attention, "__call__", original_attention_call))

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
            hidden = tracer.stage("input_embeddings", lambda: self.embed_tokens(inputs))
        else:
            hidden = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = qwen_language.create_attention_mask(hidden, cache[self.fa_idx])
        ssm_mask = qwen_language.create_ssm_mask(hidden, cache[self.ssm_idx])

        for layer, layer_cache in zip(self.layers, cache):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, layer_mask, layer_cache, position_ids)

        return tracer.stage("final_norm", lambda: self.norm(hidden))

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
            if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
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
                f"[base_split_trace {prompt_index:02d}/{len(prompt_rows):02d}] {prompt_row.id}: "
                f"prefill={row['prefill_ms']:.1f}ms, decode={row['decode_ms']:.1f}ms/tok, "
                f"wall={row['wall_ms']:.1f}ms, throughput={row['throughput']:.2f} tok/s"
            )
    finally:
        manager.restore()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "base_stage_trace_split",
        "model_dir": str(model_dir),
        "prompt_suite": str(prompt_suite),
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "summary": summarize_rows(rows),
        "stage_summary": aggregate_stage_maps(rows),
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote base split stage trace: {output_path}")


def main() -> None:
    run_trace(
        model_dir=DEFAULT_MODEL_DIR,
        prompt_suite=DEFAULT_PROMPT_SUITE,
        output_path=DEFAULT_OUTPUT,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    )


if __name__ == "__main__":
    main()