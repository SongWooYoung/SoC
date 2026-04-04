#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from mlx_vlm import load as mlx_load
from mlx_vlm.generate import stream_generate
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_reference_qwen3_5_4b_mlx_8bit import (
    EOS_ENDOFTEXT,
    IM_END,
    build_nothink_chatml_text,
    build_nothink_chatml_tokens,
    compute_prefill_topk,
    load_prompt_suite,
    safe_rate_to_ms,
    summarize_rows,
)


DEFAULT_BASE_MODEL_DIR = Path("/Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b")
DEFAULT_CANDIDATE_MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
DEFAULT_PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json")
DEFAULT_OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_pytorch_4b_vs_mlx_9b.json")
DEFAULT_OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_pytorch_4b_vs_mlx_9b.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch Qwen3.5-4B against local Qwen3.5-9B-MLX-8bit."
    )
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--candidate-model-dir", type=Path, default=DEFAULT_CANDIDATE_MODEL_DIR)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_PROMPT_SUITE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--system-message", default=None)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def maybe_sync_torch(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_torch_memory_gb(device: str) -> float:
    if device == "mps" and torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
        return round(float(torch.mps.current_allocated_memory()) / (1024**3), 3)
    return 0.0


def run_pytorch_model(
    model_dir: Path,
    prompt_suite: Path,
    max_new_tokens: int,
    system_message: str | None,
    device: str,
    limit: int | None,
) -> dict[str, Any]:
    prompt_rows = load_prompt_suite(prompt_suite, limit)
    print(f"Loading PyTorch base model from {model_dir} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    torch_dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    eos_ids = {EOS_ENDOFTEXT, IM_END}
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt_row in enumerate(prompt_rows, start=1):
        templated_text = build_nothink_chatml_text(prompt_row.prompt_text, system_message)
        prompt_tokens = build_nothink_chatml_tokens(tokenizer, prompt_row.prompt_text, system_message)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        maybe_sync_torch(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        maybe_sync_torch(device)
        t1 = time.perf_counter()

        logits = outputs.logits[0, -1, :].float().cpu().numpy()
        past_key_values = outputs.past_key_values
        argmax_token, argmax_text, top_entries = compute_prefill_topk(logits.tolist(), tokenizer)

        generated_tokens: list[int] = [argmax_token]
        next_token = torch.tensor([[argmax_token]], dtype=torch.long, device=device)

        maybe_sync_torch(device)
        t2 = time.perf_counter()
        with torch.no_grad():
            for _ in range(1, max_new_tokens):
                step_outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = step_outputs.past_key_values
                next_token_id = int(torch.argmax(step_outputs.logits[0, -1, :]).item())
                generated_tokens.append(next_token_id)
                if next_token_id in eos_ids:
                    break
                next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        maybe_sync_torch(device)
        t3 = time.perf_counter()

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        prefill_ms = (t1 - t0) * 1000.0
        decode_ms_total = (t3 - t2) * 1000.0
        decode_ms = decode_ms_total / len(generated_tokens) if generated_tokens else 0.0
        wall_ms = prefill_ms + decode_ms_total
        throughput = len(generated_tokens) / wall_ms * 1000.0 if wall_ms > 0.0 else 0.0

        row = {
            "id": prompt_row.id,
            "kind": prompt_row.kind,
            "prompt_text": prompt_row.prompt_text,
            "templated_text": templated_text,
            "prompt_token_count": len(prompt_tokens),
            "prefill_argmax_token_id": argmax_token,
            "prefill_argmax_token_text": argmax_text,
            "prefill_logprobs_top10": top_entries,
            "generated_tokens": generated_tokens,
            "generated_token_count": len(generated_tokens),
            "output_text": output_text,
            "eos_reached": bool(generated_tokens and generated_tokens[-1] in eos_ids),
            "prefill_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "wall_ms": round(wall_ms, 3),
            "throughput": round(throughput, 3),
            "peak_memory_gb": get_torch_memory_gb(device),
        }
        rows.append(row)

        print(
            f"[pytorch4b {prompt_index:02d}/{len(prompt_rows):02d}] {prompt_row.id}: "
            f"prefill={row['prefill_ms']:.1f}ms, decode={row['decode_ms']:.1f}ms/tok, "
            f"wall={row['wall_ms']:.1f}ms, throughput={row['throughput']:.2f} tok/s"
        )

    return {
        "label": "qwen3_5_4b_pytorch",
        "backend": "transformers",
        "model_dir": str(model_dir),
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def run_mlx_model(
    model_dir: Path,
    prompt_suite: Path,
    max_new_tokens: int,
    system_message: str | None,
    limit: int | None,
) -> dict[str, Any]:
    prompt_rows = load_prompt_suite(prompt_suite, limit)
    print(f"Loading MLX candidate model from {model_dir}...")
    model, processor = mlx_load(str(model_dir))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    eos_ids = {EOS_ENDOFTEXT, IM_END}
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt_row in enumerate(prompt_rows, start=1):
        templated_text = build_nothink_chatml_text(prompt_row.prompt_text, system_message)
        prompt_tokens = build_nothink_chatml_tokens(tokenizer, prompt_row.prompt_text, system_message)
        input_ids = mx.array([prompt_tokens], dtype=mx.int32)
        responses = stream_generate(
            model,
            processor,
            prompt="",
            input_ids=input_ids,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )

        generated_tokens: list[int] = []
        generated_text_segments: list[str] = []
        first_logprobs: list[float] | None = None
        last_response = None
        seen_generation_tokens = 0

        for response in responses:
            last_response = response
            if first_logprobs is None and response.logprobs is not None:
                mx.eval(response.logprobs)
                first_logprobs = response.logprobs.tolist()
            if response.generation_tokens > seen_generation_tokens and response.token is not None:
                generated_tokens.append(int(response.token))
                seen_generation_tokens = response.generation_tokens
            if response.text:
                generated_text_segments.append(response.text)

        if last_response is None or first_logprobs is None:
            raise RuntimeError(f"Generation failed for prompt {prompt_row.id}")

        output_text = "".join(generated_text_segments)
        prefill_ms = safe_rate_to_ms(last_response.prompt_tokens, last_response.prompt_tps)
        generation_wall_ms = safe_rate_to_ms(last_response.generation_tokens, last_response.generation_tps)
        decode_ms = generation_wall_ms / last_response.generation_tokens if last_response.generation_tokens > 0 else 0.0
        wall_ms = prefill_ms + generation_wall_ms
        throughput = last_response.generation_tokens / wall_ms * 1000.0 if wall_ms > 0.0 else 0.0
        argmax_token, argmax_text, top_entries = compute_prefill_topk(first_logprobs, tokenizer)

        row = {
            "id": prompt_row.id,
            "kind": prompt_row.kind,
            "prompt_text": prompt_row.prompt_text,
            "templated_text": templated_text,
            "prompt_token_count": len(prompt_tokens),
            "prefill_argmax_token_id": argmax_token,
            "prefill_argmax_token_text": argmax_text,
            "prefill_logprobs_top10": top_entries,
            "generated_tokens": generated_tokens,
            "generated_token_count": len(generated_tokens),
            "output_text": output_text,
            "eos_reached": bool(generated_tokens and generated_tokens[-1] in eos_ids),
            "prefill_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "wall_ms": round(wall_ms, 3),
            "throughput": round(throughput, 3),
            "peak_memory_gb": round(float(last_response.peak_memory), 3),
        }
        rows.append(row)

        print(
            f"[mlx9b {prompt_index:02d}/{len(prompt_rows):02d}] {prompt_row.id}: "
            f"prefill={row['prefill_ms']:.1f}ms, decode={row['decode_ms']:.1f}ms/tok, "
            f"wall={row['wall_ms']:.1f}ms, throughput={row['throughput']:.2f} tok/s"
        )

    return {
        "label": "qwen3_5_9b_mlx_8bit",
        "backend": "mlx_vlm",
        "model_dir": str(model_dir),
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def build_prompt_comparisons(base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_by_id = {row["id"]: row for row in candidate_rows}
    comparisons: list[dict[str, Any]] = []
    for base_row in base_rows:
        candidate_row = candidate_by_id[base_row["id"]]
        comparisons.append(
            {
                "id": base_row["id"],
                "kind": base_row["kind"],
                "prompt_text": base_row["prompt_text"],
                "base_prefill_ms": base_row["prefill_ms"],
                "candidate_prefill_ms": candidate_row["prefill_ms"],
                "prefill_ms_delta": round(candidate_row["prefill_ms"] - base_row["prefill_ms"], 3),
                "base_decode_ms": base_row["decode_ms"],
                "candidate_decode_ms": candidate_row["decode_ms"],
                "decode_ms_delta": round(candidate_row["decode_ms"] - base_row["decode_ms"], 3),
                "base_wall_ms": base_row["wall_ms"],
                "candidate_wall_ms": candidate_row["wall_ms"],
                "wall_ms_delta": round(candidate_row["wall_ms"] - base_row["wall_ms"], 3),
                "base_throughput": base_row["throughput"],
                "candidate_throughput": candidate_row["throughput"],
                "throughput_delta": round(candidate_row["throughput"] - base_row["throughput"], 3),
                "base_output_text": base_row["output_text"],
                "candidate_output_text": candidate_row["output_text"],
                "exact_output_match": base_row["output_text"] == candidate_row["output_text"],
            }
        )
    return comparisons


def render_markdown(report: dict[str, Any]) -> str:
    base = report["base_model"]
    candidate = report["candidate_model"]
    delta = report["summary_delta"]
    lines = [
        "# Qwen3.5 PyTorch 4B vs MLX 9B Comparison",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Prompt suite: {report['prompt_suite']}",
        f"- Base model: {base['label']} ({base['backend']}) {base['model_dir']}",
        f"- Candidate model: {candidate['label']} ({candidate['backend']}) {candidate['model_dir']}",
        f"- Max new tokens: {report['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| Model | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |",
        "|------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|",
        f"| {base['label']} | {base['summary']['rows']} | {base['summary']['avg_prompt_tokens']:.3f} | {base['summary']['avg_generated_tokens']:.3f} | {base['summary']['avg_prefill_ms']:.3f} | {base['summary']['avg_decode_ms']:.3f} | {base['summary']['avg_wall_ms']:.3f} | {base['summary']['avg_throughput']:.3f} | {base['summary']['avg_peak_memory_gb']:.3f} |",
        f"| {candidate['label']} | {candidate['summary']['rows']} | {candidate['summary']['avg_prompt_tokens']:.3f} | {candidate['summary']['avg_generated_tokens']:.3f} | {candidate['summary']['avg_prefill_ms']:.3f} | {candidate['summary']['avg_decode_ms']:.3f} | {candidate['summary']['avg_wall_ms']:.3f} | {candidate['summary']['avg_throughput']:.3f} | {candidate['summary']['avg_peak_memory_gb']:.3f} |",
        "",
        "## Candidate Minus Base",
        "",
        f"- Avg prefill ms delta: {delta['avg_prefill_ms_delta']:.3f}",
        f"- Avg decode ms/tok delta: {delta['avg_decode_ms_delta']:.3f}",
        f"- Avg wall ms delta: {delta['avg_wall_ms_delta']:.3f}",
        f"- Avg throughput delta: {delta['avg_throughput_delta']:.3f}",
        f"- Avg peak memory GB delta: {delta['avg_peak_memory_gb_delta']:.3f}",
        f"- Exact output matches: {delta['exact_output_matches']}/{delta['rows']}",
        "",
        "## Per-Prompt",
        "",
        "| Prompt | Kind | 4B Tok/s | 9B Tok/s | Delta Tok/s | 4B Wall ms | 9B Wall ms | Delta Wall ms | Match |",
        "|-------|------|---------:|---------:|------------:|-----------:|-----------:|--------------:|:------|",
    ]
    for row in report["prompt_comparisons"]:
        lines.append(
            f"| {row['id']} | {row['kind']} | {row['base_throughput']:.3f} | {row['candidate_throughput']:.3f} | {row['throughput_delta']:.3f} | "
            f"{row['base_wall_ms']:.3f} | {row['candidate_wall_ms']:.3f} | {row['wall_ms_delta']:.3f} | {'yes' if row['exact_output_match'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    base = run_pytorch_model(
        model_dir=args.base_model_dir,
        prompt_suite=args.prompt_suite,
        max_new_tokens=args.max_new_tokens,
        system_message=args.system_message,
        device=args.device,
        limit=args.limit,
    )
    candidate = run_mlx_model(
        model_dir=args.candidate_model_dir,
        prompt_suite=args.prompt_suite,
        max_new_tokens=args.max_new_tokens,
        system_message=args.system_message,
        limit=args.limit,
    )

    prompt_comparisons = build_prompt_comparisons(base["rows"], candidate["rows"])
    exact_matches = sum(1 for row in prompt_comparisons if row["exact_output_match"])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_suite": str(args.prompt_suite),
        "max_new_tokens": args.max_new_tokens,
        "base_model": {
            "label": base["label"],
            "backend": base["backend"],
            "model_dir": base["model_dir"],
            "summary": base["summary"],
        },
        "candidate_model": {
            "label": candidate["label"],
            "backend": candidate["backend"],
            "model_dir": candidate["model_dir"],
            "summary": candidate["summary"],
        },
        "summary_delta": {
            "rows": len(prompt_comparisons),
            "avg_prefill_ms_delta": round(candidate["summary"]["avg_prefill_ms"] - base["summary"]["avg_prefill_ms"], 3),
            "avg_decode_ms_delta": round(candidate["summary"]["avg_decode_ms"] - base["summary"]["avg_decode_ms"], 3),
            "avg_wall_ms_delta": round(candidate["summary"]["avg_wall_ms"] - base["summary"]["avg_wall_ms"], 3),
            "avg_throughput_delta": round(candidate["summary"]["avg_throughput"] - base["summary"]["avg_throughput"], 3),
            "avg_peak_memory_gb_delta": round(candidate["summary"]["avg_peak_memory_gb"] - base["summary"]["avg_peak_memory_gb"], 3),
            "exact_output_matches": exact_matches,
        },
        "prompt_comparisons": prompt_comparisons,
    }

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")

    print("\nComparison Summary")
    print(
        f"  base      : prefill={base['summary']['avg_prefill_ms']:.3f}ms, decode={base['summary']['avg_decode_ms']:.3f}ms/tok, "
        f"wall={base['summary']['avg_wall_ms']:.3f}ms, throughput={base['summary']['avg_throughput']:.3f} tok/s"
    )
    print(
        f"  candidate : prefill={candidate['summary']['avg_prefill_ms']:.3f}ms, decode={candidate['summary']['avg_decode_ms']:.3f}ms/tok, "
        f"wall={candidate['summary']['avg_wall_ms']:.3f}ms, throughput={candidate['summary']['avg_throughput']:.3f} tok/s"
    )
    print(
        f"  delta     : prefill={report['summary_delta']['avg_prefill_ms_delta']:.3f}ms, decode={report['summary_delta']['avg_decode_ms_delta']:.3f}ms/tok, "
        f"wall={report['summary_delta']['avg_wall_ms_delta']:.3f}ms, throughput={report['summary_delta']['avg_throughput_delta']:.3f} tok/s, "
        f"exact_matches={exact_matches}/{len(prompt_comparisons)}"
    )
    print(f"\nWrote comparison JSON: {args.output_json}")
    print(f"Wrote comparison MD:   {args.output_md}")


if __name__ == "__main__":
    main()