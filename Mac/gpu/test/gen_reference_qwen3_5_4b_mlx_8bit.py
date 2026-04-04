#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.generate import stream_generate


DEFAULT_MODEL_DIR = Path("/Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b-mlx-8bit")
DEFAULT_PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json")
DEFAULT_JSONL = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_4b_mlx_8bit_reference.jsonl")
DEFAULT_PRETTY_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_4b_mlx_8bit_reference_pretty.json")
DEFAULT_SUMMARY_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_4b_mlx_8bit_reference_summary.json")
DEFAULT_SUMMARY_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_4b_mlx_8bit_reference_summary.md")
DEFAULT_REPORT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_4b_mlx_8bit_reference_report.md")

IM_START = 248045
IM_END = 248046
EOS_ENDOFTEXT = 248044
THINK = 248068
THINK_END = 248069


@dataclass
class PromptRow:
    id: str
    kind: str
    prompt_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSONL reference outputs for mlx-community/Qwen3.5-4B-MLX-8bit."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_PROMPT_SUITE)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-pretty-json", type=Path, default=DEFAULT_PRETTY_JSON)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--output-summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--output-report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--system-message", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_prompt_suite(path: Path, limit: int | None) -> list[PromptRow]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    prompt_rows = [PromptRow(**row) for row in rows]
    if limit is not None:
        return prompt_rows[:limit]
    return prompt_rows


def build_nothink_chatml_text(user_message: str, system_message: str | None = None) -> str:
    parts: list[str] = []
    if system_message:
        parts.append(f"<|im_start|>system\n{system_message}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_message}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def build_nothink_chatml_tokens(tokenizer: Any, user_message: str, system_message: str | None = None) -> list[int]:
    newline = tokenizer.encode("\n", add_special_tokens=False)
    double_newline = tokenizer.encode("\n\n", add_special_tokens=False)
    ids: list[int] = []

    def append_role(role: str, content: str) -> None:
        ids.append(IM_START)
        ids.extend(tokenizer.encode(role, add_special_tokens=False))
        ids.extend(newline)
        ids.extend(tokenizer.encode(content, add_special_tokens=False))
        ids.append(IM_END)
        ids.extend(newline)

    if system_message:
        append_role("system", system_message)

    append_role("user", user_message)
    ids.append(IM_START)
    ids.extend(tokenizer.encode("assistant", add_special_tokens=False))
    ids.extend(newline)
    ids.append(THINK)
    ids.extend(double_newline)
    ids.append(THINK_END)
    ids.extend(double_newline)
    return ids


def compute_prefill_topk(logprobs: list[float], tokenizer: Any, k: int = 10) -> tuple[int, str, list[dict[str, Any]]]:
    array = np.asarray(logprobs, dtype=np.float32)
    top_indices = np.argsort(array)[-k:][::-1]
    top_entries = []
    for token_id in top_indices.tolist():
        top_entries.append(
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logprob": float(array[token_id]),
            }
        )
    argmax_token = int(top_indices[0])
    argmax_text = tokenizer.decode([argmax_token], skip_special_tokens=False)
    return argmax_token, argmax_text, top_entries


def safe_rate_to_ms(count: int, rate: float) -> float:
    if rate <= 0.0:
        return 0.0
    return count / rate * 1000.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "avg_prompt_tokens": 0.0,
            "avg_generated_tokens": 0.0,
            "avg_prefill_ms": 0.0,
            "avg_decode_ms": 0.0,
            "avg_wall_ms": 0.0,
            "avg_throughput": 0.0,
            "avg_peak_memory_gb": 0.0,
        }

    def avg(key: str) -> float:
        return round(sum(float(row[key]) for row in rows) / len(rows), 3)

    return {
        "rows": len(rows),
        "avg_prompt_tokens": avg("prompt_token_count"),
        "avg_generated_tokens": avg("generated_token_count"),
        "avg_prefill_ms": avg("prefill_ms"),
        "avg_decode_ms": avg("decode_ms"),
        "avg_wall_ms": avg("wall_ms"),
        "avg_throughput": avg("throughput"),
        "avg_peak_memory_gb": avg("peak_memory_gb"),
    }


def make_summary_document(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir),
        "prompt_suite": str(args.prompt_suite),
        "max_new_tokens": args.max_new_tokens,
        "system_message": args.system_message,
        "overall": summarize_rows(rows),
        "short": summarize_rows(by_kind.get("short", [])),
        "long": summarize_rows(by_kind.get("long", [])),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5-4B-MLX-8bit Reference Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Model dir: {summary['model_dir']}",
        f"- Prompt suite: {summary['prompt_suite']}",
        f"- Max new tokens: {summary['max_new_tokens']}",
        "",
        "| Split | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |",
        "|------|------:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|",
    ]
    for split in ("overall", "short", "long"):
        row = summary[split]
        lines.append(
            f"| {split} | {row['rows']} | {row['avg_prompt_tokens']:.3f} | {row['avg_generated_tokens']:.3f} | "
            f"{row['avg_prefill_ms']:.3f} | {row['avg_decode_ms']:.3f} | {row['avg_wall_ms']:.3f} | "
            f"{row['avg_throughput']:.3f} | {row['avg_peak_memory_gb']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_detailed_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5-4B-MLX-8bit Reference Report",
        "",
        "## Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Model dir: {summary['model_dir']}",
        f"- Prompt suite: {summary['prompt_suite']}",
        f"- Max new tokens: {summary['max_new_tokens']}",
        "",
        "| Split | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |",
        "|------|------:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|",
    ]
    for split in ("overall", "short", "long"):
        row = summary[split]
        lines.append(
            f"| {split} | {row['rows']} | {row['avg_prompt_tokens']:.3f} | {row['avg_generated_tokens']:.3f} | "
            f"{row['avg_prefill_ms']:.3f} | {row['avg_decode_ms']:.3f} | {row['avg_wall_ms']:.3f} | "
            f"{row['avg_throughput']:.3f} | {row['avg_peak_memory_gb']:.3f} |"
        )

    for row in rows:
        lines.extend(
            [
                "",
                f"## {row['id']} ({row['kind']})",
                "",
                "### Prompt",
                "",
                row["prompt_text"],
                "",
                "### Output",
                "",
                row["output_text"] if row["output_text"] else "(empty)",
                "",
                "### Metrics",
                "",
                f"- Prompt tokens: {row['prompt_token_count']}",
                f"- Generated tokens: {row['generated_token_count']}",
                f"- EOS reached: {row['eos_reached']}",
                f"- Prefill ms: {row['prefill_ms']}",
                f"- Decode ms/tok: {row['decode_ms']}",
                f"- Wall ms: {row['wall_ms']}",
                f"- Throughput tok/s: {row['throughput']}",
                f"- Prompt tok/s: {row['prompt_tps']}",
                f"- Generation tok/s: {row['generation_tps']}",
                f"- Peak memory GB: {row['peak_memory_gb']}",
                "",
                "### Prefill Top-10",
                "",
            ]
        )
        for entry in row["prefill_logprobs_top10"]:
            lines.append(
                f"- token_id={entry['token_id']}, token_text={json.dumps(entry['token_text'], ensure_ascii=False)}, logprob={entry['logprob']}"
            )
        lines.extend(
            [
                "",
                "### Generated Tokens",
                "",
                json.dumps(row["generated_tokens"], ensure_ascii=False),
                "",
                "### Templated Text",
                "",
                "```text",
                row["templated_text"],
                "```",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    prompt_rows = load_prompt_suite(args.prompt_suite, args.limit)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_pretty_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MLX model from {args.model_dir}...")
    model, processor = load(str(args.model_dir))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    eos_ids = {EOS_ENDOFTEXT, IM_END}
    result_rows: list[dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for prompt_index, prompt_row in enumerate(prompt_rows, start=1):
            templated_text = build_nothink_chatml_text(
                prompt_row.prompt_text,
                args.system_message,
            )
            prompt_tokens = build_nothink_chatml_tokens(
                tokenizer,
                prompt_row.prompt_text,
                args.system_message,
            )

            input_ids = mx.array([prompt_tokens], dtype=mx.int32)
            responses = stream_generate(
                model,
                processor,
                prompt="",
                input_ids=input_ids,
                max_tokens=args.max_new_tokens,
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
            decode_ms = (
                generation_wall_ms / last_response.generation_tokens
                if last_response.generation_tokens > 0
                else 0.0
            )
            wall_ms = prefill_ms + generation_wall_ms
            throughput = (
                last_response.generation_tokens / wall_ms * 1000.0
                if wall_ms > 0.0
                else 0.0
            )

            argmax_token, argmax_text, top_entries = compute_prefill_topk(first_logprobs, tokenizer)

            row = {
                "id": prompt_row.id,
                "kind": prompt_row.kind,
                "prompt_text": prompt_row.prompt_text,
                "templated_text": templated_text,
                "prompt_tokens": prompt_tokens,
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
                "prompt_tps": round(float(last_response.prompt_tps), 3),
                "generation_tps": round(float(last_response.generation_tps), 3),
                "peak_memory_gb": round(float(last_response.peak_memory), 3),
            }

            jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            result_rows.append(row)

            print(
                f"[{prompt_index:02d}/{len(prompt_rows):02d}] {prompt_row.id}: "
                f"prompt={len(prompt_tokens)} tok, gen={len(generated_tokens)} tok, "
                f"prefill={row['prefill_ms']:.1f}ms, decode={row['decode_ms']:.1f}ms/tok, "
                f"wall={row['wall_ms']:.1f}ms, throughput={row['throughput']:.2f} tok/s"
            )

    summary = make_summary_document(result_rows, args)
    args.output_pretty_json.write_text(
        json.dumps(result_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_summary_md.write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    args.output_report_md.write_text(
        render_detailed_markdown(result_rows, summary),
        encoding="utf-8",
    )

    print("\nSummary")
    for split in ("overall", "short", "long"):
        row = summary[split]
        print(
            f"  {split:>7}: rows={row['rows']}, prefill={row['avg_prefill_ms']:.3f}ms, "
            f"decode={row['avg_decode_ms']:.3f}ms/tok, wall={row['avg_wall_ms']:.3f}ms, "
            f"throughput={row['avg_throughput']:.3f} tok/s"
        )
    print(f"\nWrote machine JSONL: {args.output_jsonl}")
    print(f"Wrote pretty JSON:   {args.output_pretty_json}")
    print(f"Wrote summary JSON:  {args.output_summary}")
    print(f"Wrote summary MD:    {args.output_summary_md}")
    print(f"Wrote report MD:     {args.output_report_md}")


if __name__ == "__main__":
    main()