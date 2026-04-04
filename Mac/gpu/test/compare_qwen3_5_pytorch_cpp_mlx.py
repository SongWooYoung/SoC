#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = Path("/Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b")
DEFAULT_PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json")
DEFAULT_CPP_BIN = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/build/test_output_eval")
DEFAULT_MLX_BIN = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/build/test_mlx_output_eval")
DEFAULT_OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_pytorch_cpp_mlx_compare.json")
DEFAULT_OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_pytorch_cpp_mlx_compare.md")
DEFAULT_MLX_METAL_PATH = Path("/Volumes/990pro/Documents/SoC/.repo_cache/mlx/build/mlx/backend/metal/kernels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch baseline vs qwen3_5_py_cpp vs qwen3_5_mlx on the same Qwen3.5-4B model."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_PROMPT_SUITE)
    parser.add_argument("--cpp-bin", type=Path, default=DEFAULT_CPP_BIN)
    parser.add_argument("--mlx-bin", type=Path, default=DEFAULT_MLX_BIN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mlx-metal-path", type=Path, default=DEFAULT_MLX_METAL_PATH)
    return parser.parse_args()


def load_prompt_suite(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return rows[:limit] if limit is not None else rows


def build_nothink_chatml_tokens(tokenizer: Any, user_message: str) -> list[int]:
    im_start = 248045
    im_end = 248046
    think = 248068
    think_end = 248069
    newline = tokenizer.encode("\n", add_special_tokens=False)
    double_newline = tokenizer.encode("\n\n", add_special_tokens=False)

    ids: list[int] = [im_start]
    ids.extend(tokenizer.encode("user", add_special_tokens=False))
    ids.extend(newline)
    ids.extend(tokenizer.encode(user_message, add_special_tokens=False))
    ids.append(im_end)
    ids.extend(newline)
    ids.append(im_start)
    ids.extend(tokenizer.encode("assistant", add_special_tokens=False))
    ids.extend(newline)
    ids.append(think)
    ids.extend(double_newline)
    ids.append(think_end)
    ids.extend(double_newline)
    return ids


def maybe_sync_torch(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def run_pytorch_eval(
    model_dir: Path,
    suite_rows: list[dict[str, Any]],
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    torch_dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    eos_ids = {248044, 248046}
    rows: list[dict[str, Any]] = []
    for row in suite_rows:
        prompt_tokens = build_nothink_chatml_tokens(tokenizer, row["prompt_text"])
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        maybe_sync_torch(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        maybe_sync_torch(device)
        t1 = time.perf_counter()

        past_key_values = outputs.past_key_values
        next_token = int(torch.argmax(outputs.logits[0, -1, :]).item())
        generated_tokens = [next_token]

        maybe_sync_torch(device)
        t2 = time.perf_counter()
        with torch.no_grad():
            while len(generated_tokens) < max_new_tokens and next_token not in eos_ids:
                step_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
                step_outputs = model(
                    input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = step_outputs.past_key_values
                next_token = int(torch.argmax(step_outputs.logits[0, -1, :]).item())
                generated_tokens.append(next_token)
        maybe_sync_torch(device)
        t3 = time.perf_counter()

        prefill_ms = (t1 - t0) * 1000.0
        decode_total_ms = (t3 - t2) * 1000.0
        wall_ms = prefill_ms + decode_total_ms
        decode_ms = decode_total_ms / len(generated_tokens) if generated_tokens else 0.0
        throughput = len(generated_tokens) * 1000.0 / wall_ms if wall_ms > 0.0 else 0.0
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        rows.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "prompt_text": row["prompt_text"],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "generated_token_count": len(generated_tokens),
                "output_text": output_text,
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "wall_ms": round(wall_ms, 3),
                "throughput": round(throughput, 3),
            }
        )
        print(
            f"[pytorch] {row['id']} done: {len(generated_tokens)} tokens, prefill={prefill_ms:.1f}ms, "
            f"decode={decode_ms:.1f}ms/tok, wall={wall_ms:.1f}ms, throughput={throughput:.2f} tok/s",
            file=sys.stderr,
        )

    return {"mode": "pytorch", "model_dir": str(model_dir), "rows": rows}


def run_binary_eval(
    binary: Path,
    model_dir: Path,
    prompt_suite: Path,
    output_path: Path,
    max_new_tokens: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    command = [str(binary), str(model_dir), str(prompt_suite), str(output_path), str(max_new_tokens)]
    subprocess.run(command, check=True, env=env)
    with output_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "rows": 0,
            "avg_generated_token_count": 0.0,
            "avg_prefill_ms": 0.0,
            "avg_decode_ms": 0.0,
            "avg_wall_ms": 0.0,
            "avg_throughput": 0.0,
        }

    def avg(key: str) -> float:
        return round(sum(float(row[key]) for row in rows) / len(rows), 3)

    return {
        "rows": len(rows),
        "avg_generated_token_count": avg("generated_token_count"),
        "avg_prefill_ms": avg("prefill_ms"),
        "avg_decode_ms": avg("decode_ms"),
        "avg_wall_ms": avg("wall_ms"),
        "avg_throughput": avg("throughput"),
    }


def compare_against_base(
    base_rows: list[dict[str, Any]],
    other_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    other_by_id = {row["id"]: row for row in other_rows}
    comparisons = []
    for base_row in base_rows:
        other_row = other_by_id[base_row["id"]]
        comparisons.append(
            {
                "id": base_row["id"],
                "kind": base_row["kind"],
                "prompt_tokens_match": base_row["prompt_tokens"] == other_row["prompt_tokens"],
                "generated_tokens_match": base_row["generated_tokens"] == other_row["generated_tokens"],
                "output_text_match": base_row["output_text"] == other_row["output_text"],
                "base_wall_ms": base_row["wall_ms"],
                "other_wall_ms": other_row["wall_ms"],
                "wall_ms_delta": round(other_row["wall_ms"] - base_row["wall_ms"], 3),
                "base_throughput": base_row["throughput"],
                "other_throughput": other_row["throughput"],
                "throughput_delta": round(other_row["throughput"] - base_row["throughput"], 3),
            }
        )
    return comparisons


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5-4B Custom Backend Comparison",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Model dir: {report['model_dir']}",
        f"- Prompt suite: {report['prompt_suite']}",
        f"- Max new tokens: {report['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| Backend | Rows | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |",
        "|--------|-----:|------------:|---------------:|------------------:|------------:|----------:|",
    ]
    for name in ("base", "cpp", "mlx"):
        row = report["summaries"][name]
        lines.append(
            f"| {name} | {row['rows']} | {row['avg_generated_token_count']:.3f} | {row['avg_prefill_ms']:.3f} | {row['avg_decode_ms']:.3f} | {row['avg_wall_ms']:.3f} | {row['avg_throughput']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Match Counts Vs Base",
            "",
            f"- cpp prompt token matches: {report['match_summary']['cpp_prompt_tokens_match_count']}/{report['match_summary']['rows']}",
            f"- cpp generated token matches: {report['match_summary']['cpp_generated_tokens_match_count']}/{report['match_summary']['rows']}",
            f"- cpp output text matches: {report['match_summary']['cpp_output_text_match_count']}/{report['match_summary']['rows']}",
            f"- mlx prompt token matches: {report['match_summary']['mlx_prompt_tokens_match_count']}/{report['match_summary']['rows']}",
            f"- mlx generated token matches: {report['match_summary']['mlx_generated_tokens_match_count']}/{report['match_summary']['rows']}",
            f"- mlx output text matches: {report['match_summary']['mlx_output_text_match_count']}/{report['match_summary']['rows']}",
            "",
            "## Per-Prompt",
            "",
            "| Prompt | Kind | Base Tok/s | Cpp Tok/s | MLX Tok/s | Cpp Match | MLX Match |",
            "|-------|------|-----------:|----------:|----------:|:---------:|:---------:|",
        ]
    )

    cpp_by_id = {row["id"]: row for row in report["comparisons"]["cpp"]}
    mlx_by_id = {row["id"]: row for row in report["comparisons"]["mlx"]}
    base_by_id = {row["id"]: row for row in report["rows"]["base"]}
    for prompt_id, base_row in base_by_id.items():
        cpp_row = cpp_by_id[prompt_id]
        mlx_row = mlx_by_id[prompt_id]
        lines.append(
            f"| {prompt_id} | {base_row['kind']} | {base_row['throughput']:.3f} | {cpp_row['other_throughput']:.3f} | {mlx_row['other_throughput']:.3f} | {'yes' if cpp_row['output_text_match'] else 'no'} | {'yes' if mlx_row['output_text_match'] else 'no'} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    suite_rows = load_prompt_suite(args.prompt_suite, args.limit)
    suite_path = args.prompt_suite
    if args.limit is not None:
        suite_path = args.output_json.with_name(args.output_json.stem + "_suite.json")
        suite_path.write_text(json.dumps(suite_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base_output = args.output_json.with_name(args.output_json.stem + "_base.json")
    cpp_output = args.output_json.with_name(args.output_json.stem + "_cpp.json")
    mlx_output = args.output_json.with_name(args.output_json.stem + "_mlx.json")

    base_result = run_pytorch_eval(args.model_dir, suite_rows, args.device, args.max_new_tokens)
    base_output.write_text(json.dumps(base_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cpp_result = run_binary_eval(args.cpp_bin, args.model_dir, suite_path, cpp_output, args.max_new_tokens)

    mlx_env = dict(os.environ)
    mlx_env["MLX_METAL_PATH"] = str(args.mlx_metal_path)
    mlx_result = run_binary_eval(args.mlx_bin, args.model_dir, suite_path, mlx_output, args.max_new_tokens, env=mlx_env)

    cpp_comparisons = compare_against_base(base_result["rows"], cpp_result["rows"])
    mlx_comparisons = compare_against_base(base_result["rows"], mlx_result["rows"])

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir),
        "prompt_suite": str(suite_path),
        "max_new_tokens": args.max_new_tokens,
        "summaries": {
            "base": summarize_rows(base_result["rows"]),
            "cpp": summarize_rows(cpp_result["rows"]),
            "mlx": summarize_rows(mlx_result["rows"]),
        },
        "match_summary": {
            "rows": len(base_result["rows"]),
            "cpp_prompt_tokens_match_count": sum(1 for row in cpp_comparisons if row["prompt_tokens_match"]),
            "cpp_generated_tokens_match_count": sum(1 for row in cpp_comparisons if row["generated_tokens_match"]),
            "cpp_output_text_match_count": sum(1 for row in cpp_comparisons if row["output_text_match"]),
            "mlx_prompt_tokens_match_count": sum(1 for row in mlx_comparisons if row["prompt_tokens_match"]),
            "mlx_generated_tokens_match_count": sum(1 for row in mlx_comparisons if row["generated_tokens_match"]),
            "mlx_output_text_match_count": sum(1 for row in mlx_comparisons if row["output_text_match"]),
        },
        "rows": {
            "base": base_result["rows"],
            "cpp": cpp_result["rows"],
            "mlx": mlx_result["rows"],
        },
        "comparisons": {
            "cpp": cpp_comparisons,
            "mlx": mlx_comparisons,
        },
    }

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"summaries": report["summaries"], "match_summary": report["match_summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()