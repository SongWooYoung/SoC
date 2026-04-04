#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import load as mlx_load
from mlx_vlm.generate import stream_generate

from gen_reference_qwen3_5_4b_mlx_8bit import (
    EOS_ENDOFTEXT,
    IM_END,
    build_nothink_chatml_text,
    build_nothink_chatml_tokens,
    load_prompt_suite,
    safe_rate_to_ms,
    summarize_rows,
)


DEFAULT_MODEL_DIR = Path("/Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit")
DEFAULT_PROMPT_SUITE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/prompt_suite.json")
DEFAULT_CUSTOM_BIN = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/build/test_mlx_quantized_output_eval")
DEFAULT_OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_9b_mlx_lib_vs_custom.json")
DEFAULT_OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_9b_mlx_lib_vs_custom.md")
DEFAULT_OUTPUT_ONLY_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_9b_mlx_lib_vs_custom_outputs.md")
DEFAULT_MLX_METAL_PATH = Path("/Volumes/990pro/Documents/SoC/.repo_cache/mlx/build/mlx/backend/metal/kernels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official MLX runtime vs qwen3_5_mlx custom runtime on Qwen3.5-9B-MLX-8bit."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_PROMPT_SUITE)
    parser.add_argument("--custom-bin", type=Path, default=DEFAULT_CUSTOM_BIN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-only-md", type=Path, default=DEFAULT_OUTPUT_ONLY_MD)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--system-message", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mlx-metal-path", type=Path, default=DEFAULT_MLX_METAL_PATH)
    parser.add_argument("--custom-gated-delta-mode", choices=["ops", "compiled_ops", "metal_kernel"], default="metal_kernel")
    parser.add_argument("--custom-linear-cache-mode", choices=["legacy", "arrays"], default="legacy")
    parser.add_argument("--enable-custom-stage-trace", action="store_true")
    return parser.parse_args()


def run_mlx_library_eval(
    model_dir: Path,
    prompt_suite: Path,
    max_new_tokens: int,
    system_message: str | None,
    limit: int | None,
) -> dict[str, Any]:
    prompt_rows = load_prompt_suite(prompt_suite, limit)
    print(f"Loading MLX library model from {model_dir}...")
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
        last_response = None
        seen_generation_tokens = 0

        for response in responses:
            last_response = response
            if response.generation_tokens > seen_generation_tokens and response.token is not None:
                generated_tokens.append(int(response.token))
                seen_generation_tokens = response.generation_tokens
            if response.text:
                generated_text_segments.append(response.text)

        if last_response is None:
            raise RuntimeError(f"Generation failed for prompt {prompt_row.id}")

        output_text = "".join(generated_text_segments)
        prefill_ms = safe_rate_to_ms(last_response.prompt_tokens, last_response.prompt_tps)
        generation_wall_ms = safe_rate_to_ms(last_response.generation_tokens, last_response.generation_tps)
        decode_ms = generation_wall_ms / last_response.generation_tokens if last_response.generation_tokens > 0 else 0.0
        wall_ms = prefill_ms + generation_wall_ms
        throughput = last_response.generation_tokens / wall_ms * 1000.0 if wall_ms > 0.0 else 0.0

        row = {
            "id": prompt_row.id,
            "kind": prompt_row.kind,
            "prompt_text": prompt_row.prompt_text,
            "templated_text": templated_text,
            "prompt_tokens": prompt_tokens,
            "prompt_token_count": len(prompt_tokens),
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
            f"[mlx_lib {prompt_index:02d}/{len(prompt_rows):02d}] {prompt_row.id}: "
            f"prefill={row['prefill_ms']:.1f}ms, decode={row['decode_ms']:.1f}ms/tok, "
            f"wall={row['wall_ms']:.1f}ms, throughput={row['throughput']:.2f} tok/s"
        )

    return {
        "label": "qwen3_5_9b_mlx_library",
        "backend": "mlx_vlm",
        "model_dir": str(model_dir),
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def run_custom_eval(
    binary: Path,
    model_dir: Path,
    prompt_suite: Path,
    output_path: Path,
    max_new_tokens: int,
    mlx_metal_path: Path,
    gated_delta_mode: str,
    linear_cache_mode: str,
    trace_enabled: bool,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["MLX_METAL_PATH"] = str(mlx_metal_path)
    env["QWEN3_5_MLX_GATED_DELTA_MODE"] = gated_delta_mode
    env["QWEN3_5_MLX_LINEAR_CACHE_MODE"] = linear_cache_mode
    env["QWEN3_5_MLX_STAGE_TRACE"] = "1" if trace_enabled else "0"
    command = [str(binary), str(model_dir), str(prompt_suite), str(output_path), str(max_new_tokens)]
    subprocess.run(command, check=True, env=env)
    with output_path.open(encoding="utf-8") as handle:
        result = json.load(handle)

    rows = result["rows"]
    for row in rows:
        row["prompt_token_count"] = len(row["prompt_tokens"])
        row["peak_memory_gb"] = 0.0

    return {
        "label": "qwen3_5_9b_qwen3_5_mlx",
        "backend": "qwen3_5_mlx",
        "model_dir": str(model_dir),
        "linear_cache_mode": result.get("linear_cache_mode", linear_cache_mode),
        "gated_delta_mode": result.get("gated_delta_mode", gated_delta_mode),
        "trace_enabled": result.get("trace_enabled", trace_enabled),
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def build_prompt_comparisons(library_rows: list[dict[str, Any]], custom_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    custom_by_id = {row["id"]: row for row in custom_rows}
    comparisons: list[dict[str, Any]] = []
    for library_row in library_rows:
        custom_row = custom_by_id[library_row["id"]]
        comparisons.append(
            {
                "id": library_row["id"],
                "kind": library_row["kind"],
                "prompt_text": library_row["prompt_text"],
                "prompt_tokens_match": library_row["prompt_tokens"] == custom_row["prompt_tokens"],
                "generated_tokens_match": library_row["generated_tokens"] == custom_row["generated_tokens"],
                "output_text_match": library_row["output_text"] == custom_row["output_text"],
                "library_wall_ms": library_row["wall_ms"],
                "custom_wall_ms": custom_row["wall_ms"],
                "wall_ms_delta": round(custom_row["wall_ms"] - library_row["wall_ms"], 3),
                "library_throughput": library_row["throughput"],
                "custom_throughput": custom_row["throughput"],
                "throughput_delta": round(custom_row["throughput"] - library_row["throughput"], 3),
                "library_output_text": library_row["output_text"],
                "custom_output_text": custom_row["output_text"],
            }
        )
    return comparisons


def render_markdown(report: dict[str, Any]) -> str:
    mlx_lib = report["mlx_library"]
    custom = report["custom"]
    delta = report["summary_delta"]
    lines = [
        "# Qwen3.5-9B MLX Library vs qwen3_5_mlx",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Prompt suite: {report['prompt_suite']}",
        f"- Model dir: {report['model_dir']}",
        f"- Max new tokens: {report['max_new_tokens']}",
        f"- Custom gated_delta mode: {custom['gated_delta_mode']}",
        f"- Custom linear cache mode: {custom['linear_cache_mode']}",
        f"- Custom stage trace enabled: {custom['trace_enabled']}",
        "",
        "## Summary",
        "",
        "| Backend | Rows | Avg Prompt Tok | Avg Gen Tok | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s | Avg Peak GB |",
        "|--------|-----:|---------------:|------------:|---------------:|------------------:|------------:|----------:|------------:|",
        f"| {mlx_lib['label']} | {mlx_lib['summary']['rows']} | {mlx_lib['summary']['avg_prompt_tokens']:.3f} | {mlx_lib['summary']['avg_generated_tokens']:.3f} | {mlx_lib['summary']['avg_prefill_ms']:.3f} | {mlx_lib['summary']['avg_decode_ms']:.3f} | {mlx_lib['summary']['avg_wall_ms']:.3f} | {mlx_lib['summary']['avg_throughput']:.3f} | {mlx_lib['summary']['avg_peak_memory_gb']:.3f} |",
        f"| {custom['label']} | {custom['summary']['rows']} | {custom['summary']['avg_prompt_tokens']:.3f} | {custom['summary']['avg_generated_tokens']:.3f} | {custom['summary']['avg_prefill_ms']:.3f} | {custom['summary']['avg_decode_ms']:.3f} | {custom['summary']['avg_wall_ms']:.3f} | {custom['summary']['avg_throughput']:.3f} | {custom['summary']['avg_peak_memory_gb']:.3f} |",
        "",
        "## Custom Minus MLX Library",
        "",
        f"- Avg prefill ms delta: {delta['avg_prefill_ms_delta']:.3f}",
        f"- Avg decode ms/tok delta: {delta['avg_decode_ms_delta']:.3f}",
        f"- Avg wall ms delta: {delta['avg_wall_ms_delta']:.3f}",
        f"- Avg throughput delta: {delta['avg_throughput_delta']:.3f}",
        f"- Prompt token matches: {delta['prompt_tokens_match_count']}/{delta['rows']}",
        f"- Generated token matches: {delta['generated_tokens_match_count']}/{delta['rows']}",
        f"- Output text matches: {delta['output_text_match_count']}/{delta['rows']}",
        "",
        "## Per-Prompt",
        "",
        "| Prompt | Kind | MLX Tok/s | Custom Tok/s | Delta Tok/s | MLX Wall ms | Custom Wall ms | Delta Wall ms | Match |",
        "|-------|------|----------:|-------------:|------------:|------------:|---------------:|--------------:|:------|",
    ]
    for row in report["prompt_comparisons"]:
        lines.append(
            f"| {row['id']} | {row['kind']} | {row['library_throughput']:.3f} | {row['custom_throughput']:.3f} | {row['throughput_delta']:.3f} | "
            f"{row['library_wall_ms']:.3f} | {row['custom_wall_ms']:.3f} | {row['wall_ms_delta']:.3f} | {'yes' if row['output_text_match'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def render_output_only_markdown(report: dict[str, Any]) -> str:
    comparisons = report["prompt_comparisons"]
    lines = [
        "# Qwen3.5-9B Output Collection",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Model dir: {report['model_dir']}",
        "",
    ]
    for row in comparisons:
        lines.extend(
            [
                f"## {row['id']} ({row['kind']})",
                "",
                "### mlx_library",
                "",
                row["library_output_text"] or "",
                "",
                "### qwen3_5_mlx",
                "",
                row["custom_output_text"] or "",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_only_md.parent.mkdir(parents=True, exist_ok=True)

    suite_rows = load_prompt_suite(args.prompt_suite, args.limit)
    suite_path = args.prompt_suite
    if args.limit is not None:
        suite_path = args.output_json.with_name(args.output_json.stem + "_suite.json")
        suite_path.write_text(json.dumps([row.__dict__ for row in suite_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    library_output = args.output_json.with_name(args.output_json.stem + "_mlx_library.json")
    custom_output = args.output_json.with_name(args.output_json.stem + "_custom.json")

    mlx_library = run_mlx_library_eval(
        model_dir=args.model_dir,
        prompt_suite=args.prompt_suite,
        max_new_tokens=args.max_new_tokens,
        system_message=args.system_message,
        limit=args.limit,
    )
    library_output.write_text(json.dumps(mlx_library, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    custom = run_custom_eval(
        binary=args.custom_bin,
        model_dir=args.model_dir,
        prompt_suite=suite_path,
        output_path=custom_output,
        max_new_tokens=args.max_new_tokens,
        mlx_metal_path=args.mlx_metal_path,
        gated_delta_mode=args.custom_gated_delta_mode,
        linear_cache_mode=args.custom_linear_cache_mode,
        trace_enabled=args.enable_custom_stage_trace,
    )

    prompt_comparisons = build_prompt_comparisons(mlx_library["rows"], custom["rows"])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_suite": str(suite_path),
        "model_dir": str(args.model_dir),
        "max_new_tokens": args.max_new_tokens,
        "mlx_library": {
            "label": mlx_library["label"],
            "backend": mlx_library["backend"],
            "model_dir": mlx_library["model_dir"],
            "summary": mlx_library["summary"],
        },
        "custom": {
            "label": custom["label"],
            "backend": custom["backend"],
            "model_dir": custom["model_dir"],
            "linear_cache_mode": custom["linear_cache_mode"],
            "gated_delta_mode": custom["gated_delta_mode"],
            "trace_enabled": custom["trace_enabled"],
            "summary": custom["summary"],
        },
        "summary_delta": {
            "rows": len(prompt_comparisons),
            "avg_prefill_ms_delta": round(custom["summary"]["avg_prefill_ms"] - mlx_library["summary"]["avg_prefill_ms"], 3),
            "avg_decode_ms_delta": round(custom["summary"]["avg_decode_ms"] - mlx_library["summary"]["avg_decode_ms"], 3),
            "avg_wall_ms_delta": round(custom["summary"]["avg_wall_ms"] - mlx_library["summary"]["avg_wall_ms"], 3),
            "avg_throughput_delta": round(custom["summary"]["avg_throughput"] - mlx_library["summary"]["avg_throughput"], 3),
            "prompt_tokens_match_count": sum(1 for row in prompt_comparisons if row["prompt_tokens_match"]),
            "generated_tokens_match_count": sum(1 for row in prompt_comparisons if row["generated_tokens_match"]),
            "output_text_match_count": sum(1 for row in prompt_comparisons if row["output_text_match"]),
        },
        "prompt_comparisons": prompt_comparisons,
    }

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    args.output_only_md.write_text(render_output_only_markdown(report), encoding="utf-8")

    print("\nComparison Summary")
    print(
        f"  mlx_library : prefill={mlx_library['summary']['avg_prefill_ms']:.3f}ms, decode={mlx_library['summary']['avg_decode_ms']:.3f}ms/tok, "
        f"wall={mlx_library['summary']['avg_wall_ms']:.3f}ms, throughput={mlx_library['summary']['avg_throughput']:.3f} tok/s"
    )
    print(
        f"  custom      : prefill={custom['summary']['avg_prefill_ms']:.3f}ms, decode={custom['summary']['avg_decode_ms']:.3f}ms/tok, "
        f"wall={custom['summary']['avg_wall_ms']:.3f}ms, throughput={custom['summary']['avg_throughput']:.3f} tok/s"
    )
    print(
        f"  delta       : prefill={report['summary_delta']['avg_prefill_ms_delta']:.3f}ms, decode={report['summary_delta']['avg_decode_ms_delta']:.3f}ms/tok, "
        f"wall={report['summary_delta']['avg_wall_ms_delta']:.3f}ms, throughput={report['summary_delta']['avg_throughput_delta']:.3f} tok/s, "
        f"output_matches={report['summary_delta']['output_text_match_count']}/{report['summary_delta']['rows']}"
    )
    print(f"\nWrote comparison JSON: {args.output_json}")
    print(f"Wrote comparison MD:   {args.output_md}")
    print(f"Wrote output-only MD:  {args.output_only_md}")


if __name__ == "__main__":
    main()