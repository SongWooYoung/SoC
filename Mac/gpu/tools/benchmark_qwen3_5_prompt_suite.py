#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def render_markdown(rows: list[dict], source_name: str) -> str:
    lines: list[str] = []
    lines.append("# Prompt Suite Results")
    lines.append("")
    lines.append(f"- source: `{source_name}`")
    lines.append(f"- rows: `{len(rows)}`")
    lines.append("")
    for row in rows:
        lines.append(f"## {row['id']} ({row['category']})")
        lines.append("")
        lines.append("### Prompt")
        lines.append("")
        lines.append("```text")
        lines.append(row["prompt"])
        lines.append("```")
        lines.append("")
        lines.append("### Base")
        lines.append("")
        if row["base"].get("prepared_prompt"):
            lines.append("#### Prepared Prompt")
            lines.append("")
            lines.append("```text")
            lines.append(row["base"]["prepared_prompt"])
            lines.append("```")
            lines.append("")
        lines.append(f"- generated_token_ids: `{row['base']['generated_token_ids']}`")
        lines.append(f"- prefill_ms: `{row['base']['prefill_ms']}`")
        lines.append(f"- decode_ms: `{row['base']['decode_ms']}`")
        lines.append(f"- decode_tok_per_s: `{row['base']['decode_tok_per_s']}`")
        lines.append(f"- wall_ms: `{row['base']['wall_ms']}`")
        lines.append("")
        lines.append("```text")
        lines.append(row["base"]["generated_text"])
        lines.append("```")
        lines.append("")
        lines.append("### Cpp")
        lines.append("")
        if row["cpp"].get("prepared_prompt"):
            lines.append("#### Prepared Prompt")
            lines.append("")
            lines.append("```text")
            lines.append(row["cpp"]["prepared_prompt"])
            lines.append("```")
            lines.append("")
        lines.append(f"- generated_token_ids: `{row['cpp']['generated_token_ids']}`")
        lines.append(f"- prefill_ms: `{row['cpp']['prefill_ms']}`")
        lines.append(f"- decode_ms: `{row['cpp']['decode_ms']}`")
        lines.append(f"- decode_tok_per_s: `{row['cpp']['decode_tok_per_s']}`")
        lines.append(f"- wall_ms: `{row['cpp']['wall_ms']}`")
        lines.append(f"- gpu_ms: `{row['cpp']['gpu_ms']}`")
        lines.append(f"- wait_ms: `{row['cpp']['wait_ms']}`")
        lines.append("")
        lines.append("```text")
        lines.append(row["cpp"]["generated_text"])
        lines.append("```")
        lines.append("")
        lines.append("### Comparison")
        lines.append("")
        lines.append(f"- text_match: `{row['comparison']['text_match']}`")
        lines.append(f"- token_ids_match: `{row['comparison']['token_ids_match']}`")
        lines.append(f"- prefill_ms_ratio_cpp_over_base: `{row['comparison']['prefill_ms_ratio_cpp_over_base']}`")
        lines.append(f"- decode_tok_per_s_ratio_cpp_over_base: `{row['comparison']['decode_tok_per_s_ratio_cpp_over_base']}`")
        lines.append(f"- wall_ms_ratio_cpp_over_base: `{row['comparison']['wall_ms_ratio_cpp_over_base']}`")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def refresh_pretty_outputs(rows: list[dict], source_jsonl: Path, output_json: Path | None, output_md: Path | None) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(rows, source_jsonl.name), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare qwen3_5 C++ runtime vs PyTorch MPS over a prompt suite.")
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--infer-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--model-type", default="qwen3_5")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="current")
    parser.add_argument("--cpp-env", action="append", default=[])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def load_input_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def maybe_sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def build_qwen3_5_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def resolve_effective_max_new_tokens(prompt_tokens: int, max_new_tokens: int, max_seq_len: int) -> int:
    if max_new_tokens != 0:
        return max_new_tokens
    return max(0, max_seq_len - prompt_tokens)


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, generator: torch.Generator) -> torch.Tensor:
    if top_k <= 1:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / max(temperature, 1e-5)
    k = min(top_k, scaled.shape[-1])
    top_values, top_indices = torch.topk(scaled, k=k, dim=-1)
    probs = F.softmax(top_values, dim=-1)
    sampled_rank = torch.multinomial(probs, num_samples=1, generator=generator)
    return torch.gather(top_indices, -1, sampled_rank)


def run_base_case(model,
                  tokenizer,
                  device: str,
                  prompt: str,
                  max_new_tokens: int,
                  max_seq_len: int,
                  temperature: float,
                  top_k: int) -> dict:
    prepared_prompt = build_qwen3_5_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer.encode(prepared_prompt, return_tensors="pt").to(device)
    prompt_tokens = int(input_ids.shape[1])
    effective_max_new_tokens = resolve_effective_max_new_tokens(prompt_tokens, max_new_tokens, max_seq_len)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)

    maybe_sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        next_token = sample_next_token(next_logits.float().cpu(), temperature, top_k, generator).to(device)
    maybe_sync(device)
    t1 = time.perf_counter()

    generated_ids = [int(next_token.item())]
    current_token = next_token

    decode_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(effective_max_new_tokens - 1, 0)):
            outputs = model(current_token, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            current_token = sample_next_token(next_logits.float().cpu(), temperature, top_k, generator).to(device)
            generated_ids.append(int(current_token.item()))
            if tokenizer.eos_token_id is not None and int(current_token.item()) == int(tokenizer.eos_token_id):
                break
    maybe_sync(device)
    decode_end = time.perf_counter()

    prefill_ms = (t1 - t0) * 1000.0
    decode_ms = (decode_end - decode_start) * 1000.0
    decode_tok_per_s = len(generated_ids) / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    prefill_tok_per_s = prompt_tokens / (prefill_ms / 1000.0) if prefill_ms > 0 else 0.0

    return {
        "prepared_prompt": prepared_prompt,
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "prompt_tokens": prompt_tokens,
        "prefill_ms": round(prefill_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "prefill_tok_per_s": round(prefill_tok_per_s, 3),
        "decode_tok_per_s": round(decode_tok_per_s, 3),
        "wall_ms": round(prefill_ms + decode_ms, 3),
    }


def run_cpp_case(args: argparse.Namespace, prompt: str) -> dict:
    command = [
        str(args.infer_bin),
        "--manifest",
        str(args.manifest),
        "--model-type",
        args.model_type,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-seq-len",
        str(args.max_seq_len),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--profiling",
        "summary",
        "--json",
    ]
    env = os.environ.copy()
    for item in args.cpp_env:
        key, value = item.split("=", 1)
        env[key] = value
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def simplify_cpp_report(report: dict) -> dict:
    timing = report.get("timing", {})
    generated_ids = report.get("generated_token_ids", [])
    prompt_token_ids = report.get("prompt_token_ids", [])
    prefill_ms = timing.get("prefill_ms")
    decode_ms = timing.get("decode_ms")
    wall_ms = timing.get("wall_ms")
    return {
        "prepared_prompt": report.get("serialized_prompt", report.get("prepared_prompt", "")),
        "generated_token_ids": generated_ids,
        "generated_token_count": len(generated_ids),
        "generated_text": report.get("generated_text", ""),
        "prompt_tokens": len(prompt_token_ids),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prefill_tok_per_s": timing.get("prefill_tok_per_s"),
        "decode_tok_per_s": timing.get("decode_tok_per_s"),
        "wall_ms": wall_ms,
        "gpu_ms": timing.get("gpu_ms"),
        "wait_ms": timing.get("wait_ms"),
        "command_buffer_count": timing.get("command_buffer_count"),
        "encoder_count": timing.get("encoder_count"),
        "prefill_ms_per_token": round(float(prefill_ms) / float(len(prompt_token_ids)), 6) if prefill_ms and prompt_token_ids else None,
        "decode_ms_per_token": round(float(decode_ms) / float(len(generated_ids)), 6) if decode_ms and generated_ids else None,
        "wall_ms_per_token": round(float(wall_ms) / float(len(generated_ids)), 6) if wall_ms and generated_ids else None,
    }


def main() -> None:
    args = parse_args()
    rows = load_input_rows(args.input_list, args.limit)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.hf_model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model))
    print(f"Loading base model to {args.device} ({args.dtype}) ...", file=sys.stderr)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(args.hf_model), dtype=dtype_from_name(args.dtype))
    model = model.to(args.device)
    model.eval()
    maybe_sync(args.device)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    print(f"Base model loaded in {load_ms:.1f} ms", file=sys.stderr)

    rendered_rows: list[dict] = []
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            prompt = row["prompt"]
            print(f"[{index}/{len(rows)}] {row['id']}", file=sys.stderr)
            base_result = run_base_case(model,
                                        tokenizer,
                                        args.device,
                                        prompt,
                                        args.max_new_tokens,
                                        args.max_seq_len,
                                        args.temperature,
                                        args.top_k)
            cpp_result = simplify_cpp_report(run_cpp_case(args, prompt))
            base_generated_count = max(1, len(base_result["generated_token_ids"]))
            cpp_generated_count = max(1, len(cpp_result["generated_token_ids"]))
            result = {
                "id": row["id"],
                "category": row["category"],
                "prompt": prompt,
                "tag": args.tag,
                "base_model_load_ms": round(load_ms, 3),
                "cpp_env": args.cpp_env,
                "base": base_result,
                "cpp": cpp_result,
                "comparison": {
                    "text_match": base_result["generated_text"] == cpp_result["generated_text"],
                    "token_ids_match": base_result["generated_token_ids"] == cpp_result["generated_token_ids"],
                    "prefill_ms": None if cpp_result["prefill_ms"] is None else round(float(cpp_result["prefill_ms"]) - float(base_result["prefill_ms"]), 3),
                    "decode_ms": None if cpp_result["decode_ms"] is None else round(float(cpp_result["decode_ms"]) - float(base_result["decode_ms"]), 3),
                    "decode_tok_per_s": None if cpp_result["decode_tok_per_s"] is None else round(float(cpp_result["decode_tok_per_s"]) - float(base_result["decode_tok_per_s"]), 3),
                    "wall_ms": None if cpp_result["wall_ms"] is None else round(float(cpp_result["wall_ms"]) - float(base_result["wall_ms"]), 3),
                    "prefill_ms_ratio_cpp_over_base": None if cpp_result["prefill_ms"] in (None, 0) or base_result["prefill_ms"] == 0 else round(float(cpp_result["prefill_ms"]) / float(base_result["prefill_ms"]), 6),
                    "decode_ms_ratio_cpp_over_base": None if cpp_result["decode_ms"] in (None, 0) or base_result["decode_ms"] == 0 else round(float(cpp_result["decode_ms"]) / float(base_result["decode_ms"]), 6),
                    "decode_tok_per_s_ratio_cpp_over_base": None if cpp_result["decode_tok_per_s"] in (None, 0) or base_result["decode_tok_per_s"] == 0 else round(float(cpp_result["decode_tok_per_s"]) / float(base_result["decode_tok_per_s"]), 6),
                    "wall_ms_ratio_cpp_over_base": None if cpp_result["wall_ms"] in (None, 0) or base_result["wall_ms"] == 0 else round(float(cpp_result["wall_ms"]) / float(base_result["wall_ms"]), 6),
                    "base_answer": base_result["generated_text"],
                    "cpp_answer": cpp_result["generated_text"],
                    "base_generated_token_count": base_generated_count,
                    "cpp_generated_token_count": cpp_generated_count,
                },
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            rendered_rows.append(result)
            refresh_pretty_outputs(rendered_rows, args.output_jsonl, args.output_json, args.output_md)


if __name__ == "__main__":
    main()
