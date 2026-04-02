#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate qwen3_5 base vs cpp output quality on a prompt list.")
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--infer-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--model-type", default="qwen3_5")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
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


def load_rows(path: Path, limit: int) -> list[dict]:
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


def build_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def resolve_effective_max_new_tokens(prompt_tokens: int, max_new_tokens: int, max_seq_len: int) -> int:
    if max_new_tokens != 0:
        return max_new_tokens
    return max(0, max_seq_len - prompt_tokens)


def greedy_or_topk_sample(logits: torch.Tensor, temperature: float, top_k: int, generator: torch.Generator) -> torch.Tensor:
    if top_k <= 1 or temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / max(temperature, 1e-5)
    k = min(top_k, scaled.shape[-1])
    top_values, top_indices = torch.topk(scaled, k=k, dim=-1)
    probs = torch.softmax(top_values, dim=-1)
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
    prepared_prompt = build_prompt(tokenizer, prompt)
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
        current_token = greedy_or_topk_sample(outputs.logits[:, -1, :].float().cpu(), temperature, top_k, generator).to(device)
    maybe_sync(device)
    t1 = time.perf_counter()

    generated_ids = [int(current_token.item())]
    decode_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(effective_max_new_tokens - 1, 0)):
            outputs = model(current_token, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            current_token = greedy_or_topk_sample(outputs.logits[:, -1, :].float().cpu(), temperature, top_k, generator).to(device)
            generated_ids.append(int(current_token.item()))
            if tokenizer.eos_token_id is not None and int(current_token.item()) == int(tokenizer.eos_token_id):
                break
    maybe_sync(device)
    decode_end = time.perf_counter()

    prefill_ms = (t1 - t0) * 1000.0
    decode_ms = (decode_end - decode_start) * 1000.0
    return {
        "prepared_prompt": prepared_prompt,
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "prompt_tokens": prompt_tokens,
        "prefill_ms": round(prefill_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "decode_tok_per_s": round(len(generated_ids) / max(decode_ms / 1000.0, 1e-9), 3),
        "wall_ms": round(prefill_ms + decode_ms, 3),
    }


def run_cpp_case(args: argparse.Namespace, prompt: str) -> dict:
    command = [
        str(args.infer_bin),
        "--manifest", str(args.manifest),
        "--model-type", args.model_type,
        "--prompt", prompt,
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-seq-len", str(args.max_seq_len),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--json",
        "--profiling", "summary",
    ]
    env = dict(**subprocess.os.environ)
    for item in args.cpp_env:
        key, value = item.split("=", 1)
        env[key] = value
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    report = json.loads(completed.stdout)
    timing = report.get("timing", {})
    return {
        "prepared_prompt": report.get("serialized_prompt", ""),
        "generated_token_ids": report.get("generated_token_ids", []),
        "generated_text": report.get("generated_text", ""),
        "prompt_tokens": len(report.get("prompt_token_ids", [])),
        "prefill_ms": timing.get("prefill_ms"),
        "decode_ms": timing.get("decode_ms"),
        "decode_tok_per_s": timing.get("decode_tok_per_s"),
        "wall_ms": timing.get("wall_ms"),
        "gpu_ms": timing.get("gpu_ms"),
        "wait_ms": timing.get("wait_ms"),
        "command_buffer_count": timing.get("command_buffer_count"),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_list, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.hf_model), dtype=dtype_from_name(args.dtype))
    model = model.to(args.device)
    model.eval()

    results: list[dict] = []

    def write_payload() -> None:
        summary = {
            "tag": args.tag,
            "model_type": args.model_type,
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": args.max_seq_len,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "rows": len(results),
            "prepared_prompt_match_count": sum(1 for row in results if row["comparison"]["prepared_prompt_match"]),
            "text_match_count": sum(1 for row in results if row["comparison"]["text_match"]),
            "token_ids_match_count": sum(1 for row in results if row["comparison"]["token_ids_match"]),
            "base_avg_decode_tok_per_s": round(sum(float(row["base"]["decode_tok_per_s"]) for row in results) / max(len(results), 1), 6),
            "cpp_avg_decode_tok_per_s": round(sum(float(row["cpp"]["decode_tok_per_s"] or 0.0) for row in results) / max(len(results), 1), 6),
            "base_avg_prefill_ms": round(sum(float(row["base"]["prefill_ms"]) for row in results) / max(len(results), 1), 6),
            "cpp_avg_prefill_ms": round(sum(float(row["cpp"]["prefill_ms"] or 0.0) for row in results) / max(len(results), 1), 6),
            "base_avg_wall_ms": round(sum(float(row["base"]["wall_ms"]) for row in results) / max(len(results), 1), 6),
            "cpp_avg_wall_ms": round(sum(float(row["cpp"]["wall_ms"] or 0.0) for row in results) / max(len(results), 1), 6),
        }
        payload = {
            "summary": summary,
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['id']}", file=sys.stderr)
        base = run_base_case(model,
                             tokenizer,
                             args.device,
                             row["prompt"],
                             args.max_new_tokens,
                             args.max_seq_len,
                             args.temperature,
                             args.top_k)
        cpp = run_cpp_case(args, row["prompt"])
        results.append({
            "id": row["id"],
            "category": row["category"],
            "prompt": row["prompt"],
            "base": base,
            "cpp": cpp,
            "comparison": {
                "prepared_prompt_match": base["prepared_prompt"] == cpp["prepared_prompt"],
                "text_match": base["generated_text"] == cpp["generated_text"],
                "token_ids_match": base["generated_token_ids"] == cpp["generated_token_ids"],
            },
        })
        write_payload()


if __name__ == "__main__":
    main()
