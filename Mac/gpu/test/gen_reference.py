#!/usr/bin/env python3
"""Generate reference outputs from Qwen3.5-4B via HuggingFace Transformers.

Produces a JSON file with:
  - prompt_text: the raw user prompt
  - prompt_tokens: token IDs after chat template (non-thinking)
  - generated_tokens: full output token IDs (prompt + generated)
  - generated_text: decoded output
  - prefill_logits_top10: top10 (token_id, logit) from last prompt position
  - prefill_logits_argmax: argmax token ID from prefill

Usage:
  python gen_reference.py --model-dir <path> --output reference.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


IM_START = 248045
IM_END = 248046
THINK = 248068
THINK_END = 248069


def build_nothink_chatml_text(user_message: str, system_message: str | None = None) -> str:
    parts: list[str] = []
    if system_message:
        parts.append(f"<|im_start|>system\n{system_message}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_message}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def build_nothink_chatml_tokens(tokenizer, user_message: str, system_message: str | None = None) -> list[int]:
    ids: list[int] = []
    newline = tokenizer.encode("\n", add_special_tokens=False)
    double_newline = tokenizer.encode("\n\n", add_special_tokens=False)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--output", default="reference.json", help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # ── Load tokenizer ────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {model_dir}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # ── Build prompt with chat template (non-thinking) ────────────────
    prompts = [
        "Hello",
        "What is 2+2?",
        "Write a haiku about the moon.",
    ]

    results = []

    for prompt_text in prompts:
        templated = build_nothink_chatml_text(prompt_text)
        prompt_ids = build_nothink_chatml_tokens(tokenizer, prompt_text)

        print(f"\nPrompt: {prompt_text!r}", file=sys.stderr)
        print(f"  Templated: {templated!r}", file=sys.stderr)
        print(f"  Token IDs ({len(prompt_ids)}): {prompt_ids}", file=sys.stderr)

        results.append({
            "prompt_text": prompt_text,
            "templated_text": templated,
            "prompt_tokens": prompt_ids,
        })

    # ── Load model ────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM
    print(f"\nLoading model to {args.device}...", file=sys.stderr)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"  Model loaded in {load_s:.1f}s", file=sys.stderr)

    # ── Generate for each prompt ──────────────────────────────────────
    for entry in results:
        prompt_ids = entry["prompt_tokens"]
        prompt_text = entry["prompt_text"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

        # 1. Prefill: single forward for logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            # Last token logits
            last_logits = outputs.logits[0, -1, :].float().cpu().numpy()

        argmax_id = int(np.argmax(last_logits))
        # Top-10
        top10_idx = np.argsort(last_logits)[-10:][::-1]
        top10 = [(int(idx), float(last_logits[idx])) for idx in top10_idx]

        entry["prefill_logits_argmax"] = argmax_id
        entry["prefill_logits_top10"] = top10

        print(f"\n  [{prompt_text!r}] Prefill argmax: {argmax_id} "
              f"({tokenizer.decode([argmax_id])!r})", file=sys.stderr)
        for tid, val in top10[:5]:
            print(f"    {tid:>6d} ({tokenizer.decode([tid]):>10s}): {val:.4f}",
                  file=sys.stderr)

        # 2. Full greedy generation
        t0 = time.perf_counter()
        with torch.no_grad():
            gen_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy
                temperature=1.0,
                eos_token_id=[
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                ],
            )
        gen_s = time.perf_counter() - t0

        all_ids = gen_output[0].tolist()
        new_ids = all_ids[len(prompt_ids):]
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        entry["generated_tokens"] = all_ids
        entry["new_tokens"] = new_ids
        entry["generated_text"] = gen_text
        entry["num_new_tokens"] = len(new_ids)
        entry["generation_time_ms"] = round(gen_s * 1000, 1)

        print(f"  Generated {len(new_ids)} tokens in {gen_s*1000:.1f}ms",
              file=sys.stderr)
        print(f"  Text: {gen_text!r}", file=sys.stderr)

    # ── Save ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
