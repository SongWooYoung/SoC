#!/usr/bin/env python3
"""Benchmark Qwen3-0.6B on Apple Metal (MPS) via PyTorch / HuggingFace Transformers.

Measures:
  - Model load time
  - Prefill latency & tok/s
  - Decode latency & tok/s
  - Peak memory

Outputs a JSON report for comparison with the custom C++/Metal engine.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
    num_warmup: int,
    num_runs: int,
) -> dict:
    results: dict = {
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "num_warmup": num_warmup,
        "num_runs": num_runs,
    }

    # ── Load ──────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_id} …", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model to {device} ({dtype}) …", file=sys.stderr)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    results["load_time_s"] = round(load_time, 3)
    print(f"  Loaded in {load_time:.2f}s", file=sys.stderr)

    # ── Model info ────────────────────────────────────────────────────
    config = model.config
    results["model_config"] = {
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    param_count = sum(p.numel() for p in model.parameters())
    results["param_count"] = param_count
    results["param_count_m"] = round(param_count / 1e6, 1)

    # ── Tokenize ──────────────────────────────────────────────────────
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    results["prompt_tokens"] = prompt_len
    print(f"  Prompt tokens: {prompt_len}", file=sys.stderr)

    # ── Warmup ────────────────────────────────────────────────────────
    print(f"Warmup ({num_warmup} runs) …", file=sys.stderr)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_k=1,
            )
        if device == "mps":
            torch.mps.synchronize()

    # ── Timed runs ────────────────────────────────────────────────────
    print(f"Benchmarking ({num_runs} runs) …", file=sys.stderr)
    decode_token_counts = []
    total_times = []
    prefill_times = []
    decode_times = []
    generated_texts = []

    for run_idx in range(num_runs):
        # --- Prefill timing ---
        if device == "mps":
            torch.mps.synchronize()
        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if device == "mps":
            torch.mps.synchronize()
        t_prefill_end = time.perf_counter()

        prefill_time = t_prefill_end - t_prefill_start
        prefill_times.append(prefill_time)

        # --- Decode timing (autoregressive) ---
        generated_ids = [next_token.item()]
        current_token = next_token
        t_decode_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                outputs = model(current_token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                current_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(current_token.item())
                if current_token.item() == tokenizer.eos_token_id:
                    break
        if device == "mps":
            torch.mps.synchronize()
        t_decode_end = time.perf_counter()

        decode_time = t_decode_end - t_decode_start
        decode_tokens = len(generated_ids)
        decode_token_counts.append(decode_tokens)
        decode_times.append(decode_time)
        total_times.append(prefill_time + decode_time)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_texts.append(text)
        print(f"  Run {run_idx+1}: prefill={prefill_time*1000:.1f}ms, "
              f"decode={decode_tokens}tok in {decode_time*1000:.1f}ms "
              f"({decode_tokens/decode_time:.1f} tok/s)", file=sys.stderr)

    # ── Aggregate ─────────────────────────────────────────────────────
    avg_prefill = sum(prefill_times) / num_runs
    avg_decode_time = sum(decode_times) / num_runs
    avg_decode_tokens = sum(decode_token_counts) / num_runs
    avg_total = sum(total_times) / num_runs

    results["prefill_ms"] = round(avg_prefill * 1000, 2)
    results["prefill_tok_per_s"] = round(prompt_len / avg_prefill, 1)
    results["decode_tokens"] = round(avg_decode_tokens, 1)
    results["decode_time_ms"] = round(avg_decode_time * 1000, 2)
    results["decode_tok_per_s"] = round(avg_decode_tokens / avg_decode_time, 2)
    results["total_time_ms"] = round(avg_total * 1000, 2)
    results["generated_text"] = generated_texts[-1]

    # Per-run detail
    results["runs"] = []
    for i in range(num_runs):
        results["runs"].append({
            "prefill_ms": round(prefill_times[i] * 1000, 2),
            "decode_tokens": decode_token_counts[i],
            "decode_ms": round(decode_times[i] * 1000, 2),
            "decode_tok_per_s": round(decode_token_counts[i] / decode_times[i], 2),
        })

    # Memory
    if device == "mps":
        try:
            results["mps_peak_memory_mb"] = round(
                torch.mps.driver_allocated_memory() / 1024 / 1024, 1
            )
        except Exception:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-0.6B on Metal MPS")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model ID")
    parser.add_argument("--prompt", default="Hello",
                        help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("--device", default="mps",
                        choices=["mps", "cpu"],
                        help="Device to run on")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=5,
                        help="Timed iterations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    results = benchmark(
        model_id=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=dtype_map[args.dtype],
        num_warmup=args.warmup,
        num_runs=args.runs,
    )

    output_json = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json + "\n")
        print(f"\nResults saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Model:           {results['model_id']}", file=sys.stderr)
    print(f"  Device/Dtype:    {results['device']} / {results['dtype']}", file=sys.stderr)
    print(f"  Params:          {results['param_count_m']}M", file=sys.stderr)
    print(f"  Prompt tokens:   {results['prompt_tokens']}", file=sys.stderr)
    print(f"  Prefill:         {results['prefill_ms']:.1f} ms "
          f"({results['prefill_tok_per_s']:.0f} tok/s)", file=sys.stderr)
    print(f"  Decode:          {results['decode_tok_per_s']:.1f} tok/s "
          f"({results['decode_tokens']:.0f} tokens in {results['decode_time_ms']:.0f} ms)",
          file=sys.stderr)
    if "mps_peak_memory_mb" in results:
        print(f"  MPS Memory:      {results['mps_peak_memory_mb']:.0f} MB", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
