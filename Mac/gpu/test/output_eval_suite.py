#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt_suite(path: Path):
    with path.open() as f:
        return json.load(f)


def run_cpp_eval(binary: Path, model_dir: Path, prompt_suite: Path, output_path: Path, max_new_tokens: int):
    command = [str(binary), str(model_dir), str(prompt_suite), str(output_path), str(max_new_tokens)]
    subprocess.run(command, check=True)
    with output_path.open() as f:
        return json.load(f)


def run_base_eval(model_dir: Path, prompt_suite_rows, device: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    eos_ids = [
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ]
    rows = []
    for row in prompt_suite_rows:
        messages = [{"role": "user", "content": row["prompt_text"]}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            prefill_outputs = model(input_ids=input_ids)
        t1 = time.perf_counter()
        prefill_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                eos_token_id=eos_ids,
            )
        t3 = time.perf_counter()

        all_ids = generated[0].tolist()
        new_ids = all_ids[len(prompt_tokens):]
        wall_ms = (t3 - t2) * 1000.0
        decode_ms = (wall_ms / len(new_ids)) if new_ids else 0.0
        throughput = (len(new_ids) * 1000.0 / wall_ms) if new_ids else 0.0
        output_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        rows.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "prompt_text": row["prompt_text"],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": new_ids,
                "generated_token_count": len(new_ids),
                "output_text": output_text,
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "wall_ms": round(wall_ms, 3),
                "throughput": round(throughput, 3),
            }
        )
        print(
            f"[base] {row['id']} done: {len(new_ids)} tokens, "
            f"prefill={prefill_ms:.1f}ms, decode={decode_ms:.1f}ms/tok, "
            f"wall={wall_ms:.1f}ms, throughput={throughput:.2f} tok/s",
            file=sys.stderr,
        )

    return {"model_dir": str(model_dir), "mode": "base", "max_new_tokens": max_new_tokens, "rows": rows}


def summarize(rows):
    count = len(rows)
    if count == 0:
        return {
            "rows": 0,
            "prepared_prompt_match_count": 0,
            "output_text_match_count": 0,
            "generated_tokens_match_count": 0,
            "base_avg_prefill_ms": 0.0,
            "cpp_avg_prefill_ms": 0.0,
            "base_avg_decode_ms": 0.0,
            "cpp_avg_decode_ms": 0.0,
            "base_avg_wall_ms": 0.0,
            "cpp_avg_wall_ms": 0.0,
            "base_avg_throughput": 0.0,
            "cpp_avg_throughput": 0.0,
        }

    def avg(key, side):
        return round(sum(row[side][key] for row in rows) / count, 3)

    return {
        "rows": count,
        "prepared_prompt_match_count": sum(1 for row in rows if row["prompt_tokens_match"]),
        "output_text_match_count": sum(1 for row in rows if row["output_text_match"]),
        "generated_tokens_match_count": sum(1 for row in rows if row["generated_tokens_match"]),
        "base_avg_prefill_ms": avg("prefill_ms", "base"),
        "cpp_avg_prefill_ms": avg("prefill_ms", "cpp"),
        "base_avg_decode_ms": avg("decode_ms", "base"),
        "cpp_avg_decode_ms": avg("decode_ms", "cpp"),
        "base_avg_wall_ms": avg("wall_ms", "base"),
        "cpp_avg_wall_ms": avg("wall_ms", "cpp"),
        "base_avg_throughput": avg("throughput", "base"),
        "cpp_avg_throughput": avg("throughput", "cpp"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-suite", required=True)
    parser.add_argument("--cpp-bin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    prompt_suite = Path(args.prompt_suite)
    cpp_bin = Path(args.cpp_bin)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cpp_output_path = output_path.with_name(output_path.stem + "_cpp.json")
    base_output_path = output_path.with_name(output_path.stem + "_base.json")

    suite_rows = load_prompt_suite(prompt_suite)
    cpp_result = run_cpp_eval(cpp_bin, model_dir, prompt_suite, cpp_output_path, args.max_new_tokens)
    base_result = run_base_eval(model_dir, suite_rows, args.device, args.max_new_tokens)
    with base_output_path.open("w") as f:
        json.dump(base_result, f, indent=2, ensure_ascii=False)

    cpp_rows = {row["id"]: row for row in cpp_result["rows"]}
    base_rows = {row["id"]: row for row in base_result["rows"]}

    merged_rows = []
    for suite_row in suite_rows:
        rid = suite_row["id"]
        base_row = base_rows[rid]
        cpp_row = cpp_rows[rid]
        merged_rows.append(
            {
                "id": rid,
                "kind": suite_row["kind"],
                "prompt_text": suite_row["prompt_text"],
                "prompt_tokens_match": base_row["prompt_tokens"] == cpp_row["prompt_tokens"],
                "generated_tokens_match": base_row["generated_tokens"] == cpp_row["generated_tokens"],
                "output_text_match": base_row["output_text"] == cpp_row["output_text"],
                "base": base_row,
                "cpp": cpp_row,
            }
        )

    result = {
        "model_dir": str(model_dir),
        "prompt_suite": str(prompt_suite),
        "max_new_tokens": args.max_new_tokens,
        "summary": summarize(merged_rows),
        "rows": merged_rows,
    }
    with output_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
