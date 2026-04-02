#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare qwen3_5 prompt-boundary logits between HF and cpp.")
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--infer-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--hf-probe-script", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="current")
    parser.add_argument("--cpp-env", action="append", default=[])
    return parser.parse_args()


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


def run_command(command: list[str], env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, env=env)


def run_hf_probe(script: Path, model: Path, prompt: str, output_json: Path, output_pt: Path) -> dict:
    run_command([
        sys.executable,
        str(script),
        "--model", str(model),
        "--prompt", prompt,
        "--output-json", str(output_json),
        "--output-pt", str(output_pt),
        "--device", "mps",
        "--dtype", "float16",
        "--max-layers", "4",
    ])
    return json.loads(output_json.read_text(encoding="utf-8"))


def run_cpp_probe(args: argparse.Namespace, prompt: str, output_json: Path) -> dict:
    command = [
        str(args.infer_bin),
        "--manifest", str(args.manifest),
        "--model-type", "qwen3_5",
        "--prompt", prompt,
        "--json",
        "--qwen3_5-boundary-probe", str(output_json),
        "--profiling", "off",
    ]
    env = dict(**subprocess.os.environ)
    for item in args.cpp_env:
        key, value = item.split("=", 1)
        env[key] = value
    completed = run_command(command, env=env)
    if completed.stdout.strip():
        return json.loads(completed.stdout)
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_list, args.limit)
    results: list[dict] = []
    tmp_dir = args.output_json.parent / f".{args.tag}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def write_payload() -> None:
        payload = {
            "summary": {
                "tag": args.tag,
                "rows": len(results),
                "argmax_match_count": sum(
                    1 for row in results
                    if row["comparison"]["hf_vs_cpp_full_argmax_match"] and row["comparison"]["hf_vs_cpp_replay_argmax_match"]
                ),
                "hf_vs_cpp_full_argmax_match_count": sum(
                    1 for row in results if row["comparison"]["hf_vs_cpp_full_argmax_match"]
                ),
                "hf_vs_cpp_replay_argmax_match_count": sum(
                    1 for row in results if row["comparison"]["hf_vs_cpp_replay_argmax_match"]
                ),
                "cpp_full_vs_replay_argmax_match_count": sum(
                    1 for row in results if row["comparison"]["cpp_full_vs_replay_argmax_match"]
                ),
                "cpp_avg_max_abs_logit_diff": round(
                    sum(float(row["cpp"]["boundary"]["max_abs_logit_diff"]) for row in results) / max(len(results), 1), 6
                ),
                "cpp_avg_mean_abs_logit_diff": round(
                    sum(float(row["cpp"]["boundary"]["mean_abs_logit_diff"]) for row in results) / max(len(results), 1), 6
                ),
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['id']}", file=sys.stderr)
        hf_json = tmp_dir / f"{row['id']}_hf.json"
        hf_pt = tmp_dir / f"{row['id']}_hf.pt"
        cpp_json = tmp_dir / f"{row['id']}_cpp.json"
        hf = run_hf_probe(args.hf_probe_script, args.hf_model, row["prompt"], hf_json, hf_pt)
        cpp = run_cpp_probe(args, row["prompt"], cpp_json)
        hf_argmax_id = int(hf["next_token_argmax_id"])
        cpp_full_argmax_id = int(cpp["boundary"]["full_prompt_argmax_id"])
        cpp_replay_argmax_id = int(cpp["boundary"]["replay_warm_argmax_id"])
        results.append({
            "id": row["id"],
            "category": row["category"],
            "prompt": row["prompt"],
            "hf": hf,
            "cpp": cpp,
            "comparison": {
                "hf_vs_cpp_full_argmax_match": hf_argmax_id == cpp_full_argmax_id,
                "hf_vs_cpp_replay_argmax_match": hf_argmax_id == cpp_replay_argmax_id,
                "cpp_full_vs_replay_argmax_match": cpp_full_argmax_id == cpp_replay_argmax_id,
            },
        })
        write_payload()


if __name__ == "__main__":
    main()
