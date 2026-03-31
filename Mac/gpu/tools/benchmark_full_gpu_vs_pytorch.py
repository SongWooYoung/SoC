#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full-GPU C++ runtime vs PyTorch MPS on the same prompt.")
    parser.add_argument("--infer-bin", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu-runs", type=int, default=3)
    parser.add_argument("--pytorch-runs", type=int, default=3)
    parser.add_argument("--pytorch-warmup", type=int, default=1)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--pytorch-script", required=True)
    parser.add_argument("--pytorch-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_gpu_case(infer_bin: Path,
                 manifest: Path,
                 prompt: str,
                 max_new_tokens: int,
                 output_path: Path) -> dict:
    command = [
        str(infer_bin),
        "--manifest",
        str(manifest),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--layer",
        "-1",
        "--json",
        "--output-file",
        str(output_path),
    ]
    run_command(command)
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_pytorch_case(python_bin: str,
                     script_path: Path,
                     hf_model: Path,
                     prompt: str,
                     max_new_tokens: int,
                     runs: int,
                     warmup: int,
                     dtype: str,
                     output_path: Path) -> dict:
    command = [
        python_bin,
        str(script_path),
        "--model",
        str(hf_model),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        "mps",
        "--dtype",
        dtype,
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--output",
        str(output_path),
    ]
    run_command(command)
    return json.loads(output_path.read_text(encoding="utf-8"))


def summarize_gpu_runs(payloads: list[dict]) -> dict:
    wall_ms = [float(payload["timing"]["wall_ms"]) for payload in payloads]
    gpu_ms = [float(payload["timing"]["gpu_ms"]) for payload in payloads]
    command_buffer_count = [int(payload["timing"].get("command_buffer_count", 0)) for payload in payloads]
    encoder_count = [int(payload["timing"].get("encoder_count", 0)) for payload in payloads]
    generated_counts = [len(payload["generated_token_ids"]) for payload in payloads]
    total_tok_s = [count / max(ms / 1000.0, 1.0e-9) for count, ms in zip(generated_counts, wall_ms)]
    aggregated_entries: dict[str, dict[str, float]] = {}
    for payload in payloads:
        for entry in payload["timing"].get("entries", []):
            label = str(entry["label"])
            bucket = aggregated_entries.setdefault(label, {"gpu_ms": 0.0, "command_buffer_count": 0.0, "encoder_count": 0.0})
            bucket["gpu_ms"] += float(entry["gpu_ms"])
            bucket["command_buffer_count"] += float(entry["command_buffer_count"])
            bucket["encoder_count"] += float(entry["encoder_count"])
    mean_entries = [
        {
            "label": label,
            "gpu_ms_avg": bucket["gpu_ms"] / max(len(payloads), 1),
            "command_buffer_count_avg": bucket["command_buffer_count"] / max(len(payloads), 1),
            "encoder_count_avg": bucket["encoder_count"] / max(len(payloads), 1),
        }
        for label, bucket in aggregated_entries.items()
    ]
    mean_entries.sort(key=lambda entry: entry["gpu_ms_avg"], reverse=True)
    return {
        "runs": len(payloads),
        "generated_tokens_avg": statistics.mean(generated_counts),
        "wall_ms_avg": statistics.mean(wall_ms),
        "wall_ms_min": min(wall_ms),
        "gpu_ms_avg": statistics.mean(gpu_ms),
        "command_buffer_count_avg": statistics.mean(command_buffer_count),
        "encoder_count_avg": statistics.mean(encoder_count),
        "total_tok_per_s_avg": statistics.mean(total_tok_s),
        "total_tok_per_s_max": max(total_tok_s),
        "timing_entries_avg": mean_entries,
        "last_generated_text": payloads[-1]["generated_text"],
    }


def summarize_pytorch_run(payload: dict) -> dict:
    decode_tokens = float(payload["decode_tokens"])
    total_time_ms = float(payload["total_time_ms"])
    total_tok_per_s = decode_tokens / max(total_time_ms / 1000.0, 1.0e-9)
    return {
        "runs": int(payload["num_runs"]),
        "warmup": int(payload["num_warmup"]),
        "prompt_tokens": int(payload["prompt_tokens"]),
        "generated_tokens_avg": decode_tokens,
        "prefill_ms": float(payload["prefill_ms"]),
        "decode_time_ms": float(payload["decode_time_ms"]),
        "decode_tok_per_s": float(payload["decode_tok_per_s"]),
        "total_time_ms": total_time_ms,
        "total_tok_per_s": total_tok_per_s,
        "dtype": payload["dtype"],
        "generated_text": payload["generated_text"],
    }


def write_report(report_path: Path,
                 prompt: str,
                 max_new_tokens: int,
                 manifest: Path,
                 hf_model: Path,
                 gpu_summary: dict,
                 pytorch_summary: dict) -> None:
    speedup = gpu_summary["total_tok_per_s_avg"] / max(pytorch_summary["total_tok_per_s"], 1.0e-9)
    lines = [
        "# Full GPU vs PyTorch MPS Benchmark",
        "",
        f"- Prompt: `{prompt}`",
        f"- max_new_tokens: `{max_new_tokens}`",
        f"- C++ manifest: `{manifest}`",
        f"- HF model: `{hf_model}`",
        "",
        "| Runtime | Metric | Value |",
        "| --- | --- | ---: |",
        f"| C++ full GPU | runs | {gpu_summary['runs']} |",
        f"| C++ full GPU | generated_tokens_avg | {gpu_summary['generated_tokens_avg']:.2f} |",
        f"| C++ full GPU | wall_ms_avg | {gpu_summary['wall_ms_avg']:.2f} |",
        f"| C++ full GPU | wall_ms_min | {gpu_summary['wall_ms_min']:.2f} |",
        f"| C++ full GPU | gpu_ms_avg | {gpu_summary['gpu_ms_avg']:.2f} |",
        f"| C++ full GPU | command_buffer_count_avg | {gpu_summary['command_buffer_count_avg']:.2f} |",
        f"| C++ full GPU | encoder_count_avg | {gpu_summary['encoder_count_avg']:.2f} |",
        f"| C++ full GPU | total_tok_per_s_avg | {gpu_summary['total_tok_per_s_avg']:.3f} |",
        f"| C++ full GPU | total_tok_per_s_max | {gpu_summary['total_tok_per_s_max']:.3f} |",
        f"| PyTorch MPS | runs | {pytorch_summary['runs']} |",
        f"| PyTorch MPS | dtype | {pytorch_summary['dtype']} |",
        f"| PyTorch MPS | prefill_ms | {pytorch_summary['prefill_ms']:.2f} |",
        f"| PyTorch MPS | decode_time_ms | {pytorch_summary['decode_time_ms']:.2f} |",
        f"| PyTorch MPS | decode_tok_per_s | {pytorch_summary['decode_tok_per_s']:.3f} |",
        f"| PyTorch MPS | total_time_ms | {pytorch_summary['total_time_ms']:.2f} |",
        f"| PyTorch MPS | total_tok_per_s | {pytorch_summary['total_tok_per_s']:.3f} |",
        "",
        "## Readout",
        "",
        f"- C++ full GPU vs PyTorch total throughput ratio: `{speedup:.3f}x`",
        f"- C++ full GPU preview: `{gpu_summary['last_generated_text'].strip()}`",
        f"- PyTorch MPS preview: `{pytorch_summary['generated_text'].strip()}`",
        "",
        "## C++ GPU Timing Breakdown",
        "",
        "| Label | gpu_ms_avg | command_buffer_count_avg | encoder_count_avg |",
        "| --- | ---: | ---: | ---: |",
    ]
    for entry in gpu_summary["timing_entries_avg"]:
        lines.append(
            f"| {entry['label']} | {entry['gpu_ms_avg']:.2f} | {entry['command_buffer_count_avg']:.2f} | {entry['encoder_count_avg']:.2f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    infer_bin = Path(args.infer_bin).resolve()
    manifest = Path(args.manifest).resolve()
    hf_model = Path(args.hf_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    pytorch_script = Path(args.pytorch_script).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_payloads: list[dict] = []
    for index in range(args.gpu_runs):
        gpu_output_path = output_dir / f"gpu_full_run_{index + 1}.json"
        gpu_payloads.append(run_gpu_case(infer_bin, manifest, args.prompt, args.max_new_tokens, gpu_output_path))

    pytorch_output_path = output_dir / "pytorch_mps.json"
    pytorch_payload = run_pytorch_case(args.python_bin,
                                       pytorch_script,
                                       hf_model,
                                       args.prompt,
                                       args.max_new_tokens,
                                       args.pytorch_runs,
                                       args.pytorch_warmup,
                                       args.pytorch_dtype,
                                       pytorch_output_path)

    gpu_summary = summarize_gpu_runs(gpu_payloads)
    pytorch_summary = summarize_pytorch_run(pytorch_payload)

    summary = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "cpp_full_gpu": gpu_summary,
        "pytorch_mps": pytorch_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(report_path, args.prompt, args.max_new_tokens, manifest, hf_model, gpu_summary, pytorch_summary)


if __name__ == "__main__":
    main()
