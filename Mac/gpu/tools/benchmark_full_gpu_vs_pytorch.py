#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


def bucket_timing_entry(label: str) -> str | None:
    if label == "PrefillAttentionBatch":
        return "prefill_attention_batch"
    if label in {"QProjPrefill", "KProjPrefill", "VProjPrefill"}:
        return "prefill_attention_qkv"
    if label == "OProjPrefill":
        return "prefill_attention_output"
    if label in {"GateProjPrefill", "UpProjPrefill", "DownProjPrefill"}:
        return "prefill_mlp"
    if label == "LMHeadPrefill":
        return "prefill_lm_head"
    if label in {"DecodeAttnPrepBatch", "DecodeAttentionTailBatch", "DecodeAttentionOutputBatch", "DecodeAttentionFullBatch", "DecodeAttnContextBatch", "DecodeBlockAttentionBatch", "DecodeBlockPrepBatch"}:
        return "decode_attention_batches"
    if label in {"QProjDecode", "QProjDecodeQ4", "KProjDecode", "KProjDecodeQ4", "VProjDecode", "VProjDecodeQ4", "OProjDecode", "OProjDecodeQ4"}:
        return "decode_attention_projection"
    if label in {"GateProjDecode", "GateProjDecodeQ4", "UpProjDecode", "UpProjDecodeQ4", "DownProjDecodeQ4", "DecodePostNormBatch", "DecodePostNormMlpBatch", "DecodeMlpBatch"}:
        return "decode_mlp"
    if label in {"LMHeadDecode", "LMHeadDecodeQ4", "DecodeFinalNormLmHeadBatch"}:
        return "decode_logits"
    if label == "KVCacheBlitBatch":
        return "kv_cache"
    if label in {"LayerBatch", "FullRangeBatch"}:
        return "range_batches"
    return None


def summarize_bucket_entries(entries: list[dict], run_count: int) -> list[dict]:
    buckets: dict[str, dict[str, float]] = {}
    for entry in entries:
        bucket_name = bucket_timing_entry(str(entry["label"]))
        if bucket_name is None:
            continue
        bucket = buckets.setdefault(bucket_name, {"gpu_ms": 0.0, "wait_ms": 0.0, "command_buffer_count": 0.0, "encoder_count": 0.0})
        bucket["gpu_ms"] += float(entry["gpu_ms_avg"])
        bucket["wait_ms"] += float(entry["wait_ms_avg"])
        bucket["command_buffer_count"] += float(entry["command_buffer_count_avg"])
        bucket["encoder_count"] += float(entry["encoder_count_avg"])
    summarized = [
        {
            "bucket": bucket_name,
            "gpu_ms_avg": values["gpu_ms"],
            "wait_ms_avg": values["wait_ms"],
            "command_buffer_count_avg": values["command_buffer_count"],
            "encoder_count_avg": values["encoder_count"],
        }
        for bucket_name, values in buckets.items()
    ]
    summarized.sort(key=lambda entry: entry["gpu_ms_avg"], reverse=True)
    return summarized


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
    parser.add_argument("--gpu-cached-runs", type=int, default=-1)
    parser.add_argument("--pytorch-runs", type=int, default=3)
    parser.add_argument("--pytorch-warmup", type=int, default=1)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--pytorch-script", required=True)
    parser.add_argument("--pytorch-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--gpu-timeout-seconds", type=int, default=180)
    parser.add_argument("--gpu-env", action="append", default=[])
    return parser.parse_args()


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        key, separator, payload = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid --gpu-env entry: {value!r}")
        overrides[key] = payload
    return overrides


def run_command(command: list[str], env: dict[str, str] | None = None, timeout_seconds: int | None = None) -> None:
    subprocess.run(command, check=True, env=env, timeout=timeout_seconds)


def run_gpu_case(infer_bin: Path,
                 manifest: Path,
                 prompt: str,
                 max_new_tokens: int,
                 output_path: Path,
                 extra_args: list[str],
                 env_overrides: dict[str, str],
                 timeout_seconds: int) -> dict:
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
    command.extend(extra_args)
    env = os.environ.copy()
    env.update(env_overrides)
    run_command(command, env=env, timeout_seconds=timeout_seconds)
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
    wait_ms = [float(payload["timing"].get("wait_ms", 0.0)) for payload in payloads]
    command_buffer_count = [int(payload["timing"].get("command_buffer_count", 0)) for payload in payloads]
    encoder_count = [int(payload["timing"].get("encoder_count", 0)) for payload in payloads]
    prefill_ms = [float(payload["timing"].get("prefill_ms", payload["timing"]["wall_ms"])) for payload in payloads]
    decode_ms = [float(payload["timing"].get("decode_ms", payload["timing"]["wall_ms"])) for payload in payloads]
    generated_counts = [len(payload["generated_token_ids"]) for payload in payloads]
    prompt_counts = [len(payload["prompt_token_ids"]) for payload in payloads]
    total_tok_s = [count / max(ms / 1000.0, 1.0e-9) for count, ms in zip(generated_counts, wall_ms)]
    prefill_tok_s = [count / max(ms / 1000.0, 1.0e-9) for count, ms in zip(prompt_counts, prefill_ms)]
    decode_tok_s = [count / max(ms / 1000.0, 1.0e-9) for count, ms in zip(generated_counts, decode_ms)]
    command_buffers_per_generated_token = [
        count / max(token_count, 1)
        for count, token_count in zip(command_buffer_count, generated_counts)
    ]
    encoders_per_generated_token = [
        count / max(token_count, 1)
        for count, token_count in zip(encoder_count, generated_counts)
    ]
    decode_plan_execution_group_count = [int(payload.get("decode_plan", {}).get("execution_group_count", 0)) for payload in payloads]
    decode_plan_merged_range_count = [int(payload.get("decode_plan", {}).get("merged_range_count", 0)) for payload in payloads]
    decode_plan_merged_stage_count = [int(payload.get("decode_plan", {}).get("merged_stage_count", 0)) for payload in payloads]
    decode_plan_max_group_size = [int(payload.get("decode_plan", {}).get("max_group_size", 0)) for payload in payloads]
    aggregated_entries: dict[str, dict[str, float]] = {}
    for payload in payloads:
        for entry in payload["timing"].get("entries", []):
            label = str(entry["label"])
            bucket = aggregated_entries.setdefault(label, {"gpu_ms": 0.0, "wait_ms": 0.0, "command_buffer_count": 0.0, "encoder_count": 0.0})
            bucket["gpu_ms"] += float(entry["gpu_ms"])
            bucket["wait_ms"] += float(entry.get("wait_ms", 0.0))
            bucket["command_buffer_count"] += float(entry["command_buffer_count"])
            bucket["encoder_count"] += float(entry["encoder_count"])
    mean_entries = [
        {
            "label": label,
            "gpu_ms_avg": bucket["gpu_ms"] / max(len(payloads), 1),
            "wait_ms_avg": bucket["wait_ms"] / max(len(payloads), 1),
            "command_buffer_count_avg": bucket["command_buffer_count"] / max(len(payloads), 1),
            "encoder_count_avg": bucket["encoder_count"] / max(len(payloads), 1),
        }
        for label, bucket in aggregated_entries.items()
    ]
    mean_entries.sort(key=lambda entry: entry["gpu_ms_avg"], reverse=True)
    bucket_entries = summarize_bucket_entries(mean_entries, len(payloads))
    return {
        "runs": len(payloads),
        "generated_tokens_avg": statistics.mean(generated_counts),
        "prompt_tokens_avg": statistics.mean(prompt_counts),
        "prefill_ms_avg": statistics.mean(prefill_ms),
        "decode_ms_avg": statistics.mean(decode_ms),
        "prefill_tok_per_s_avg": statistics.mean(prefill_tok_s),
        "decode_tok_per_s_avg": statistics.mean(decode_tok_s),
        "wall_ms_avg": statistics.mean(wall_ms),
        "wall_ms_min": min(wall_ms),
        "gpu_ms_avg": statistics.mean(gpu_ms),
        "wait_ms_avg": statistics.mean(wait_ms),
        "command_buffer_count_avg": statistics.mean(command_buffer_count),
        "encoder_count_avg": statistics.mean(encoder_count),
        "command_buffers_per_generated_token_avg": statistics.mean(command_buffers_per_generated_token),
        "encoders_per_generated_token_avg": statistics.mean(encoders_per_generated_token),
        "total_tok_per_s_avg": statistics.mean(total_tok_s),
        "total_tok_per_s_max": max(total_tok_s),
        "decode_plan_execution_group_count_avg": statistics.mean(decode_plan_execution_group_count),
        "decode_plan_merged_range_count_avg": statistics.mean(decode_plan_merged_range_count),
        "decode_plan_merged_stage_count_avg": statistics.mean(decode_plan_merged_stage_count),
        "decode_plan_max_group_size_max": max(decode_plan_max_group_size) if decode_plan_max_group_size else 0,
        "timing_entries_avg": mean_entries,
        "timing_buckets_avg": bucket_entries,
        "last_generated_text": payloads[-1]["generated_text"],
        "prompt_cache_mode": str(payloads[-1].get("prompt_cache", {}).get("mode", "disabled")),
        "runtime_policy": payloads[-1].get("runtime_policy", {}),
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
        "prefill_tok_per_s": float(payload["prefill_tok_per_s"]),
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
                 gpu_cached_summary: dict | None,
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
        f"| C++ full GPU | prefill_ms_avg | {gpu_summary['prefill_ms_avg']:.2f} |",
        f"| C++ full GPU | decode_ms_avg | {gpu_summary['decode_ms_avg']:.2f} |",
        f"| C++ full GPU | prefill_tok_per_s_avg | {gpu_summary['prefill_tok_per_s_avg']:.3f} |",
        f"| C++ full GPU | decode_tok_per_s_avg | {gpu_summary['decode_tok_per_s_avg']:.3f} |",
        f"| C++ full GPU | generated_tokens_avg | {gpu_summary['generated_tokens_avg']:.2f} |",
        f"| C++ full GPU | wall_ms_avg | {gpu_summary['wall_ms_avg']:.2f} |",
        f"| C++ full GPU | wall_ms_min | {gpu_summary['wall_ms_min']:.2f} |",
        f"| C++ full GPU | gpu_ms_avg | {gpu_summary['gpu_ms_avg']:.2f} |",
        f"| C++ full GPU | wait_ms_avg | {gpu_summary['wait_ms_avg']:.2f} |",
        f"| C++ full GPU | command_buffer_count_avg | {gpu_summary['command_buffer_count_avg']:.2f} |",
        f"| C++ full GPU | encoder_count_avg | {gpu_summary['encoder_count_avg']:.2f} |",
        f"| C++ full GPU | command_buffers_per_generated_token_avg | {gpu_summary['command_buffers_per_generated_token_avg']:.2f} |",
        f"| C++ full GPU | encoders_per_generated_token_avg | {gpu_summary['encoders_per_generated_token_avg']:.2f} |",
        f"| C++ full GPU | decode_plan_execution_group_count_avg | {gpu_summary['decode_plan_execution_group_count_avg']:.2f} |",
        f"| C++ full GPU | decode_plan_merged_range_count_avg | {gpu_summary['decode_plan_merged_range_count_avg']:.2f} |",
        f"| C++ full GPU | decode_plan_merged_stage_count_avg | {gpu_summary['decode_plan_merged_stage_count_avg']:.2f} |",
        f"| C++ full GPU | decode_plan_max_group_size_max | {gpu_summary['decode_plan_max_group_size_max']} |",
        f"| C++ full GPU | runtime_prefill_step_size | {gpu_summary.get('runtime_policy', {}).get('prefill_step_size', 0)} |",
        f"| C++ full GPU | runtime_command_stream_encoder_budget | {gpu_summary.get('runtime_policy', {}).get('command_stream_encoder_budget', 0)} |",
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
        f"- C++ full GPU vs PyTorch prefill throughput ratio: `{gpu_summary['prefill_tok_per_s_avg'] / max(pytorch_summary['prefill_tok_per_s'], 1.0e-9):.3f}x`",
        f"- C++ full GPU vs PyTorch decode throughput ratio: `{gpu_summary['decode_tok_per_s_avg'] / max(pytorch_summary['decode_tok_per_s'], 1.0e-9):.3f}x`",
        f"- C++ full GPU preview: `{gpu_summary['last_generated_text'].strip()}`",
        f"- PyTorch MPS preview: `{pytorch_summary['generated_text'].strip()}`",
        "",
        "## C++ GPU Timing Breakdown",
        "",
        "| Label | gpu_ms_avg | wait_ms_avg | command_buffer_count_avg | encoder_count_avg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in gpu_summary["timing_entries_avg"]:
        lines.append(
            f"| {entry['label']} | {entry['gpu_ms_avg']:.2f} | {entry['wait_ms_avg']:.2f} | {entry['command_buffer_count_avg']:.2f} | {entry['encoder_count_avg']:.2f} |"
        )
    lines.extend([
        "",
        "## C++ GPU Timing Buckets",
        "",
        "| Bucket | gpu_ms_avg | wait_ms_avg | command_buffer_count_avg | encoder_count_avg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for entry in gpu_summary["timing_buckets_avg"]:
        lines.append(
            f"| {entry['bucket']} | {entry['gpu_ms_avg']:.2f} | {entry['wait_ms_avg']:.2f} | {entry['command_buffer_count_avg']:.2f} | {entry['encoder_count_avg']:.2f} |"
        )
    if gpu_cached_summary is not None:
        cached_total_ratio = gpu_cached_summary["total_tok_per_s_avg"] / max(pytorch_summary["total_tok_per_s"], 1.0e-9)
        cached_decode_ratio = gpu_cached_summary["decode_tok_per_s_avg"] / max(pytorch_summary["decode_tok_per_s"], 1.0e-9)
        cached_prefill_ratio = gpu_cached_summary["prefill_tok_per_s_avg"] / max(pytorch_summary["prefill_tok_per_s"], 1.0e-9)
        lines.extend([
            "",
            "## C++ GPU Cached Prompt",
            "",
            "- Artifact creation happens out-of-band once; cached runs below measure artifact load as prefill and cached decode separately.",
            "",
            "| Runtime | Metric | Value |",
            "| --- | --- | ---: |",
            f"| C++ GPU cached | runs | {gpu_cached_summary['runs']} |",
            f"| C++ GPU cached | prompt_cache_mode | {gpu_cached_summary['prompt_cache_mode']} |",
            f"| C++ GPU cached | prefill_ms_avg | {gpu_cached_summary['prefill_ms_avg']:.2f} |",
            f"| C++ GPU cached | decode_ms_avg | {gpu_cached_summary['decode_ms_avg']:.2f} |",
            f"| C++ GPU cached | prefill_tok_per_s_avg | {gpu_cached_summary['prefill_tok_per_s_avg']:.3f} |",
            f"| C++ GPU cached | decode_tok_per_s_avg | {gpu_cached_summary['decode_tok_per_s_avg']:.3f} |",
            f"| C++ GPU cached | total_tok_per_s_avg | {gpu_cached_summary['total_tok_per_s_avg']:.3f} |",
            f"| C++ GPU cached | wall_ms_avg | {gpu_cached_summary['wall_ms_avg']:.2f} |",
            f"| C++ GPU cached | runtime_prefill_step_size | {gpu_cached_summary.get('runtime_policy', {}).get('prefill_step_size', 0)} |",
            "",
            "## Cached Readout",
            "",
            f"- C++ GPU cached vs PyTorch total throughput ratio: `{cached_total_ratio:.3f}x`",
            f"- C++ GPU cached vs PyTorch prefill throughput ratio: `{cached_prefill_ratio:.3f}x`",
            f"- C++ GPU cached vs PyTorch decode throughput ratio: `{cached_decode_ratio:.3f}x`",
            f"- C++ GPU cached preview: `{gpu_cached_summary['last_generated_text'].strip()}`",
        ])
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
    gpu_env = parse_env_overrides(args.gpu_env)
    gpu_cached_runs = args.gpu_runs if args.gpu_cached_runs < 0 else args.gpu_cached_runs

    gpu_payloads: list[dict] = []
    for index in range(args.gpu_runs):
        gpu_output_path = output_dir / f"gpu_full_run_{index + 1}.json"
        gpu_payloads.append(run_gpu_case(infer_bin,
                                         manifest,
                                         args.prompt,
                                         args.max_new_tokens,
                                         gpu_output_path,
                                         [],
                                         gpu_env,
                                         args.gpu_timeout_seconds))

    artifact_path = output_dir / "prompt_cache_artifact.bin"
    artifact_build_output_path = output_dir / "gpu_cached_artifact_build.json"
    run_gpu_case(infer_bin,
                 manifest,
                 args.prompt,
                 0,
                 artifact_build_output_path,
                 ["--prompt-cache-artifact-save", str(artifact_path)],
                 gpu_env,
                 args.gpu_timeout_seconds)

    gpu_cached_payloads: list[dict] = []
    for index in range(gpu_cached_runs):
        gpu_cached_output_path = output_dir / f"gpu_cached_run_{index + 1}.json"
        gpu_cached_payloads.append(run_gpu_case(infer_bin,
                                                manifest,
                                                args.prompt,
                                                args.max_new_tokens,
                                                gpu_cached_output_path,
                                                ["--prompt-cache-artifact-load", str(artifact_path)],
                                                gpu_env,
                                                args.gpu_timeout_seconds))

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
    gpu_cached_summary = summarize_gpu_runs(gpu_cached_payloads)
    pytorch_summary = summarize_pytorch_run(pytorch_payload)

    summary = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "cpp_full_gpu": gpu_summary,
        "cpp_gpu_cached": gpu_cached_summary,
        "pytorch_mps": pytorch_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(report_path, args.prompt, args.max_new_tokens, manifest, hf_model, gpu_summary, gpu_cached_summary, pytorch_summary)


if __name__ == "__main__":
    main()
