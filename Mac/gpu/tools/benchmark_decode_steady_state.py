#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


def bucket_timing_entry(label: str) -> str | None:
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


def diff_bucket_entries(short_entries: list[dict], full_entries: list[dict]) -> list[dict]:
    def to_map(entries: list[dict]) -> dict[str, dict[str, float]]:
        return {
            str(entry["label"]): {
                "gpu_ms": float(entry.get("gpu_ms", 0.0)),
                "wait_ms": float(entry.get("wait_ms", 0.0)),
                "command_buffer_count": float(entry.get("command_buffer_count", 0.0)),
                "encoder_count": float(entry.get("encoder_count", 0.0)),
            }
            for entry in entries
        }

    short_map = to_map(short_entries)
    full_map = to_map(full_entries)
    bucket_totals: dict[str, dict[str, float]] = {}
    for label, full_metrics in full_map.items():
        bucket_name = bucket_timing_entry(label)
        if bucket_name is None:
            continue
        short_metrics = short_map.get(label, {"gpu_ms": 0.0, "wait_ms": 0.0, "command_buffer_count": 0.0, "encoder_count": 0.0})
        bucket = bucket_totals.setdefault(bucket_name, {"gpu_ms": 0.0, "wait_ms": 0.0, "command_buffer_count": 0.0, "encoder_count": 0.0})
        bucket["gpu_ms"] += max(full_metrics["gpu_ms"] - short_metrics["gpu_ms"], 0.0)
        bucket["wait_ms"] += max(full_metrics["wait_ms"] - short_metrics["wait_ms"], 0.0)
        bucket["command_buffer_count"] += max(full_metrics["command_buffer_count"] - short_metrics["command_buffer_count"], 0.0)
        bucket["encoder_count"] += max(full_metrics["encoder_count"] - short_metrics["encoder_count"], 0.0)
    summarized = [
        {
            "bucket": bucket_name,
            "gpu_ms_avg": values["gpu_ms"],
            "wait_ms_avg": values["wait_ms"],
            "command_buffer_count_avg": values["command_buffer_count"],
            "encoder_count_avg": values["encoder_count"],
        }
        for bucket_name, values in bucket_totals.items()
    ]
    summarized.sort(key=lambda entry: entry["gpu_ms_avg"], reverse=True)
    return summarized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate decode-only steady-state throughput by differencing 1-token and N-token runs.")
    parser.add_argument("--infer-bin", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
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


def run_case(infer_bin: Path,
             manifest: Path,
             prompt: str,
             max_new_tokens: int,
             output_path: Path,
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
    env = os.environ.copy()
    env.update(env_overrides)
    subprocess.run(command, check=True, env=env, timeout=timeout_seconds)
    return json.loads(output_path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize(pairs: list[tuple[dict, dict]], max_new_tokens: int) -> dict:
    warm_wall = [float(short_run["timing"]["wall_ms"]) for short_run, _ in pairs]
    warm_gpu = [float(short_run["timing"]["gpu_ms"]) for short_run, _ in pairs]
    warm_wait = [float(short_run["timing"].get("wait_ms", 0.0)) for short_run, _ in pairs]

    full_wall = [float(full_run["timing"]["wall_ms"]) for _, full_run in pairs]
    full_gpu = [float(full_run["timing"]["gpu_ms"]) for _, full_run in pairs]
    full_wait = [float(full_run["timing"].get("wait_ms", 0.0)) for _, full_run in pairs]
    full_cb = [int(full_run["timing"].get("command_buffer_count", 0)) for _, full_run in pairs]
    full_enc = [int(full_run["timing"].get("encoder_count", 0)) for _, full_run in pairs]
    decode_tokens = max(max_new_tokens - 1, 0)
    full_cb_per_decode_token = [
        count / max(decode_tokens, 1)
        for count in full_cb
    ]
    full_enc_per_decode_token = [
        count / max(decode_tokens, 1)
        for count in full_enc
    ]
    decode_plan_groups = [int(full_run.get("decode_plan", {}).get("execution_group_count", 0)) for _, full_run in pairs]
    decode_plan_merged_ranges = [int(full_run.get("decode_plan", {}).get("merged_range_count", 0)) for _, full_run in pairs]
    decode_plan_merged_stages = [int(full_run.get("decode_plan", {}).get("merged_stage_count", 0)) for _, full_run in pairs]
    decode_plan_max_group_sizes = [int(full_run.get("decode_plan", {}).get("max_group_size", 0)) for _, full_run in pairs]
    decode_plan_hidden_slot0_blockers = [int(full_run.get("decode_plan", {}).get("hidden_slot0_blocker_count", 0)) for _, full_run in pairs]
    decode_plan_hidden_slot1_blockers = [int(full_run.get("decode_plan", {}).get("hidden_slot1_blocker_count", 0)) for _, full_run in pairs]
    decode_plan_logits_blockers = [int(full_run.get("decode_plan", {}).get("logits_blocker_count", 0)) for _, full_run in pairs]
    decode_plan_kv_keys_blockers = [int(full_run.get("decode_plan", {}).get("kv_keys_blocker_count", 0)) for _, full_run in pairs]
    decode_plan_kv_values_blockers = [int(full_run.get("decode_plan", {}).get("kv_values_blocker_count", 0)) for _, full_run in pairs]
    decode_plan_raw_blockers = [int(full_run.get("decode_plan", {}).get("read_after_write_blocker_count", 0)) for _, full_run in pairs]
    decode_plan_war_blockers = [int(full_run.get("decode_plan", {}).get("write_after_read_blocker_count", 0)) for _, full_run in pairs]
    decode_plan_waw_blockers = [int(full_run.get("decode_plan", {}).get("write_after_write_blocker_count", 0)) for _, full_run in pairs]

    steady_wall = [max(f - s, 0.0) for s, f in zip(warm_wall, full_wall)]
    steady_gpu = [max(f - s, 0.0) for s, f in zip(warm_gpu, full_gpu)]
    steady_wait = [max(f - s, 0.0) for s, f in zip(warm_wait, full_wait)]
    steady_tok_s = [
        decode_tokens / max(delta_ms / 1000.0, 1.0e-9)
        for delta_ms in steady_wall
    ] if decode_tokens > 0 else [0.0 for _ in steady_wall]
    bucket_runs = [
        diff_bucket_entries(short_run["timing"].get("entries", []), full_run["timing"].get("entries", []))
        for short_run, full_run in pairs
    ]
    bucket_aggregate: dict[str, dict[str, float]] = {}
    for bucket_run in bucket_runs:
        for entry in bucket_run:
            bucket = bucket_aggregate.setdefault(entry["bucket"], {"gpu_ms": 0.0, "wait_ms": 0.0, "command_buffer_count": 0.0, "encoder_count": 0.0})
            bucket["gpu_ms"] += float(entry["gpu_ms_avg"])
            bucket["wait_ms"] += float(entry["wait_ms_avg"])
            bucket["command_buffer_count"] += float(entry["command_buffer_count_avg"])
            bucket["encoder_count"] += float(entry["encoder_count_avg"])
    steady_buckets = [
        {
            "bucket": bucket_name,
            "gpu_ms_avg": values["gpu_ms"] / max(len(bucket_runs), 1),
            "wait_ms_avg": values["wait_ms"] / max(len(bucket_runs), 1),
            "command_buffer_count_avg": values["command_buffer_count"] / max(len(bucket_runs), 1),
            "encoder_count_avg": values["encoder_count"] / max(len(bucket_runs), 1),
        }
        for bucket_name, values in bucket_aggregate.items()
    ]
    steady_buckets.sort(key=lambda entry: entry["gpu_ms_avg"], reverse=True)

    return {
        "runs": len(pairs),
        "prompt": pairs[-1][1]["prompt"] if pairs else "",
        "max_new_tokens": max_new_tokens,
        "decode_tokens_measured": decode_tokens,
        "warmup_run_wall_ms_avg": mean(warm_wall),
        "warmup_run_gpu_ms_avg": mean(warm_gpu),
        "warmup_run_wait_ms_avg": mean(warm_wait),
        "full_run_wall_ms_avg": mean(full_wall),
        "full_run_gpu_ms_avg": mean(full_gpu),
        "full_run_wait_ms_avg": mean(full_wait),
        "steady_state_wall_ms_avg": mean(steady_wall),
        "steady_state_gpu_ms_avg": mean(steady_gpu),
        "steady_state_wait_ms_avg": mean(steady_wait),
        "steady_state_tok_per_s_avg": mean(steady_tok_s),
        "steady_state_tok_per_s_max": max(steady_tok_s) if steady_tok_s else 0.0,
        "full_run_command_buffer_count_avg": mean(full_cb),
        "full_run_encoder_count_avg": mean(full_enc),
        "full_run_command_buffers_per_decode_token_avg": mean(full_cb_per_decode_token),
        "full_run_encoders_per_decode_token_avg": mean(full_enc_per_decode_token),
        "decode_plan_execution_group_count_avg": mean(decode_plan_groups),
        "decode_plan_merged_range_count_avg": mean(decode_plan_merged_ranges),
        "decode_plan_merged_stage_count_avg": mean(decode_plan_merged_stages),
        "decode_plan_max_group_size_max": max(decode_plan_max_group_sizes) if decode_plan_max_group_sizes else 0,
        "decode_plan_hidden_slot0_blocker_count_avg": mean(decode_plan_hidden_slot0_blockers),
        "decode_plan_hidden_slot1_blocker_count_avg": mean(decode_plan_hidden_slot1_blockers),
        "decode_plan_logits_blocker_count_avg": mean(decode_plan_logits_blockers),
        "decode_plan_kv_keys_blocker_count_avg": mean(decode_plan_kv_keys_blockers),
        "decode_plan_kv_values_blocker_count_avg": mean(decode_plan_kv_values_blockers),
        "decode_plan_read_after_write_blocker_count_avg": mean(decode_plan_raw_blockers),
        "decode_plan_write_after_read_blocker_count_avg": mean(decode_plan_war_blockers),
        "decode_plan_write_after_write_blocker_count_avg": mean(decode_plan_waw_blockers),
        "steady_state_timing_buckets_avg": steady_buckets,
        "runtime_policy": pairs[-1][1].get("runtime_policy", {}) if pairs else {},
        "preview_text": pairs[-1][1]["generated_text"] if pairs else "",
    }


def write_report(report_path: Path, manifest: Path, summary: dict, env_overrides: dict[str, str]) -> None:
    lines = [
        "# Decode Steady-State Benchmark",
        "",
        f"- Manifest: `{manifest}`",
        f"- Prompt: `{summary['prompt']}`",
        f"- max_new_tokens: `{summary['max_new_tokens']}`",
        f"- decode_tokens_measured: `{summary['decode_tokens_measured']}`",
        f"- gpu_env: `{json.dumps(env_overrides, ensure_ascii=False, sort_keys=True)}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| runs | {summary['runs']} |",
        f"| warmup_run_wall_ms_avg | {summary['warmup_run_wall_ms_avg']:.2f} |",
        f"| warmup_run_gpu_ms_avg | {summary['warmup_run_gpu_ms_avg']:.2f} |",
        f"| warmup_run_wait_ms_avg | {summary['warmup_run_wait_ms_avg']:.2f} |",
        f"| full_run_wall_ms_avg | {summary['full_run_wall_ms_avg']:.2f} |",
        f"| full_run_gpu_ms_avg | {summary['full_run_gpu_ms_avg']:.2f} |",
        f"| full_run_wait_ms_avg | {summary['full_run_wait_ms_avg']:.2f} |",
        f"| steady_state_wall_ms_avg | {summary['steady_state_wall_ms_avg']:.2f} |",
        f"| steady_state_gpu_ms_avg | {summary['steady_state_gpu_ms_avg']:.2f} |",
        f"| steady_state_wait_ms_avg | {summary['steady_state_wait_ms_avg']:.2f} |",
        f"| steady_state_tok_per_s_avg | {summary['steady_state_tok_per_s_avg']:.3f} |",
        f"| steady_state_tok_per_s_max | {summary['steady_state_tok_per_s_max']:.3f} |",
        f"| full_run_command_buffer_count_avg | {summary['full_run_command_buffer_count_avg']:.2f} |",
        f"| full_run_encoder_count_avg | {summary['full_run_encoder_count_avg']:.2f} |",
        f"| full_run_command_buffers_per_decode_token_avg | {summary['full_run_command_buffers_per_decode_token_avg']:.2f} |",
        f"| full_run_encoders_per_decode_token_avg | {summary['full_run_encoders_per_decode_token_avg']:.2f} |",
        f"| runtime_prefill_step_size | {summary.get('runtime_policy', {}).get('prefill_step_size', 0)} |",
        f"| runtime_command_stream_encoder_budget | {summary.get('runtime_policy', {}).get('command_stream_encoder_budget', 0)} |",
        f"| decode_plan_execution_group_count_avg | {summary['decode_plan_execution_group_count_avg']:.2f} |",
        f"| decode_plan_merged_range_count_avg | {summary['decode_plan_merged_range_count_avg']:.2f} |",
        f"| decode_plan_merged_stage_count_avg | {summary['decode_plan_merged_stage_count_avg']:.2f} |",
        f"| decode_plan_max_group_size_max | {summary['decode_plan_max_group_size_max']} |",
        f"| decode_plan_hidden_slot0_blocker_count_avg | {summary['decode_plan_hidden_slot0_blocker_count_avg']:.2f} |",
        f"| decode_plan_hidden_slot1_blocker_count_avg | {summary['decode_plan_hidden_slot1_blocker_count_avg']:.2f} |",
        f"| decode_plan_logits_blocker_count_avg | {summary['decode_plan_logits_blocker_count_avg']:.2f} |",
        f"| decode_plan_kv_keys_blocker_count_avg | {summary['decode_plan_kv_keys_blocker_count_avg']:.2f} |",
        f"| decode_plan_kv_values_blocker_count_avg | {summary['decode_plan_kv_values_blocker_count_avg']:.2f} |",
        f"| decode_plan_read_after_write_blocker_count_avg | {summary['decode_plan_read_after_write_blocker_count_avg']:.2f} |",
        f"| decode_plan_write_after_read_blocker_count_avg | {summary['decode_plan_write_after_read_blocker_count_avg']:.2f} |",
        f"| decode_plan_write_after_write_blocker_count_avg | {summary['decode_plan_write_after_write_blocker_count_avg']:.2f} |",
        "",
        f"- Preview: `{summary['preview_text'].strip()}`",
        "",
        "| Bucket | gpu_ms_avg | wait_ms_avg | command_buffer_count_avg | encoder_count_avg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["steady_state_timing_buckets_avg"]:
        lines.append(
            f"| {entry['bucket']} | {entry['gpu_ms_avg']:.2f} | {entry['wait_ms_avg']:.2f} | {entry['command_buffer_count_avg']:.2f} | {entry['encoder_count_avg']:.2f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    infer_bin = Path(args.infer_bin).resolve()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    env_overrides = parse_env_overrides(args.gpu_env)

    pairs: list[tuple[dict, dict]] = []
    for index in range(args.runs):
        short_path = output_dir / f"run_{index + 1}_1tok.json"
        full_path = output_dir / f"run_{index + 1}_{args.max_new_tokens}tok.json"
        short_payload = run_case(infer_bin,
                                 manifest,
                                 args.prompt,
                                 1,
                                 short_path,
                                 env_overrides,
                                 args.timeout_seconds)
        full_payload = run_case(infer_bin,
                                manifest,
                                args.prompt,
                                args.max_new_tokens,
                                full_path,
                                env_overrides,
                                args.timeout_seconds)
        pairs.append((short_payload, full_payload))

    summary = summarize(pairs, args.max_new_tokens)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(report_path, manifest, summary, env_overrides)


if __name__ == "__main__":
    main()
