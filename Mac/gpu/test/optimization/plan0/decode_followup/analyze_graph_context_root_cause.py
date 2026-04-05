#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


BASE_SPLIT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_stage_trace_split.json")
CUSTOM_LEGACY_SPLIT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_stage_trace_linear_split.json")
CUSTOM_STEP_SPLIT = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_stage_trace_linear_split_step_buffer.json")
CUSTOM_NO_TRACE_LEGACY = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_no_trace_legacy_4prompt.json")
CUSTOM_NO_TRACE_STEP = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_no_trace_step_buffer_4prompt.json")
OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/graph_context_root_cause.json")
OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/graph_context_root_cause.md")

LINEAR_STAGES = [
    "linear_attention_in_proj_qkv",
    "linear_attention_in_proj_z",
    "linear_attention_in_proj_a",
    "linear_attention_in_proj_b",
    "linear_cache_conv_state_update",
    "linear_attention_conv1d",
    "linear_attention_q_norm",
    "linear_attention_k_norm",
    "linear_attention_gated_delta",
    "linear_cache_rec_state_update",
    "linear_attention_norm_gated",
    "linear_attention_out_proj",
    "linear_cache_update",
    "linear_attention",
]

ATTENTION_STAGES = [
    "full_attention_q_proj",
    "full_attention_cache_update",
    "full_attention_o_proj",
    "full_attention_sdpa",
    "full_attention",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean_decode_ms(rows: list[dict[str, Any]]) -> float:
    return round(mean(float(row["decode_ms"]) for row in rows), 3)


def mean_stage(rows: list[dict[str, Any]], stage: str, metric: str = "sync_ms") -> float:
    values = [float(row["stage_trace"]["decode"].get(stage, {}).get(metric, 0.0)) for row in rows]
    return round(mean(values), 3)


def stage_delta_rows(base_rows: list[dict[str, Any]], custom_rows: list[dict[str, Any]], stages: list[str]) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        base_sync = mean_stage(base_rows, stage, "sync_ms")
        custom_sync = mean_stage(custom_rows, stage, "sync_ms")
        rows.append(
            {
                "stage": stage,
                "base_sync_ms": base_sync,
                "custom_sync_ms": custom_sync,
                "sync_delta_ms": round(custom_sync - base_sync, 3),
            }
        )
    rows.sort(key=lambda row: row["sync_delta_ms"], reverse=True)
    return rows


def step_impact_rows(legacy_rows: list[dict[str, Any]], step_rows: list[dict[str, Any]], stages: list[str]) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        legacy_sync = mean_stage(legacy_rows, stage, "sync_ms")
        step_sync = mean_stage(step_rows, stage, "sync_ms")
        rows.append(
            {
                "stage": stage,
                "legacy_sync_ms": legacy_sync,
                "step_sync_ms": step_sync,
                "step_minus_legacy_ms": round(step_sync - legacy_sync, 3),
            }
        )
    rows.sort(key=lambda row: abs(row["step_minus_legacy_ms"]), reverse=True)
    return rows


def table_stage_delta(rows: list[dict[str, Any]], left: str, right: str, delta: str) -> list[str]:
    lines = [
        f"| Stage | {left} | {right} | Delta ms |",
        "|------|---------:|---------:|---------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row[left]:.3f} | {row[right]:.3f} | {row[delta]:+.3f} |"
        )
    return lines


def main() -> None:
    base_split = load_json(BASE_SPLIT)
    custom_legacy_split = load_json(CUSTOM_LEGACY_SPLIT)
    custom_step_split = load_json(CUSTOM_STEP_SPLIT)
    custom_no_trace_legacy = load_json(CUSTOM_NO_TRACE_LEGACY)
    custom_no_trace_step = load_json(CUSTOM_NO_TRACE_STEP)

    legacy_linear = stage_delta_rows(base_split["rows"], custom_legacy_split["rows"], LINEAR_STAGES)
    legacy_attention = stage_delta_rows(base_split["rows"], custom_legacy_split["rows"], ATTENTION_STAGES)
    step_attention = step_impact_rows(custom_legacy_split["rows"], custom_step_split["rows"], ATTENTION_STAGES)

    legacy_trace_decode = mean_decode_ms(custom_legacy_split["rows"])
    step_trace_decode = mean_decode_ms(custom_step_split["rows"])
    legacy_no_trace_decode = mean_decode_ms(custom_no_trace_legacy["rows"])
    step_no_trace_decode = mean_decode_ms(custom_no_trace_step["rows"])

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_vs_custom_legacy": {
            "trace_decode_ms": {
                "base": mean_decode_ms(base_split["rows"]),
                "custom_legacy": legacy_trace_decode,
                "delta_ms": round(legacy_trace_decode - mean_decode_ms(base_split["rows"]), 3),
            },
            "linear_stage_sync_delta": legacy_linear,
            "attention_stage_sync_delta": legacy_attention,
        },
        "step_buffer_effect": {
            "trace_decode_ms": {
                "legacy": legacy_trace_decode,
                "step_buffer": step_trace_decode,
                "delta_ms": round(step_trace_decode - legacy_trace_decode, 3),
            },
            "no_trace_decode_ms": {
                "legacy": legacy_no_trace_decode,
                "step_buffer": step_no_trace_decode,
                "delta_ms": round(step_no_trace_decode - legacy_no_trace_decode, 3),
            },
            "attention_stage_sync_change": step_attention,
        },
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Graph Context Root Cause",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Base split trace: {BASE_SPLIT}",
        f"- Custom legacy split trace: {CUSTOM_LEGACY_SPLIT}",
        f"- Custom step-buffer split trace: {CUSTOM_STEP_SPLIT}",
        f"- Custom no-trace legacy: {CUSTOM_NO_TRACE_LEGACY}",
        f"- Custom no-trace step-buffer: {CUSTOM_NO_TRACE_STEP}",
        "",
        "## Base vs Custom Legacy: Linear Path Decode Delta",
        "",
        f"- trace decode average: base {report['base_vs_custom_legacy']['trace_decode_ms']['base']:.3f} ms/tok, custom {report['base_vs_custom_legacy']['trace_decode_ms']['custom_legacy']:.3f} ms/tok, delta {report['base_vs_custom_legacy']['trace_decode_ms']['delta_ms']:+.3f} ms/tok.",
        "",
    ]
    lines.extend(table_stage_delta(legacy_linear, "base_sync_ms", "custom_sync_ms", "sync_delta_ms"))
    lines.extend([
        "",
        "## Base vs Custom Legacy: Attention Path Decode Delta",
        "",
    ])
    lines.extend(table_stage_delta(legacy_attention, "base_sync_ms", "custom_sync_ms", "sync_delta_ms"))
    lines.extend([
        "",
        "## Step-Buffer Effect On Attention Path",
        "",
        f"- custom trace decode average: legacy {legacy_trace_decode:.3f} ms/tok, step-buffer {step_trace_decode:.3f} ms/tok, delta {step_trace_decode - legacy_trace_decode:+.3f} ms/tok.",
        f"- custom no-trace decode average: legacy {legacy_no_trace_decode:.3f} ms/tok, step-buffer {step_no_trace_decode:.3f} ms/tok, delta {step_no_trace_decode - legacy_no_trace_decode:+.3f} ms/tok.",
        "",
    ])
    lines.extend(table_stage_delta(step_attention, "legacy_sync_ms", "step_sync_ms", "step_minus_legacy_ms"))
    lines.extend([
        "",
        "## Findings",
        "",
        f"- legacy full-model linear path에서 base 대비 가장 큰 양의 delta는 {legacy_linear[0]['stage']} ({legacy_linear[0]['sync_delta_ms']:+.3f} ms), {legacy_linear[1]['stage']} ({legacy_linear[1]['sync_delta_ms']:+.3f} ms), {legacy_linear[2]['stage']} ({legacy_linear[2]['sync_delta_ms']:+.3f} ms)였다.",
        f"- `linear_cache_conv_state_update` delta는 {next(row['sync_delta_ms'] for row in legacy_linear if row['stage'] == 'linear_cache_conv_state_update'):+.3f} ms였지만, `linear_attention_in_proj_qkv` delta는 {next(row['sync_delta_ms'] for row in legacy_linear if row['stage'] == 'linear_attention_in_proj_qkv'):+.3f} ms, `linear_attention_out_proj` delta는 {next(row['sync_delta_ms'] for row in legacy_linear if row['stage'] == 'linear_attention_out_proj'):+.3f} ms였다. 즉 conv-state update 하나만의 문제가 아니라 그 앞뒤 projection path 전체가 같이 비싸다.",
        f"- attention cache를 step-buffer로 바꾸면 trace 기준 `full_attention_cache_update`는 {next(row['step_minus_legacy_ms'] for row in step_attention if row['stage'] == 'full_attention_cache_update'):+.3f} ms, `full_attention_q_proj`는 {next(row['step_minus_legacy_ms'] for row in step_attention if row['stage'] == 'full_attention_q_proj'):+.3f} ms, `full_attention_o_proj`는 {next(row['step_minus_legacy_ms'] for row in step_attention if row['stage'] == 'full_attention_o_proj'):+.3f} ms 바뀐다.",
        f"- 그런데 같은 변경의 no-trace decode 평균은 {step_no_trace_decode - legacy_no_trace_decode:+.3f} ms/tok로 거의 개선되지 않았다. 따라서 attention cache concat은 trace-forced sync에서는 큰 영향을 주지만, 현재 end-to-end decode gap의 주원인이라고 보기는 어렵다.",
        "- 현재 증거상 남은 핵심 차이는 full-attention KV cache 자체보다는 linear attention 안의 quantized projection/conv/out-proj 경로와 그 주변 graph-context 비용이 더 크다. conv-state update delta도 존재하지만, 그것만 따로 떼어 최상위 원인으로 보기는 어렵다.",
        "",
    ])

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()