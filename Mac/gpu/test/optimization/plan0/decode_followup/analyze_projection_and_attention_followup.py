#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_SPLIT_TRACE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_stage_trace_split.json")
CUSTOM_SPLIT_TRACE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_stage_trace_split.json")
BASE_PROJECTION = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_decode_projection_microbench.json")
CUSTOM_PROJECTION = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_decode_projection_microbench.json")
OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/projection_attention_followup.json")
OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/projection_attention_followup.md")

FULL_ATTENTION_STAGES = [
    "full_attention",
    "full_attention_q_proj",
    "full_attention_k_proj",
    "full_attention_v_proj",
    "full_attention_q_norm",
    "full_attention_k_norm",
    "full_attention_rope",
    "full_attention_cache_update",
    "full_attention_sdpa",
    "full_attention_o_proj",
]

LINEAR_CACHE_STAGES = [
    "linear_cache_update",
    "linear_cache_conv_state_update",
    "linear_cache_rec_state_update",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean_stage(rows: list[dict[str, Any]], phase: str, stage: str) -> dict[str, float]:
    total_calls = 0.0
    total_dispatch = 0.0
    total_sync = 0.0
    for row in rows:
        stats = row["stage_trace"][phase].get(stage, {})
        total_calls += float(stats.get("calls", 0.0))
        total_dispatch += float(stats.get("dispatch_ms", 0.0))
        total_sync += float(stats.get("sync_ms", 0.0))
    count = len(rows) or 1
    return {
        "calls": round(total_calls / count, 3),
        "dispatch_ms": round(total_dispatch / count, 3),
        "sync_ms": round(total_sync / count, 3),
    }


def compare_stage_group(base_rows: list[dict[str, Any]], custom_rows: list[dict[str, Any]], stages: list[str]) -> list[dict[str, Any]]:
    results = []
    for stage in stages:
        base = mean_stage(base_rows, "decode", stage)
        custom = mean_stage(custom_rows, "decode", stage)
        results.append(
            {
                "stage": stage,
                "base_dispatch_ms": base["dispatch_ms"],
                "custom_dispatch_ms": custom["dispatch_ms"],
                "dispatch_delta_ms": round(custom["dispatch_ms"] - base["dispatch_ms"], 3),
                "base_sync_ms": base["sync_ms"],
                "custom_sync_ms": custom["sync_ms"],
                "sync_delta_ms": round(custom["sync_ms"] - base["sync_ms"], 3),
            }
        )
    return results


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Stage | Base dispatch ms | Custom dispatch ms | Delta dispatch ms | Base sync ms | Custom sync ms | Delta sync ms |",
        "|------|------------------:|-------------------:|------------------:|-------------:|---------------:|--------------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['base_dispatch_ms']:.3f} | {row['custom_dispatch_ms']:.3f} | {row['dispatch_delta_ms']:.3f} | {row['base_sync_ms']:.3f} | {row['custom_sync_ms']:.3f} | {row['sync_delta_ms']:.3f} |"
        )
    return lines


def signed(value: float) -> str:
    return f"{value:+.3f}"


def main() -> None:
    base_split = load_json(BASE_SPLIT_TRACE)
    custom_split = load_json(CUSTOM_SPLIT_TRACE)
    base_projection = load_json(BASE_PROJECTION)
    custom_projection = load_json(CUSTOM_PROJECTION)

    full_attention = compare_stage_group(base_split["rows"], custom_split["rows"], FULL_ATTENTION_STAGES)
    linear_cache = compare_stage_group(base_split["rows"], custom_split["rows"], LINEAR_CACHE_STAGES)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "projection_microbench": {
            "base_mlp_average": base_projection["mlp_average"],
            "custom_mlp_average": custom_projection["mlp_average"],
            "mlp_sync_delta_ms": round(custom_projection["mlp_average"]["sync_ms"] - base_projection["mlp_average"]["sync_ms"], 3),
            "base_lm_head_decode": base_projection["lm_head_decode"],
            "custom_lm_head_decode": custom_projection["lm_head_decode"],
            "lm_head_sync_delta_ms": round(custom_projection["lm_head_decode"]["sync_ms"] - base_projection["lm_head_decode"]["sync_ms"], 3),
        },
        "linear_cache_split_decode": linear_cache,
        "full_attention_split_decode": full_attention,
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Projection And Attention Follow-Up",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Base split trace: {BASE_SPLIT_TRACE}",
        f"- Custom split trace: {CUSTOM_SPLIT_TRACE}",
        f"- Base projection microbench: {BASE_PROJECTION}",
        f"- Custom projection microbench: {CUSTOM_PROJECTION}",
        "",
        "## Decode Projection Microbench",
        "",
        f"- MLP average sync: base {base_projection['mlp_average']['sync_ms']:.3f} ms, custom {custom_projection['mlp_average']['sync_ms']:.3f} ms, delta {signed(report['projection_microbench']['mlp_sync_delta_ms'])} ms.",
        f"- LM head decode sync: base {base_projection['lm_head_decode']['sync_ms']:.3f} ms, custom {custom_projection['lm_head_decode']['sync_ms']:.3f} ms, delta {signed(report['projection_microbench']['lm_head_sync_delta_ms'])} ms.",
        "",
        "## Linear Cache Split Decode",
        "",
    ]
    lines.extend(render_table(linear_cache))
    lines.extend([
        "",
        "## Full Attention Split Decode",
        "",
    ])
    lines.extend(render_table(full_attention))
    lines.extend([
        "",
        "## Findings",
        "",
        f"- `linear_cache_conv_state_update` delta는 {next(row['sync_delta_ms'] for row in linear_cache if row['stage'] == 'linear_cache_conv_state_update'):.3f} ms이고, `linear_cache_rec_state_update` delta는 {next(row['sync_delta_ms'] for row in linear_cache if row['stage'] == 'linear_cache_rec_state_update'):.3f} ms다.",
        f"- isolated decode projection microbench에서는 `mlp`와 `lm_head` 자체가 base보다 느리지 않았다. 따라서 full-model trace의 `mlp`/`lm_head` delta는 순수 projection kernel 자체보다 상위 graph/context 차이와 더 관련 있을 가능성이 높다.",
        f"- `full_attention` 내부에서는 `full_attention_q_proj`, `full_attention_cache_update`, `full_attention_o_proj`가 양의 sync delta로 남고, `full_attention_sdpa` 자체는 거의 차이가 없었다.",
        "- 이 문서는 base와 custom이 decode에서 실제로 갈라지는 sub-stage만 다시 확인하기 위한 follow-up 산출물이다.",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()