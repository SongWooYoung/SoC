#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


BASE_TRACE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/base_stage_trace.json")
CUSTOM_TRACE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/metal_kernel_stage_trace.json")
NO_TRACE_COMPARE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_9b_mlx_lib_vs_custom.json")
OUTPUT_JSON = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_custom_decode_gap.json")
OUTPUT_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_custom_decode_gap.md")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def average_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    count = len(rows)
    if count == 0:
        return {
            "rows": 0,
            "avg_prefill_ms": 0.0,
            "avg_decode_ms": 0.0,
            "avg_wall_ms": 0.0,
            "avg_throughput": 0.0,
        }
    return {
        "rows": count,
        "avg_prefill_ms": round(mean(float(row["prefill_ms"]) for row in rows), 3),
        "avg_decode_ms": round(mean(float(row["decode_ms"]) for row in rows), 3),
        "avg_wall_ms": round(mean(float(row["wall_ms"]) for row in rows), 3),
        "avg_throughput": round(mean(float(row["throughput"]) for row in rows), 3),
    }


def mean_stage_stats(rows: list[dict[str, Any]], phase: str) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "dispatch_ms": 0.0, "sync_ms": 0.0})
    count = len(rows)
    if count == 0:
        return {}
    for row in rows:
        for stage, stats in row["stage_trace"][phase].items():
            acc = totals[stage]
            acc["calls"] += float(stats.get("calls", 0.0))
            acc["dispatch_ms"] += float(stats.get("dispatch_ms", 0.0))
            acc["sync_ms"] += float(stats.get("sync_ms", 0.0))
    return {
        stage: {
            "calls": round(stats["calls"] / count, 3),
            "dispatch_ms": round(stats["dispatch_ms"] / count, 3),
            "sync_ms": round(stats["sync_ms"] / count, 3),
        }
        for stage, stats in totals.items()
    }


def pair_rows(base_rows: list[dict[str, Any]], custom_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    custom_by_id = {row["id"]: row for row in custom_rows}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for base_row in base_rows:
        custom_row = custom_by_id.get(base_row["id"])
        if custom_row is not None:
            pairs.append((base_row, custom_row))
    return pairs


def build_stage_delta_rows(pairs: list[tuple[dict[str, Any], dict[str, Any]]], phase: str, metric: str) -> list[dict[str, Any]]:
    per_stage: dict[str, list[float]] = defaultdict(list)
    for base_row, custom_row in pairs:
        stage_names = sorted(set(base_row["stage_trace"][phase].keys()) | set(custom_row["stage_trace"][phase].keys()))
        for stage in stage_names:
            base_value = float(base_row["stage_trace"][phase].get(stage, {}).get(metric, 0.0))
            custom_value = float(custom_row["stage_trace"][phase].get(stage, {}).get(metric, 0.0))
            per_stage[stage].append(custom_value - base_value)

    rows: list[dict[str, Any]] = []
    matched_count = len(pairs)
    for stage, deltas in per_stage.items():
        positive_count = sum(1 for value in deltas if value > 0.0)
        rows.append(
            {
                "stage": stage,
                "avg_delta_ms": round(mean(deltas), 3),
                "min_delta_ms": round(min(deltas), 3),
                "max_delta_ms": round(max(deltas), 3),
                "positive_count": positive_count,
                "matched_count": matched_count,
            }
        )
    rows.sort(key=lambda row: row["avg_delta_ms"], reverse=True)
    return rows


def to_markdown_table(rows: list[dict[str, Any]], value_key: str) -> list[str]:
    lines = [
        "| Stage | Avg delta ms | Min delta ms | Max delta ms | Positive prompts |",
        "|------|-------------:|-------------:|-------------:|-----------------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row[value_key]:.3f} | {row['min_delta_ms']:.3f} | {row['max_delta_ms']:.3f} | {row['positive_count']}/{row['matched_count']} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    trace = report["trace_decode_gap"]
    runner = report["runner_gap"]
    prompt_rows = report["prompt_decode_delta"]
    top_sync = ", ".join(row["stage"] for row in trace["decode_sync_delta_by_stage"][:3])
    top_dispatch = ", ".join(row["stage"] for row in trace["decode_dispatch_delta_by_stage"][:3])
    sync_by_stage = {row["stage"]: row for row in trace["decode_sync_delta_by_stage"]}
    dispatch_by_stage = {row["stage"]: row for row in trace["decode_dispatch_delta_by_stage"]}
    lines = [
        "# Base vs Custom Decode Gap",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Base trace: {BASE_TRACE}",
        f"- Custom trace: {CUSTOM_TRACE}",
        f"- No-trace comparison: {NO_TRACE_COMPARE}",
        "- Scope: decode에서 base official MLX와 custom이 어디서 다르게 느려지는지만 정리한다.",
        "",
        "## Overall Decode Gap",
        "",
        f"- No-trace 최신 평균: base {runner['base_no_trace_decode_ms']:.3f} ms/tok, custom {runner['custom_no_trace_decode_ms']:.3f} ms/tok, delta +{runner['no_trace_decode_gap_ms']:.3f} ms/tok.",
        f"- Trace 평균: base {trace['base_trace_summary']['avg_decode_ms']:.3f} ms/tok, custom {trace['custom_trace_summary']['avg_decode_ms']:.3f} ms/tok, delta +{trace['trace_decode_gap_ms']:.3f} ms/tok.",
        f"- Runner benefit: base {runner['base_runner_benefit_ms']:.3f} ms/tok, custom {runner['custom_runner_benefit_ms']:.3f} ms/tok.",
        "",
        "## Prompt-Level Decode Delta",
        "",
        "| Prompt | Base decode ms/tok | Custom decode ms/tok | Delta ms/tok |",
        "|-------|--------------------:|---------------------:|-------------:|",
    ]
    for row in prompt_rows:
        lines.append(
            f"| {row['id']} | {row['base_decode_ms']:.3f} | {row['custom_decode_ms']:.3f} | {row['delta_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decode Sync Delta by Stage",
            "",
            f"상위 양의 delta stage: {top_sync}.",
            "",
        ]
    )
    lines.extend(to_markdown_table(trace["decode_sync_delta_by_stage"], "avg_delta_ms"))
    lines.extend(
        [
            "",
            "## Decode Dispatch Delta by Stage",
            "",
            f"host-side dispatch 차이가 큰 stage: {top_dispatch}.",
            "",
        ]
    )
    lines.extend(to_markdown_table(trace["decode_dispatch_delta_by_stage"], "avg_delta_ms"))
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- `linear_cache_update`는 decode sync에서 평균 +{sync_by_stage['linear_cache_update']['avg_delta_ms']:.3f} ms로 4/4 prompt 모두 base보다 높다. 다만 이 stage는 nested 계측이라 outer `linear_attention`와 단순 합산하면 안 된다.",
            f"- `mlp`는 decode sync에서 평균 +{sync_by_stage['mlp']['avg_delta_ms']:.3f} ms로 4/4 prompt 모두 base보다 느리다.",
            f"- `lm_head`는 decode sync에서 평균 +{sync_by_stage['lm_head']['avg_delta_ms']:.3f} ms로 4/4 prompt 모두 느리다.",
            f"- `full_attention`는 평균 +{sync_by_stage['full_attention']['avg_delta_ms']:.3f} ms로 남아 있지만, outer `linear_attention`는 평균 {sync_by_stage['linear_attention']['avg_delta_ms']:.3f} ms라서 decode gap의 주범이 linear_attention 전체라고 보기는 어렵다.",
            f"- dispatch 기준으로는 `linear_attention` 평균 +{dispatch_by_stage['linear_attention']['avg_delta_ms']:.3f} ms가 가장 커서 host-side graph composition 차이도 따로 남아 있다.",
            "- 결론: 현재 decode gap은 runner가 아니라 model-core 차이로 재현되며, 가장 일관되게 벌어지는 구간은 `mlp`, `lm_head`, 그리고 일부 `full_attention`/`linear_cache_update`다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    base = load_json(BASE_TRACE)
    custom = load_json(CUSTOM_TRACE)
    no_trace = load_json(NO_TRACE_COMPARE)

    base_rows = base["rows"]
    custom_rows = custom["rows"]
    pairs = pair_rows(base_rows, custom_rows)

    base_trace_summary = average_summary(base_rows)
    custom_trace_summary = average_summary(custom_rows)
    prompt_decode_delta = []
    for base_row, custom_row in pairs:
        prompt_decode_delta.append(
            {
                "id": base_row["id"],
                "base_decode_ms": round(float(base_row["decode_ms"]), 3),
                "custom_decode_ms": round(float(custom_row["decode_ms"]), 3),
                "delta_ms": round(float(custom_row["decode_ms"]) - float(base_row["decode_ms"]), 3),
            }
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trace_decode_gap": {
            "matched_prompts": len(pairs),
            "base_trace_summary": base_trace_summary,
            "custom_trace_summary": custom_trace_summary,
            "trace_decode_gap_ms": round(custom_trace_summary["avg_decode_ms"] - base_trace_summary["avg_decode_ms"], 3),
            "decode_sync_delta_by_stage": build_stage_delta_rows(pairs, "decode", "sync_ms"),
            "decode_dispatch_delta_by_stage": build_stage_delta_rows(pairs, "decode", "dispatch_ms"),
            "base_decode_mean_stage_stats": mean_stage_stats(base_rows, "decode"),
            "custom_decode_mean_stage_stats": mean_stage_stats(custom_rows, "decode"),
        },
        "runner_gap": {
            "base_no_trace_decode_ms": float(no_trace["mlx_library"]["summary"]["avg_decode_ms"]),
            "custom_no_trace_decode_ms": float(no_trace["custom"]["summary"]["avg_decode_ms"]),
            "no_trace_decode_gap_ms": round(
                float(no_trace["custom"]["summary"]["avg_decode_ms"]) - float(no_trace["mlx_library"]["summary"]["avg_decode_ms"]),
                3,
            ),
            "base_runner_benefit_ms": round(
                base_trace_summary["avg_decode_ms"] - float(no_trace["mlx_library"]["summary"]["avg_decode_ms"]),
                3,
            ),
            "custom_runner_benefit_ms": round(
                custom_trace_summary["avg_decode_ms"] - float(no_trace["custom"]["summary"]["avg_decode_ms"]),
                3,
            ),
        },
        "prompt_decode_delta": prompt_decode_delta,
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()