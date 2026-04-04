#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_TRACE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/base_stage_trace.json")
CUSTOM_TRACE = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/custom_stage_trace.json")
SUMMARY_MD = Path("/Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/summary.md")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "rows": 0,
            "avg_prefill_ms": 0.0,
            "avg_decode_ms": 0.0,
            "avg_wall_ms": 0.0,
            "avg_throughput": 0.0,
        }

    count = len(rows)
    return {
        "rows": count,
        "avg_prefill_ms": round(sum(float(row["prefill_ms"]) for row in rows) / count, 3),
        "avg_decode_ms": round(sum(float(row["decode_ms"]) for row in rows) / count, 3),
        "avg_wall_ms": round(sum(float(row["wall_ms"]) for row in rows) / count, 3),
        "avg_throughput": round(sum(float(row["throughput"]) for row in rows) / count, 3),
    }


def mean_stage_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    totals: dict[str, dict[str, dict[str, float]]] = {
        "prefill": defaultdict(lambda: {"calls": 0.0, "dispatch_ms": 0.0, "sync_ms": 0.0}),
        "decode": defaultdict(lambda: {"calls": 0.0, "dispatch_ms": 0.0, "sync_ms": 0.0}),
    }
    count = len(rows)
    for row in rows:
        for phase in ("prefill", "decode"):
            for stage, stats in row["stage_trace"][phase].items():
                acc = totals[phase][stage]
                acc["calls"] += float(stats["calls"])
                acc["dispatch_ms"] += float(stats["dispatch_ms"])
                acc["sync_ms"] += float(stats["sync_ms"])

    if count == 0:
        return {"prefill": {}, "decode": {}}

    out: dict[str, dict[str, dict[str, float]]] = {"prefill": {}, "decode": {}}
    for phase, stage_map in totals.items():
        for stage, stats in stage_map.items():
            out[phase][stage] = {
                "calls": round(stats["calls"] / count, 3),
                "dispatch_ms": round(stats["dispatch_ms"] / count, 3),
                "sync_ms": round(stats["sync_ms"] / count, 3),
            }
    return out


def rank_stage_deltas(base: dict[str, Any], custom: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    base_stage = mean_stage_stats(base["rows"])[phase]
    custom_stage = mean_stage_stats(custom["rows"])[phase]
    names = sorted(set(base_stage.keys()) | set(custom_stage.keys()))
    rows: list[dict[str, Any]] = []
    for name in names:
        base_sync = float(base_stage.get(name, {}).get("sync_ms", 0.0))
        custom_sync = float(custom_stage.get(name, {}).get("sync_ms", 0.0))
        delta = round(custom_sync - base_sync, 3)
        ratio = round((custom_sync / base_sync), 3) if base_sync > 0.0 else None
        rows.append(
            {
                "stage": name,
                "base_sync_ms": round(base_sync, 3),
                "custom_sync_ms": round(custom_sync, 3),
                "delta_sync_ms": delta,
                "ratio": ratio,
            }
        )
    rows.sort(key=lambda row: row["delta_sync_ms"], reverse=True)
    return rows


def render_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [title, "", "| Stage | Base sync ms | Custom sync ms | Delta ms | Ratio |", "|------|-------------:|---------------:|---------:|------:|"]
    for row in rows:
        ratio = "n/a" if row["ratio"] is None else f"{row['ratio']:.3f}x"
        lines.append(
            f"| {row['stage']} | {row['base_sync_ms']:.3f} | {row['custom_sync_ms']:.3f} | {row['delta_sync_ms']:.3f} | {ratio} |"
        )
    lines.append("")
    return lines


def render_summary(base: dict[str, Any], custom: dict[str, Any]) -> str:
    base_summary = base.get("summary", summarize_rows(base["rows"]))
    custom_summary = custom.get("summary", summarize_rows(custom["rows"]))
    base_mean = mean_stage_stats(base["rows"])
    custom_mean = mean_stage_stats(custom["rows"])
    prefill_rows = rank_stage_deltas(base, custom, "prefill")
    decode_rows = rank_stage_deltas(base, custom, "decode")

    top_prefill = ", ".join(row["stage"] for row in prefill_rows[:3])
    top_decode = ", ".join(row["stage"] for row in decode_rows[:3])

    lines = [
        "# Plan 0 Stage Trace Summary",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Base trace: {BASE_TRACE}",
        f"- Custom trace: {CUSTOM_TRACE}",
        "- Notes: `full_attention` includes nested `rope` and `kv_cache_update`, and `linear_attention` includes nested `linear_cache_update`. Stage deltas are therefore interpreted per-stage, not summed into a single exclusive total.",
        "",
        "## Overall Metrics",
        "",
        "| Backend | Rows | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |",
        "|--------|-----:|---------------:|------------------:|------------:|----------:|",
        f"| base | {base_summary['rows']} | {base_summary['avg_prefill_ms']:.3f} | {base_summary['avg_decode_ms']:.3f} | {base_summary['avg_wall_ms']:.3f} | {base_summary['avg_throughput']:.3f} |",
        f"| custom | {custom_summary['rows']} | {custom_summary['avg_prefill_ms']:.3f} | {custom_summary['avg_decode_ms']:.3f} | {custom_summary['avg_wall_ms']:.3f} | {custom_summary['avg_throughput']:.3f} |",
        "",
        "## High-Level Read",
        "",
        f"Custom prefill is slower mainly in: {top_prefill}.",
        f"Custom decode is slower mainly in: {top_decode}.",
        (
            "Custom linear_attention also shows a large dispatch-side anomaly: "
            f"prefill dispatch {custom_mean['prefill'].get('linear_attention', {}).get('dispatch_ms', 0.0):.3f} ms "
            f"vs base {base_mean['prefill'].get('linear_attention', {}).get('dispatch_ms', 0.0):.3f} ms, "
            f"decode dispatch {custom_mean['decode'].get('linear_attention', {}).get('dispatch_ms', 0.0):.3f} ms "
            f"vs base {base_mean['decode'].get('linear_attention', {}).get('dispatch_ms', 0.0):.3f} ms."
        ),
        "",
    ]

    lines.extend(render_table("## Prefill Stage Delta", prefill_rows))
    lines.extend(render_table("## Decode Stage Delta", decode_rows))

    lines.extend(
        [
            "## Mean Stage Stats",
            "",
            "### Base Prefill",
            "",
            json.dumps(base_mean["prefill"], indent=2, ensure_ascii=False),
            "",
            "### Custom Prefill",
            "",
            json.dumps(custom_mean["prefill"], indent=2, ensure_ascii=False),
            "",
            "### Base Decode",
            "",
            json.dumps(base_mean["decode"], indent=2, ensure_ascii=False),
            "",
            "### Custom Decode",
            "",
            json.dumps(custom_mean["decode"], indent=2, ensure_ascii=False),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    base = load_json(BASE_TRACE)
    custom = load_json(CUSTOM_TRACE)
    SUMMARY_MD.write_text(render_summary(base, custom), encoding="utf-8")
    print(f"Wrote summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()