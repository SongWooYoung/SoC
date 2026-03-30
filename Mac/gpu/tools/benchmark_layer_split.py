#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark gpu_infer across GPU layer split settings.")
    parser.add_argument("--infer-bin", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def load_layer_count(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return int(manifest["config"]["num_hidden_layers"])


def run_case(infer_bin: Path, manifest: Path, prompt: str, max_new_tokens: int, layer: int, output_path: Path) -> dict:
    command = [
        str(infer_bin),
        "--manifest",
        str(manifest),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--layer",
        str(layer),
        "--json",
        "--output-file",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_preview(text: str, max_chars: int = 48) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def write_report(report_path: Path, prompt: str, max_new_tokens: int, rows: list[dict]) -> None:
    lines = [
        "# Layer Split Benchmark",
        "",
        f"- Prompt: `{prompt}`",
        f"- max_new_tokens: `{max_new_tokens}`",
        f"- Cases: `{len(rows)}` (`1..N-1` plus `-1` full GPU)",
        "",
        "| requested_layer | mode | resolved_gpu_layers | wall_ms | gpu_ms | tokens_per_sec | preview |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {requested_layer} | {mode} | {resolved_gpu_layers} | {wall_ms:.2f} | {gpu_ms:.2f} | {tokens_per_sec:.3f} | {preview} |".format(
                requested_layer=row["requested_layer"],
                mode=row["mode"],
                resolved_gpu_layers=row["resolved_gpu_layers"],
                wall_ms=row["wall_ms"],
                gpu_ms=row["gpu_ms"],
                tokens_per_sec=row["tokens_per_sec"],
                preview=format_preview(row["generated_text"]),
            )
        )

    fastest = min(rows, key=lambda row: row["wall_ms"])
    lines.extend(
        [
            "",
            "## Fastest Case",
            "",
            f"- requested_layer: `{fastest['requested_layer']}`",
            f"- mode: `{fastest['mode']}`",
            f"- wall_ms: `{fastest['wall_ms']:.2f}`",
            f"- gpu_ms: `{fastest['gpu_ms']:.2f}`",
            f"- tokens_per_sec: `{fastest['tokens_per_sec']:.3f}`",
        ]
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

    layer_count = load_layer_count(manifest)
    requested_layers = list(range(1, layer_count)) + [-1]
    rows: list[dict] = []
    for layer in requested_layers:
        output_path = output_dir / f"layer_{layer}.json"
        payload = run_case(infer_bin, manifest, args.prompt, args.max_new_tokens, layer, output_path)
        timing = payload["timing"]
        generated_token_count = max(1, len(payload["generated_token_ids"]))
        rows.append(
            {
                "requested_layer": payload["execution"]["requested_layer"],
                "mode": payload["execution"]["mode"],
                "resolved_gpu_layers": payload["execution"]["resolved_gpu_layers"],
                "wall_ms": float(timing["wall_ms"]),
                "gpu_ms": float(timing["gpu_ms"]),
                "tokens_per_sec": generated_token_count / max(float(timing["wall_ms"]) / 1000.0, 1.0e-9),
                "generated_text": payload["generated_text"],
            }
        )

    write_report(report_path, args.prompt, args.max_new_tokens, rows)


if __name__ == "__main__":
    main()