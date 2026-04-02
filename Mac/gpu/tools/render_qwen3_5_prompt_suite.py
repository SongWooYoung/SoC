#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render qwen3_5 prompt suite JSONL into pretty JSON and Markdown.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_markdown(rows: list[dict], source_name: str) -> str:
    lines: list[str] = []
    lines.append(f"# Prompt Suite Results")
    lines.append("")
    lines.append(f"- source: `{source_name}`")
    lines.append(f"- rows: `{len(rows)}`")
    lines.append("")
    for row in rows:
        lines.append(f"## {row['id']} ({row['category']})")
        lines.append("")
        lines.append("### Prompt")
        lines.append("")
        lines.append("```text")
        lines.append(row["prompt"])
        lines.append("```")
        lines.append("")
        lines.append("### Base")
        lines.append("")
        if row["base"].get("prepared_prompt"):
            lines.append("#### Prepared Prompt")
            lines.append("")
            lines.append("```text")
            lines.append(row["base"]["prepared_prompt"])
            lines.append("```")
            lines.append("")
        lines.append(f"- generated_token_ids: `{row['base']['generated_token_ids']}`")
        lines.append(f"- prefill_ms: `{row['base']['prefill_ms']}`")
        lines.append(f"- decode_ms: `{row['base']['decode_ms']}`")
        lines.append(f"- decode_tok_per_s: `{row['base']['decode_tok_per_s']}`")
        lines.append(f"- wall_ms: `{row['base']['wall_ms']}`")
        lines.append("")
        lines.append("```text")
        lines.append(row["base"]["generated_text"])
        lines.append("```")
        lines.append("")
        lines.append("### Cpp")
        lines.append("")
        if row["cpp"].get("prepared_prompt"):
            lines.append("#### Prepared Prompt")
            lines.append("")
            lines.append("```text")
            lines.append(row["cpp"]["prepared_prompt"])
            lines.append("```")
            lines.append("")
        lines.append(f"- generated_token_ids: `{row['cpp']['generated_token_ids']}`")
        lines.append(f"- prefill_ms: `{row['cpp']['prefill_ms']}`")
        lines.append(f"- decode_ms: `{row['cpp']['decode_ms']}`")
        lines.append(f"- decode_tok_per_s: `{row['cpp']['decode_tok_per_s']}`")
        lines.append(f"- wall_ms: `{row['cpp']['wall_ms']}`")
        lines.append(f"- gpu_ms: `{row['cpp']['gpu_ms']}`")
        lines.append(f"- wait_ms: `{row['cpp']['wait_ms']}`")
        lines.append("")
        lines.append("```text")
        lines.append(row["cpp"]["generated_text"])
        lines.append("```")
        lines.append("")
        lines.append("### Comparison")
        lines.append("")
        lines.append(f"- text_match: `{row['comparison']['text_match']}`")
        lines.append(f"- token_ids_match: `{row['comparison']['token_ids_match']}`")
        lines.append(f"- prefill_ms_ratio_cpp_over_base: `{row['comparison']['prefill_ms_ratio_cpp_over_base']}`")
        lines.append(f"- decode_tok_per_s_ratio_cpp_over_base: `{row['comparison']['decode_tok_per_s_ratio_cpp_over_base']}`")
        lines.append(f"- wall_ms_ratio_cpp_over_base: `{row['comparison']['wall_ms_ratio_cpp_over_base']}`")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.write_text(render_markdown(rows, args.input_jsonl.name), encoding="utf-8")


if __name__ == "__main__":
    main()
