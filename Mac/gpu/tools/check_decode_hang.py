#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded full-GPU decode and fail on hang/timeout.")
    parser.add_argument("--infer-bin", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--output", required=True)
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


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(parse_env_overrides(args.gpu_env))
    command = [
        str(Path(args.infer_bin).resolve()),
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--layer",
        "-1",
        "--json",
        "--output-file",
        str(output_path),
    ]
    started_at = time.perf_counter()
    result = {
        "command": command,
        "timeout_seconds": args.timeout_seconds,
        "env": {key: env[key] for key in sorted(parse_env_overrides(args.gpu_env))},
    }
    try:
        subprocess.run(command, check=True, env=env, timeout=args.timeout_seconds)
        result["status"] = "ok"
        result["elapsed_wall_ms"] = (time.perf_counter() - started_at) * 1000.0
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["elapsed_wall_ms"] = (time.perf_counter() - started_at) * 1000.0
    except subprocess.CalledProcessError as exc:
        result["status"] = "error"
        result["returncode"] = exc.returncode
        result["elapsed_wall_ms"] = (time.perf_counter() - started_at) * 1000.0
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
