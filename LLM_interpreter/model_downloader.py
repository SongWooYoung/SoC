from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_ALLOW_PATTERNS = [
	"*.json",
	"*.safetensors",
	"*.model",
	"*.tiktoken",
	"*.py",
	"README.md",
	"LICENSE*",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download Qwen3-0.6B from Hugging Face and optionally export a C++-friendly bundle."
	)
	parser.add_argument(
		"--model-id",
		default=DEFAULT_MODEL_ID,
		help="Hugging Face repository id to download.",
	)
	parser.add_argument(
		"--download-dir",
		type=Path,
		default=Path("models/raw/qwen3-0.6b"),
		help="Directory where the Hugging Face snapshot will be stored.",
	)
	parser.add_argument(
		"--revision",
		default=None,
		help="Optional branch, tag, or commit hash.",
	)
	parser.add_argument(
		"--hf-token",
		default=None,
		help="Optional Hugging Face token. If omitted, HUGGINGFACE_HUB_TOKEN/HF_TOKEN are used.",
	)
	parser.add_argument(
		"--export-dir",
		type=Path,
		default=Path("models/cpp/qwen3-0.6b"),
		help="Directory where the exported C++ assets will be written.",
	)
	parser.add_argument(
		"--skip-export",
		action="store_true",
		help="Only download the model snapshot without generating the C++ export bundle.",
	)
	parser.add_argument(
		"--dtype",
		choices=("native", "float32", "float16"),
		default="native",
		help="Floating-point export dtype used when generating the C++ bundle.",
	)
	return parser.parse_args()


def resolve_hf_token(explicit_token: str | None) -> str | None:
	if explicit_token:
		return explicit_token
	return os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")


def load_snapshot_download() -> Any:
	try:
		huggingface_hub = importlib.import_module("huggingface_hub")
	except ImportError as exc:
		raise RuntimeError(
			"Missing downloader dependency. Install packages from LLM_interpreter/requirements.txt before downloading."
		) from exc

	return huggingface_hub.snapshot_download


def download_model_snapshot(model_id: str, download_dir: Path, revision: str | None, token: str | None) -> Path:
	download_dir = download_dir.resolve()
	download_dir.mkdir(parents=True, exist_ok=True)
	snapshot_download = load_snapshot_download()

	snapshot_path = snapshot_download(
		repo_id=model_id,
		local_dir=str(download_dir),
		revision=revision,
		token=token,
		allow_patterns=DEFAULT_ALLOW_PATTERNS,
	)

	metadata = {
		"model_id": model_id,
		"revision": revision,
		"snapshot_path": snapshot_path,
		"allow_patterns": DEFAULT_ALLOW_PATTERNS,
	}
	(download_dir / "download_metadata.json").write_text(
		json.dumps(metadata, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	return Path(snapshot_path)


def main() -> None:
	args = parse_args()
	token = resolve_hf_token(args.hf_token)

	snapshot_path = download_model_snapshot(
		model_id=args.model_id,
		download_dir=args.download_dir,
		revision=args.revision,
		token=token,
	)
	print(f"Downloaded snapshot: {snapshot_path}")

	if args.skip_export:
		return

	from convert_py_to_cpp import export_hf_checkpoint_for_cpp

	manifest_path = export_hf_checkpoint_for_cpp(
		model_dir=snapshot_path,
		output_dir=args.export_dir,
		export_dtype=args.dtype,
		model_id=args.model_id,
	)
	print(f"C++ export manifest: {manifest_path}")


if __name__ == "__main__":
	main()
