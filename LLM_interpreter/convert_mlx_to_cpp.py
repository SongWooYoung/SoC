from __future__ import annotations

import argparse
from pathlib import Path

from convert_py_to_cpp import export_hf_checkpoint_for_cpp


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export an MLX-style local model directory into the SoC C++ manifest format."
	)
	parser.add_argument("--model-dir", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--model-id", default=None)
	parser.add_argument("--dtype", choices=("native", "float32", "float16"), default="native")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model_dir = args.model_dir.resolve()
	if not model_dir.exists():
		raise FileNotFoundError(f"model directory does not exist: {model_dir}")

	# Most MLX model repos still materialize tokenizer/config plus safetensors shards.
	# Reuse the HF exporter path when those files are present.
	has_safetensors = any(model_dir.glob("*.safetensors")) or (model_dir / "model.safetensors.index.json").exists()
	if not has_safetensors:
		raise RuntimeError(
			"Unsupported MLX directory layout for direct export. Expected *.safetensors or model.safetensors.index.json."
		)

	manifest_path = export_hf_checkpoint_for_cpp(
		model_dir=model_dir,
		output_dir=args.output_dir.resolve(),
		export_dtype=args.dtype,
		model_id=args.model_id,
	)
	print(f"C++ export manifest: {manifest_path}")


if __name__ == "__main__":
	main()
