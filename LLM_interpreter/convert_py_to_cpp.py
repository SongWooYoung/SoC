from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


TOKENIZER_ASSET_NAMES = (
	"tokenizer.json",
	"tokenizer_config.json",
	"special_tokens_map.json",
	"generation_config.json",
	"config.json",
	"vocab.json",
	"merges.txt",
	"added_tokens.json",
	"tokenizer.model",
	"chat_template.jinja",
)


@dataclass
class TensorExportRecord:
	name: str
	file: str
	dtype: str
	shape: list[int]
	byte_size: int
	source_shard: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export a Hugging Face causal LM checkpoint into a C++-friendly weight/tokenizer bundle."
	)
	parser.add_argument(
		"--model-dir",
		type=Path,
		required=True,
		help="Directory containing the downloaded Hugging Face model snapshot.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		required=True,
		help="Directory where the exported C++ assets will be written.",
	)
	parser.add_argument(
		"--model-id",
		default=None,
		help="Optional Hugging Face repository id to store in the export manifest.",
	)
	parser.add_argument(
		"--dtype",
		choices=("native", "float32", "float16"),
		default="native",
		help="Target dtype for exported floating-point tensors.",
	)
	return parser.parse_args()


def load_export_dependencies() -> tuple[Any, Any, Any]:
	try:
		torch_module = importlib.import_module("torch")
		safetensors_module = importlib.import_module("safetensors")
		transformers_module = importlib.import_module("transformers")
	except ImportError as exc:
		raise RuntimeError(
			"Missing export dependency. Install packages from LLM_interpreter/requirements.txt before exporting."
		) from exc

	return torch_module, safetensors_module.safe_open, transformers_module.AutoTokenizer


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None

	return json.loads(path.read_text(encoding="utf-8"))


def discover_safetensor_files(model_dir: Path) -> list[Path]:
	index_path = model_dir / "model.safetensors.index.json"
	if index_path.exists():
		index_data = json.loads(index_path.read_text(encoding="utf-8"))
		shard_names: list[str] = []
		for shard_name in index_data.get("weight_map", {}).values():
			if shard_name not in shard_names:
				shard_names.append(shard_name)
		return [model_dir / shard_name for shard_name in shard_names]

	shard_files = sorted(model_dir.glob("*.safetensors"))
	if shard_files:
		return shard_files

	raise FileNotFoundError(f"No safetensors files found under {model_dir}")


def sanitize_tensor_name(name: str, used_names: set[str]) -> str:
	sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-")
	if not sanitized:
		sanitized = "tensor"

	candidate = sanitized
	suffix = 1
	while candidate in used_names:
		suffix += 1
		candidate = f"{sanitized}_{suffix}"

	used_names.add(candidate)
	return candidate


def torch_dtype_name(dtype: Any, torch_module: Any) -> str:
	dtype_names = {
		torch_module.float16: "float16",
		torch_module.float32: "float32",
		torch_module.float64: "float64",
		torch_module.bfloat16: "bfloat16",
		torch_module.int8: "int8",
		torch_module.uint8: "uint8",
		torch_module.int16: "int16",
		torch_module.int32: "int32",
		torch_module.int64: "int64",
		torch_module.bool: "bool",
	}
	if dtype not in dtype_names:
		raise TypeError(f"Unsupported tensor dtype for export: {dtype}")
	return dtype_names[dtype]


def convert_tensor_bytes(tensor: Any, export_dtype: str, torch_module: Any) -> tuple[bytes, str, list[int]]:
	tensor = tensor.detach().contiguous().cpu()
	original_dtype_name = torch_dtype_name(tensor.dtype, torch_module)
	shape = list(tensor.shape)

	if tensor.dtype.is_floating_point:
		if export_dtype == "float32":
			converted = tensor.to(torch_module.float32).contiguous().numpy()
			return converted.tobytes(), "float32", shape
		if export_dtype == "float16":
			converted = tensor.to(torch_module.float16).contiguous().numpy()
			return converted.tobytes(), "float16", shape
		if tensor.dtype == torch_module.bfloat16:
			raw = tensor.view(torch_module.uint16).numpy()
			return raw.tobytes(), "bfloat16", shape

		converted = tensor.numpy()
		return converted.tobytes(), original_dtype_name, shape

	converted = tensor.numpy()
	return converted.tobytes(), original_dtype_name, shape


def copy_tokenizer_assets(model_dir: Path, tokenizer_dir: Path, auto_tokenizer_cls: Any) -> list[str]:
	tokenizer_dir.mkdir(parents=True, exist_ok=True)
	copied_files: list[str] = []

	for asset_name in TOKENIZER_ASSET_NAMES:
		source = model_dir / asset_name
		if source.exists():
			shutil.copy2(source, tokenizer_dir / asset_name)
			copied_files.append(asset_name)

	if not (tokenizer_dir / "tokenizer.json").exists():
		tokenizer = auto_tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
		tokenizer.backend_tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
		copied_files.append("tokenizer.json")

	return copied_files


def build_tokenizer_manifest(model_dir: Path, copied_files: list[str], auto_tokenizer_cls: Any) -> dict[str, Any]:
	tokenizer = auto_tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
	special_tokens_map = tokenizer.special_tokens_map

	return {
		"tokenizer_class": tokenizer.__class__.__name__,
		"vocab_size": tokenizer.vocab_size,
		"model_max_length": tokenizer.model_max_length,
		"bos_token": tokenizer.bos_token,
		"eos_token": tokenizer.eos_token,
		"pad_token": tokenizer.pad_token,
		"unk_token": tokenizer.unk_token,
		"special_tokens_map": special_tokens_map,
		"special_token_ids": build_special_token_ids(tokenizer, special_tokens_map),
		"chat_template": getattr(tokenizer, "chat_template", None),
		"copied_files": copied_files,
	}


def build_special_token_ids(tokenizer: Any, special_tokens_map: dict[str, Any]) -> dict[str, int]:
	special_token_ids: dict[str, int] = {}
	for token_name, token_value in special_tokens_map.items():
		if not isinstance(token_value, str):
			continue

		try:
			token_id = tokenizer.convert_tokens_to_ids(token_value)
		except KeyError:
			continue

		if token_id is None:
			continue

		special_token_ids[token_name] = int(token_id)

	return special_token_ids


def serialize_added_tokens(tokenizer: Any) -> list[dict[str, Any]]:
	added_tokens: list[dict[str, Any]] = []
	for token_id, token_info in getattr(tokenizer, "added_tokens_decoder", {}).items():
		content = getattr(token_info, "content", None)
		if content is None:
			content = str(token_info)

		added_tokens.append(
			{
				"id": int(token_id),
				"content": content,
				"special": bool(getattr(token_info, "special", False)),
				"single_word": bool(getattr(token_info, "single_word", False)),
				"lstrip": bool(getattr(token_info, "lstrip", False)),
				"rstrip": bool(getattr(token_info, "rstrip", False)),
				"normalized": bool(getattr(token_info, "normalized", True)),
			}
		)

	added_tokens.sort(key=lambda item: item["id"])
	return added_tokens


def build_vocab_entries(tokenizer: Any) -> list[dict[str, Any]]:
	vocab = tokenizer.get_vocab()
	vocab_entries = [{"token": token, "id": int(token_id)} for token, token_id in vocab.items()]
	vocab_entries.sort(key=lambda item: item["id"])
	return vocab_entries


def detect_qwen3_template_runtime(tokenizer: Any) -> dict[str, Any]:
	special_tokens_map = tokenizer.special_tokens_map
	im_start = special_tokens_map.get("additional_special_tokens", ["<|im_start|>"])
	im_start_token = "<|im_start|>"
	im_end_token = "<|im_end|>"
	if isinstance(im_start, list):
		for token in im_start:
			if token == "<|im_start|>":
				im_start_token = token
			if token == "<|im_end|>":
				im_end_token = token

	think_start = "<think>"
	think_end = "</think>"
	for token in getattr(tokenizer, "additional_special_tokens", []) or []:
		if token == "<think>":
			think_start = token
		if token == "</think>":
			think_end = token

	return {
		"type": "qwen3",
		"im_start": im_start_token,
		"im_end": im_end_token,
		"think_start": think_start,
		"think_end": think_end,
		"default_system_prompt": "You are a helpful assistant.",
	}


def build_tokenizer_runtime_manifest(tokenizer: Any, copied_files: list[str]) -> dict[str, Any]:
	special_tokens_map = tokenizer.special_tokens_map
	return {
		"format": "soc.cpp.tokenizer_runtime",
		"format_version": 1,
		"tokenizer_class": tokenizer.__class__.__name__,
		"vocab_size": tokenizer.vocab_size,
		"model_max_length": tokenizer.model_max_length,
		"special_tokens_map": special_tokens_map,
		"special_token_ids": build_special_token_ids(tokenizer, special_tokens_map),
		"added_tokens": serialize_added_tokens(tokenizer),
		"vocab": build_vocab_entries(tokenizer),
		"chat_template": getattr(tokenizer, "chat_template", None),
		"template_runtime": detect_qwen3_template_runtime(tokenizer),
		"copied_files": copied_files,
	}


def write_tokenizer_runtime(tokenizer_dir: Path, runtime_manifest: dict[str, Any]) -> str:
	runtime_path = tokenizer_dir / "tokenizer_runtime.json"
	runtime_path.write_text(json.dumps(runtime_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
	return runtime_path.name


def export_weights(model_dir: Path, weights_dir: Path, export_dtype: str, torch_module: Any, safe_open_fn: Any) -> list[TensorExportRecord]:
	weights_dir.mkdir(parents=True, exist_ok=True)
	records: list[TensorExportRecord] = []
	used_file_names: set[str] = set()

	for shard_path in discover_safetensor_files(model_dir):
		with safe_open_fn(str(shard_path), framework="pt", device="cpu") as handle:
			for tensor_name in handle.keys():
				tensor = handle.get_tensor(tensor_name)
				tensor_bytes, tensor_dtype, shape = convert_tensor_bytes(tensor, export_dtype, torch_module)
				file_stem = sanitize_tensor_name(tensor_name, used_file_names)
				relative_path = Path("weights") / f"{file_stem}.bin"
				destination = weights_dir / f"{file_stem}.bin"
				destination.write_bytes(tensor_bytes)

				records.append(
					TensorExportRecord(
						name=tensor_name,
						file=relative_path.as_posix(),
						dtype=tensor_dtype,
						shape=shape,
						byte_size=len(tensor_bytes),
						source_shard=shard_path.name,
					)
				)

	return records


def export_hf_checkpoint_for_cpp(
	model_dir: Path,
	output_dir: Path,
	export_dtype: str = "native",
	model_id: str | None = None,
) -> Path:
	torch_module, safe_open_fn, auto_tokenizer_cls = load_export_dependencies()

	model_dir = model_dir.resolve()
	output_dir = output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	tokenizer_dir = output_dir / "tokenizer"
	weights_dir = output_dir / "weights"

	copied_tokenizer_files = copy_tokenizer_assets(model_dir, tokenizer_dir, auto_tokenizer_cls)
	tokenizer_manifest = build_tokenizer_manifest(model_dir, copied_tokenizer_files, auto_tokenizer_cls)
	tokenizer = auto_tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
	tokenizer_runtime = build_tokenizer_runtime_manifest(tokenizer, copied_tokenizer_files)
	tokenizer_runtime_file = write_tokenizer_runtime(tokenizer_dir, tokenizer_runtime)
	tokenizer_manifest["runtime_file"] = tokenizer_runtime_file
	tensor_records = export_weights(model_dir, weights_dir, export_dtype, torch_module, safe_open_fn)

	config = load_json_if_exists(model_dir / "config.json")
	generation_config = load_json_if_exists(model_dir / "generation_config.json")
	source_index = load_json_if_exists(model_dir / "model.safetensors.index.json")

	manifest = {
		"format": "soc.cpp.llm_export",
		"format_version": 2,
		"model_id": model_id,
		"source_dir": str(model_dir),
		"export_dtype": export_dtype,
		"tensor_count": len(tensor_records),
		"config": config,
		"generation_config": generation_config,
		"source_index": source_index,
		"tokenizer_runtime_file": f"tokenizer/{tokenizer_runtime_file}",
		"tensors": [asdict(record) for record in tensor_records],
		"tokenizer": tokenizer_manifest,
	}

	manifest_path = output_dir / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
	return manifest_path


def main() -> None:
	args = parse_args()
	manifest_path = export_hf_checkpoint_for_cpp(
		model_dir=args.model_dir,
		output_dir=args.output_dir,
		export_dtype=args.dtype,
		model_id=args.model_id,
	)
	print(f"C++ export written to {manifest_path.parent}")
	print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
	main()
