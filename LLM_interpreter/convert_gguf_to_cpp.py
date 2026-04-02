from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from gguf import GGUFReader

from convert_py_to_cpp import build_byte_to_unicode_map


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export a GGUF model into the SoC C++ manifest format using file-offset tensor records."
	)
	parser.add_argument("--gguf-file", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--model-id", default=None)
	parser.add_argument(
		"--config-json",
		type=Path,
		default=None,
		help="Optional HF config.json to embed as the manifest config. If omitted, a minimal config is synthesized from GGUF metadata.",
	)
	return parser.parse_args()


def _field_contents(reader: GGUFReader, key: str, default: Any = None) -> Any:
	field = reader.fields.get(key)
	if field is None:
		return default
	return field.contents()


def _split_merge(merge: str) -> dict[str, str] | None:
	parts = merge.split()
	if len(parts) != 2:
		return None
	return {"left": parts[0], "right": parts[1]}


def _build_special_tokens(tokens: list[str], eos_id: int | None, pad_id: int | None, bos_id: int | None) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
	special_tokens_map: dict[str, Any] = {}
	special_token_ids: dict[str, int] = {}

	def add_named(name: str, token_id: int | None) -> None:
		if token_id is None or token_id < 0 or token_id >= len(tokens):
			return
		special_tokens_map[name] = tokens[token_id]
		special_token_ids[name] = token_id

	add_named("eos_token", eos_id)
	add_named("pad_token", pad_id)
	add_named("bos_token", bos_id)

	added_tokens: list[dict[str, Any]] = []
	seen_ids: set[int] = set()

	for token_id, token in enumerate(tokens):
		is_special = (
			token.startswith("<|") and token.endswith("|>")
			or token in ("<think>", "</think>")
			or token_id in special_token_ids.values()
		)
		if not is_special or token_id in seen_ids:
			continue
		seen_ids.add(token_id)
		added_tokens.append(
			{
				"id": token_id,
				"content": token,
				"special": True,
				"single_word": False,
				"lstrip": False,
				"rstrip": False,
				"normalized": True,
			}
		)

	return special_tokens_map, special_token_ids, added_tokens


def _template_runtime(tokens: list[str]) -> dict[str, Any]:
	im_start = "<|im_start|>" if "<|im_start|>" in tokens else "<|im_start|>"
	im_end = "<|im_end|>" if "<|im_end|>" in tokens else "<|im_end|>"
	think_start = "<think>" if "<think>" in tokens else "<think>"
	think_end = "</think>" if "</think>" in tokens else "</think>"
	return {
		"type": "qwen3",
		"im_start": im_start,
		"im_end": im_end,
		"think_start": think_start,
		"think_end": think_end,
		"default_system_prompt": "You are a helpful assistant.",
	}


def _build_tokenizer_runtime(reader: GGUFReader) -> dict[str, Any]:
	tokens = list(_field_contents(reader, "tokenizer.ggml.tokens", []))
	merges_raw = list(_field_contents(reader, "tokenizer.ggml.merges", []))
	chat_template = _field_contents(reader, "tokenizer.chat_template", "") or ""
	eos_id = _field_contents(reader, "tokenizer.ggml.eos_token_id", None)
	pad_id = _field_contents(reader, "tokenizer.ggml.padding_token_id", None)
	bos_id = _field_contents(reader, "tokenizer.ggml.bos_token_id", None)
	context_length = int(_field_contents(reader, "qwen35.context_length", 0) or 0)

	special_tokens_map, special_token_ids, added_tokens = _build_special_tokens(tokens, eos_id, pad_id, bos_id)
	merges = []
	for merge in merges_raw:
		record = _split_merge(str(merge))
		if record is not None:
			merges.append(record)

	return {
		"format": "soc.cpp.tokenizer_runtime",
		"format_version": 1,
		"tokenizer_class": "GGUFTokenizer",
		"vocab_size": len(tokens),
		"model_max_length": context_length,
		"special_tokens_map": special_tokens_map,
		"special_token_ids": special_token_ids,
		"added_tokens": added_tokens,
		"vocab": [{"token": token, "id": token_id} for token_id, token in enumerate(tokens)],
		"chat_template": chat_template,
		"template_runtime": _template_runtime(tokens),
		"bpe_model": {
			"type": "bpe",
			"enabled": True,
			"unk_token": "",
			"continuing_subword_prefix": "",
			"end_of_word_suffix": "",
			"merges": merges,
		},
		"pre_tokenizer": {
			"type": "ByteLevel",
			"enabled": True,
			"byte_level": {
				"enabled": True,
				"add_prefix_space": False,
				"use_regex": True,
			},
		},
		"decoder": {
			"type": "Sequence",
			"enabled": True,
			"byte_level": {
				"enabled": True,
				"add_prefix_space": False,
				"trim_offsets": False,
				"use_regex": True,
				"byte_to_unicode": build_byte_to_unicode_map(),
			},
			"bpe": {
				"enabled": True,
				"suffix": "",
			},
		},
	}


def _build_qwen35_config_from_gguf(reader: GGUFReader) -> dict[str, Any]:
	block_count = int(_field_contents(reader, "qwen35.block_count", 0) or 0)
	full_attention_interval = int(_field_contents(reader, "qwen35.full_attention_interval", 4) or 4)
	head_dim = int(_field_contents(reader, "qwen35.attention.key_length", 0) or 0)
	rotary_dim = int(_field_contents(reader, "qwen35.rope.dimension_count", 0) or 0)
	partial_rotary_factor = float(rotary_dim) / float(head_dim) if head_dim else 0.25
	layer_types = [
		"full_attention" if (layer_index % full_attention_interval) == (full_attention_interval - 1) else "linear_attention"
		for layer_index in range(block_count)
	]

	return {
		"model_type": "qwen3_5",
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"tie_word_embeddings": False,
		"gguf_backend": {
			"format": "gguf",
			"architecture": _field_contents(reader, "general.architecture", ""),
			"file_type": _field_contents(reader, "general.file_type", None),
		},
		"text_config": {
			"model_type": "qwen3_5_text",
			"vocab_size": len(list(_field_contents(reader, "tokenizer.ggml.tokens", []))),
			"hidden_size": int(_field_contents(reader, "qwen35.embedding_length", 0) or 0),
			"intermediate_size": int(_field_contents(reader, "qwen35.feed_forward_length", 0) or 0),
			"num_hidden_layers": block_count,
			"num_attention_heads": int(_field_contents(reader, "qwen35.attention.head_count", 0) or 0),
			"num_key_value_heads": int(_field_contents(reader, "qwen35.attention.head_count_kv", 0) or 0),
			"head_dim": head_dim,
			"linear_num_key_heads": int(_field_contents(reader, "qwen35.ssm.group_count", 0) or 0),
			"linear_num_value_heads": int(_field_contents(reader, "qwen35.ssm.time_step_rank", 0) or 0),
			"linear_key_head_dim": int(_field_contents(reader, "qwen35.ssm.state_size", 0) or 0),
			"linear_value_head_dim": int(_field_contents(reader, "qwen35.ssm.state_size", 0) or 0),
			"linear_conv_kernel_dim": int(_field_contents(reader, "qwen35.ssm.conv_kernel", 0) or 0),
			"max_position_embeddings": int(_field_contents(reader, "qwen35.context_length", 0) or 0),
			"full_attention_interval": full_attention_interval,
			"rms_norm_eps": float(_field_contents(reader, "qwen35.attention.layer_norm_rms_epsilon", 1e-6) or 1e-6),
			"mamba_ssm_dtype": "float32",
			"rope_parameters": {
				"rope_type": "default",
				"mrope_interleaved": True,
				"mrope_section": list(_field_contents(reader, "qwen35.rope.dimension_sections", [])),
				"rope_theta": float(_field_contents(reader, "qwen35.rope.freq_base", 10000000.0) or 10000000.0),
				"partial_rotary_factor": partial_rotary_factor,
			},
			"layer_types": layer_types,
		},
	}


def _link_or_copy(source: Path, destination: Path) -> None:
	destination.parent.mkdir(parents=True, exist_ok=True)
	if destination.exists():
		destination.unlink()
	try:
		os.link(source, destination)
	except OSError:
		shutil.copy2(source, destination)


def main() -> None:
	args = parse_args()
	gguf_file = args.gguf_file.resolve()
	if not gguf_file.exists():
		raise FileNotFoundError(f"gguf file does not exist: {gguf_file}")

	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	weights_dir = output_dir / "weights"
	tokenizer_dir = output_dir / "tokenizer"
	weights_dir.mkdir(parents=True, exist_ok=True)
	tokenizer_dir.mkdir(parents=True, exist_ok=True)

	reader = GGUFReader(gguf_file)
	tokenizer_runtime = _build_tokenizer_runtime(reader)
	tokenizer_runtime_path = tokenizer_dir / "tokenizer_runtime.json"
	tokenizer_runtime_path.write_text(json.dumps(tokenizer_runtime, indent=2, ensure_ascii=False), encoding="utf-8")

	if args.config_json is not None:
		config = json.loads(args.config_json.resolve().read_text(encoding="utf-8"))
	else:
		config = _build_qwen35_config_from_gguf(reader)
	generation_config = {
		"eos_token_id": _field_contents(reader, "tokenizer.ggml.eos_token_id", -1),
		"pad_token_id": _field_contents(reader, "tokenizer.ggml.padding_token_id", -1),
	}

	linked_gguf_path = weights_dir / gguf_file.name
	_link_or_copy(gguf_file, linked_gguf_path)

	tensor_records = []
	for tensor in reader.tensors:
		tensor_records.append(
			{
				"name": tensor.name,
				"file": (Path("weights") / gguf_file.name).as_posix(),
				"dtype": str(tensor.tensor_type).lower(),
				"shape": [int(dim) for dim in tensor.shape.tolist()],
				"file_offset": int(tensor.data_offset),
				"byte_size": int(tensor.n_bytes),
				"source_shard": gguf_file.name,
			}
		)

	gguf_metadata = {
		"field_count": len(reader.fields),
		"tensor_count": len(reader.tensors),
		"fields": {name: _field_contents(reader, name, None) for name in reader.fields.keys()},
	}
	(output_dir / "gguf_metadata.json").write_text(json.dumps(gguf_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

	manifest = {
		"format": "soc.cpp.llm_export",
		"format_version": 2,
		"model_id": args.model_id,
		"source_dir": str(gguf_file.parent),
		"export_dtype": "gguf",
		"tensor_count": len(tensor_records),
		"config": config,
		"generation_config": generation_config,
		"tokenizer_runtime_file": "tokenizer/tokenizer_runtime.json",
		"tensors": tensor_records,
		"tokenizer": {
			"tokenizer_class": "GGUFTokenizer",
			"vocab_size": tokenizer_runtime["vocab_size"],
			"model_max_length": tokenizer_runtime["model_max_length"],
			"runtime_file": "tokenizer_runtime.json",
			"copied_files": [],
			"chat_template": tokenizer_runtime["chat_template"],
			"special_tokens_map": tokenizer_runtime["special_tokens_map"],
			"special_token_ids": tokenizer_runtime["special_token_ids"],
		},
	}
	manifest_path = output_dir / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"C++ GGUF export written to {output_dir}")
	print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
	main()
