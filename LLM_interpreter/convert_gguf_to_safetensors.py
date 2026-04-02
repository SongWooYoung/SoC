from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.numpy import save_file

from convert_gguf_to_cpp import (
    _build_qwen35_config_from_gguf,
    _build_tokenizer_runtime,
    _field_contents,
)


@dataclass(frozen=True)
class MappingEntry:
    gguf_name: str
    hf_name: str
    expected_shape: tuple[int, ...]
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Qwen3.5 GGUF -> HF-style safetensors converter."
    )
    parser.add_argument("--gguf-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=2.0,
        help="Approximate safetensors shard size limit in GiB.",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="Optional mapped tensor limit for smoke tests.",
    )
    parser.add_argument(
        "--include-regex",
        default="",
        help="Optional regex over HF tensor names to restrict conversion.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write config/tokenizer/map/report without writing safetensors shards.",
    )
    return parser.parse_args()


def _tensor_field(reader: GGUFReader, key: str, default: Any) -> Any:
    value = _field_contents(reader, key, default)
    return default if value is None else value


def _layer_type(reader: GGUFReader, layer_index: int) -> str:
    interval = int(_tensor_field(reader, "qwen35.full_attention_interval", 4))
    return "full_attention" if (layer_index % interval) == (interval - 1) else "linear_attention"


def _build_mapping(reader: GGUFReader) -> list[MappingEntry]:
    vocab_size = len(list(_tensor_field(reader, "tokenizer.ggml.tokens", [])))
    hidden_size = int(_tensor_field(reader, "qwen35.embedding_length", 0))
    intermediate_size = int(_tensor_field(reader, "qwen35.feed_forward_length", 0))
    num_layers = int(_tensor_field(reader, "qwen35.block_count", 0))
    num_attention_heads = int(_tensor_field(reader, "qwen35.attention.head_count", 0))
    num_key_value_heads = int(_tensor_field(reader, "qwen35.attention.head_count_kv", 0))
    attention_head_dim = int(_tensor_field(reader, "qwen35.attention.key_length", 0))
    linear_num_key_heads = int(_tensor_field(reader, "qwen35.ssm.group_count", 0))
    linear_num_value_heads = int(_tensor_field(reader, "qwen35.ssm.time_step_rank", 0))
    linear_key_head_dim = int(_tensor_field(reader, "qwen35.ssm.state_size", 0))
    linear_value_head_dim = int(_tensor_field(reader, "qwen35.ssm.state_size", 0))
    linear_conv_kernel_dim = int(_tensor_field(reader, "qwen35.ssm.conv_kernel", 0))

    attention_proj_dim = num_attention_heads * attention_head_dim
    attention_q_proj_dim = attention_proj_dim * 2
    kv_proj_dim = num_key_value_heads * attention_head_dim
    linear_key_dim = linear_num_key_heads * linear_key_head_dim
    linear_value_dim = linear_num_value_heads * linear_value_head_dim
    linear_qkv_dim = linear_key_dim * 2 + linear_value_dim

    entries = [
        MappingEntry("token_embd.weight", "model.language_model.embed_tokens.weight", (vocab_size, hidden_size)),
        MappingEntry("output_norm.weight", "model.language_model.norm.weight", (hidden_size,)),
        MappingEntry("output.weight", "lm_head.weight", (vocab_size, hidden_size)),
    ]

    for layer_index in range(num_layers):
        prefix = f"model.language_model.layers.{layer_index}."
        gguf_prefix = f"blk.{layer_index}."
        entries.extend(
            [
                MappingEntry(f"{gguf_prefix}attn_norm.weight", f"{prefix}input_layernorm.weight", (hidden_size,)),
                MappingEntry(
                    f"{gguf_prefix}post_attention_norm.weight",
                    f"{prefix}post_attention_layernorm.weight",
                    (hidden_size,),
                ),
                MappingEntry(f"{gguf_prefix}ffn_gate.weight", f"{prefix}mlp.gate_proj.weight", (intermediate_size, hidden_size)),
                MappingEntry(f"{gguf_prefix}ffn_up.weight", f"{prefix}mlp.up_proj.weight", (intermediate_size, hidden_size)),
                MappingEntry(f"{gguf_prefix}ffn_down.weight", f"{prefix}mlp.down_proj.weight", (hidden_size, intermediate_size)),
            ]
        )
        if _layer_type(reader, layer_index) == "full_attention":
            entries.extend(
                [
                    MappingEntry(f"{gguf_prefix}attn_q.weight", f"{prefix}self_attn.q_proj.weight", (attention_q_proj_dim, hidden_size)),
                    MappingEntry(f"{gguf_prefix}attn_k.weight", f"{prefix}self_attn.k_proj.weight", (kv_proj_dim, hidden_size)),
                    MappingEntry(f"{gguf_prefix}attn_v.weight", f"{prefix}self_attn.v_proj.weight", (kv_proj_dim, hidden_size)),
                    MappingEntry(f"{gguf_prefix}attn_output.weight", f"{prefix}self_attn.o_proj.weight", (hidden_size, attention_proj_dim)),
                    MappingEntry(f"{gguf_prefix}attn_q_norm.weight", f"{prefix}self_attn.q_norm.weight", (attention_head_dim,)),
                    MappingEntry(f"{gguf_prefix}attn_k_norm.weight", f"{prefix}self_attn.k_norm.weight", (attention_head_dim,)),
                ]
            )
        else:
            entries.extend(
                [
                    MappingEntry(f"{gguf_prefix}ssm_norm.weight", f"{prefix}linear_attn.norm.weight", (linear_value_head_dim,)),
                    MappingEntry(f"{gguf_prefix}attn_qkv.weight", f"{prefix}linear_attn.in_proj_qkv.weight", (linear_qkv_dim, hidden_size)),
                    MappingEntry(f"{gguf_prefix}attn_gate.weight", f"{prefix}linear_attn.in_proj_z.weight", (linear_value_dim, hidden_size), notes="Experimental semantic mapping from GGUF attn_gate -> HF in_proj_z"),
                    MappingEntry(f"{gguf_prefix}ssm_alpha.weight", f"{prefix}linear_attn.in_proj_a.weight", (linear_num_value_heads, hidden_size)),
                    MappingEntry(f"{gguf_prefix}ssm_beta.weight", f"{prefix}linear_attn.in_proj_b.weight", (linear_num_value_heads, hidden_size)),
                    MappingEntry(f"{gguf_prefix}ssm_out.weight", f"{prefix}linear_attn.out_proj.weight", (hidden_size, linear_value_dim)),
                    MappingEntry(f"{gguf_prefix}ssm_conv1d.weight", f"{prefix}linear_attn.conv1d.weight", (linear_qkv_dim, 1, linear_conv_kernel_dim)),
                    MappingEntry(f"{gguf_prefix}ssm_a", f"{prefix}linear_attn.A_log", (linear_num_value_heads,), notes="Experimental semantic mapping from GGUF ssm_a -> HF A_log"),
                    MappingEntry(f"{gguf_prefix}ssm_dt.bias", f"{prefix}linear_attn.dt_bias", (linear_num_value_heads,)),
                ]
            )
    return entries


def _load_array(tensor, output_dtype: np.dtype) -> np.ndarray:
    tensor_type_name = tensor.tensor_type.name
    if tensor_type_name == "Q8_0":
        array = dequantize(tensor.data, tensor.tensor_type)
    elif tensor_type_name == "F32":
        array = np.asarray(tensor.data)
    else:
        raise ValueError(f"unsupported GGUF tensor type: {tensor_type_name}")
    return np.asarray(array, dtype=output_dtype, order="C")


def _reshape_special(entry: MappingEntry, array: np.ndarray) -> np.ndarray:
    if entry.hf_name.endswith("linear_attn.conv1d.weight"):
        if array.ndim != 2:
            raise ValueError(f"unexpected conv1d rank for {entry.gguf_name}: {array.shape}")
        return np.reshape(array, (array.shape[0], 1, array.shape[1]), order="C")
    return array


def _save_shard(output_dir: Path,
                shard_index: int,
                total_shards: int,
                tensors: dict[str, np.ndarray],
                weight_map: dict[str, str]) -> None:
    shard_name = f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
    save_file(tensors, str(output_dir / shard_name), metadata={"format": "pt"})
    for tensor_name in tensors.keys():
        weight_map[tensor_name] = shard_name


def _estimate_total_shards(mapped_arrays: list[tuple[MappingEntry, np.ndarray]], max_shard_bytes: int) -> int:
    if not mapped_arrays:
        return 0
    shard_count = 1
    current_size = 0
    for _, array in mapped_arrays:
        tensor_bytes = int(array.nbytes)
        if current_size > 0 and current_size + tensor_bytes > max_shard_bytes:
            shard_count += 1
            current_size = 0
        current_size += tensor_bytes
    return shard_count


def _gguf_metadata_dump(reader: GGUFReader) -> dict[str, Any]:
    return {
        "field_count": len(reader.fields),
        "tensor_count": len(reader.tensors),
        "fields": {name: _field_contents(reader, name, None) for name in reader.fields.keys()},
    }


def main() -> None:
    args = parse_args()
    gguf_file = args.gguf_file.resolve()
    if not gguf_file.exists():
        raise FileNotFoundError(f"gguf file does not exist: {gguf_file}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(gguf_file))
    lookup = {tensor.name: tensor for tensor in reader.tensors}
    tokenizer_runtime = _build_tokenizer_runtime(reader)
    config = _build_qwen35_config_from_gguf(reader)
    gguf_metadata = _gguf_metadata_dump(reader)

    (tokenizer_dir / "tokenizer_runtime.json").write_text(
        json.dumps(tokenizer_runtime, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "gguf_metadata.json").write_text(
        json.dumps(gguf_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    include_pattern = re.compile(args.include_regex) if args.include_regex else None
    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    mapped_entries = _build_mapping(reader)
    mapped_arrays: list[tuple[MappingEntry, np.ndarray]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for entry in mapped_entries:
        if include_pattern is not None and include_pattern.search(entry.hf_name) is None:
            skipped.append({"gguf_name": entry.gguf_name, "hf_name": entry.hf_name, "reason": "filtered"})
            continue
        if args.max_tensors > 0 and len(mapped_arrays) >= args.max_tensors:
            skipped.append({"gguf_name": entry.gguf_name, "hf_name": entry.hf_name, "reason": "max_tensors"})
            continue
        tensor = lookup.get(entry.gguf_name)
        if tensor is None:
            errors.append({"gguf_name": entry.gguf_name, "hf_name": entry.hf_name, "reason": "missing_gguf_tensor"})
            continue
        try:
            array = _reshape_special(entry, _load_array(tensor, output_dtype))
            if tuple(int(dim) for dim in array.shape) != entry.expected_shape:
                raise ValueError(
                    f"shape mismatch for {entry.gguf_name}: got {tuple(int(dim) for dim in array.shape)}, "
                    f"expected {entry.expected_shape}"
                )
            mapped_arrays.append((entry, array))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "gguf_name": entry.gguf_name,
                    "hf_name": entry.hf_name,
                    "reason": "conversion_error",
                    "message": str(exc),
                }
            )

    tensor_map_payload = [
        {
            "gguf_name": entry.gguf_name,
            "hf_name": entry.hf_name,
            "expected_shape": list(entry.expected_shape),
            "notes": entry.notes,
        }
        for entry in mapped_entries
    ]
    (output_dir / "tensor_map.json").write_text(
        json.dumps(tensor_map_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    weight_map: dict[str, str] = {}
    metadata = {
        "total_size": int(sum(array.nbytes for _, array in mapped_arrays)),
    }
    safetensor_index = {
        "metadata": metadata,
        "weight_map": weight_map,
    }

    if not args.metadata_only:
        max_shard_bytes = max(1, int(args.max_shard_size_gb * (1024 ** 3)))
        total_shards = _estimate_total_shards(mapped_arrays, max_shard_bytes)
        shard_tensors: dict[str, np.ndarray] = {}
        shard_size = 0
        shard_index = 1
        for entry, array in mapped_arrays:
            tensor_bytes = int(array.nbytes)
            if shard_tensors and shard_size + tensor_bytes > max_shard_bytes:
                _save_shard(output_dir, shard_index, total_shards, shard_tensors, weight_map)
                shard_tensors = {}
                shard_size = 0
                shard_index += 1
            shard_tensors[entry.hf_name] = array
            shard_size += tensor_bytes
        if shard_tensors:
            _save_shard(output_dir, shard_index, total_shards, shard_tensors, weight_map)

        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(safetensor_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    report = {
        "format": "soc.gguf_to_safetensors",
        "model_id": args.model_id,
        "gguf_file": str(gguf_file),
        "output_dir": str(output_dir),
        "output_dtype": args.dtype,
        "metadata_only": args.metadata_only,
        "mapped_tensor_count": len(mapped_arrays),
        "skipped_tensor_count": len(skipped),
        "error_count": len(errors),
        "mapped_tensors": [
            {
                "gguf_name": entry.gguf_name,
                "hf_name": entry.hf_name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "notes": entry.notes,
            }
            for entry, array in mapped_arrays
        ],
        "skipped": skipped,
        "errors": errors,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Experimental GGUF->safetensors export written to {output_dir}")
    print(f"Mapped tensors: {len(mapped_arrays)}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
