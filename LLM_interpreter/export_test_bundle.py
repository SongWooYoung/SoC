from __future__ import annotations

import argparse
import json
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import convert_py_to_cpp


@dataclass(frozen=True)
class FakeDType:
	name: str
	is_floating_point: bool


class FakeArray:
	def __init__(self, values, dtype_name: str):
		self._values = list(values)
		self._dtype_name = dtype_name

	def tobytes(self) -> bytes:
		if self._dtype_name == "float32":
			return b"".join(struct.pack("<f", float(value)) for value in self._values)
		if self._dtype_name == "float16":
			return b"".join(struct.pack("<e", float(value)) for value in self._values)
		if self._dtype_name == "uint16":
			return b"".join(struct.pack("<H", int(value)) for value in self._values)
		if self._dtype_name == "int32":
			return b"".join(struct.pack("<i", int(value)) for value in self._values)
		raise TypeError(f"unsupported fake array dtype: {self._dtype_name}")


class FakeTensor:
	def __init__(self, values, shape, dtype: FakeDType):
		self._values = list(values)
		self.shape = tuple(shape)
		self.dtype = dtype

	def detach(self):
		return self

	def contiguous(self):
		return self

	def cpu(self):
		return self

	def to(self, dtype: FakeDType):
		return FakeTensor(self._values, self.shape, dtype)

	def view(self, dtype: FakeDType):
		if self.dtype.name != "bfloat16" or dtype.name != "uint16":
			raise TypeError("unsupported fake tensor view")

		def float32_to_bfloat16_bits(value: float) -> int:
			bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
			rounding_bias = 0x7FFF + ((bits >> 16) & 1)
			return (bits + rounding_bias) >> 16

		return FakeTensor([float32_to_bfloat16_bits(value) for value in self._values], self.shape, dtype)

	def numpy(self):
		return FakeArray(self._values, self.dtype.name)


class FakeBackendTokenizer:
	def save(self, path: str) -> None:
		Path(path).write_text("{\"version\":\"fake\"}", encoding="utf-8")


class FakeAddedToken:
	def __init__(self, content: str):
		self.content = content
		self.special = True
		self.single_word = False
		self.lstrip = False
		self.rstrip = False
		self.normalized = False


class FakeTokenizer:
	vocab_size = 8
	model_max_length = 64
	bos_token = "<|im_start|>"
	eos_token = "<|im_end|>"
	pad_token = "<|im_end|>"
	unk_token = None
	chat_template = "fake-qwen3-template"
	special_tokens_map = {
		"bos_token": "<|im_start|>",
		"eos_token": "<|im_end|>",
		"pad_token": "<|im_end|>",
		"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
	}
	additional_special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
	added_tokens_decoder = {
		0: FakeAddedToken("<|im_start|>"),
		1: FakeAddedToken("<|im_end|>"),
		2: FakeAddedToken("<think>"),
		3: FakeAddedToken("</think>"),
	}
	backend_tokenizer = FakeBackendTokenizer()
	_vocab = {
		"<|im_start|>": 0,
		"<|im_end|>": 1,
		"<think>": 2,
		"</think>": 3,
		"A": 4,
		"B": 5,
		"C": 6,
		"D": 7,
	}

	@classmethod
	def from_pretrained(cls, model_dir: Path, trust_remote_code: bool = True):
		return cls()

	def get_vocab(self):
		return dict(self._vocab)

	def convert_tokens_to_ids(self, token):
		return self._vocab[token]


class FakeTokenizerFactory:
	@staticmethod
	def from_pretrained(model_dir: Path, trust_remote_code: bool = True):
		return FakeTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)


class FakeSafeOpenHandle:
	def __init__(self, tensors):
		self._tensors = dict(tensors)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		return False

	def keys(self):
		return list(self._tensors.keys())

	def get_tensor(self, name: str):
		return self._tensors[name]


class FakeTorchModule:
	float16 = FakeDType("float16", True)
	float32 = FakeDType("float32", True)
	float64 = FakeDType("float64", True)
	bfloat16 = FakeDType("bfloat16", True)
	uint16 = FakeDType("uint16", False)
	int8 = FakeDType("int8", False)
	uint8 = FakeDType("uint8", False)
	int16 = FakeDType("int16", False)
	int32 = FakeDType("int32", False)
	int64 = FakeDType("int64", False)
	bool = FakeDType("bool", False)


def build_fake_tensors(fake_torch: FakeTorchModule):
	hidden_size = 2
	intermediate_size = 2
	q_proj_shape = (2, hidden_size)
	zeros_2x2 = [0.0, 0.0, 0.0, 0.0]
	zeros = FakeTensor(zeros_2x2, q_proj_shape, fake_torch.float32)
	ones = FakeTensor([1.0, 1.0], (2,), fake_torch.float32)
	return {
		"model.embed_tokens.weight": FakeTensor([
			0.0, 0.0,
			0.0, 0.0,
			0.0, 0.0,
			0.0, 0.0,
			1.0, 0.0,
			0.0, 1.0,
			1.0, 1.0,
			-1.0, 1.0,
		], (8, hidden_size), fake_torch.float32),
		"model.layers.0.input_layernorm.weight": ones,
		"model.layers.0.self_attn.q_proj.weight": zeros,
		"model.layers.0.self_attn.k_proj.weight": zeros,
		"model.layers.0.self_attn.v_proj.weight": zeros,
		"model.layers.0.self_attn.o_proj.weight": zeros,
		"model.layers.0.post_attention_layernorm.weight": ones,
		"model.layers.0.mlp.gate_proj.weight": FakeTensor(zeros_2x2, (intermediate_size, hidden_size), fake_torch.float32),
		"model.layers.0.mlp.up_proj.weight": FakeTensor(zeros_2x2, (intermediate_size, hidden_size), fake_torch.float32),
		"model.layers.0.mlp.down_proj.weight": FakeTensor(zeros_2x2, (hidden_size, intermediate_size), fake_torch.float32),
		"model.norm.weight": ones,
		"lm_head.weight": FakeTensor([
			0.0, 0.0,
			0.0, 0.0,
			0.0, 0.0,
			0.0, 0.0,
			0.0, 0.0,
			0.1, 0.2,
			0.0, 2.0,
			2.0, 2.0,
		], (8, hidden_size), fake_torch.float32),
	}


def build_fake_dependency_loader():
	fake_torch = FakeTorchModule()
	fake_tensors = build_fake_tensors(fake_torch)

	def safe_open(path: str, framework: str = "pt", device: str = "cpu"):
		return FakeSafeOpenHandle(fake_tensors)

	return fake_torch, safe_open, FakeTokenizerFactory


def create_fake_model_dir(model_dir: Path) -> None:
	model_dir.mkdir(parents=True, exist_ok=True)
	(model_dir / "model.safetensors").write_bytes(b"fake-safetensors")
	(model_dir / "config.json").write_text(
		json.dumps(
			{
				"model_type": "qwen3",
				"hidden_size": 2,
				"intermediate_size": 2,
				"num_hidden_layers": 1,
				"num_attention_heads": 1,
				"num_key_value_heads": 1,
				"head_dim": 2,
				"rms_norm_eps": 1e-6,
				"rope_theta": 10000,
				"tie_word_embeddings": False,
				"torch_dtype": "float32",
				"vocab_size": 8,
			}
		),
		encoding="utf-8",
	)
	(model_dir / "generation_config.json").write_text(json.dumps({"max_new_tokens": 2}), encoding="utf-8")


def export_test_bundle(output_dir: Path, dtype: str = "float16") -> Path:
	with tempfile.TemporaryDirectory(prefix="soc_fake_export_model_") as temp_dir:
		model_dir = Path(temp_dir)
		create_fake_model_dir(model_dir)

		original_loader = convert_py_to_cpp.load_export_dependencies
		convert_py_to_cpp.load_export_dependencies = build_fake_dependency_loader
		try:
			return convert_py_to_cpp.export_hf_checkpoint_for_cpp(
				model_dir=model_dir,
				output_dir=output_dir,
				export_dtype=dtype,
				model_id="fake/qwen3-test",
			)
		finally:
			convert_py_to_cpp.load_export_dependencies = original_loader


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Emit a tiny exporter-generated runtime bundle for C++ smoke tests.")
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--dtype", choices=("native", "float32", "float16"), default="float16")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	manifest_path = export_test_bundle(args.output_dir.resolve(), args.dtype)
	print(manifest_path)


if __name__ == "__main__":
	main()