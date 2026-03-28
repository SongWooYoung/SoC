import json
import tempfile
import unittest
from pathlib import Path

from convert_py_to_cpp import build_tokenizer_runtime_manifest, write_tokenizer_runtime


class FakeAddedToken:
    def __init__(self, content, special=False, single_word=False, lstrip=False, rstrip=False, normalized=True):
        self.content = content
        self.special = special
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = normalized


class FakeTokenizer:
    def __init__(self):
        self.special_tokens_map = {
            "bos_token": "<|endoftext|>",
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
        }
        self.additional_special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        self.added_tokens_decoder = {
            151644: FakeAddedToken("<|im_start|>", special=True, normalized=False),
            151645: FakeAddedToken("<|im_end|>", special=True, normalized=False),
        }
        self.chat_template = "dummy-template"
        self.vocab_size = 4
        self.model_max_length = 131072
        self.backend_tokenizer = type(
            "FakeBackendTokenizer",
            (),
            {
                "to_str": staticmethod(
                    lambda: json.dumps(
                        {
                            "pre_tokenizer": {
                                "type": "ByteLevel",
                                "add_prefix_space": True,
                                "use_regex": True,
                            },
                            "decoder": {
                                "type": "Sequence",
                                "decoders": [
                                    {
                                        "type": "ByteLevel",
                                        "add_prefix_space": True,
                                        "trim_offsets": False,
                                        "use_regex": True,
                                    },
                                    {
                                        "type": "BPEDecoder",
                                        "suffix": "</w>",
                                    },
                                ],
                            },
                            "model": {
                                "type": "BPE",
                                "unk_token": "<unk>",
                                "continuing_subword_prefix": "",
                                "end_of_word_suffix": "",
                                "merges": [["h", "e"], ["he", "llo"], [" ", "w"], [" w", "orld"]],
                            }
                        }
                    )
                )
            },
        )()
        self._vocab = {
            "!": 0,
            "hello": 1,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        }

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]


class ConvertPyToCppTests(unittest.TestCase):
    def test_build_tokenizer_runtime_manifest(self):
        tokenizer = FakeTokenizer()

        runtime_manifest = build_tokenizer_runtime_manifest(tokenizer, Path("."), ["tokenizer.json", "config.json"])

        self.assertEqual(runtime_manifest["format"], "soc.cpp.tokenizer_runtime")
        self.assertEqual(runtime_manifest["format_version"], 1)
        self.assertEqual(runtime_manifest["template_runtime"]["type"], "qwen3")
        self.assertEqual(runtime_manifest["template_runtime"]["im_start"], "<|im_start|>")
        self.assertEqual(len(runtime_manifest["added_tokens"]), 2)
        self.assertEqual(runtime_manifest["vocab_size"], 151646)
        self.assertEqual(runtime_manifest["vocab"][0], {"token": "!", "id": 0})
        self.assertEqual(runtime_manifest["bpe_model"]["type"], "bpe")
        self.assertEqual(runtime_manifest["bpe_model"]["unk_token"], "<unk>")
        self.assertEqual(runtime_manifest["bpe_model"]["merges"][0], {"left": "h", "right": "e"})
        self.assertEqual(runtime_manifest["pre_tokenizer"]["type"], "ByteLevel")
        self.assertTrue(runtime_manifest["pre_tokenizer"]["byte_level"]["add_prefix_space"])
        self.assertEqual(runtime_manifest["decoder"]["type"], "Sequence")
        self.assertEqual(runtime_manifest["decoder"]["bpe"]["suffix"], "</w>")
        self.assertEqual(len(runtime_manifest["decoder"]["byte_level"]["byte_to_unicode"]), 256)

    def test_write_tokenizer_runtime(self):
        tokenizer = FakeTokenizer()
        runtime_manifest = build_tokenizer_runtime_manifest(tokenizer, Path("."), ["tokenizer.json"])

        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_dir = Path(temp_dir)
            file_name = write_tokenizer_runtime(tokenizer_dir, runtime_manifest)

            self.assertEqual(file_name, "tokenizer_runtime.json")
            written_manifest = json.loads((tokenizer_dir / file_name).read_text(encoding="utf-8"))
            self.assertEqual(written_manifest["vocab_size"], 151646)
            self.assertEqual(written_manifest["chat_template"], "dummy-template")
            self.assertEqual(written_manifest["bpe_model"]["type"], "bpe")
            self.assertEqual(written_manifest["pre_tokenizer"]["type"], "ByteLevel")
            self.assertEqual(written_manifest["decoder"]["bpe"]["suffix"], "</w>")


if __name__ == "__main__":
    unittest.main()