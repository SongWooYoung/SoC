import json
import tempfile
import unittest
from pathlib import Path

from export_test_bundle import export_test_bundle


class ExportTestBundleTests(unittest.TestCase):
    def test_export_test_bundle_writes_manifest_and_runtime_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "exported"
            manifest_path = export_test_bundle(output_dir, "float16")

            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "soc.cpp.llm_export")
            self.assertEqual(manifest["export_dtype"], "float16")
            self.assertEqual(manifest["tensor_count"], len(manifest["tensors"]))
            self.assertTrue((output_dir / "tokenizer" / "tokenizer_runtime.json").exists())
            self.assertTrue((output_dir / "weights").exists())


if __name__ == "__main__":
    unittest.main()