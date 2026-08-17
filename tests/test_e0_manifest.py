import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_manifest.py"
SPEC = importlib.util.spec_from_file_location("e0_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManifestTests(unittest.TestCase):
    def test_partition_must_cover_debugmodel_once(self) -> None:
        MODULE.validate_partition([
            ["tok_embeddings", "layers.0", "layers.1", "layers.2"],
            ["layers.3", "layers.4", "layers.5", "norm", "lm_head"],
        ])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            MODULE.validate_partition([["tok_embeddings", "layers.0", "norm", "lm_head"]])

    def test_canonical_hash_is_key_order_independent(self) -> None:
        self.assertEqual(MODULE.canonical_bytes({"b": 1, "a": 2}), MODULE.canonical_bytes({"a": 2, "b": 1}))

    def test_finds_default_adamw_in_selected_config(self) -> None:
        import tempfile

        source = "def selected():\n    return Trainer.Config(optimizer=default_adamw(lr=1e-3))\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.py"
            path.write_text(source, encoding="utf-8")
            self.assertTrue(MODULE.config_uses_default_adamw(path, "selected"))
            self.assertFalse(MODULE.config_uses_default_adamw(path, "missing"))

    def test_extracts_default_adamw_contract(self) -> None:
        import tempfile

        source = '''
class Container:
    class Config:
        implementation: str = "fused"
def default_adamw():
    return ParamGroupConfig(optimizer_name="AdamW")
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.py"
            path.write_text(source, encoding="utf-8")
            self.assertEqual(
                MODULE.default_adamw_contract(path),
                {"name": "AdamW", "implementation": "fused"},
            )

    def test_extracts_keyword_only_function_default(self) -> None:
        import tempfile

        source = 'def model(*, backend: str = "standard"):\n    pass\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.py"
            path.write_text(source, encoding="utf-8")
            self.assertEqual(
                MODULE.function_parameter_default(path, "model", "backend"),
                "standard",
            )

    def test_extracts_unique_call_keyword_constant(self) -> None:
        import tempfile

        source = "def model():\n    return build(fuse_qkv=True)\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.py"
            path.write_text(source, encoding="utf-8")
            self.assertIs(
                MODULE.function_keyword_constant(path, "model", "fuse_qkv"), True
            )


if __name__ == "__main__":
    unittest.main()
