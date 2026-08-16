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


if __name__ == "__main__":
    unittest.main()
