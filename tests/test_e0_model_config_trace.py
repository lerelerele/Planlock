import importlib.util
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_model_config_trace.py"
SPEC = importlib.util.spec_from_file_location("e0_model_config_trace", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModelConfigTraceTests(unittest.TestCase):
    def test_walks_nested_dataclass_configs(self) -> None:
        @dataclass
        class Leaf:
            sharding_config: object | None

        @dataclass
        class Root:
            children: list[Leaf]

        rows = list(MODULE.walk_configs(Root([Leaf(object()), Leaf(None)])))
        self.assertEqual([path for path, _ in rows], ["model.children.0", "model.children.1"])

    def test_rejects_output_inside_checkout(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.external_output(SCRIPT.parent / "report.json")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                MODULE.external_output(Path(directory) / "report.json"),
                (Path(directory) / "report.json").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
