import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_hsdp_trace.py"
SPEC = importlib.util.spec_from_file_location("e0_hsdp_trace", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HsdpTraceTests(unittest.TestCase):
    def test_rejects_output_inside_checkout(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the Planlock checkout"):
            MODULE.external_output(SCRIPT.parent / "trace")

    def test_accepts_external_output(self) -> None:
        target = Path(tempfile.gettempdir()) / "planlock-hsdp-trace"
        self.assertEqual(MODULE.external_output(target), target.resolve())


if __name__ == "__main__":
    unittest.main()
