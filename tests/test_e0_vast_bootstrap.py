import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_vast_bootstrap.sh"


class VastBootstrapTests(unittest.TestCase):
    def test_bootstrap_pins_reference_and_preserves_failed_report(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("9a711521ac2973fe230a3f38efc6aedfc7d1f9c6", source)
        self.assertIn("nightly/cu130", source)
        self.assertIn("set +e", source)
        self.assertIn("sha256sum", source)
        self.assertIn('exit "$validation_status"', source)

    def test_bootstrap_uses_external_work_root(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('$HOME/planlock-e0-rental', source)
        self.assertIn('test -z "$(git -C "$REFERENCE_REPO" status --porcelain)"', source)


if __name__ == "__main__":
    unittest.main()
