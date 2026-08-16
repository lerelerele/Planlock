import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_reference_extractor.py"
SPEC = importlib.util.spec_from_file_location("e0_reference_extractor", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InventoryVisitorTests(unittest.TestCase):
    def test_collects_boundaries_and_explicit_communication(self) -> None:
        source = """\
def configure():
    cfg = ShardingConfig(out_src_shardings=layout)
    dist.all_to_all_single(output, input)
    tensor.redistribute(mesh, placements)
"""
        visitor = MODULE.InventoryVisitor("PE_moe", "fixture.py", source)
        visitor.visit(ast.parse(source))
        self.assertEqual(
            [item.kind for item in visitor.candidates],
            ["sharding_boundary", "explicit_communication", "explicit_redistribution"],
        )
        self.assertTrue(all(item.enclosing_function == "configure" for item in visitor.candidates))

    def test_classifies_reviewed_dispatcher_helpers(self) -> None:
        source = """\
def _dispatch_token_exchange():
    all_to_all_single(payload)
    spmd.all_to_all(payload)
def _combine_token_exchange():
    all_to_all_single(payload)
"""
        visitor = MODULE.InventoryVisitor("PE_moe", "dispatcher.py", source)
        visitor.visit(ast.parse(source))
        self.assertEqual(
            [item.transition for item in visitor.candidates],
            ["Dispatch", "Dispatch", "Combine"],
        )
        logical = MODULE.logical_transitions(visitor.candidates)
        dispatch = next(item for item in logical if item["transition"] == "Dispatch")
        self.assertEqual(dispatch["implementation_call_count"], 2)

    def test_classifies_reviewed_role_helpers(self) -> None:
        source = "def colwise_config():\n    return ShardingConfig()\n"
        visitor = MODULE.InventoryVisitor("PE_dense", "sharding.py", source)
        visitor.visit(ast.parse(source))
        self.assertEqual(visitor.candidates[0].role, "ColLinear")
        self.assertEqual(visitor.candidates[0].status, "RULE_CLASSIFIED_PROTOTYPE")

    def test_ignores_unrelated_calls(self) -> None:
        source = "def f():\n    ordinary_call()\n"
        visitor = MODULE.InventoryVisitor("PE_dense", "fixture.py", source)
        visitor.visit(ast.parse(source))
        self.assertEqual(visitor.candidates, [])

    def test_rejects_output_inside_checkout(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the Planlock checkout"):
            MODULE.external_output(SCRIPT.parent / "report.json")

    def test_accepts_external_output(self) -> None:
        target = Path(tempfile.gettempdir()) / "planlock-report.json"
        self.assertEqual(MODULE.external_output(target), target.resolve())

    def test_route_hypotheses_do_not_pose_as_frozen_manifests(self) -> None:
        for route in MODULE.ROUTE_HYPOTHESES.values():
            self.assertIsNone(route["function_config"])
            self.assertIsNone(route["overrides"])
            self.assertIsNone(route["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
