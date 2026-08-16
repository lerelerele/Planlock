import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_reference_extractor.py"
sys.path.insert(0, str(SCRIPT.parent))
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

    def test_reachability_propagates_conditional_edges(self) -> None:
        source = """\
def root(flag):
    direct()
    if flag:
        optional()
def direct():
    leaf()
def leaf():
    pass
def optional():
    pass
def unused():
    pass
"""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.py"
            path.write_text(source, encoding="utf-8")
            status = MODULE.route_reachability(Path(directory), ("route.py",), {"root"})
        self.assertEqual(status["direct"], "ACTIVE_STATIC")
        self.assertEqual(status["leaf"], "ACTIVE_STATIC")
        self.assertEqual(status["optional"], "CONDITIONAL_RUNTIME")
        self.assertNotIn("unused", status)

    def test_manifest_resolves_deepseek_layer_branches(self) -> None:
        status = {
            "_set_deepseek_v3_mtp_sharding": "CONDITIONAL_RUNTIME",
            "set_dense_ffn_sharding": "CONDITIONAL_RUNTIME",
            "_moe_sharding_config": "CONDITIONAL_RUNTIME",
        }
        manifest_pe = {
            "arquitectura": {
                "layers": 6,
                "dense_layers": 1,
                "moe_layers": 5,
                "mtp_layers": 0,
            }
        }
        resolved = MODULE.resolve_manifest_conditions("PE_moe", status, manifest_pe)
        self.assertEqual(
            resolved["_set_deepseek_v3_mtp_sharding"], "UNREACHABLE_MANIFEST"
        )
        self.assertEqual(resolved["set_dense_ffn_sharding"], "ACTIVE_MANIFEST")
        self.assertEqual(resolved["_moe_sharding_config"], "ACTIVE_MANIFEST")

    def test_resolves_reviewed_boundary_transition(self) -> None:
        candidate = MODULE.Candidate(
            pe="PE_dense",
            kind="sharding_boundary",
            symbol="ShardingConfig",
            source="torchtitan/models/common/decoder_sharding.py",
            line=91,
            enclosing_function="rowwise_config",
            evidence="fixture",
            route_status="ACTIVE_STATIC",
        )
        resolved = MODULE.resolve_candidate_semantics(candidate)
        self.assertEqual(resolved.transition, "ReduceScatter")

    def test_excludes_inactive_dist_gemm_declaration(self) -> None:
        candidate = MODULE.Candidate(
            pe="PE_dense",
            kind="sharding_boundary",
            symbol="ShardingConfig",
            source="torchtitan/models/common/decoder_sharding.py",
            line=297,
            enclosing_function="set_dense_ffn_sharding",
            evidence="fixture",
            route_status="ACTIVE_MANIFEST",
        )
        resolved = MODULE.resolve_candidate_semantics(candidate)
        self.assertEqual(resolved.route_status, "UNREACHABLE_MANIFEST")

    def test_transition_inventory_deduplicates_backend_calls(self) -> None:
        items = [
            MODULE.Candidate(
                pe="PE_moe",
                kind="explicit_communication",
                symbol=symbol,
                source="dispatcher.py",
                line=line,
                enclosing_function="_dispatch_token_exchange",
                evidence="fixture",
                transition="Dispatch",
                route_status="ACTIVE_STATIC",
            )
            for symbol, line in (("all_to_all_single", 1), ("spmd.all_to_all", 2))
        ]
        inventory = MODULE.transition_inventory(items)
        self.assertEqual(inventory["PE_moe"]["Dispatch"], 1)

    def test_framework_candidates_keep_symbolic_multiplicity(self) -> None:
        manifest = {
            "pes": {
                "PE_dense": {
                    "overrides": {
                        "module_fqns_per_model_part": [["a"], ["b"]],
                    },
                    "arquitectura": {
                        "layers": 6,
                        "dense_layers": 6,
                        "moe_layers": 0,
                    },
                    "grados": {"dp_r": 2},
                }
            }
        }
        events = MODULE.framework_candidates(manifest)
        pp = next(item for item in events if item.subsystem == "pipeline")
        self.assertEqual(pp.multiplicity, "P - 1")
        self.assertEqual(pp.transition, "SendRecv")
        self.assertTrue(
            all(item.status != "COMPLETE_TEMPLATE" for item in events)
        )


if __name__ == "__main__":
    unittest.main()
