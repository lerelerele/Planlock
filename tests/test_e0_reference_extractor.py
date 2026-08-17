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

    def test_valid_hsdp_trace_confirms_mechanics_only(self) -> None:
        trace = {
            "status": "REAL_CPU_GLOO_HSDP_MECHANICS_ONLY",
            "backend": "gloo",
            "device": "cpu",
            "world_size": 4,
            "mesh": {"dp_replicate": 2, "fsdp": 2},
            "all_reduce_observed": True,
            "all_gather_observed": True,
            "reduce_scatter_observed": True,
            "all_ranks_observed_all_reduce": True,
            "all_ranks_observed_all_gather": True,
            "all_ranks_observed_reduce_scatter": True,
            "e0_closed": False,
            "population_touched": False,
        }
        MODULE.validate_hsdp_trace(trace)
        invalid = dict(trace, all_reduce_observed=False)
        with self.assertRaisesRegex(ValueError, "all_reduce_observed"):
            MODULE.validate_hsdp_trace(invalid)

    def test_dense_storage_catalog_preserves_w1_w3_coefficient(self) -> None:
        manifest = {
            "pes": {"PE_dense": {"dtype_classes": {"param": "f16"}}}
        }
        catalog = MODULE.dense_storage_catalog(manifest)
        swiglu_inputs = next(
            item for item in catalog if "{w1,w3}" in item.logical_parameter
        )
        self.assertEqual(swiglu_inputs.multiplicity, "2*L")
        self.assertEqual(swiglu_inputs.role, "ColLinear")
        self.assertEqual(swiglu_inputs.dtype_class, "f16")
        self.assertEqual(swiglu_inputs.tensor_class, "param")
        self.assertEqual(
            swiglu_inputs.status, "SEMANTIC_TENSOR_SIGNATURE_CATALOGED"
        )

    def test_moe_storage_catalog_separates_dense_and_sparse_families(self) -> None:
        manifest = {"pes": {"PE_moe": {"dtype_classes": {"param": "f16"}}}}
        catalog = MODULE.moe_storage_catalog(manifest)
        routed = [item for item in catalog if item.role == "GroupedGEMM"]
        shared = [item for item in catalog if "shared_experts" in item.logical_parameter]
        self.assertEqual(len(routed), 2)
        self.assertTrue(all(item.tp_placement.startswith("sparse:") for item in routed))
        self.assertTrue(all(item.tp_placement.startswith("dense:") for item in shared))
        self.assertEqual(routed[0].multiplicity, "2*L_moe")
        self.assertTrue(all(item.tensor_class == "param" for item in catalog))

    def test_storage_signature_rejects_repeated_known_axis(self) -> None:
        item = MODULE.StorageSemantic(
            pe="PE_dense",
            logical_parameter="fixture.weight",
            role="ColLinear",
            normalized_form=(("input_feature", "D"), ("input_feature", "F")),
            tp_placement="tp:Shard(output_feature)",
            multiplicity="1",
            dtype_class="f16",
            tensor_class="param",
            provenance=("fixture",),
            status="SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
        )
        with self.assertRaisesRegex(ValueError, "repeats a known axis"):
            MODULE.validate_storage_signatures([item])

    def test_gradient_signatures_preserve_form_role_and_multiplicity(self) -> None:
        manifest = {
            "pes": {
                "PE_dense": {
                    "dtype_classes": {"param": "f16", "grad_reduce": "f32"}
                }
            }
        }
        parameters = MODULE.dense_storage_catalog(manifest)
        gradients = MODULE.gradient_signature_catalog(manifest, parameters)
        self.assertEqual(len(gradients), len(parameters))
        for parameter, gradient in zip(parameters, gradients, strict=True):
            self.assertEqual(gradient.normalized_form, parameter.normalized_form)
            self.assertEqual(gradient.role, parameter.role)
            self.assertEqual(gradient.multiplicity, parameter.multiplicity)
            self.assertEqual(gradient.tensor_class, "grad")
            self.assertEqual(gradient.dtype_class, "f32")
            self.assertEqual(gradient.logical_parameter, f"{parameter.logical_parameter}::grad")

    def test_adamw_state_signatures_include_two_moments_and_scalar_step(self) -> None:
        manifest = {
            "pes": {
                "PE_dense": {
                    "dtype_classes": {"param": "f16", "grad_reduce": "f32"},
                    "optimizer": {
                        "name": "AdamW",
                        "implementation": "fused",
                        "amsgrad": False,
                        "state_tensors": {
                            "exp_avg": "same_as_param",
                            "exp_avg_sq": "same_as_param",
                            "step": "f32",
                        },
                    },
                }
            }
        }
        parameter = MODULE.dense_storage_catalog(manifest)[0]
        states = MODULE.optimizer_state_signature_catalog(manifest, [parameter])
        self.assertEqual(len(states), 3)
        moments = [item for item in states if not item.logical_parameter.endswith("step")]
        step = next(item for item in states if item.logical_parameter.endswith("step"))
        self.assertTrue(all(item.normalized_form == parameter.normalized_form for item in moments))
        self.assertTrue(all(item.dtype_class == "f16" for item in moments))
        self.assertTrue(all(item.role == "OptimizerUpdate" for item in states))
        self.assertEqual(step.normalized_form, ())
        self.assertEqual(step.dtype_class, "f32")

    def test_standard_moe_control_metadata_is_integer_and_nondifferentiable(self) -> None:
        manifest = {
            "pes": {"PE_moe": {"overrides": {"moe_comm_backend": "standard"}}}
        }
        catalog = MODULE.moe_control_metadata_catalog(manifest)
        self.assertEqual(len(catalog), 6)
        self.assertTrue(all(item.tensor_class == "control_metadata" for item in catalog))
        self.assertEqual({item.dtype_class for item in catalog}, {"i64", "bool"})
        topk = next(item for item in catalog if item.logical_parameter == "topk_expert_ids_TK")
        self.assertEqual(
            topk.normalized_form,
            (("batch", "B"), ("seq", "S"), ("topk", "K")),
        )
        routed = [item for item in catalog if item.normalized_form == (("routed_item", "B*S*K"),)]
        self.assertEqual(len(routed), 2)

    def test_moe_routing_activations_separate_scores_and_payload_dtypes(self) -> None:
        manifest = {
            "pes": {
                "PE_moe": {
                    "overrides": {"moe_comm_backend": "standard"},
                    "dtype_classes": {"param": "f16"},
                }
            }
        }
        catalog = MODULE.moe_routing_activation_catalog(manifest)
        self.assertEqual(len(catalog), 6)
        self.assertTrue(all(item.tensor_class == "activation" for item in catalog))
        scores = [item for item in catalog if "scores" in item.logical_parameter]
        payloads = [item for item in catalog if item.logical_parameter.startswith("routed_")]
        self.assertTrue(all(item.dtype_class == "f32" for item in scores))
        self.assertTrue(all(item.dtype_class == "f16" for item in payloads))
        flattened = next(item for item in scores if "sorted" in item.logical_parameter)
        self.assertEqual(flattened.normalized_form, (("routed_item", "B*S*K"),))

    def test_dense_nonattention_activations_preserve_swiglu_coefficient(self) -> None:
        manifest = {"pes": {"PE_dense": {"dtype_classes": {"param": "f16"}}}}
        catalog = MODULE.dense_nonattention_activation_catalog(manifest)
        self.assertEqual(len(catalog), 9)
        self.assertTrue(all(item.tensor_class == "activation" for item in catalog))
        collinear = next(item for item in catalog if "{w1,w3}" in item.logical_parameter)
        self.assertEqual(collinear.multiplicity, "2*L")
        self.assertEqual(
            collinear.normalized_form,
            (("batch", "B"), ("seq", "S"), ("ffn_hidden", "F")),
        )
        logits = next(item for item in catalog if item.role == "LMHead")
        self.assertIn(("vocab", "V"), logits.normalized_form)
        self.assertTrue(all("input_feature" not in dict(item.normalized_form) for item in catalog))

    def test_dense_fused_qkv_split_preserves_query_and_kv_identities(self) -> None:
        manifest = {
            "pes": {
                "PE_dense": {
                    "overrides": {"attn_backend": "flex"},
                    "arquitectura": {"fuse_qkv": True},
                    "dtype_classes": {"param": "f16"},
                }
            }
        }
        catalog = MODULE.dense_attention_activation_catalog(manifest)
        self.assertEqual(len(catalog), 5)
        fused = next(item for item in catalog if "fused_output" in item.logical_parameter)
        self.assertIn(("output_feature", "(H+2*Hkv)*Dh"), fused.normalized_form)
        kv = next(item for item in catalog if "{key,value}" in item.logical_parameter)
        self.assertEqual(kv.multiplicity, "2*L")
        self.assertIn(("kv_head", "Hkv"), kv.normalized_form)
        query = next(item for item in catalog if "query_BLNH" in item.logical_parameter)
        self.assertIn(("head", "H"), query.normalized_form)

    def test_mla_audit_blocks_unequal_head_dims_and_unsymbolized_latent(self) -> None:
        manifest = {
            "pes": {
                "PE_moe": {
                    "dimensiones_mla": {
                        "qk_nope_head_dim": 128,
                        "qk_rope_head_dim": 64,
                        "v_head_dim": 128,
                        "kv_lora_rank": 512,
                    },
                    "simbolos": {"D": 256, "H": 16},
                }
            }
        }
        audit = MODULE.mla_signature_audit(manifest)
        self.assertEqual(audit["status"], "HUELLA_NO_DERIVABLE")
        self.assertTrue(audit["e0_blocking"])
        reasons = " ".join(item["reason"] for item in audit["failures"])
        self.assertIn("unequal QK and V", reasons)
        self.assertIn("kv_lora_rank", reasons)
        self.assertFalse(audit["e0_closed"])


if __name__ == "__main__":
    unittest.main()
