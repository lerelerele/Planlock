#!/usr/bin/env python3
"""Prototype a provenance-first inventory for E0 reference fingerprints.

The prototype is intentionally conservative: it inventories declarative
``ShardingConfig`` boundaries and explicit communication calls at the pinned
TorchTitan HEAD.  It does not claim completeness, infer tensor signatures, or
close E0.  Its output is an auditable work queue for the full extractor.
"""

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from e0_manifest import validate as validate_manifest

REFERENCE_SHA = "9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"

PE_FILES = {
    "PE_dense": (
        "torchtitan/models/llama3/parallelize.py",
        "torchtitan/models/llama3/sharding.py",
        "torchtitan/models/common/decoder_sharding.py",
        "torchtitan/distributed/context_parallel/api.py",
        "torchtitan/distributed/pipeline_parallel.py",
    ),
    "PE_moe": (
        "torchtitan/models/deepseek_v3/parallelize.py",
        "torchtitan/models/deepseek_v3/sharding.py",
        "torchtitan/models/common/decoder_sharding.py",
        "torchtitan/models/common/moe_sharding.py",
        "torchtitan/models/common/token_dispatcher.py",
        "torchtitan/distributed/pipeline_parallel.py",
    ),
}

ROUTE_ROOTS = {
    "PE_dense": {"set_llama3_sharding_config"},
    "PE_moe": {
        "set_deepseek_v3_sharding_config",
        "_token_count_exchange",
        "_dispatch_token_exchange",
        "_combine_token_exchange",
    },
}

COMMUNICATION_NAMES = {
    "all_gather", "all_gather_into_tensor", "all_reduce", "all_to_all",
    "all_to_all_single", "broadcast", "recv", "reduce_scatter",
    "reduce_scatter_tensor", "send",
}


@dataclass(frozen=True)
class Candidate:
    pe: str
    kind: str
    symbol: str
    source: str
    line: int
    enclosing_function: str
    evidence: str
    role: str | None = None
    transition: str | None = None
    classification_basis: str | None = None
    route_status: str = "NOT_EVALUATED"
    status: str = "UNCLASSIFIED_PROTOTYPE"


@dataclass(frozen=True)
class FrameworkCandidate:
    pe: str
    subsystem: str
    transition: str
    group: str
    tensor_class: str
    structural_scope: str
    multiplicity: str
    provenance: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class StorageSemantic:
    pe: str
    logical_parameter: str
    role: str
    normalized_form: tuple[tuple[str, str], ...]
    tp_placement: str
    multiplicity: str
    dtype_class: str
    provenance: tuple[str, ...]
    status: str


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class InventoryVisitor(ast.NodeVisitor):
    def __init__(self, pe: str, source: str, text: str) -> None:
        self.pe = pe
        self.source = source
        self.lines = text.splitlines()
        self.functions: list[str] = []
        self.candidates: list[Candidate] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        leaf = name.rsplit(".", 1)[-1]
        kind = None
        if leaf == "ShardingConfig":
            kind = "sharding_boundary"
        elif leaf in COMMUNICATION_NAMES:
            kind = "explicit_communication"
        elif leaf == "redistribute":
            kind = "explicit_redistribution"
        if kind:
            function = self.functions[-1] if self.functions else "<module>"
            role, transition, basis = classify_candidate(kind, function)
            end = getattr(node, "end_lineno", node.lineno)
            evidence = " ".join(
                part.strip() for part in self.lines[node.lineno - 1 : end] if part.strip()
            )
            self.candidates.append(
                Candidate(
                    pe=self.pe,
                    kind=kind,
                    symbol=name,
                    source=self.source,
                    line=node.lineno,
                    enclosing_function=function,
                    evidence=evidence[:500],
                    role=role,
                    transition=transition,
                    classification_basis=basis,
                    status=(
                        "RULE_CLASSIFIED_PROTOTYPE"
                        if role is not None or transition is not None
                        else "UNCLASSIFIED_PROTOTYPE"
                    ),
                )
            )
        self.generic_visit(node)


class CallGraphVisitor(ast.NodeVisitor):
    """Collect direct function calls and whether their edge is conditional."""

    def __init__(self) -> None:
        self.function: str | None = None
        self.conditional_depth = 0
        self.edges: dict[str, list[tuple[str, bool]]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous = self.function
        self.function = node.name
        self.edges.setdefault(node.name, [])
        for statement in node.body:
            self.visit(statement)
        self.function = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self.conditional_depth += 1
        for statement in (*node.body, *node.orelse):
            self.visit(statement)
        self.conditional_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if self.function:
            leaf = dotted_name(node.func).rsplit(".", 1)[-1]
            self.edges[self.function].append((leaf, self.conditional_depth > 0))
        self.generic_visit(node)


def route_reachability(repo: Path, files: tuple[str, ...], roots: set[str]) -> dict[str, str]:
    graph: dict[str, list[tuple[str, bool]]] = {}
    for relative in files:
        tree = ast.parse((repo / relative).read_text(encoding="utf-8"), filename=relative)
        visitor = CallGraphVisitor()
        visitor.visit(tree)
        for function, edges in visitor.edges.items():
            graph.setdefault(function, []).extend(edges)

    status = {root: "ACTIVE_STATIC" for root in roots}
    queue = list(roots)
    while queue:
        caller = queue.pop(0)
        caller_status = status[caller]
        for callee, conditional_edge in graph.get(caller, []):
            if callee not in graph:
                continue
            proposed = (
                "CONDITIONAL_RUNTIME"
                if conditional_edge or caller_status == "CONDITIONAL_RUNTIME"
                else "ACTIVE_STATIC"
            )
            current = status.get(callee)
            if current == "ACTIVE_STATIC" or current == proposed:
                continue
            status[callee] = proposed
            queue.append(callee)
    return status


def resolve_manifest_conditions(
    pe: str, status: dict[str, str], manifest_pe: dict[str, object]
) -> dict[str, str]:
    """Resolve reviewed config branches using frozen candidate architecture facts."""
    resolved = dict(status)
    architecture = manifest_pe["arquitectura"]
    if pe == "PE_moe":
        if architecture["mtp_layers"] == 0:
            resolved["_set_deepseek_v3_mtp_sharding"] = "UNREACHABLE_MANIFEST"
        if architecture["dense_layers"] > 0:
            for function in ("set_dense_ffn_sharding",):
                resolved[function] = "ACTIVE_MANIFEST"
        if architecture["moe_layers"] > 0:
            for function in (
                "set_moe_sharding_config",
                "_router_sharding_config",
                "_shared_expert_colwise_config",
                "_shared_expert_rowwise_config",
                "_shared_experts_sharding_configs",
                "_routed_experts_sharding_configs",
                "_moe_sharding_config",
            ):
                resolved[function] = "ACTIVE_MANIFEST"
    return resolved


def resolve_candidate_semantics(candidate: Candidate) -> Candidate:
    """Resolve reviewed per-declaration branches and boundary transitions."""
    route_status = candidate.route_status
    key = (Path(candidate.source).name, candidate.enclosing_function, candidate.line)
    inactive_standard_backend = {
        ("decoder_sharding.py", "set_gqa_attention_sharding", 219),
        ("decoder_sharding.py", "set_dense_ffn_sharding", 297),
        ("moe_sharding.py", "_router_sharding_config", 114),
    }
    if key in inactive_standard_backend and route_status in {
        "ACTIVE_STATIC",
        "ACTIVE_MANIFEST",
    }:
        return replace(candidate, route_status="UNREACHABLE_MANIFEST")

    transition_rules = {
        ("decoder_sharding.py", "rowwise_config", 91): (
            "ReduceScatter",
            "Partial(tp)→Shard(seq) with sequence parallel enabled",
        ),
        ("decoder_sharding.py", "pre_lm_head_norm_config", 132): (
            "AllGather",
            "Shard(seq)→Replicate(tp) before LMHead",
        ),
        ("decoder_sharding.py", "set_gqa_attention_sharding", 192): (
            "AllGather",
            "Shard(seq)→Replicate(tp) at attention boundary",
        ),
        ("decoder_sharding.py", "set_gqa_inner_attention_local_map", 247): (
            "AllGather",
            "Shard(seq)→Replicate(cp) for K/V local-map inputs",
        ),
        ("decoder_sharding.py", "set_dense_ffn_sharding", 288): (
            "AllGather",
            "Shard(seq)→Replicate(tp) at dense FFN boundary",
        ),
        ("decoder_sharding.py", "set_decoder_sharding_config", 320): (
            "ReduceScatter",
            "Partial(tp)→Shard(seq) at embedding output",
        ),
        ("sharding.py", "_set_deepseek_v3_layer_sharding", 98): (
            "AllGather",
            "Shard(seq)→Replicate(tp) at MLA attention boundary",
        ),
        ("moe_sharding.py", "_shared_expert_rowwise_config", 149): (
            "ReduceScatter",
            "Partial(tp)→Shard(seq) at shared-expert output",
        ),
        ("moe_sharding.py", "_shared_experts_sharding_configs", 184): (
            "AllGather",
            "Shard(seq)→Replicate(tp) at shared-expert input",
        ),
    }
    if candidate.kind == "sharding_boundary" and key in transition_rules:
        transition, basis = transition_rules[key]
        existing = candidate.classification_basis
        combined = f"{existing}; {basis}" if existing else basis
        return replace(
            candidate,
            transition=transition,
            classification_basis=combined,
            status="RULE_CLASSIFIED_PROTOTYPE",
        )
    if (
        candidate.kind == "sharding_boundary"
        and route_status in {"ACTIVE_STATIC", "ACTIVE_MANIFEST"}
        and candidate.transition is None
    ):
        return replace(
            candidate,
            transition="NONE",
            classification_basis=(candidate.classification_basis or "")
            + ("; " if candidate.classification_basis else "")
            + "no src→dst placement change in reviewed declaration",
            status="RULE_CLASSIFIED_PROTOTYPE",
        )
    return candidate


def classify_candidate(
    kind: str, function: str
) -> tuple[str | None, str | None, str | None]:
    """Apply only classifications directly supported by §1 and provenance.

    Function names are not treated as general semantic evidence.  These rules
    cover narrowly reviewed helpers whose implementations and docstrings state
    the relevant placement or distribution change at the pinned HEAD.
    """
    if kind == "explicit_communication":
        if function == "_dispatch_token_exchange":
            return None, "Dispatch", "§1.2 token→expert; reviewed dispatcher helper"
        if function == "_combine_token_exchange":
            return None, "Combine", "§1.2 expert→token; reviewed dispatcher helper"
        if function == "_token_count_exchange":
            return None, "AllToAll", "§1.4 residual exchange; control-count payload"
    if kind == "sharding_boundary":
        role_rules = {
            "colwise_config": ("ColLinear", "§1.3.2 TP Shard(output_feature)"),
            "rowwise_config": ("RowLinear", "§1.3.2 TP Shard(input_feature)"),
            "norm_config": ("Norm", "§1.3.1 feature normalization helper"),
            "pre_lm_head_norm_config": (
                "Norm",
                "§1.3.1 pre-LMHead feature normalization helper",
            ),
            "_router_sharding_config": (
                "Router",
                "§1.3.1 expert-score routing projection",
            ),
        }
        if function in role_rules:
            role, basis = role_rules[function]
            return role, None, basis
    return None, None, None


def logical_transitions(candidates: list[Candidate]) -> list[dict[str, object]]:
    """Collapse alternate backend calls into one logical transition template."""
    grouped: dict[tuple[str, str, str, str], list[Candidate]] = {}
    for item in candidates:
        if item.transition is None or item.kind != "explicit_communication":
            continue
        key = (item.pe, item.source, item.enclosing_function, item.transition)
        grouped.setdefault(key, []).append(item)
    return [
        {
            "pe": key[0],
            "source": key[1],
            "enclosing_function": key[2],
            "transition": key[3],
            "implementation_call_count": len(items),
            "implementation_lines": sorted(item.line for item in items),
            "classification_basis": items[0].classification_basis,
            "status": "LOGICAL_TEMPLATE_PROTOTYPE",
        }
        for key, items in sorted(grouped.items())
    ]


def transition_inventory(candidates: list[Candidate]) -> dict[str, dict[str, int]]:
    """Count active boundary declarations plus deduplicated logical comms."""
    result: dict[str, dict[str, int]] = {pe: {} for pe in PE_FILES}
    active = {"ACTIVE_STATIC", "ACTIVE_MANIFEST"}
    for item in candidates:
        if (
            item.kind == "sharding_boundary"
            and item.route_status in active
            and item.transition is not None
        ):
            counts = result[item.pe]
            counts[item.transition] = counts.get(item.transition, 0) + 1
    for item in logical_transitions(candidates):
        counts = result[item["pe"]]
        transition = item["transition"]
        counts[transition] = counts.get(transition, 0) + 1
    return {pe: dict(sorted(counts.items())) for pe, counts in result.items()}


def validate_hsdp_trace(trace: dict[str, object]) -> None:
    required = {
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
    for field, expected in required.items():
        if trace.get(field) != expected:
            raise ValueError(f"invalid HSDP trace field {field}: {trace.get(field)!r}")


def framework_candidates(
    manifest: dict[str, object], hsdp_trace: dict[str, object] | None = None
) -> list[FrameworkCandidate]:
    """Emit symbolic framework events without pretending they are templates."""
    if hsdp_trace is not None:
        validate_hsdp_trace(hsdp_trace)
    events: list[FrameworkCandidate] = []
    for pe, spec in manifest["pes"].items():
        parts = spec["overrides"]["module_fqns_per_model_part"]
        if len(parts) > 1:
            events.append(
                FrameworkCandidate(
                    pe=pe,
                    subsystem="pipeline",
                    transition="SendRecv",
                    group="pp",
                    tensor_class="activation",
                    structural_scope="pipeline_stage_edge",
                    multiplicity="P - 1",
                    provenance=(
                        "§1.7 multiplicity vocabulary",
                        "§1.8 Q2 each pp edge once, independent of microbatches",
                        "candidate manifest module_fqns_per_model_part",
                    ),
                    status="SYMBOLIC_FRAMEWORK_CANDIDATE",
                )
            )

        architecture = spec["arquitectura"]
        dense_layers = architecture["dense_layers"]
        moe_layers = architecture["moe_layers"]
        fsdp_scopes = [
            ("decoder_root_input_output", "1"),
            ("decoder_nonexpert_layer", "L"),
        ]
        if moe_layers:
            fsdp_scopes.append(("routed_expert", "L_moe"))
        for scope, multiplicity in fsdp_scopes:
            group = "efsdp" if scope == "routed_expert" else "dp_s"
            for transition, tensor_class in (
                ("AllGather", "param"),
                ("ReduceScatter", "grad"),
            ):
                events.append(
                    FrameworkCandidate(
                        pe=pe,
                        subsystem="fsdp",
                        transition=transition,
                        group=group,
                        tensor_class=tensor_class,
                        structural_scope=scope,
                        multiplicity=multiplicity,
                        provenance=(
                            "torchtitan/distributed/fsdp.py::apply_fsdp_to_decoder",
                            "§1.4 Partial/Shard/Replicate transition guards",
                            "§1.6.6 parameter-gradient provenance",
                        ),
                        status="REQUIRES_SEMANTIC_DECOMPOSITION",
                    )
                )
        if spec["grados"]["dp_r"] is not None:
            for scope, multiplicity in (
                ("decoder_root_input_output", "1"),
                ("decoder_nonexpert_layer", "L"),
            ):
                events.append(
                    FrameworkCandidate(
                        pe=pe,
                        subsystem="hsdp",
                        transition="AllReduce",
                        group="dp_r",
                        tensor_class="grad",
                        structural_scope=scope,
                        multiplicity=multiplicity,
                        provenance=(
                            "torchtitan/distributed/fsdp.py HSDP dp_replicate mesh",
                            "§1.4 Partial→Replicate",
                        ),
                        status=(
                            "CONFIRMED_CPU_GLOO_MECHANICS"
                            if hsdp_trace is not None
                            else "REQUIRES_RUNTIME_CROSSCHECK"
                        ),
                    )
                )
        if dense_layers + moe_layers != architecture["layers"]:
            raise ValueError(f"{pe} architecture scopes do not cover all layers")
    return events


def dense_storage_catalog(manifest: dict[str, object]) -> list[StorageSemantic]:
    """Catalog the unambiguous Llama3 debugmodel logical parameter families."""
    dtype_class = manifest["pes"]["PE_dense"]["dtype_classes"]["param"]
    common = {"pe": "PE_dense", "dtype_class": dtype_class}
    entries = [
        ("tok_embeddings.weight", "Embedding", (("vocab", "V"), ("output_feature", "D")), "tp:Shard(vocab)", "1"),
        ("norm.weight", "Norm", (("model", "D"),), "tp:Replicate", "1"),
        ("lm_head.weight", "LMHead", (("vocab", "V"), ("input_feature", "D")), "tp:Shard(vocab)", "1"),
        ("layers.*.attention_norm.weight", "Norm", (("model", "D"),), "tp:Replicate", "L"),
        ("layers.*.ffn_norm.weight", "Norm", (("model", "D"),), "tp:Replicate", "L"),
        (
            "layers.*.attention.qkv_linear.wqkv.weight",
            "ColLinear",
            (("output_feature", "(H+2*Hkv)*Dh"), ("input_feature", "D")),
            "tp:Shard(output_feature)",
            "L",
        ),
        (
            "layers.*.attention.wo.weight",
            "RowLinear",
            (("output_feature", "D"), ("input_feature", "D")),
            "tp:Shard(input_feature)",
            "L",
        ),
        (
            "layers.*.feed_forward.{w1,w3}.weight",
            "ColLinear",
            (("output_feature", "F"), ("input_feature", "D")),
            "tp:Shard(output_feature)",
            "2*L",
        ),
        (
            "layers.*.feed_forward.w2.weight",
            "RowLinear",
            (("output_feature", "D"), ("input_feature", "F")),
            "tp:Shard(input_feature)",
            "L",
        ),
    ]
    return [
        StorageSemantic(
            **common,
            logical_parameter=name,
            role=role,
            normalized_form=form,
            tp_placement=placement,
            multiplicity=multiplicity,
            provenance=(
                "torchtitan/models/llama3/__init__.py::_debugmodel",
                "torchtitan/models/common/decoder_sharding.py",
                "preregistration §1.3 and §1.6.4.A",
            ),
            status="SEMANTIC_STORAGE_CATALOGED",
        )
        for name, role, form, placement, multiplicity in entries
    ]


def moe_storage_catalog(manifest: dict[str, object]) -> list[StorageSemantic]:
    """Catalog MoE-specific DeepSeek debugmodel logical parameter families."""
    dtype_class = manifest["pes"]["PE_moe"]["dtype_classes"]["param"]
    common = {"pe": "PE_moe", "dtype_class": dtype_class}
    entries = [
        (
            "layers.moe.*.router.gate.weight",
            "Router",
            (("expert", "E"), ("input_feature", "D")),
            "dense:{dp_s:Shard(expert),tp:Replicate}",
            "L_moe",
        ),
        (
            "layers.moe.*.shared_experts.{w1,w3}.weight",
            "ColLinear",
            (("output_feature", "2*F"), ("input_feature", "D")),
            "dense:{dp_s:Shard(output_feature),tp:Shard(output_feature)}",
            "2*L_moe",
        ),
        (
            "layers.moe.*.shared_experts.w2.weight",
            "RowLinear",
            (("output_feature", "D"), ("input_feature", "2*F")),
            "dense:{dp_s:Shard(output_feature),tp:Shard(input_feature)}",
            "L_moe",
        ),
        (
            "layers.moe.*.routed_experts.inner_experts.{w1_EFD,w3_EFD}",
            "GroupedGEMM",
            (("expert", "E"), ("output_feature", "F"), ("input_feature", "D")),
            "sparse:{efsdp:Shard(expert),ep:Shard(expert)}",
            "2*L_moe",
        ),
        (
            "layers.moe.*.routed_experts.inner_experts.w2_EDF",
            "GroupedGEMM",
            (("expert", "E"), ("output_feature", "D"), ("input_feature", "F")),
            "sparse:{efsdp:Shard(expert),ep:Shard(expert)}",
            "L_moe",
        ),
    ]
    return [
        StorageSemantic(
            **common,
            logical_parameter=name,
            role=role,
            normalized_form=form,
            tp_placement=placement,
            multiplicity=multiplicity,
            provenance=(
                "torchtitan/models/deepseek_v3/__init__.py::_debugmodel",
                "torchtitan/models/common/moe.py::GroupedExperts",
                "torchtitan/models/common/moe_sharding.py",
                "preregistration §1.3 and §1.6.4.A",
            ),
            status="SEMANTIC_STORAGE_CATALOGED",
        )
        for name, role, form, placement, multiplicity in entries
    ]
def verify_reference(repo: Path) -> str:
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect reference repo {repo}: {exc}") from exc
    if actual != REFERENCE_SHA:
        raise ValueError(f"reference HEAD mismatch: expected {REFERENCE_SHA}, got {actual}")
    return actual


def inventory_file(repo: Path, pe: str, relative: str) -> list[Candidate]:
    path = repo / relative
    if not path.is_file():
        raise ValueError(f"required reference file missing: {relative}")
    text = path.read_text(encoding="utf-8")
    visitor = InventoryVisitor(pe, relative, text)
    visitor.visit(ast.parse(text, filename=relative))
    return visitor.candidates


def run(
    repo: Path,
    manifest: dict[str, object],
    hsdp_trace: dict[str, object] | None = None,
) -> dict[str, object]:
    actual = verify_reference(repo)
    manifest_report = validate_manifest(repo, manifest)
    candidates = []
    for pe, files in PE_FILES.items():
        reachability = route_reachability(repo, files, ROUTE_ROOTS[pe])
        reachability = resolve_manifest_conditions(
            pe, reachability, manifest["pes"][pe]
        )
        for relative in files:
            for candidate in inventory_file(repo, pe, relative):
                route_status = reachability.get(candidate.enclosing_function, "UNREACHABLE_STATIC")
                candidates.append(
                    resolve_candidate_semantics(Candidate(
                        **{
                            **asdict(candidate),
                            "route_status": route_status,
                        }
                    ))
                )
    by_pe = {
        pe: {
            "files_scanned": len(files),
            "candidate_count": sum(item.pe == pe for item in candidates),
            "rule_classified_count": sum(
                item.pe == pe and item.status == "RULE_CLASSIFIED_PROTOTYPE"
                for item in candidates
            ),
            "active_static_count": sum(
                item.pe == pe
                and item.route_status in {"ACTIVE_STATIC", "ACTIVE_MANIFEST"}
                for item in candidates
            ),
            "conditional_runtime_count": sum(
                item.pe == pe and item.route_status == "CONDITIONAL_RUNTIME"
                for item in candidates
            ),
            "unreachable_static_count": sum(
                item.pe == pe
                and item.route_status in {
                    "UNREACHABLE_STATIC",
                    "UNREACHABLE_MANIFEST",
                }
                for item in candidates
            ),
        }
        for pe, files in PE_FILES.items()
    }
    return {
        "status": "PROTOTYPE_PARTIAL_CLASSIFICATION",
        "e0_closed": False,
        "e6_computed": False,
        "population_touched": False,
        "reference_sha": actual,
        "manifest_sha256": manifest_report["manifest_sha256"],
        "coverage_claim": "NONE_UNTIL_MANUAL_AND_RUNTIME_CROSSCHECK",
        "manifest_status": manifest["status"],
        "manifest_routes": {
            pe: {
                "modulo_registro": spec["modulo_registro"],
                "funcion_config": spec["funcion_config"],
            }
            for pe, spec in manifest["pes"].items()
        },
        "blocking_gaps": [
            "Candidate PE manifest is validated but not yet frozen into the preregistration.",
            "Placements and tensor signatures are not yet normalized into all seven template fields.",
            "Framework-generated FSDP/HSDP and pipeline communications are not yet expanded into templates.",
            "Static candidates have not yet been cross-checked against runtime execution paths.",
        ],
        "pe_summary": by_pe,
        "logical_transition_candidates": logical_transitions(candidates),
        "transition_inventory_prototype": transition_inventory(candidates),
        "framework_transition_candidates": [
            asdict(item) for item in framework_candidates(manifest, hsdp_trace)
        ],
        "dense_storage_semantics": [
            asdict(item) for item in dense_storage_catalog(manifest)
        ],
        "moe_storage_semantics": [
            asdict(item) for item in moe_storage_catalog(manifest)
        ],
        "hsdp_runtime_crosscheck": (
            "CONFIRMED_CPU_GLOO_MECHANICS"
            if hsdp_trace is not None
            else "NOT_PROVIDED"
        ),
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def external_output(path: Path) -> Path:
    """Reject generated reports inside this instrumentation checkout."""
    resolved = path.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(checkout)
    except ValueError:
        return resolved
    raise ValueError(f"output must be outside the Planlock checkout: {resolved}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("e0-manifest-candidate.json")
    )
    parser.add_argument("--hsdp-trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        hsdp_trace = (
            json.loads(args.hsdp_trace.read_text(encoding="utf-8"))
            if args.hsdp_trace
            else None
        )
        report = run(args.reference_repo.resolve(), manifest, hsdp_trace)
        output = external_output(args.output) if args.output else None
    except (OSError, json.JSONDecodeError, SyntaxError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
