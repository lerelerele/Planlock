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
import re
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
    tensor_class: str
    provenance: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class SevenFieldTemplate:
    pe: str
    producer_role: str
    producer_placement: tuple[tuple[str, str], ...]
    transition: str
    consumer_placement: tuple[tuple[str, str], ...]
    consumer_role: str
    tensor_signature: tuple[tuple[tuple[str, str], ...], str, str]
    communication_group: str
    multiplicity: str
    provenance: tuple[str, ...]
    status: str


KNOWN_AXIS_IDENTITIES = {
    "batch", "seq", "token", "query_pos", "key_pos", "model",
    "input_feature", "output_feature", "head", "kv_head", "head_dim",
    "ffn_hidden", "vocab", "expert", "topk", "capacity",
    "expert_offset", "layer", "routed_item", "kv_latent",
    "attention_feature", "axis_opaque",
}


def validate_storage_signatures(items: list[StorageSemantic]) -> None:
    """Validate the three normative components of every §1.6 signature."""
    allowed_dtypes = {"f32", "f16", "f8", "f4", "i64", "i32", "i8", "bool"}
    allowed_tensor_classes = {
        "activation", "control_metadata", "param", "grad", "optimizer_state"
    }
    for item in items:
        identities = [identity for identity, _ in item.normalized_form]
        unknown = set(identities) - KNOWN_AXIS_IDENTITIES
        if unknown:
            raise ValueError(
                f"{item.logical_parameter} has unknown axis identities: {sorted(unknown)}"
            )
        known = [identity for identity in identities if identity != "axis_opaque"]
        if len(known) != len(set(known)):
            raise ValueError(
                f"{item.logical_parameter} repeats a known axis identity"
            )
        if any(not expression for _, expression in item.normalized_form):
            raise ValueError(f"{item.logical_parameter} has an empty axis expression")
        if item.dtype_class not in allowed_dtypes:
            raise ValueError(
                f"{item.logical_parameter} has unsupported dtype class {item.dtype_class}"
            )
        if item.tensor_class not in allowed_tensor_classes:
            raise ValueError(
                f"{item.logical_parameter} has unsupported tensor class {item.tensor_class}"
            )


def dense_tp_component(item: StorageSemantic) -> tuple[str, str]:
    prefix = "tp:"
    if not item.tp_placement.startswith(prefix):
        raise ValueError(f"unsupported dense TP placement: {item.tp_placement}")
    component = item.tp_placement[len(prefix):]
    if component == "Replicate":
        return ("tp", "Replicate")
    if component.startswith("Shard(") and component.endswith(")"):
        return ("tp", component)
    raise ValueError(f"unsupported dense TP component: {component}")


def dense_framework_templates(
    manifest: dict[str, object],
    parameters: list[StorageSemantic],
    gradients: list[StorageSemantic],
) -> list[SevenFieldTemplate]:
    """Compose dense FSDP/HSDP events into the normative seven fields."""
    spec = manifest["pes"]["PE_dense"]
    if spec["grados"].get("dp_r") is None or spec["grados"].get("cp") is None:
        raise ValueError("PE_dense template composition requires HSDP and CP")
    gradient_by_parameter = {
        item.logical_parameter.removesuffix("::grad"): item for item in gradients
        if item.pe == "PE_dense"
    }
    templates: list[SevenFieldTemplate] = []
    for parameter in parameters:
        if parameter.pe != "PE_dense":
            continue
        gradient = gradient_by_parameter[parameter.logical_parameter]
        shard_axis = parameter.normalized_form[0][0]
        tp = dense_tp_component(parameter)
        sharded = (
            ("dp_r", "Replicate"),
            ("dp_s", f"Shard({shard_axis})"),
            ("cp", f"Shard({shard_axis})"),
            tp,
        )
        gathered = (
            ("dp_r", "Replicate"),
            ("dp_s", "Replicate"),
            ("cp", "Replicate"),
            tp,
        )
        partial_grad = (
            ("dp_r", "Replicate"),
            ("dp_s", "Partial(Sum)"),
            ("cp", "Partial(Sum)"),
            tp,
        )
        hsdp_partial = (
            ("dp_r", "Partial(Sum)"),
            ("dp_s", f"Shard({shard_axis})"),
            ("cp", f"Shard({shard_axis})"),
            tp,
        )
        param_signature = (
            parameter.normalized_form,
            parameter.dtype_class,
            parameter.tensor_class,
        )
        grad_signature = (
            gradient.normalized_form,
            gradient.dtype_class,
            gradient.tensor_class,
        )
        common_provenance = parameter.provenance + (
            "torchtitan/distributed/fsdp.py::apply_fsdp_to_decoder",
            "torchtitan/distributed/parallel_dims.py fsdp=dp_shard*cp",
            "real CPU/Gloo HSDP mechanics trace",
        )
        templates.extend(
            (
                SevenFieldTemplate(
                    pe="PE_dense",
                    producer_role="OptimizerUpdate",
                    producer_placement=sharded,
                    transition="AllGather",
                    consumer_placement=gathered,
                    consumer_role=parameter.role,
                    tensor_signature=param_signature,
                    communication_group="product(dp_s,cp)",
                    multiplicity=parameter.multiplicity,
                    provenance=common_provenance,
                    status="SEVEN_FIELD_TEMPLATE_CANDIDATE",
                ),
                SevenFieldTemplate(
                    pe="PE_dense",
                    producer_role=parameter.role,
                    producer_placement=partial_grad,
                    transition="ReduceScatter",
                    consumer_placement=sharded,
                    consumer_role="OptimizerUpdate",
                    tensor_signature=grad_signature,
                    communication_group="product(dp_s,cp)",
                    multiplicity=parameter.multiplicity,
                    provenance=common_provenance,
                    status="SEVEN_FIELD_TEMPLATE_CANDIDATE",
                ),
                SevenFieldTemplate(
                    pe="PE_dense",
                    producer_role=parameter.role,
                    producer_placement=hsdp_partial,
                    transition="AllReduce",
                    consumer_placement=sharded,
                    consumer_role="OptimizerUpdate",
                    tensor_signature=grad_signature,
                    communication_group="dp_r",
                    multiplicity=parameter.multiplicity,
                    provenance=common_provenance,
                    status="SEVEN_FIELD_TEMPLATE_CANDIDATE",
                ),
            )
        )
    return templates


def parse_moe_storage_placement(item: StorageSemantic) -> tuple[str, dict[str, str]]:
    match = re.fullmatch(r"(dense|sparse):\{(.+)\}", item.tp_placement)
    if match is None:
        raise ValueError(f"unsupported MoE storage placement: {item.tp_placement}")
    family, body = match.groups()
    components = dict(
        re.findall(r"([a-z_]+):(Shard\([a-z_]+\)|Replicate)", body)
    )
    expected = {"dp_s", "tp"} if family == "dense" else {"efsdp", "ep"}
    if set(components) != expected:
        raise ValueError(f"incomplete {family} MoE placement: {components}")
    return family, components


def moe_framework_templates(
    manifest: dict[str, object],
    parameters: list[StorageSemantic],
    gradients: list[StorageSemantic],
) -> list[SevenFieldTemplate]:
    """Compose reviewed MoE parameter families over dense/sparse FSDP meshes."""
    if manifest["pes"]["PE_moe"]["grados"].get("dp_r") is not None:
        raise ValueError("PE_moe template composition does not expect HSDP")
    gradient_by_parameter = {
        item.logical_parameter.removesuffix("::grad"): item for item in gradients
        if item.pe == "PE_moe"
    }
    templates: list[SevenFieldTemplate] = []
    for parameter in parameters:
        if parameter.pe != "PE_moe":
            continue
        gradient = gradient_by_parameter[parameter.logical_parameter]
        family, components = parse_moe_storage_placement(parameter)
        fsdp_axis = "dp_s" if family == "dense" else "efsdp"
        other_axis = "tp" if family == "dense" else "ep"
        group = fsdp_axis
        sharded = (
            (fsdp_axis, components[fsdp_axis]),
            (other_axis, components[other_axis]),
        )
        gathered = (
            (fsdp_axis, "Replicate"),
            (other_axis, components[other_axis]),
        )
        partial = (
            (fsdp_axis, "Partial(Sum)"),
            (other_axis, components[other_axis]),
        )
        common_provenance = parameter.provenance + (
            "torchtitan/models/common/moe_sharding.py",
            "torchtitan/distributed/fsdp.py::apply_fsdp_to_decoder",
        )
        templates.extend(
            (
                SevenFieldTemplate(
                    pe="PE_moe",
                    producer_role="OptimizerUpdate",
                    producer_placement=sharded,
                    transition="AllGather",
                    consumer_placement=gathered,
                    consumer_role=parameter.role,
                    tensor_signature=(
                        parameter.normalized_form,
                        parameter.dtype_class,
                        parameter.tensor_class,
                    ),
                    communication_group=group,
                    multiplicity=parameter.multiplicity,
                    provenance=common_provenance,
                    status="SEVEN_FIELD_TEMPLATE_CANDIDATE",
                ),
                SevenFieldTemplate(
                    pe="PE_moe",
                    producer_role=parameter.role,
                    producer_placement=partial,
                    transition="ReduceScatter",
                    consumer_placement=sharded,
                    consumer_role="OptimizerUpdate",
                    tensor_signature=(
                        gradient.normalized_form,
                        gradient.dtype_class,
                        gradient.tensor_class,
                    ),
                    communication_group=group,
                    multiplicity=parameter.multiplicity,
                    provenance=common_provenance,
                    status="SEVEN_FIELD_TEMPLATE_CANDIDATE",
                ),
            )
        )
    return templates


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
        dense_fsdp_group = (
            "product(dp_s,cp)" if spec["grados"].get("cp") is not None else "dp_s"
        )
        fsdp_scopes = [
            ("decoder_root_input_output", "1"),
            ("decoder_nonexpert_layer", "L"),
        ]
        if moe_layers:
            fsdp_scopes.append(("routed_expert", "L_moe"))
        for scope, multiplicity in fsdp_scopes:
            group = "efsdp" if scope == "routed_expert" else dense_fsdp_group
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
                            "torchtitan/distributed/parallel_dims.py fsdp=dp_shard*cp",
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
    common = {
        "pe": "PE_dense", "dtype_class": dtype_class, "tensor_class": "param"
    }
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
    catalog = [
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
            status="SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
        )
        for name, role, form, placement, multiplicity in entries
    ]
    validate_storage_signatures(catalog)
    return catalog


def moe_storage_catalog(manifest: dict[str, object]) -> list[StorageSemantic]:
    """Catalog MoE-specific DeepSeek debugmodel logical parameter families."""
    dtype_class = manifest["pes"]["PE_moe"]["dtype_classes"]["param"]
    common = {
        "pe": "PE_moe", "dtype_class": dtype_class, "tensor_class": "param"
    }
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
    catalog = [
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
            status="SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
        )
        for name, role, form, placement, multiplicity in entries
    ]
    validate_storage_signatures(catalog)
    return catalog


def gradient_signature_catalog(
    manifest: dict[str, object], parameter_catalog: list[StorageSemantic]
) -> list[StorageSemantic]:
    """Derive logical gradient signatures without claiming transition placements."""
    gradients = [
        replace(
            item,
            logical_parameter=f"{item.logical_parameter}::grad",
            dtype_class=manifest["pes"][item.pe]["dtype_classes"]["grad_reduce"],
            tensor_class="grad",
            provenance=item.provenance
            + (
                "candidate manifest dtype_classes.grad_reduce",
                "preregistration §1.6.6 parameter-gradient provenance",
            ),
            status="SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
        )
        for item in parameter_catalog
    ]
    validate_storage_signatures(gradients)
    return gradients


def optimizer_state_signature_catalog(
    manifest: dict[str, object], parameter_catalog: list[StorageSemantic]
) -> list[StorageSemantic]:
    """Derive AdamW moment and scalar-step signatures from reviewed config."""
    states: list[StorageSemantic] = []
    for item in parameter_catalog:
        optimizer = manifest["pes"][item.pe]["optimizer"]
        if optimizer["name"] != "AdamW" or optimizer["amsgrad"]:
            raise ValueError(f"unsupported optimizer state for {item.pe}")
        state_tensors = optimizer["state_tensors"]
        for state_name in ("exp_avg", "exp_avg_sq"):
            if state_tensors[state_name] != "same_as_param":
                raise ValueError(f"unsupported {state_name} dtype rule for {item.pe}")
            states.append(
                replace(
                    item,
                    logical_parameter=f"{item.logical_parameter}::optimizer.{state_name}",
                    role="OptimizerUpdate",
                    tensor_class="optimizer_state",
                    provenance=item.provenance
                    + (
                        "selected config registry function uses default_adamw",
                        "torchtitan/components/optimizer.py::default_adamw",
                        "PyTorch AdamW state follows parameter dtype",
                    ),
                    status="SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
                )
            )
        if state_tensors["step"] != "f32":
            raise ValueError(f"unsupported AdamW step dtype rule for {item.pe}")
        states.append(
            replace(
                item,
                logical_parameter=f"{item.logical_parameter}::optimizer.step",
                role="OptimizerUpdate",
                normalized_form=(),
                tp_placement="scalar:no_tensor_axis_placement",
                dtype_class="f32",
                tensor_class="optimizer_state",
                provenance=item.provenance
                + (
                    "selected config registry function uses default_adamw",
                    "torch.optim.AdamW per-parameter scalar step state",
                ),
                status="SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
            )
        )
    validate_storage_signatures(states)
    return states


def moe_control_metadata_catalog(manifest: dict[str, object]) -> list[StorageSemantic]:
    """Catalog standard-dispatcher routing metadata with reviewed logical forms."""
    spec = manifest["pes"]["PE_moe"]
    if spec["overrides"].get("moe_comm_backend") != "standard":
        raise ValueError("control metadata catalog requires the standard dispatcher")
    common = {
        "pe": "PE_moe",
        "tp_placement": "NOT_COMPOSED",
        "multiplicity": "L_moe",
        "tensor_class": "control_metadata",
        "provenance": (
            "torchtitan/models/deepseek_v3/__init__.py::model_registry moe_comm_backend=standard",
            "torchtitan/models/common/config_utils.py::make_token_dispatcher_config",
            "torchtitan/models/common/token_dispatcher.py::AllToAllTokenDispatcher",
            "preregistration §1.6.5/§1.6.6",
        ),
        "status": "SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
    }
    entries = [
        (
            "topk_expert_ids_TK",
            "Router",
            (("batch", "B"), ("seq", "S"), ("topk", "K")),
            "i64",
        ),
        (
            "routing_map_BLE",
            "Router",
            (("batch", "B"), ("seq", "S"), ("expert", "E")),
            "bool",
        ),
        (
            "token_indices_experts_sorted_N",
            "Dispatch",
            (("routed_item", "B*S*K"),),
            "i64",
        ),
        (
            "num_local_tokens_per_expert_E",
            "Dispatch",
            (("expert", "E"),),
            "i64",
        ),
        (
            "num_global_tokens_per_local_expert_EP_e",
            "Dispatch",
            (("expert", "E"),),
            "i64",
        ),
        (
            "permuted_indices_R",
            "Dispatch",
            (("routed_item", "B*S*K"),),
            "i64",
        ),
    ]
    catalog = [
        StorageSemantic(
            **common,
            logical_parameter=name,
            role=role,
            normalized_form=form,
            dtype_class=dtype_class,
        )
        for name, role, form, dtype_class in entries
    ]
    validate_storage_signatures(catalog)
    return catalog


def moe_routing_activation_catalog(manifest: dict[str, object]) -> list[StorageSemantic]:
    """Catalog differentiable tensors on the reviewed standard MoE route."""
    spec = manifest["pes"]["PE_moe"]
    if spec["overrides"].get("moe_comm_backend") != "standard":
        raise ValueError("routing activation catalog requires the standard dispatcher")
    low_precision = spec["dtype_classes"]["param"]
    common = {
        "pe": "PE_moe",
        "tp_placement": "NOT_COMPOSED",
        "multiplicity": "L_moe",
        "tensor_class": "activation",
        "provenance": (
            "torchtitan/models/common/moe.py::TokenChoiceTopKRouter.forward",
            "torchtitan/models/common/moe.py::RoutedExperts.forward",
            "torchtitan/models/common/token_dispatcher.py::AllToAllTokenDispatcher",
            "preregistration §1.6.4.B/§1.6.6",
        ),
        "status": "SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
    }
    entries = [
        ("scores_BLE", "Router", (("batch", "B"), ("seq", "S"), ("expert", "E")), "f32"),
        ("topk_scores_BLK", "Router", (("batch", "B"), ("seq", "S"), ("topk", "K")), "f32"),
        ("topk_scores_experts_sorted_N", "Router", (("routed_item", "B*S*K"),), "f32"),
        ("routed_input_RD", "Dispatch", (("routed_item", "B*S*K"), ("model", "D")), low_precision),
        ("routed_output_RD", "GroupedGEMM", (("routed_item", "B*S*K"), ("model", "D")), low_precision),
        ("out_BLD", "Combine", (("batch", "B"), ("seq", "S"), ("model", "D")), low_precision),
    ]
    catalog = [
        StorageSemantic(
            **common,
            logical_parameter=name,
            role=role,
            normalized_form=form,
            dtype_class=dtype_class,
        )
        for name, role, form, dtype_class in entries
    ]
    validate_storage_signatures(catalog)
    return catalog


def dense_nonattention_activation_catalog(
    manifest: dict[str, object],
) -> list[StorageSemantic]:
    """Catalog unambiguous Llama activations outside QKV/attention internals."""
    dtype_class = manifest["pes"]["PE_dense"]["dtype_classes"]["param"]
    common = {
        "pe": "PE_dense",
        "tp_placement": "NOT_COMPOSED",
        "tensor_class": "activation",
        "dtype_class": dtype_class,
        "provenance": (
            "torchtitan/models/common/decoder.py::Decoder.forward",
            "torchtitan/models/llama3/model.py::Llama3TransformerBlock.forward",
            "torchtitan/models/common/feed_forward.py::FeedForward.forward",
            "preregistration §1.3/§1.6.4.B/§1.6.6",
        ),
        "status": "SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
    }
    residual = (("batch", "B"), ("seq", "S"), ("model", "D"))
    hidden = (("batch", "B"), ("seq", "S"), ("ffn_hidden", "F"))
    entries = [
        ("tok_embeddings.output", "Embedding", residual, "1"),
        ("layers.*.attention_norm.output", "Norm", residual, "L"),
        ("layers.*.attention.output", "Attention", residual, "L"),
        ("layers.*.ffn_norm.output", "Norm", residual, "L"),
        ("layers.*.feed_forward.{w1,w3}.output", "ColLinear", hidden, "2*L"),
        ("layers.*.feed_forward.swiglu_product", "Opaque", hidden, "L"),
        ("layers.*.feed_forward.w2.output", "RowLinear", residual, "L"),
        ("norm.output", "Norm", residual, "1"),
        (
            "lm_head.output",
            "LMHead",
            (("batch", "B"), ("seq", "S"), ("vocab", "V")),
            "1",
        ),
    ]
    catalog = [
        StorageSemantic(
            **common,
            logical_parameter=name,
            role=role,
            normalized_form=form,
            multiplicity=multiplicity,
        )
        for name, role, form, multiplicity in entries
    ]
    validate_storage_signatures(catalog)
    return catalog


def dense_attention_activation_catalog(
    manifest: dict[str, object],
) -> list[StorageSemantic]:
    """Catalog the reviewed fused-QKV Llama attention boundary tensors."""
    spec = manifest["pes"]["PE_dense"]
    if spec["overrides"].get("attn_backend") != "flex":
        raise ValueError("dense attention catalog requires flex attention")
    if spec["arquitectura"].get("fuse_qkv") is not True:
        raise ValueError("dense attention catalog requires fused QKV")
    dtype_class = spec["dtype_classes"]["param"]
    common = {
        "pe": "PE_dense",
        "tp_placement": "NOT_COMPOSED",
        "tensor_class": "activation",
        "dtype_class": dtype_class,
        "provenance": (
            "torchtitan/models/llama3/__init__.py::_debugmodel fuse_qkv=True",
            "torchtitan/models/common/attention.py::FusedQKVLinear.forward",
            "torchtitan/models/common/attention.py::GQA.forward",
            "preregistration §1.6.4.B fused-linear fallback and QKV split",
        ),
        "status": "SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
    }
    prefix = (("batch", "B"), ("seq", "S"))
    entries = [
        (
            "layers.*.attention.qkv_linear.fused_output",
            "ColLinear",
            prefix + (("output_feature", "(H+2*Hkv)*Dh"),),
            "L",
        ),
        (
            "layers.*.attention.query_BLNH",
            "Attention",
            prefix + (("head", "H"), ("head_dim", "Dh")),
            "L",
        ),
        (
            "layers.*.attention.{key,value}_BLNH",
            "Attention",
            prefix + (("kv_head", "Hkv"), ("head_dim", "Dh")),
            "2*L",
        ),
        (
            "layers.*.attention.inner_output_BLNH",
            "Attention",
            prefix + (("head", "H"), ("head_dim", "Dh")),
            "L",
        ),
        (
            "layers.*.attention.flattened_output_BLD",
            "Attention",
            prefix + (("model", "D"),),
            "L",
        ),
    ]
    catalog = [
        StorageSemantic(
            **common,
            logical_parameter=name,
            role=role,
            normalized_form=form,
            multiplicity=multiplicity,
        )
        for name, role, form, multiplicity in entries
    ]
    validate_storage_signatures(catalog)
    return catalog


def mla_signature_audit(manifest: dict[str, object]) -> dict[str, object]:
    """Prove whether DeepSeek MLA fits the preregistered signature vocabulary."""
    spec = manifest["pes"]["PE_moe"]
    dimensions = spec["dimensiones_mla"]
    latent_dim = dimensions["kv_lora_rank"]
    symbols = spec["simbolos"]
    required = {
        "Qn": dimensions["qk_nope_head_dim"],
        "Qr": dimensions["qk_rope_head_dim"],
        "Dv": dimensions["v_head_dim"],
        "Rkv": latent_dim,
    }
    failures = [
        {
            "tensor_paths": ["q", "k", "v", "kv_latent"],
            "reason": f"missing or invalid MLA symbol {name}",
            "observed": {"expected": expected, "actual": symbols.get(name)},
        }
        for name, expected in required.items()
        if symbols.get(name) != expected
    ]
    return {
        "status": "HUELLA_NO_DERIVABLE" if failures else "MLA_VOCABULARY_SUFFICIENT",
        "e0_blocking": bool(failures),
        "e0_closed": False,
        "reference_sha": REFERENCE_SHA,
        "implementation": "torchtitan/models/deepseek_v3/model.py::Attention.forward",
        "failures": failures,
        "forbidden_shortcuts": [
            "do not map both unequal QK and V dimensions to Dh",
            "do not promote architectural literals 128/64/512 without normative symbols and provenance",
            "do not hide a sharded or split latent dimension as axis_opaque to claim completeness",
        ],
    }


def moe_mla_activation_catalog(manifest: dict[str, object]) -> list[StorageSemantic]:
    """Catalog MLA activations after the E0 vocabulary extension."""
    audit = mla_signature_audit(manifest)
    if audit["e0_blocking"]:
        raise ValueError("MLA activation catalog requires a sufficient vocabulary")
    spec = manifest["pes"]["PE_moe"]
    dtype_class = spec["dtype_classes"]["param"]
    common = {
        "pe": "PE_moe",
        "tp_placement": "NOT_COMPOSED",
        "tensor_class": "activation",
        "dtype_class": dtype_class,
        "multiplicity": "L",
        "provenance": (
            "torchtitan/models/deepseek_v3/model.py::Attention.forward",
            "torchtitan/models/deepseek_v3/__init__.py::_debugmodel",
            "candidate manifest dimensiones_mla and symbols Qn/Qr/Dv/Rkv",
            "preregistration §1.6.4.B MLA extension",
        ),
        "status": "SEMANTIC_TENSOR_SIGNATURE_CATALOGED",
    }
    prefix = (("batch", "B"), ("seq", "S"))
    entries = [
        ("layers.*.attention.wq.output", "ColLinear", prefix + (("output_feature", "H*(Qn+Qr)"),)),
        ("layers.*.attention.q", "Attention", prefix + (("head", "H"), ("head_dim", "Qn+Qr"))),
        ("layers.*.attention.q_nope", "Attention", prefix + (("head", "H"), ("head_dim", "Qn"))),
        ("layers.*.attention.q_pe", "Attention", prefix + (("head", "H"), ("head_dim", "Qr"))),
        ("layers.*.attention.wkv_a.fused_output", "TPReplicatedLinear", prefix + (("output_feature", "Rkv+Qr"),)),
        ("layers.*.attention.kv_latent", "Attention", prefix + (("kv_latent", "Rkv"),)),
        ("layers.*.attention.k_pe", "Attention", prefix + (("head_dim", "Qr"),)),
        ("layers.*.attention.wkv_b.fused_output", "ColLinear", prefix + (("output_feature", "H*(Qn+Dv)"),)),
        ("layers.*.attention.k_nope", "Attention", prefix + (("head", "H"), ("head_dim", "Qn"))),
        ("layers.*.attention.k", "Attention", prefix + (("head", "H"), ("head_dim", "Qn+Qr"))),
        ("layers.*.attention.v", "Attention", prefix + (("head", "H"), ("head_dim", "Dv"))),
        ("layers.*.attention.inner_output", "Attention", prefix + (("head", "H"), ("head_dim", "Dv"))),
        ("layers.*.attention.flattened_output", "Attention", prefix + (("attention_feature", "H*Dv"),)),
        ("layers.*.attention.wo.output", "RowLinear", prefix + (("model", "D"),)),
    ]
    catalog = [
        StorageSemantic(
            **common,
            logical_parameter=name,
            role=role,
            normalized_form=form,
        )
        for name, role, form in entries
    ]
    validate_storage_signatures(catalog)
    return catalog


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
    dense_parameters = dense_storage_catalog(manifest)
    moe_parameters = moe_storage_catalog(manifest)
    gradient_signatures = gradient_signature_catalog(
        manifest, dense_parameters + moe_parameters
    )
    dense_seven_field_templates = dense_framework_templates(
        manifest, dense_parameters, gradient_signatures
    )
    moe_seven_field_templates = moe_framework_templates(
        manifest, moe_parameters, gradient_signatures
    )
    optimizer_state_signatures = optimizer_state_signature_catalog(
        manifest, dense_parameters + moe_parameters
    )
    control_metadata_signatures = moe_control_metadata_catalog(manifest)
    routing_activation_signatures = moe_routing_activation_catalog(manifest)
    dense_activation_signatures = dense_nonattention_activation_catalog(manifest)
    dense_attention_signatures = dense_attention_activation_catalog(manifest)
    mla_audit = mla_signature_audit(manifest)
    mla_activation_signatures = moe_mla_activation_catalog(manifest)
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
            "Parameter, logical gradient, AdamW optimizer-state, standard MoE routing, and dense fused-QKV attention-boundary signatures are cataloged; attention score internals and MLA remain incomplete.",
            "Dense FSDP/HSDP and reviewed MoE-specific parameter/gradient signatures are composed into seven-field candidates; common PE_moe parameters and activation transitions remain incomplete.",
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
            asdict(item) for item in dense_parameters
        ],
        "moe_storage_semantics": [
            asdict(item) for item in moe_parameters
        ],
        "gradient_tensor_signatures": [
            asdict(item) for item in gradient_signatures
        ],
        "dense_framework_seven_field_templates": [
            asdict(item) for item in dense_seven_field_templates
        ],
        "moe_framework_seven_field_templates": [
            asdict(item) for item in moe_seven_field_templates
        ],
        "optimizer_state_tensor_signatures": [
            asdict(item) for item in optimizer_state_signatures
        ],
        "control_metadata_tensor_signatures": [
            asdict(item) for item in control_metadata_signatures
        ],
        "moe_routing_activation_tensor_signatures": [
            asdict(item) for item in routing_activation_signatures
        ],
        "dense_nonattention_activation_tensor_signatures": [
            asdict(item) for item in dense_activation_signatures
        ],
        "dense_attention_activation_tensor_signatures": [
            asdict(item) for item in dense_attention_signatures
        ],
        "mla_signature_audit": mla_audit,
        "moe_mla_activation_tensor_signatures": [
            asdict(item) for item in mla_activation_signatures
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
