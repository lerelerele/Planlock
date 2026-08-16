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


def run(repo: Path, manifest: dict[str, object]) -> dict[str, object]:
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = run(args.reference_repo.resolve(), manifest)
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
