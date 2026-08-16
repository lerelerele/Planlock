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
from dataclasses import asdict, dataclass
from pathlib import Path

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

# These routes are the current prototype hypothesis, not a frozen PE manifest.
# The preregistration defines the manifest fields but does not yet assign their
# definitive values (§1.0).  Consequently this script must not claim coverage.
ROUTE_HYPOTHESES = {
    "PE_dense": {
        "model_family": "llama3",
        "config_registry_module": "torchtitan.models.llama3.config_registry",
        "function_config": None,
        "overrides": None,
        "manifest_hash": None,
    },
    "PE_moe": {
        "model_family": "deepseek_v3",
        "config_registry_module": "torchtitan.models.deepseek_v3.config_registry",
        "function_config": None,
        "overrides": None,
        "manifest_hash": None,
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
        if item.transition is None:
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


def run(repo: Path) -> dict[str, object]:
    actual = verify_reference(repo)
    candidates = [
        candidate
        for pe, files in PE_FILES.items()
        for relative in files
        for candidate in inventory_file(repo, pe, relative)
    ]
    by_pe = {
        pe: {
            "files_scanned": len(files),
            "candidate_count": sum(item.pe == pe for item in candidates),
            "rule_classified_count": sum(
                item.pe == pe and item.status == "RULE_CLASSIFIED_PROTOTYPE"
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
        "coverage_claim": "NONE_UNTIL_MANUAL_AND_RUNTIME_CROSSCHECK",
        "route_hypotheses": ROUTE_HYPOTHESES,
        "blocking_gaps": [
            "Definitive PE manifest does not yet fix function_config, overrides, or manifest_hash.",
            "Placements and tensor signatures are not yet normalized into all seven template fields.",
            "Static candidates have not yet been cross-checked against runtime execution paths.",
        ],
        "pe_summary": by_pe,
        "logical_transition_candidates": logical_transitions(candidates),
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.reference_repo.resolve())
        output = external_output(args.output) if args.output else None
    except (SyntaxError, UnicodeError, ValueError) as exc:
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
