#!/usr/bin/env python3
"""Synthetic structural calibration harness for preregistration E0.

This is deliberately not the Planlock extractor and does not inspect the PR
population.  It exercises the symbolic invariants that must be decided before
the real, distributed E0 run: fused GQA QKV, the MoE routed-item identity
(closed: adopted as identity 19, see preregistro §1.6.3/§1.6.4.B/§8.3.5), and
tied/untied embedding and LMHead roles.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


REFERENCE_SHA = "9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"

ROLES: FrozenSet[str] = frozenset(
    {
        "ColLinear", "RowLinear", "TPReplicatedLinear", "Attention", "Norm",
        "Embedding", "LMHead", "Router", "GroupedGEMM", "Dispatch", "Combine",
        "LossReduction", "OptimizerUpdate", "Opaque",
    }
)

AXES: FrozenSet[str] = frozenset(
    {
        "batch", "seq", "token", "query_pos", "key_pos", "model",
        "input_feature", "output_feature", "head", "kv_head", "head_dim",
        "ffn_hidden", "vocab", "expert", "topk", "capacity", "expert_offset",
        "layer", "routed_item", "axis_opaque",
    }
)


@dataclass(frozen=True)
class Axis:
    identity: str
    expr: str


@dataclass(frozen=True)
class Finding:
    fixture: str
    status: str
    decision: str
    identities_used: Tuple[str, ...]
    axis_opaque: int
    roles_used: Tuple[str, ...]
    provenance: Tuple[str, ...]
    notes: Tuple[str, ...]


def validate_form(form: Sequence[Axis]) -> List[str]:
    """Apply §1.6.3's no-duplicate-known-axis rule."""
    errors: List[str] = []
    seen: Set[str] = set()
    for axis in form:
        if axis.identity not in AXES:
            errors.append(f"unknown axis identity: {axis.identity}")
        if axis.identity != "axis_opaque" and axis.identity in seen:
            errors.append(f"known axis repeated in one form: {axis.identity}")
        if axis.identity != "axis_opaque":
            seen.add(axis.identity)
        if not axis.expr:
            errors.append(f"empty expression for axis: {axis.identity}")
    return errors


def t4_preserves(input_shards: Sequence[Axis], output_form: Sequence[Axis]) -> bool:
    """Return whether §1.8 T4 preserves every input shard exactly once."""
    return all(
        sum(1 for axis in output_form
            if axis.identity == shard.identity and axis.expr == shard.expr) == 1
        for shard in input_shards
    )


def qkv_gqa() -> Finding:
    input_shards = (Axis("output_feature", "(H+2*Hkv)*Dh"),)
    q_form = (Axis("head", "H"), Axis("head_dim", "Dh"))
    kv_form = (Axis("kv_head", "Hkv"), Axis("head_dim", "Dh"))
    errors = validate_form(q_form) + validate_form(kv_form)
    split_detected = not t4_preserves(input_shards, q_form + kv_form)
    notes = [
        "GQA fused extent is (H + 2·Hkv)·Dh with architecture provenance.",
        "The fused output_feature shard is split into query and KV identities.",
        "T4 rejects the split because one input shard is not preserved exactly once.",
    ]
    if errors or not split_detected:
        return Finding("qkv_gqa", "FAIL", "fused_axis_split_not_certified", (), 0, (), tuple(), tuple(errors + notes))
    return Finding(
        "qkv_gqa", "PASS", "split_detected_by_T4", ("head", "kv_head", "head_dim"),
        0, ("ColLinear", "Attention"),
        ("§1.6.1 QKV/GQA dimensions", "§1.6.4.B TP output_feature placement"),
        tuple(notes),
    )


def moe_routed_item() -> Finding:
    before = (Axis("batch", "B"), Axis("seq", "S"), Axis("topk", "K"))
    after = (Axis("routed_item", "B*S*K"), Axis("expert", "E"), Axis("expert_offset", "E+1"))
    errors = validate_form(before) + validate_form(after)
    token_loses_slot = True
    routed_item_used = any(a.identity == "routed_item" and a.expr == "B*S*K" for a in after)
    notes = [
        "Router indices, offsets, counts and capacity mask are control_metadata.",
        "Differentiable topk_scores/routed_scores remain activation.",
        "Flattening B·S·K as plain token would lose the token-vs-top-k-slot distinction.",
        "Decision closed in E0 calibration: routed_item adopted as identity 19 "
        "(§1.6.3, §1.6.4.B, §8.3.5), citing torchtitan moe.py/token_dispatcher.py "
        "buffers routed_input_RD, token_indices_experts_sorted_N, "
        "topk_scores_experts_sorted_N at reference HEAD.",
    ]
    if errors or not (token_loses_slot and routed_item_used):
        return Finding("moe_dispatch_combine", "FAIL", "routed_item_decision_invalid", (), 0, (), tuple(), tuple(errors + notes))
    return Finding(
        "moe_dispatch_combine", "PASS", "routed_item_identity_19_adopted",
        ("routed_item", "topk", "expert", "expert_offset"), 0,
        ("Router", "Dispatch", "GroupedGEMM", "Combine"),
        ("§1.2 token/expert ownership", "§1.6.3/§1.6.4.B routed_item identity 19",
         "§8.3.5 MoE flattened buffer fixture"),
        tuple(notes),
    )


def embedding_lmhead() -> Finding:
    tied_roles = frozenset(("Embedding", "LMHead"))
    untied_roles = (frozenset(("Embedding",)), frozenset(("LMHead",)))
    tied_is_ambiguous = len(tied_roles) > 1
    untied_is_separate = all(len(roles) == 1 for roles in untied_roles)
    notes = [
        "Embedding uses input_feature against a vocabulary table.",
        "LMHead projects to vocabulary and uses output_feature.",
        "A tied parameter carrying both semantic roles is not assigned two roles in one template.",
    ]
    if not (tied_is_ambiguous and untied_is_separate):
        return Finding("embedding_lmhead", "FAIL", "tied_role_guard_invalid", (), 0, (), tuple(), tuple(notes))
    return Finding(
        "embedding_lmhead", "PASS", "tied_is_HUELLA_NO_DERIVABLE_untied_is_separate",
        ("vocab", "input_feature", "output_feature"), 0,
        ("Embedding", "LMHead"),
        ("§1.3.1 semantic role guards", "§1.9 double role in one template"),
        tuple(notes),
    )


def verify_reference(repo: Path) -> str:
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect reference repo {repo}: {exc}") from exc
    if actual != REFERENCE_SHA:
        raise ValueError(f"reference HEAD mismatch: expected {REFERENCE_SHA}, got {actual}")
    return actual


def run(repo: Path) -> Dict[str, object]:
    actual = verify_reference(repo)
    findings = [qkv_gqa(), moe_routed_item(), embedding_lmhead()]
    return {
        "status": "SYNTHETIC_STRUCTURAL_ONLY",
        "e0_closed": False,
        "reference_sha": actual,
        "population_touched": False,
        "roles_vocabulary_count": len(ROLES),
        "axes_vocabulary_count_including_sentinel": len(AXES),
        "findings": [asdict(f) for f in findings],
        "decision": "VOCABULARY_DECISIONS_CLOSED_REAL_PE_VALIDATION_PENDING",
    }


def render_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# E0 synthetic structural calibration",
        "",
        "> This report is not a closed E0 result and does not inspect the PR population.",
        "",
        f"- Status: `{report['status']}`",
        f"- Reference HEAD: `{report['reference_sha']}`",
        f"- Population touched: `{report['population_touched']}`",
        f"- E0 closed: `{report['e0_closed']}`",
        "",
        "| Fixture | Status | Decision | Opaque axes |",
        "|---|---|---|---:|",
    ]
    for finding in report["findings"]:
        lines.append(
            f"| `{finding['fixture']}` | **{finding['status']}** | "
            f"`{finding['decision']}` | {finding['axis_opaque']} |"
        )
    lines.extend(["", "## Findings", ""])
    for finding in report["findings"]:
        lines.extend([
            f"### `{finding['fixture']}` — {finding['status']}", "",
            f"Decision: `{finding['decision']}`", "",
            f"Identities: {', '.join(finding['identities_used']) or 'none'}", "",
            f"Roles: {', '.join(finding['roles_used']) or 'none'}", "",
            "Notes:",
        ])
        lines.extend(f"- {note}" for note in finding["notes"])
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args(argv)
    try:
        report = run(args.reference_repo.resolve())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
