#!/usr/bin/env python3
"""Validate and hash the candidate E0 PE manifest without importing TorchTitan."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REFERENCE_SHA = "9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"
EXPECTED_WORLD_SIZE = {"PE_dense": 32, "PE_moe": 8}


def canonical_bytes(manifest: dict[str, object]) -> bytes:
    return json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def validate_partition(parts: list[list[str]], layer_count: int = 6) -> None:
    flattened = [name for part in parts for name in part]
    expected = ["tok_embeddings", *(f"layers.{i}" for i in range(layer_count)), "norm", "lm_head"]
    if sorted(flattened) != sorted(expected):
        raise ValueError("pipeline partition must cover each debugmodel module exactly once")
    if any(not part for part in parts):
        raise ValueError("pipeline partition contains an empty model part")


def validate(repo: Path, manifest: dict[str, object]) -> dict[str, object]:
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REFERENCE_SHA or manifest.get("reference_sha") != actual:
        raise ValueError("reference SHA mismatch")
    if manifest.get("status") != "CANDIDATE_NOT_FROZEN":
        raise ValueError("this validator only accepts an unfrozen candidate manifest")

    results = {}
    pes = manifest.get("pes")
    if not isinstance(pes, dict) or set(pes) != set(EXPECTED_WORLD_SIZE):
        raise ValueError("manifest must contain exactly PE_dense and PE_moe")
    for pe_name, expected_world_size in EXPECTED_WORLD_SIZE.items():
        pe = pes[pe_name]
        module_path = repo / (str(pe["modulo_registro"]).replace(".", "/") + ".py")
        if not module_path.is_file():
            raise ValueError(f"missing config registry for {pe_name}: {module_path}")
        if pe["funcion_config"] not in function_names(module_path):
            raise ValueError(f"missing function_config for {pe_name}: {pe['funcion_config']}")
        overrides = pe["overrides"]
        architecture = pe["arquitectura"]
        if architecture["layers"] != 6:
            raise ValueError(f"{pe_name} candidate must use the six-layer debugmodel")
        if architecture["dense_layers"] + architecture["moe_layers"] != 6:
            raise ValueError(f"{pe_name} dense/MoE layer counts do not cover the model")
        if architecture["mtp_layers"] != 0:
            raise ValueError(f"{pe_name} candidate unexpectedly enables MTP")
        world_size = (
            overrides["data_parallel_replicate_degree"]
            * overrides["data_parallel_shard_degree"]
            * overrides["context_parallel_degree"]
            * overrides["tensor_parallel_degree"]
            * overrides["pipeline_parallel_degree"]
        )
        if world_size != expected_world_size:
            raise ValueError(f"{pe_name} world size: expected {expected_world_size}, got {world_size}")
        degrees = pe["grados"]
        expected_efsdp = (
            overrides["data_parallel_shard_degree"]
            * overrides["context_parallel_degree"]
            * overrides["tensor_parallel_degree"]
            // overrides["expert_parallel_degree"]
        )
        if pe_name == "PE_moe" and degrees["efsdp"] != expected_efsdp:
            raise ValueError(
                f"{pe_name} efsdp: expected {expected_efsdp}, got {degrees['efsdp']}"
            )
        degree_mapping = {
            "dp_r": overrides["data_parallel_replicate_degree"],
            "dp_s": overrides["data_parallel_shard_degree"],
            "cp": overrides["context_parallel_degree"],
            "tp": overrides["tensor_parallel_degree"],
            "ep": overrides["expert_parallel_degree"],
            "pp": overrides["pipeline_parallel_degree"],
        }
        for axis, configured in degree_mapping.items():
            declared = degrees[axis]
            normalized = None if configured == 1 else configured
            if declared != normalized:
                raise ValueError(
                    f"{pe_name} degree mismatch for {axis}: {declared} != {normalized}"
                )
        parts = overrides["module_fqns_per_model_part"]
        validate_partition(parts)
        schedule = overrides["pipeline_parallel_schedule"]
        expected_parts = overrides["pipeline_parallel_degree"] * (2 if schedule == "Interleaved1F1B" else 1)
        if len(parts) != expected_parts:
            raise ValueError(f"{pe_name} partition is incompatible with {schedule}")
        results[pe_name] = {
            "function_exists": True,
            "world_size": world_size,
            "pipeline_parts": len(parts),
            "partition_complete": True,
            "degrees_consistent": True,
            "architecture_consistent": True,
        }
    return {
        "status": "VALID_CANDIDATE_NOT_FROZEN",
        "reference_sha": actual,
        "manifest_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
        "pes": results,
        "e0_closed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("e0-manifest-candidate.json"))
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = validate(args.reference_repo.resolve(), manifest)
    except (OSError, subprocess.CalledProcessError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
