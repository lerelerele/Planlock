#!/usr/bin/env python3
"""Trace real TorchTitan model-config sharding routes before device setup."""

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path

REFERENCE_SHA = "9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"
EXPECTED_COUNTS = {"PE_dense": 69, "PE_moe": 107}
OVERRIDE_FIELDS = (
    "data_parallel_replicate_degree",
    "data_parallel_shard_degree",
    "context_parallel_degree",
    "tensor_parallel_degree",
    "pipeline_parallel_degree",
    "expert_parallel_degree",
    "enable_sequence_parallel",
    "spmd_backend",
    "pipeline_parallel_schedule",
    "module_fqns_per_model_part",
)


def external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(checkout)
    except ValueError:
        return resolved
    raise ValueError(f"output must be outside the Planlock checkout: {resolved}")


def walk_configs(value: object, path: str = "model"):
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_configs(child, f"{path}.{index}")
    elif is_dataclass(value):
        if hasattr(value, "sharding_config"):
            yield path, value
        for field in fields(value):
            if field.name not in {"param_init", "sharding_config"}:
                yield from walk_configs(getattr(value, field.name), f"{path}.{field.name}")


def run(reference_repo: Path, manifest: dict[str, object]) -> dict[str, object]:
    actual = subprocess.check_output(
        ["git", "-C", str(reference_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REFERENCE_SHA or manifest.get("reference_sha") != actual:
        raise ValueError("reference SHA mismatch")
    sys.path.insert(0, str(reference_repo))
    from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    reports = {}
    for pe, factory in (
        ("PE_dense", llama3_debugmodel),
        ("PE_moe", deepseek_v3_debugmodel),
    ):
        trainer_config = factory()
        overrides = manifest["pes"][pe]["overrides"]
        for name in OVERRIDE_FIELDS:
            setattr(trainer_config.parallelism, name, overrides[name])
        model_config = trainer_config.model_spec.model
        model_config.update_from_config(config=trainer_config)
        routes = [
            {"path": path, "config_type": type(config).__qualname__}
            for path, config in walk_configs(model_config)
            if config.sharding_config is not None
        ]
        if len(routes) != EXPECTED_COUNTS[pe]:
            raise ValueError(
                f"{pe} sharding-config count drift: {len(routes)} != {EXPECTED_COUNTS[pe]}"
            )
        reports[pe] = {
            "configured_route_count": len(routes),
            "config_type_counts": dict(Counter(row["config_type"] for row in routes)),
            "routes": routes,
        }
    return {
        "status": "REAL_MODEL_CONFIG_ROUTES_CONFIRMED",
        "reference_sha": actual,
        "population_touched": False,
        "device_execution": False,
        "e0_closed": False,
        "pes": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = run(args.reference_repo.resolve(), manifest)
        output = external_output(args.output)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (ImportError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
