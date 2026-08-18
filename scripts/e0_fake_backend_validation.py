#!/usr/bin/env python3
"""Run the frozen E0 candidates through TorchTitan's CUDA fake backend."""

import argparse
import hashlib
import json
import os
import pprint
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REFERENCE_SHA = "9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"
PE_NAMES = ("PE_dense", "PE_moe")
PARALLELISM_FIELDS = (
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


def canonical_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(checkout)
    except ValueError:
        return resolved
    raise ValueError(f"output must be outside the Planlock checkout: {resolved}")


def git_state(reference_repo: Path) -> str:
    actual = subprocess.check_output(
        ["git", "-C", str(reference_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(reference_repo), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError("reference checkout must be clean")
    return actual


def world_size(pe: dict[str, object]) -> int:
    overrides = pe["overrides"]
    return (
        overrides["data_parallel_replicate_degree"]
        * overrides["data_parallel_shard_degree"]
        * overrides["context_parallel_degree"]
        * overrides["tensor_parallel_degree"]
        * overrides["pipeline_parallel_degree"]
    )


def runtime_module(pe_name: str, pe: dict[str, object]) -> str:
    overrides = pe["overrides"]
    assignments = "\n".join(
        f"    config.parallelism.{name} = {pprint.pformat(overrides[name], width=100)}"
        for name in PARALLELISM_FIELDS
    )
    return (
        f"from {pe['modulo_registro']} import {pe['funcion_config']} as _factory\n\n"
        f"def {pe_name}():\n"
        "    config = _factory()\n"
        f"{assignments}\n"
        "    return config\n"
    )


def cuda_probe(reference_repo: Path) -> dict[str, object]:
    source = (
        "import json, torch; "
        "ok=torch.cuda.is_available(); "
        "print(json.dumps({'torch': torch.__version__, 'cuda_available': ok, "
        "'device_count': torch.cuda.device_count(), "
        "'device_name': torch.cuda.get_device_name(0) if ok else None}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=reference_repo,
        text=True,
        capture_output=True,
        check=True,
    )
    probe = json.loads(result.stdout)
    if not probe["cuda_available"] or probe["device_count"] < 1:
        raise RuntimeError("CUDA device required for fake_backend validation")
    return probe


def run_candidate(
    reference_repo: Path,
    pe_name: str,
    pe: dict[str, object],
    timeout: int,
) -> dict[str, object]:
    size = world_size(pe)
    with tempfile.TemporaryDirectory(prefix="planlock-e0-") as directory:
        module_path = Path(directory) / "planlock_e0_runtime.py"
        module_path.write_text(runtime_module(pe_name, pe), encoding="utf-8")
        env = os.environ.copy()
        env.update({"NGPU": str(size), "LOCAL_RANK": "0", "LOG_RANK": "0"})
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (directory, str(reference_repo), env.get("PYTHONPATH")))
        )
        command = [
            sys.executable,
            "-m",
            "torchtitan.train",
            "--module",
            "planlock_e0_runtime",
            "--config",
            pe_name,
            "--comm.mode=fake_backend",
            "--training.disable_cuda_graphs",
            "--training.steps",
            "1",
        ]
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=reference_repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    combined = result.stdout + result.stderr
    completed = "Training completed" in combined
    mesh_built = "Building device mesh with parallelism:" in combined
    fake_pp_dynamic_shape_block = (
        "requires dynamic shape inference, which is not supported with a fake process group"
        in combined
    )
    if result.returncode != 0 and mesh_built and fake_pp_dynamic_shape_block:
        return {
            "status": "BLOCKED_FAKE_BACKEND_PIPELINE_DYNAMIC_SHAPES",
            "world_size": size,
            "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "mesh_built": True,
            "training_completed": False,
            "blocker": (
                "Torch distributed pipelining cannot infer dynamic stage shapes "
                "with a fake process group"
            ),
            "log_tail": combined[-4000:],
        }
    if result.returncode != 0 or not completed or not mesh_built:
        raise RuntimeError(
            f"{pe_name} fake_backend run failed: returncode={result.returncode}, "
            f"mesh_built={mesh_built}, training_completed={completed}\n"
            + combined[-4000:]
        )
    return {
        "status": "CONFIRMED_CUDA_FAKE_BACKEND",
        "world_size": size,
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "mesh_built": mesh_built,
        "training_completed": completed,
        "log_tail": combined[-4000:],
    }


def run(
    reference_repo: Path,
    manifest: dict[str, object],
    selected: tuple[str, ...] = PE_NAMES,
    timeout: int = 600,
) -> dict[str, object]:
    actual = git_state(reference_repo)
    if actual != REFERENCE_SHA or manifest.get("reference_sha") != actual:
        raise ValueError("reference SHA mismatch")
    if any(name not in PE_NAMES for name in selected):
        raise ValueError(f"unknown PE selection: {selected}")
    probe = cuda_probe(reference_repo)
    reports = {
        name: run_candidate(reference_repo, name, manifest["pes"][name], timeout)
        for name in selected
    }
    all_completed = all(report["training_completed"] for report in reports.values())
    return {
        "status": (
            "CONFIRMED_CUDA_FAKE_BACKEND"
            if all_completed
            else "BLOCKED_FAKE_BACKEND_PIPELINE_DYNAMIC_SHAPES"
        ),
        "reference_sha": actual,
        "manifest_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
        "runtime": probe,
        "physical_gpu_count_used": 1,
        "fake_world_sizes": {name: report["world_size"] for name, report in reports.items()},
        "nccl_collectives_validated": False,
        "physical_multi_gpu_validated": False,
        "training_completed": all_completed,
        "e0_closed": False,
        "pes": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("e0-manifest-candidate.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pe", choices=("all", *PE_NAMES), default="all")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selected = PE_NAMES if args.pe == "all" else (args.pe,)
        report = run(args.reference_repo.resolve(), manifest, selected, args.timeout)
        output = external_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
