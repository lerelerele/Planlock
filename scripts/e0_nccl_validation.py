#!/usr/bin/env python3
"""Validate the frozen PE_moe on eight physical CUDA devices with NCCL."""

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
PE_NAME = "PE_moe"
EXPECTED_WORLD_SIZE = 8
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


def git_head(repo: Path, *, require_clean: bool = False) -> str:
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if require_clean:
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).strip()
        if dirty:
            raise ValueError(f"checkout must be clean: {repo}")
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


def validate_manifest(manifest: dict[str, object]) -> dict[str, object]:
    if manifest.get("reference_sha") != REFERENCE_SHA:
        raise ValueError("manifest reference SHA mismatch")
    pe = manifest.get("pes", {}).get(PE_NAME)
    if not isinstance(pe, dict):
        raise TypeError("manifest does not contain PE_moe")
    if world_size(pe) != EXPECTED_WORLD_SIZE:
        raise ValueError("PE_moe must have world size 8")
    overrides = pe["overrides"]
    if overrides.get("moe_comm_backend") != "standard":
        raise ValueError("PE_moe must use the standard dispatcher")
    if overrides.get("attn_backend") != "flex":
        raise ValueError("PE_moe must use flex attention")
    return pe


def runtime_module(pe: dict[str, object]) -> str:
    overrides = pe["overrides"]
    assignments = "\n".join(
        f"    config.parallelism.{name} = {pprint.pformat(overrides[name], width=100)}"
        for name in PARALLELISM_FIELDS
    )
    return (
        f"from {pe['modulo_registro']} import {pe['funcion_config']} as _factory\n\n"
        f"def {PE_NAME}():\n"
        "    config = _factory()\n"
        f"{assignments}\n"
        "    return config\n"
    )


def cuda_probe(reference_repo: Path) -> dict[str, object]:
    source = """
import json
import torch
ok = torch.cuda.is_available()
nccl = torch.cuda.nccl.version() if ok and torch.distributed.is_nccl_available() else None
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": ok,
    "device_count": torch.cuda.device_count(),
    "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "nccl_available": torch.distributed.is_nccl_available(),
    "nccl_version": nccl,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=reference_repo,
        text=True,
        capture_output=True,
        check=True,
    )
    probe = json.loads(result.stdout)
    if not probe["cuda_available"] or probe["device_count"] != EXPECTED_WORLD_SIZE:
        raise RuntimeError("exactly eight visible CUDA devices are required")
    if not probe["nccl_available"]:
        raise RuntimeError("PyTorch NCCL support is required")
    return probe


def nvidia_topology(reference_repo: Path) -> dict[str, str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        cwd=reference_repo,
        text=True,
        capture_output=True,
        check=True,
    )
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        cwd=reference_repo,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [line for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != EXPECTED_WORLD_SIZE:
        raise RuntimeError("nvidia-smi did not report exactly eight GPUs")
    return {"inventory": query.stdout, "topology": topology.stdout}


def nccl_probe_source() -> str:
    return """import json
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
dist.all_reduce(value)
torch.cuda.synchronize(local_rank)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
payload = {
    "backend": dist.get_backend(),
    "world_size": dist.get_world_size(),
    "expected_sum": expected,
    "observed_sum": value.item(),
    "passed": value.item() == expected,
}
if rank == 0:
    print("PLANLOCK_NCCL_PROBE=" + json.dumps(payload, sort_keys=True), flush=True)
dist.destroy_process_group()
if not payload["passed"]:
    raise SystemExit(2)
"""


def torchrun_command(module: str, *args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={EXPECTED_WORLD_SIZE}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        "--local-ranks-filter=0",
        "--tee=3",
        "-m",
        module,
        *args,
    ]


def run_nccl_probe(reference_repo: Path, directory: str, timeout: int) -> dict[str, object]:
    probe_path = Path(directory) / "planlock_nccl_probe.py"
    probe_path.write_text(nccl_probe_source(), encoding="utf-8")
    env = runtime_env(reference_repo, directory)
    started = time.monotonic()
    result = subprocess.run(
        torchrun_command("planlock_nccl_probe"),
        cwd=reference_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    combined = result.stdout + result.stderr
    marker = next(
        (line.split("PLANLOCK_NCCL_PROBE=", 1)[1] for line in combined.splitlines() if "PLANLOCK_NCCL_PROBE=" in line),
        None,
    )
    payload = json.loads(marker) if marker else None
    passed = result.returncode == 0 and payload is not None and payload.get("passed") is True
    return {
        "status": "CONFIRMED_NCCL_ALL_REDUCE" if passed else "FAILED_NCCL_PROBE",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "result": payload,
        "log_tail": combined[-4000:],
    }


def runtime_env(reference_repo: Path, directory: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LOG_RANK": "0",
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "INFO"),
            "NCCL_DEBUG_SUBSYS": env.get("NCCL_DEBUG_SUBSYS", "INIT,COLL"),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (directory, str(reference_repo), env.get("PYTHONPATH")))
    )
    return env


def run_training(
    reference_repo: Path, directory: str, pe: dict[str, object], timeout: int
) -> dict[str, object]:
    module_path = Path(directory) / "planlock_e0_runtime.py"
    module_path.write_text(runtime_module(pe), encoding="utf-8")
    env = runtime_env(reference_repo, directory)
    command = torchrun_command(
        "torchtitan.train",
        "--module",
        "planlock_e0_runtime",
        "--config",
        PE_NAME,
        "--training.disable_cuda_graphs",
        "--training.steps",
        "1",
    )
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
    mesh_built = "Building device mesh with parallelism:" in combined
    completed = "Training completed" in combined
    passed = result.returncode == 0 and mesh_built and completed
    return {
        "status": "CONFIRMED_PHYSICAL_NCCL_PE_MOE" if passed else "FAILED_PE_MOE_TRAINING",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "mesh_built": mesh_built,
        "training_completed": completed,
        "log_tail": combined[-8000:],
    }


def run(
    planlock_repo: Path,
    reference_repo: Path,
    manifest: dict[str, object],
    timeout: int = 1200,
) -> dict[str, object]:
    reference_head = git_head(reference_repo, require_clean=True)
    if reference_head != REFERENCE_SHA:
        raise ValueError("reference SHA mismatch")
    pe = validate_manifest(manifest)
    runtime = cuda_probe(reference_repo)
    topology = nvidia_topology(reference_repo)
    with tempfile.TemporaryDirectory(prefix="planlock-e0-nccl-") as directory:
        nccl = run_nccl_probe(reference_repo, directory, timeout)
        training = (
            run_training(reference_repo, directory, pe, timeout)
            if nccl["status"] == "CONFIRMED_NCCL_ALL_REDUCE"
            else {
                "status": "SKIPPED_AFTER_NCCL_FAILURE",
                "mesh_built": False,
                "training_completed": False,
            }
        )
    confirmed = training["status"] == "CONFIRMED_PHYSICAL_NCCL_PE_MOE"
    return {
        "status": (
            "CONFIRMED_PHYSICAL_NCCL_PE_MOE"
            if confirmed
            else "FAILED_PHYSICAL_NCCL_PE_MOE"
        ),
        "planlock_sha": git_head(planlock_repo, require_clean=True),
        "reference_sha": reference_head,
        "manifest_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
        "runtime": runtime,
        "nvidia": topology,
        "physical_gpu_count_used": EXPECTED_WORLD_SIZE,
        "nccl_probe": nccl,
        "training": training,
        "claims": {
            "pe_moe_physical_nccl_validated": confirmed,
            "pe_dense_physical_nccl_validated": False,
            "e0_closed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("e0-manifest-candidate.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args(argv)
    planlock_repo = Path(__file__).resolve().parents[1]
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = run(
            planlock_repo,
            args.reference_repo.expanduser().resolve(),
            manifest,
            args.timeout,
        )
        output = external_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "CONFIRMED_PHYSICAL_NCCL_PE_MOE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
