#!/usr/bin/env python3
"""Validate frozen PE_dense on a uniform 32-GPU multi-node NCCL cluster."""

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
PE_NAME = "PE_dense"
DEFAULT_NNODES = 4
DEFAULT_LOCAL_WORLD_SIZE = 8
EXPECTED_WORLD_SIZE = 32
EXPECTED_ALL_REDUCE_SUM = 528
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
        raise TypeError("manifest does not contain PE_dense")
    if world_size(pe) != EXPECTED_WORLD_SIZE:
        raise ValueError("PE_dense must have world size 32")
    overrides = pe["overrides"]
    expected = {
        "data_parallel_replicate_degree": 2,
        "data_parallel_shard_degree": 2,
        "context_parallel_degree": 2,
        "tensor_parallel_degree": 2,
        "pipeline_parallel_degree": 2,
        "expert_parallel_degree": 1,
        "attn_backend": "flex",
        "pipeline_parallel_schedule": "Interleaved1F1B",
    }
    for key, value in expected.items():
        if overrides.get(key) != value:
            raise ValueError(f"PE_dense override mismatch: {key}")
    return pe


def validate_pe_moe_evidence(
    evidence: dict[str, object], manifest_digest: str
) -> dict[str, object]:
    claims = evidence.get("claims", {})
    digest = evidence.get("source_artifact", {}).get("sha256", "")
    valid = (
        evidence.get("status") == "CONFIRMED_PHYSICAL_NCCL_PE_MOE"
        and evidence.get("reference_sha") == REFERENCE_SHA
        and evidence.get("manifest_sha256") == manifest_digest
        and claims.get("pe_moe_physical_nccl_validated") is True
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    if not valid:
        raise ValueError("PE_moe physical NCCL evidence is not valid")
    return {
        "status": evidence["status"],
        "source_artifact_sha256": digest,
        "planlock_sha": evidence.get("planlock_sha"),
    }


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


def cuda_probe(reference_repo: Path, expected_device_count: int) -> dict[str, object]:
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
    if not probe["cuda_available"] or probe["device_count"] != expected_device_count:
        raise RuntimeError(
            f"exactly {expected_device_count} visible CUDA devices are required per node"
        )
    if not probe["nccl_available"]:
        raise RuntimeError("PyTorch NCCL support is required")
    return probe


def local_nvidia_topology(
    reference_repo: Path, expected_device_count: int
) -> dict[str, str]:
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
    if len(rows) != expected_device_count:
        raise RuntimeError(
            f"nvidia-smi did not report exactly {expected_device_count} local GPUs"
        )
    return {"inventory": query.stdout, "topology": topology.stdout}


def validate_rank_inventory(
    inventory: list[dict[str, object]], *, nnodes: int, nproc_per_node: int
) -> bool:
    if len(inventory) != EXPECTED_WORLD_SIZE:
        return False
    uuids = {item.get("device_uuid") for item in inventory}
    if None in uuids or len(uuids) != EXPECTED_WORLD_SIZE:
        return False
    host_counts: dict[object, int] = {}
    for item in inventory:
        hostname = item.get("hostname")
        if not hostname:
            return False
        host_counts[hostname] = host_counts.get(hostname, 0) + 1
    return len(host_counts) == nnodes and set(host_counts.values()) == {nproc_per_node}


def nccl_probe_source() -> str:
    return """import json
import os
import socket
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
props = torch.cuda.get_device_properties(local_rank)
identity = {
    "rank": rank,
    "local_rank": local_rank,
    "hostname": socket.gethostname(),
    "device_name": props.name,
    "device_uuid": str(props.uuid),
    "total_memory": props.total_memory,
}
identities = [None] * dist.get_world_size()
dist.all_gather_object(identities, identity)
value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
dist.all_reduce(value)
torch.cuda.synchronize(local_rank)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
payload = {
    "backend": dist.get_backend(),
    "world_size": dist.get_world_size(),
    "expected_sum": expected,
    "observed_sum": value.item(),
    "rank_inventory": identities,
    "passed": value.item() == expected and len(identities) == dist.get_world_size(),
}
if rank == 0:
    print("PLANLOCK_NCCL_PROBE=" + json.dumps(payload, sort_keys=True), flush=True)
dist.destroy_process_group()
if not payload["passed"]:
    raise SystemExit(2)
"""


def runtime_env(reference_repo: Path, directory: str, network_interface: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LOG_RANK": "0",
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "INFO"),
            "NCCL_DEBUG_SUBSYS": env.get("NCCL_DEBUG_SUBSYS", "INIT,COLL"),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if network_interface:
        env["NCCL_SOCKET_IFNAME"] = network_interface
        env["GLOO_SOCKET_IFNAME"] = network_interface
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (directory, str(reference_repo), env.get("PYTHONPATH")))
    )
    return env


def torchrun_command(
    module: str,
    *args: str,
    nnodes: int,
    node_rank: int,
    nproc_per_node: int,
    rdzv_endpoint: str,
    rdzv_id: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--nproc_per_node={nproc_per_node}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={rdzv_endpoint}",
        f"--rdzv_id={rdzv_id}",
        "--rdzv_conf=timeout=1800",
        "--local-ranks-filter=0",
        "--tee=3",
        "-m",
        module,
        *args,
    ]


def run_nccl_probe(
    reference_repo: Path,
    directory: str,
    timeout: int,
    *,
    nnodes: int,
    node_rank: int,
    nproc_per_node: int,
    rdzv_endpoint: str,
    network_interface: str | None,
) -> dict[str, object]:
    (Path(directory) / "planlock_dense_nccl_probe.py").write_text(
        nccl_probe_source(), encoding="utf-8"
    )
    env = runtime_env(reference_repo, directory, network_interface)
    started = time.monotonic()
    result = subprocess.run(
        torchrun_command(
            "planlock_dense_nccl_probe",
            nnodes=nnodes,
            node_rank=node_rank,
            nproc_per_node=nproc_per_node,
            rdzv_endpoint=rdzv_endpoint,
            rdzv_id="planlock-e0-dense-probe",
        ),
        cwd=reference_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    combined = result.stdout + result.stderr
    marker = next(
        (
            line.split("PLANLOCK_NCCL_PROBE=", 1)[1]
            for line in combined.splitlines()
            if "PLANLOCK_NCCL_PROBE=" in line
        ),
        None,
    )
    payload = json.loads(marker) if marker else None
    marker_ok = node_rank != 0 or (
        payload is not None
        and payload.get("passed") is True
        and payload.get("world_size") == EXPECTED_WORLD_SIZE
        and payload.get("observed_sum") == EXPECTED_ALL_REDUCE_SUM
        and validate_rank_inventory(
            payload.get("rank_inventory", []),
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
        )
    )
    passed = result.returncode == 0 and marker_ok
    return {
        "status": "CONFIRMED_NCCL_ALL_REDUCE" if passed else "FAILED_NCCL_PROBE",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "result": payload,
        "log_tail": combined[-6000:],
    }


def run_training(
    reference_repo: Path,
    directory: str,
    pe: dict[str, object],
    timeout: int,
    *,
    nnodes: int,
    node_rank: int,
    nproc_per_node: int,
    rdzv_endpoint: str,
    network_interface: str | None,
) -> dict[str, object]:
    (Path(directory) / "planlock_e0_dense_runtime.py").write_text(
        runtime_module(pe), encoding="utf-8"
    )
    env = runtime_env(reference_repo, directory, network_interface)
    command = torchrun_command(
        "torchtitan.train",
        "--module",
        "planlock_e0_dense_runtime",
        "--config",
        PE_NAME,
        "--training.disable_cuda_graphs",
        "--training.steps",
        "1",
        nnodes=nnodes,
        node_rank=node_rank,
        nproc_per_node=nproc_per_node,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_id="planlock-e0-dense-training",
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
    passed = result.returncode == 0 and (node_rank != 0 or (mesh_built and completed))
    return {
        "status": (
            "CONFIRMED_PHYSICAL_NCCL_PE_DENSE" if passed else "FAILED_PE_DENSE_TRAINING"
        ),
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "mesh_built": mesh_built if node_rank == 0 else None,
        "training_completed": completed if node_rank == 0 else None,
        "log_tail": combined[-10000:],
    }


def run(
    planlock_repo: Path,
    reference_repo: Path,
    manifest: dict[str, object],
    pe_moe_evidence: dict[str, object],
    *,
    nnodes: int,
    node_rank: int,
    nproc_per_node: int,
    rdzv_endpoint: str,
    network_interface: str | None = None,
    timeout: int = 1800,
) -> dict[str, object]:
    if nnodes < 2:
        raise ValueError("PE_dense physical validation requires at least two nodes")
    if nproc_per_node < 1 or nnodes * nproc_per_node != EXPECTED_WORLD_SIZE:
        raise ValueError("PE_dense requires a uniform multi-node layout of 32 GPUs")
    if not 0 <= node_rank < nnodes:
        raise ValueError("node rank is outside the cluster")
    if not rdzv_endpoint or rdzv_endpoint.startswith("localhost"):
        raise ValueError("multi-node rendezvous must use the head node address")
    reference_head = git_head(reference_repo, require_clean=True)
    if reference_head != REFERENCE_SHA:
        raise ValueError("reference SHA mismatch")
    pe = validate_manifest(manifest)
    manifest_digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    prior_moe = validate_pe_moe_evidence(pe_moe_evidence, manifest_digest)
    runtime = cuda_probe(reference_repo, nproc_per_node)
    topology = local_nvidia_topology(reference_repo, nproc_per_node)
    with tempfile.TemporaryDirectory(prefix="planlock-e0-dense-") as directory:
        nccl = run_nccl_probe(
            reference_repo,
            directory,
            timeout,
            nnodes=nnodes,
            node_rank=node_rank,
            nproc_per_node=nproc_per_node,
            rdzv_endpoint=rdzv_endpoint,
            network_interface=network_interface,
        )
        training = (
            run_training(
                reference_repo,
                directory,
                pe,
                timeout,
                nnodes=nnodes,
                node_rank=node_rank,
                nproc_per_node=nproc_per_node,
                rdzv_endpoint=rdzv_endpoint,
                network_interface=network_interface,
            )
            if nccl["status"] == "CONFIRMED_NCCL_ALL_REDUCE"
            else {
                "status": "SKIPPED_AFTER_NCCL_FAILURE",
                "mesh_built": False,
                "training_completed": False,
            }
        )
    dense_confirmed = training["status"] == "CONFIRMED_PHYSICAL_NCCL_PE_DENSE"
    e0_closed = dense_confirmed and prior_moe["status"] == "CONFIRMED_PHYSICAL_NCCL_PE_MOE"
    return {
        "status": (
            "CONFIRMED_PHYSICAL_NCCL_PE_DENSE"
            if dense_confirmed
            else "FAILED_PHYSICAL_NCCL_PE_DENSE"
        ),
        "planlock_sha": git_head(planlock_repo, require_clean=True),
        "reference_sha": reference_head,
        "manifest_sha256": manifest_digest,
        "cluster": {
            "nnodes": nnodes,
            "nproc_per_node": nproc_per_node,
            "world_size": nnodes * nproc_per_node,
            "node_rank": node_rank,
            "rdzv_endpoint": rdzv_endpoint,
            "network_interface": network_interface,
        },
        "runtime": runtime,
        "local_nvidia": topology,
        "physical_gpu_count_used": EXPECTED_WORLD_SIZE,
        "prior_pe_moe_evidence": prior_moe,
        "nccl_probe": nccl,
        "training": training,
        "claims": {
            "pe_moe_physical_nccl_validated": True,
            "pe_dense_physical_nccl_validated": dense_confirmed,
            "e0_closed": e0_closed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("e0-manifest-candidate.json"))
    parser.add_argument("--pe-moe-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nnodes", type=int, default=DEFAULT_NNODES)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument(
        "--nproc-per-node", type=int, default=DEFAULT_LOCAL_WORLD_SIZE
    )
    parser.add_argument("--rdzv-endpoint", required=True)
    parser.add_argument("--network-interface")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    planlock_repo = Path(__file__).resolve().parents[1]
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        pe_moe_evidence = json.loads(args.pe_moe_evidence.read_text(encoding="utf-8"))
        report = run(
            planlock_repo,
            args.reference_repo.expanduser().resolve(),
            manifest,
            pe_moe_evidence,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            nproc_per_node=args.nproc_per_node,
            rdzv_endpoint=args.rdzv_endpoint,
            network_interface=args.network_interface,
            timeout=args.timeout,
        )
        if args.node_rank == 0:
            output = external_output(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(
                json.dumps(
                    {
                        "node_rank": args.node_rank,
                        "status": report["status"],
                        "report_written": False,
                    }
                )
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
    return 0 if report["status"] == "CONFIRMED_PHYSICAL_NCCL_PE_DENSE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
