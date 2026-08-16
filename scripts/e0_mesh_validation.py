#!/usr/bin/env python3
"""Real (CPU/gloo) distributed mesh validation for E0 (preregistro §8.3, point 2).

This is NOT NCCL/GPU validation. It spawns the literal declared world size of
PE_dense (32) or PE_moe (8) as local CPU processes using torch.distributed's
gloo backend, imports torchtitan's real, unmodified ParallelDims class from
the pinned reference HEAD, calls its real build_mesh(), and for every
resulting one-dimensional mesh with size > 1 runs a real all_reduce over it
to confirm the group actually carries communication — not just topology
metadata.

Requires an external Python environment with torch (CPU build is enough) and
spmd_types==0.2.3 installed; this repository does not vendor or install
those. See REPRODUCIBILITY.md.

Answers: "does the reference code, run for real, actually form the
communication groups the preregistro's §1.0 axis table declares" — not "does
it perform correctly at scale over real interconnects". The latter still
needs real multi-GPU/NCCL and is out of scope for this script.

Usage:
    python scripts/e0_mesh_validation.py \
        --pe-name PE_moe \
        --dp-replicate 1 --dp-shard 2 --cp 1 --tp 2 --pp 2 --ep 2 \
        --world-size 8 \
        --torchtitan-repo <path_to_pinned_torchtitan_checkout> \
        --out-root <external-output>
"""

import argparse
import json
import os
import sys
from pathlib import Path

from paths import external_out_root


def _worker(rank: int, world_size: int, degrees: dict, spmd_backend: str, out_dir: str, pe_name: str):
    import torch
    import torch.distributed as dist

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ["E0_MASTER_PORT"]

    sys.path.insert(0, os.environ["TORCHTITAN_REPO"])
    import torch.testing._internal.distributed.fake_pg  # noqa: F401 -- registers the "fake" c10d backend torchtitan uses for degree-1 axes
    import torchtitan.distributed.parallel_dims as pd

    pd.device_type = "cpu"  # torchtitan falls back to the literal string "cuda" when no accelerator is detected

    dims = pd.ParallelDims(
        dp_replicate=degrees["dp_replicate"],
        dp_shard=degrees["dp_shard"],
        cp=degrees["cp"],
        tp=degrees["tp"],
        pp=degrees["pp"],
        ep=degrees["ep"],
        world_size=world_size,
        spmd_backend=spmd_backend,
    )
    dims.build_mesh()

    active = dims.get_all_one_dimensional_meshes()
    axis_report = {}
    for name, mesh in active.items():
        pg = mesh.get_group()
        t = torch.ones(1)
        dist.all_reduce(t, group=pg)
        expected_sum = float(mesh.size())
        axis_report[name] = {
            "size": mesh.size(),
            "all_reduce_result": t.item(),
            "collective_confirmed": t.item() == expected_sum,
        }

    if rank == 0:
        result = {
            "pe_name": pe_name,
            "world_size": world_size,
            "degrees": degrees,
            "spmd_backend": spmd_backend,
            "torch_version": torch.__version__,
            "backend": "gloo (CPU) -- NOT NCCL/GPU",
            "active_one_dimensional_meshes": axis_report,
        }
        Path(out_dir, f"result_{pe_name}_{spmd_backend}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-name", required=True, choices=["PE_dense", "PE_moe"])
    parser.add_argument("--dp-replicate", type=int, required=True)
    parser.add_argument("--dp-shard", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--spmd-backend", default="default", choices=["default", "full_dtensor", "spmd_types"])
    parser.add_argument("--torchtitan-repo", required=True, type=Path)
    parser.add_argument("--out-root", required=True, help="Must be outside this Git checkout")
    parser.add_argument("--master-port", default="29511")
    args = parser.parse_args()

    import torch.multiprocessing as mp  # deferred: only the parent needs the launcher

    out_dir = external_out_root(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCHTITAN_REPO"] = str(args.torchtitan_repo.resolve())
    os.environ["E0_MASTER_PORT"] = args.master_port

    degrees = {
        "dp_replicate": args.dp_replicate,
        "dp_shard": args.dp_shard,
        "cp": args.cp,
        "tp": args.tp,
        "pp": args.pp,
        "ep": args.ep,
    }

    mp.spawn(
        _worker,
        args=(args.world_size, degrees, args.spmd_backend, str(out_dir), args.pe_name),
        nprocs=args.world_size,
        join=True,
    )

    result_path = out_dir / f"result_{args.pe_name}_{args.spmd_backend}.json"
    print(result_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
