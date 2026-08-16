#!/usr/bin/env python3
"""Trace real CPU/Gloo HSDP collectives for the E0 framework hypothesis."""

import argparse
import json
import os
import socket
import sys
from pathlib import Path


def external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(checkout)
    except ValueError:
        return resolved
    raise ValueError(f"output must be outside the Planlock checkout: {resolved}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def worker(rank: int, world_size: int, port: int, result_dir: str) -> None:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.profiler import ProfilerActivity, profile

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh(
            "cpu",
            (2, 2),
            mesh_dim_names=("dp_replicate", "fsdp"),
        )
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        fully_shard(model[0], mesh=mesh)
        fully_shard(model[2], mesh=mesh)
        fully_shard(model, mesh=mesh)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        torch.manual_seed(1000 + rank)
        x = torch.randn(4, 8)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            loss = model(x).square().mean()
            loss.backward()
            optimizer.step()
        keys = sorted(
            {
                event.key
                for event in prof.key_averages()
                if any(
                    marker in event.key.lower()
                    for marker in ("allreduce", "all_reduce", "allgather", "all_gather", "reduce_scatter")
                )
            }
        )
        payload = {"rank": rank, "collective_operator_keys": keys}
        (Path(result_dir) / f"rank-{rank}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run(output: Path) -> dict[str, object]:
    try:
        import torch
        import torch.multiprocessing as mp
    except ImportError as exc:
        raise ValueError("PyTorch is required in the external E0 environment") from exc

    output.mkdir(parents=True, exist_ok=False)
    world_size = 4
    mp.spawn(worker, args=(world_size, free_port(), str(output)), nprocs=world_size, join=True)
    ranks = [
        json.loads((output / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    all_keys = sorted({key for rank in ranks for key in rank["collective_operator_keys"]})
    lower = " ".join(all_keys).lower()
    rank_lowers = [
        " ".join(rank["collective_operator_keys"]).lower() for rank in ranks
    ]
    report = {
        "status": "REAL_CPU_GLOO_HSDP_MECHANICS_ONLY",
        "backend": "gloo",
        "device": "cpu",
        "world_size": world_size,
        "mesh": {"dp_replicate": 2, "fsdp": 2},
        "torch_version": torch.__version__,
        "collective_operator_keys": all_keys,
        "all_reduce_observed": "allreduce" in lower or "all_reduce" in lower,
        "all_gather_observed": "allgather" in lower or "all_gather" in lower,
        "reduce_scatter_observed": "reduce_scatter" in lower,
        "all_ranks_observed_all_reduce": all(
            "allreduce" in keys or "all_reduce" in keys for keys in rank_lowers
        ),
        "all_ranks_observed_all_gather": all(
            "allgather" in keys or "all_gather" in keys for keys in rank_lowers
        ),
        "all_ranks_observed_reduce_scatter": all(
            "reduce_scatter" in keys for keys in rank_lowers
        ),
        "e0_closed": False,
        "population_touched": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(external_output(args.output))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
