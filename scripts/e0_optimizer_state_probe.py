#!/usr/bin/env python3
"""Probe the reviewed fused AdamW state tensors without touching PR population."""

import argparse
import json
import sys
import uuid
from pathlib import Path


def external_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(checkout)
    except ValueError:
        return resolved
    raise ValueError(f"output must be outside the Planlock checkout: {resolved}")


def probe() -> dict[str, object]:
    import torch

    results = {}
    for label, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3, fused=True)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = optimizer.state[parameter]
        observed = {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(state.items())
        }
        if observed["step"] != {"dtype": "torch.float32", "shape": []}:
            raise ValueError(f"unexpected {label} AdamW step state: {observed['step']}")
        for name in ("exp_avg", "exp_avg_sq"):
            if observed[name] != {"dtype": str(dtype), "shape": [2]}:
                raise ValueError(f"unexpected {label} AdamW {name}: {observed[name]}")
        results[label] = observed
    return {
        "status": "REAL_PYTORCH_FUSED_ADAMW_STATE_PROBE",
        "population_touched": False,
        "e0_closed": False,
        "torch_version": torch.__version__,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = external_directory(args.output_root)
        run_dir = root / f"optimizer-state-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)
        report = probe()
        output = run_dir / "report.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (ImportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
