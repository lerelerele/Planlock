#!/usr/bin/env python3
"""
anchor.py — Selects anchor controls from conjunto_B (par_id list).

Reads out/permutation.json to obtain conjunto_B (the full par_id list),
computes a deterministic hash for each par_id, selects the top
ceil(0.2 * N) entries with the smallest hash values, and writes:
  out/sealed/anchor_controls.json  — selected par_ids + ISO-8601 UTC timestamp

The script refuses to overwrite an existing output file.

Usage:
    python scripts/anchor.py --out-root <external-output>
"""

import json
import math
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from paths import external_out_root

def refuse_overwrite(path: Path) -> None:
    if path.exists():
        print(f"ERROR: {path} already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)


def anchor_hash(par_id: str) -> int:
    return int(sha256(f"20260815|anchor|{par_id}".encode()).hexdigest(), 16)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        required=True,
        help="External output directory; must be outside the Git checkout",
    )
    args = parser.parse_args()
    try:
        out_root = external_out_root(args.out_root)
    except ValueError as exc:
        parser.error(str(exc))
    permutation_path = out_root / "permutation.json"
    sealed_dir = out_root / "sealed"
    anchor_path = sealed_dir / "anchor_controls.json"

    refuse_overwrite(anchor_path)

    if not permutation_path.exists():
        print(f"ERROR: {permutation_path} not found.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(permutation_path.read_text(encoding="utf-8"))
    conjunto_b: list[str] = data["par_ids"]

    n_select = math.ceil(0.2 * len(conjunto_b))
    sorted_by_hash = sorted(conjunto_b, key=anchor_hash)
    controls = sorted_by_hash[:n_select]

    now_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    sealed_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": now_utc,
        "n_controls": n_select,
        "anchor_controls": controls,
    }
    anchor_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Selected {n_select} anchor controls from {len(conjunto_b)} entries.")
    print(f"Written: {anchor_path}")


if __name__ == "__main__":
    main()
