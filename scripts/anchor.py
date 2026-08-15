#!/usr/bin/env python3
"""
anchor.py — Selects anchor controls from conjunto_B (par_id list).

Reads out/permutation.json to obtain conjunto_B (the full par_id list),
computes a deterministic hash for each par_id, selects the top
ceil(0.2 * N) entries with the smallest hash values, and writes:
  out/sealed/anchor_controls.json  — selected par_ids + ISO-8601 UTC timestamp

The script refuses to overwrite an existing output file.

Usage:
    python scripts/anchor.py
"""

import json
import math
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

PERMUTATION_PATH = Path("out/permutation.json")
SEALED_DIR = Path("out/sealed")
ANCHOR_PATH = SEALED_DIR / "anchor_controls.json"


def refuse_overwrite(path: Path) -> None:
    if path.exists():
        print(f"ERROR: {path} already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)


def anchor_hash(par_id: str) -> int:
    return int(sha256(f"20260815|anchor|{par_id}".encode()).hexdigest(), 16)


def main() -> None:
    refuse_overwrite(ANCHOR_PATH)

    if not PERMUTATION_PATH.exists():
        print(f"ERROR: {PERMUTATION_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(PERMUTATION_PATH.read_text(encoding="utf-8"))
    conjunto_b: list[str] = data["par_ids"]

    n_select = math.ceil(0.2 * len(conjunto_b))
    sorted_by_hash = sorted(conjunto_b, key=anchor_hash)
    controls = sorted_by_hash[:n_select]

    now_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": now_utc,
        "n_controls": n_select,
        "anchor_controls": controls,
    }
    ANCHOR_PATH.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Selected {n_select} anchor controls from {len(conjunto_b)} entries.")
    print(f"Written: {ANCHOR_PATH}")


if __name__ == "__main__":
    main()
