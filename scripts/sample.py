#!/usr/bin/env python3
"""
sample.py — Deterministic sampling script for Planlock.

Reads out/population.json, shuffles qualifying commits with a fixed seed,
assigns opaque par_ids, and writes:
  out/permutation.json          — seed + ordered par_id list (no SHAs)
  out/sealed/parid_to_sha.json  — mapping par_id -> sha
  out/sealed/side_map.json      — per-pair randomized side order

Usage:
    python scripts/sample.py --out-root <external-output>
"""

import json
import random
import sys
from hashlib import sha256
from pathlib import Path

from paths import external_out_root

SEED = 20260815
def refuse_overwrite(path: Path) -> None:
    if path.exists():
        print(f"ERROR: {path} already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)


def make_par_id(sha: str) -> str:
    return sha256(f"20260815|parid|{sha}".encode()).hexdigest()[:12]


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
    parid_to_sha_path = sealed_dir / "parid_to_sha.json"
    side_map_path = sealed_dir / "side_map.json"
    population_path = out_root / "population.json"

    # Guard: refuse to overwrite any output file
    for p in (permutation_path, parid_to_sha_path, side_map_path):
        refuse_overwrite(p)

    # Load population
    if not population_path.exists():
        print(f"ERROR: {population_path} not found.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(population_path.read_text(encoding="utf-8"))
    qualifying = data["qualifying"]

    # 1. Sort chronologically ascending
    commits = sorted(qualifying, key=lambda c: c["date"])
    indices = list(range(len(commits)))

    # 2. Shuffle with fixed seed; record order
    rng = random.Random(SEED)
    rng.shuffle(indices)

    # 3. Assign par_ids and assert no collisions
    par_ids = []
    seen: set[str] = set()
    for idx in indices:
        sha = commits[idx]["sha"]
        pid = make_par_id(sha)
        assert pid not in seen, f"par_id collision on {pid}"
        seen.add(pid)
        par_ids.append((pid, sha))

    # 4. Randomize each pair's side order with the same rng; build side_map
    # A "pair" here is each entry in the ordered list.
    # Side order: each entry gets two sides [A, B] or [B, A] determined by rng.
    side_map: dict[str, list[str]] = {}
    for pid, _ in par_ids:
        sides = ["A", "B"]
        if rng.randint(0, 1):
            sides = ["B", "A"]
        side_map[pid] = sides

    # Create output dirs
    sealed_dir.mkdir(parents=True, exist_ok=True)
    permutation_path.parent.mkdir(parents=True, exist_ok=True)

    # 5a. Write out/permutation.json (no SHAs)
    permutation = {
        "seed": SEED,
        "par_ids": [pid for pid, _ in par_ids],
    }
    permutation_path.write_text(
        json.dumps(permutation, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    # 5b. Write out/sealed/parid_to_sha.json
    parid_to_sha = {pid: sha for pid, sha in par_ids}
    parid_to_sha_path.write_text(
        json.dumps(parid_to_sha, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    # 5c. Write out/sealed/side_map.json
    side_map_path.write_text(
        json.dumps(side_map, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Sampled {len(par_ids)} entries.")
    print(f"Written: {permutation_path}")
    print(f"Written: {parid_to_sha_path}")
    print(f"Written: {side_map_path}")


if __name__ == "__main__":
    main()
