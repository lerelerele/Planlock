#!/usr/bin/env python3
"""
sample.py — Deterministic sampling script for Planlock.

Reads out/population.json, shuffles qualifying commits with a fixed seed,
assigns opaque par_ids, and writes:
  out/permutation.json          — seed + ordered par_id list (no SHAs)
  out/sealed/parid_to_sha.json  — mapping par_id -> sha
  out/sealed/side_map.json      — per-pair randomized side order

Usage:
    python scripts/sample.py
"""

import json
import random
import sys
from hashlib import sha256
from pathlib import Path

SEED = 20260815
POPULATION_PATH = Path("out/population.json")
PERMUTATION_PATH = Path("out/permutation.json")
SEALED_DIR = Path("out/sealed")
PARID_TO_SHA_PATH = SEALED_DIR / "parid_to_sha.json"
SIDE_MAP_PATH = SEALED_DIR / "side_map.json"


def refuse_overwrite(path: Path) -> None:
    if path.exists():
        print(f"ERROR: {path} already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)


def make_par_id(sha: str) -> str:
    return sha256(f"20260815|parid|{sha}".encode()).hexdigest()[:12]


def main() -> None:
    # Guard: refuse to overwrite any output file
    for p in (PERMUTATION_PATH, PARID_TO_SHA_PATH, SIDE_MAP_PATH):
        refuse_overwrite(p)

    # Load population
    if not POPULATION_PATH.exists():
        print(f"ERROR: {POPULATION_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(POPULATION_PATH.read_text(encoding="utf-8"))
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
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    PERMUTATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 5a. Write out/permutation.json (no SHAs)
    permutation = {
        "seed": SEED,
        "par_ids": [pid for pid, _ in par_ids],
    }
    PERMUTATION_PATH.write_text(
        json.dumps(permutation, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    # 5b. Write out/sealed/parid_to_sha.json
    parid_to_sha = {pid: sha for pid, sha in par_ids}
    PARID_TO_SHA_PATH.write_text(
        json.dumps(parid_to_sha, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    # 5c. Write out/sealed/side_map.json
    SIDE_MAP_PATH.write_text(
        json.dumps(side_map, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Sampled {len(par_ids)} entries.")
    print(f"Written: {PERMUTATION_PATH}")
    print(f"Written: {PARID_TO_SHA_PATH}")
    print(f"Written: {SIDE_MAP_PATH}")


if __name__ == "__main__":
    main()
