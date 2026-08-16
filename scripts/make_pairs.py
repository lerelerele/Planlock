#!/usr/bin/env python3
"""
make_pairs.py — Produce anonymised source-tree pairs for each par_id.

Reads:
  out/permutation.json           — seed + ordered par_id list
  out/sealed/parid_to_sha.json   — par_id -> qualifying commit SHA
  out/sealed/side_map.json       — par_id -> [side_before, side_after]
                                   (side_before = label for parent commit,
                                    side_after  = label for qualifying commit)

For each par_id writes two source trees:
  out/pairs/<par_id>/<side_before>/   — files at the parent commit
  out/pairs/<par_id>/<side_after>/    — files at the qualifying commit

Only files matching the §4.1 selector are included.  The directory
structure inside each side mirrors the original repository layout.
The blind reviewer cannot infer the direction from the tree layout alone
(no commit messages, PR numbers, dates, or diff annotations are written).

Usage:
    python scripts/make_pairs.py --repo <path_to_torchtitan> --out-root <external-output>
"""

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

from paths import external_out_root

# ── Paths ────────────────────────────────────────────────────────────────────

# ── §4.1 selector rules (literal, IMMUTABLE) ─────────────────────────────────
# Copied verbatim from population.py so that make_pairs.py is self-contained.

SELECTOR_RULES = [
    ("torchtitan/distributed/**/*.py", None),
    (
        "torchtitan/models/**/",
        {
            "model.py", "moe.py", "parallelize.py", "sharding.py",
            "expert_parallel.py", "token_dispatcher.py", "attention.py",
            "decoder.py", "feed_forward.py", "embedding.py", "linear.py",
            "nn_modules.py", "rmsnorm.py", "vision_encoder.py",
            "vision_encoder_sharding.py", "mtp.py", "dist_gemm.py", "layers.py",
        },
    ),
    ("torchtitan/models/common/config_utils.py", None),
    (
        "torchtitan/experiments/**/",
        {
            "model.py", "parallelize.py", "pipeline.py", "moe_replacement.py",
            "hf_sharding.py", "module_conversion.py",
        },
    ),
    ("torchtitan/experiments/**/ep_*.py", None),
    ("torchtitan/protocols/module.py", None),
    ("torchtitan/protocols/sharding.py", None),
    ("torchtitan/overrides/moe_token_dispatcher.py", None),
]


def glob_match(pattern: str, path: str) -> bool:
    """Match *path* against *pattern* supporting '**' (any depth of dirs)."""
    pattern_parts = pattern.split("/")
    path_parts    = path.split("/")

    def _match(pp: list, tp: list) -> bool:
        if not pp and not tp:
            return True
        if not pp:
            return False
        if pp[0] == "**":
            rest_pp = pp[1:]
            for i in range(len(tp) + 1):
                if _match(rest_pp, tp[i:]):
                    return True
            return False
        if not tp:
            return False
        if fnmatch.fnmatch(tp[0], pp[0]):
            return _match(pp[1:], tp[1:])
        return False

    return _match(pattern_parts, path_parts)


def path_matches_selector(path: str) -> bool:
    """Return True if *path* matches any §4.1 selector rule."""
    for pattern, allowed_names in SELECTOR_RULES:
        if allowed_names is None:
            if glob_match(pattern, path):
                return True
        else:
            parts    = path.split("/")
            filename = parts[-1]
            if filename not in allowed_names:
                continue
            dir_pattern = pattern.rstrip("/")
            dirpart     = "/".join(parts[:-1])
            if glob_match(dir_pattern, dirpart):
                return True
    return False


# ── Git helpers ───────────────────────────────────────────────────────────────

def git(args: list[str], cwd: str) -> str:
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def git_bytes(args: list[str], cwd: str) -> bytes:
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def list_selector_files(repo: str, sha: str) -> list[str]:
    """Return all selector-matching paths present in *sha* of *repo*."""
    raw = git(["ls-tree", "-r", "--name-only", sha], cwd=repo)
    return [p for p in raw.splitlines() if p and path_matches_selector(p)]


def get_parent_sha(repo: str, sha: str) -> str:
    return git(["rev-parse", f"{sha}^"], cwd=repo).strip()


def write_tree(repo: str, sha: str, dest: Path) -> int:
    """Extract selector files from *sha* into *dest*; return file count."""
    paths = list_selector_files(repo, sha)
    count = 0
    for rel_path in paths:
        content = git_bytes(["show", f"{sha}:{rel_path}"], cwd=repo)
        out_file = dest / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(content)
        count += 1
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to pytorch/torchtitan checkout",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="External output directory; must be outside the Git checkout",
    )
    args = parser.parse_args()
    repo = args.repo
    try:
        out_root = external_out_root(args.out_root)
    except ValueError as exc:
        parser.error(str(exc))
    permutation_path = out_root / "permutation.json"
    parid_to_sha_path = out_root / "sealed/parid_to_sha.json"
    side_map_path = out_root / "sealed/side_map.json"
    pairs_dir = out_root / "pairs"

    # Guard: refuse to overwrite the pairs directory
    if pairs_dir.exists():
        print(f"ERROR: {pairs_dir} already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)

    # Load inputs
    for p in (permutation_path, parid_to_sha_path, side_map_path):
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    permutation  = json.loads(permutation_path.read_text(encoding="utf-8"))
    parid_to_sha = json.loads(parid_to_sha_path.read_text(encoding="utf-8"))
    side_map     = json.loads(side_map_path.read_text(encoding="utf-8"))

    par_ids: list[str] = permutation["par_ids"]

    # Pre-flight: ensure every par_id is present in both sealed maps
    missing_sha  = [p for p in par_ids if p not in parid_to_sha]
    missing_side = [p for p in par_ids if p not in side_map]
    if missing_sha or missing_side:
        if missing_sha:
            print(f"ERROR: {len(missing_sha)} par_id(s) missing from {parid_to_sha_path}:", file=sys.stderr)
            for p in missing_sha:
                print(f"  {p}", file=sys.stderr)
        if missing_side:
            print(f"ERROR: {len(missing_side)} par_id(s) missing from {side_map_path}:", file=sys.stderr)
            for p in missing_side:
                print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    pairs_dir.mkdir(parents=True, exist_ok=True)

    total_pairs = len(par_ids)
    for i, par_id in enumerate(par_ids, 1):
        sha        = parid_to_sha[par_id]
        sides      = side_map[par_id]          # [side_before, side_after]
        side_before, side_after = sides[0], sides[1]

        parent_sha = get_parent_sha(repo, sha)

        pair_dir = pairs_dir / par_id

        before_dir = pair_dir / side_before
        after_dir  = pair_dir / side_after

        n_before = write_tree(repo, parent_sha, before_dir)
        n_after  = write_tree(repo, sha,        after_dir)

        print(f"[{i}/{total_pairs}] {par_id}  {side_before}={n_before} files  {side_after}={n_after} files")

    print(f"\nDone. {total_pairs} pairs written to {pairs_dir}/")


if __name__ == "__main__":
    main()
