#!/usr/bin/env python3
"""
population.py — Deterministic population script for pytorch/torchtitan.

Enumerates first-parent commits of main in [2026-05-17T17:00:00Z, 2026-08-15T17:00:00Z),
identifies qualifying commits (those touching §4.1 selector paths), excludes
docs/tests/CI/logging/dependency-only commits, and writes out/population.json.

Usage:
    python scripts/population.py --repo <path_to_torchtitan> --out-root <external-output>
    python scripts/population.py --repo <path_to_torchtitan> --out-root <external-output> --verify
"""

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from paths import external_out_root

# ── Constants ───────────────────────────────────────────────────────────────

WINDOW_START = "2026-05-17T17:00:00Z"
WINDOW_END   = "2026-08-15T17:00:00Z"
REF_COMMIT   = "9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"
REPO_DEFAULT = "."

# §4.1 selector — literal, IMMUTABLE
# Each entry is an (include_glob, allowed_filenames_set_or_None) tuple.
# None means "all filenames under that glob prefix".
SELECTOR_RULES = [
    # torchtitan/distributed/**/*.py  — any .py under distributed
    ("torchtitan/distributed/**/*.py", None),

    # torchtitan/models/**/<named files>
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

    # torchtitan/models/common/config_utils.py — exact path
    ("torchtitan/models/common/config_utils.py", None),

    # torchtitan/experiments/**/<named files>
    (
        "torchtitan/experiments/**/",
        {
            "model.py", "parallelize.py", "pipeline.py", "moe_replacement.py",
            "hf_sharding.py", "module_conversion.py",
        },
    ),

    # torchtitan/experiments/**/ep_*.py  — handled separately below
    ("torchtitan/experiments/**/ep_*.py", None),

    # exact paths
    ("torchtitan/protocols/module.py", None),
    ("torchtitan/protocols/sharding.py", None),
    ("torchtitan/overrides/moe_token_dispatcher.py", None),
]

# Patterns for docs/tests/CI/logging/dependency-only exclusions.
EXCLUDE_PATTERNS = [
    # docs
    "*.md", "*.rst", "*.txt", "docs/**",
    # tests
    "test/**", "tests/**", "**/test_*.py", "**/*_test.py",
    # CI / GitHub Actions / configs
    ".github/**", "*.yaml", "*.yml", "*.toml", "*.cfg", "*.ini", "*.json",
    # logging  (standalone logging helpers, not matching selector)
    "**/logging_utils.py", "**/logger.py",
    # dependency / packaging
    "requirements*.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "Makefile", "Dockerfile*", "*.sh",
    # misc non-code
    "LICENSE", "NOTICE", "CONTRIBUTING*", "CHANGELOG*",
    "*.png", "*.jpg", "*.gif", "*.svg",
]


def git(args, cwd):
    """Run a git command in *cwd* and return stdout as a string."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def glob_match(pattern: str, path: str) -> bool:
    """Match *path* against *pattern* supporting '**' (any depth of dirs).

    Uses a recursive approach: split on '**' segments and verify each
    fixed segment with fnmatch.
    """
    pattern_parts = pattern.split("/")
    path_parts    = path.split("/")

    def _match(pp: list, tp: list) -> bool:
        # Both exhausted — success
        if not pp and not tp:
            return True
        # Pattern exhausted but path still has parts
        if not pp:
            return False
        # Pattern has '**' at current position
        if pp[0] == "**":
            rest_pp = pp[1:]
            # '**' can match zero or more path components
            for i in range(len(tp) + 1):
                if _match(rest_pp, tp[i:]):
                    return True
            return False
        # Path exhausted but pattern still has non-'**' parts
        if not tp:
            return False
        # Normal segment match via fnmatch
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
            # Pattern is a directory prefix ending with '**/' — check filename
            parts = path.split("/")
            filename = parts[-1]
            if filename not in allowed_names:
                continue
            # The pattern "torchtitan/models/**/" should match the directory part
            dir_pattern = pattern.rstrip("/")  # strip trailing slash
            dirpart = "/".join(parts[:-1])
            if glob_match(dir_pattern, dirpart):
                return True
    return False


def path_is_excluded_only(path: str) -> bool:
    """Return True if *path* looks like docs/tests/CI/logging/dependency."""
    for pat in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(path, pat):
            return True
        # Also match basename alone
        basename = Path(path).name
        if fnmatch.fnmatch(basename, pat):
            return True
    return False


def get_changed_files(repo: str, sha: str) -> list:
    """Return list of files changed in *sha* relative to its first parent."""
    raw = git(
        ["diff-tree", "--no-commit-id", "-r", "--name-only", "-m", "--first-parent", sha],
        cwd=repo,
    )
    return [line for line in raw.splitlines() if line]


def parse_iso(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def iso_week(dt: datetime) -> str:
    return dt.strftime("%Y-W%V")


def run(repo: str, out_path: Path):
    # Resolve window bounds
    window_start = parse_iso(WINDOW_START)
    window_end   = parse_iso(WINDOW_END)

    # List first-parent commits of the reference HEAD within the window
    raw_log = git(
        [
            "log",
            REF_COMMIT,
            "--first-parent",
            "--format=%H %aI %s",
            f"--after={WINDOW_START}",
            f"--before={WINDOW_END}",
        ],
        cwd=repo,
    )

    all_commits = []
    pr_pattern  = re.compile(r"\(#(\d+)\)\s*$")

    for line in raw_log.splitlines():
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        sha, iso_date, subject = parts[0], parts[1], parts[2]

        # Parse and double-check window (git --after/--before can be off by one)
        dt = parse_iso(iso_date)
        if not (window_start <= dt < window_end):
            continue

        m = pr_pattern.search(subject)
        if not m:
            continue
        pr_number = int(m.group(1))

        all_commits.append(
            {
                "sha": sha,
                "pr_number": pr_number,
                "date": iso_date,
                "subject": subject,
            }
        )

    total_count = len(all_commits)

    qualifying = []
    for commit in all_commits:
        files = get_changed_files(repo, commit["sha"])
        if not files:
            continue

        # Check exclusion: skip commits where ALL files are excluded
        non_excluded = [f for f in files if not path_is_excluded_only(f)]
        if not non_excluded:
            continue

        # Check selector match
        matched = [f for f in files if path_matches_selector(f)]
        if matched:
            qualifying.append(
                {
                    "sha": commit["sha"],
                    "pr_number": commit["pr_number"],
                    "date": commit["date"],
                    "matched_paths": sorted(matched),
                }
            )

    qualifying_count = len(qualifying)

    # Compute per-week rates
    total_weeks = (window_end - window_start).days / 7.0
    total_rate  = round(total_count / total_weeks, 4)
    qual_rate   = round(qualifying_count / total_weeks, 4)

    # Build output
    output = {
        "window_start": WINDOW_START,
        "window_end":   WINDOW_END,
        "ref_commit":   REF_COMMIT,
        "total_count":  total_count,
        "total_rate_per_week": total_rate,
        "qualifying_count": qualifying_count,
        "qualifying_rate_per_week": qual_rate,
        "qualifying": qualifying,
    }

    # Write JSON (canonical: sorted keys, 2-space indent, newline at end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = (
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    out_path.write_bytes(json_bytes)

    # Print summary
    print(f"Window:          {WINDOW_START} – {WINDOW_END}")
    print(f"Total commits:   {total_count}  ({total_rate:.2f}/week)")
    print(f"Qualifying:      {qualifying_count}  ({qual_rate:.2f}/week)")
    print(f"Output:          {out_path}")

    # Per-week breakdown
    weeks: dict = {}
    for c in qualifying:
        w = iso_week(parse_iso(c["date"]))
        weeks[w] = weeks.get(w, 0) + 1
    if weeks:
        print("\nQualifying commits per ISO week:")
        for w in sorted(weeks):
            print(f"  {w}: {weeks[w]}")

    return json_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=REPO_DEFAULT,
        help="Path to pytorch/torchtitan checkout (default: current directory)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (must be outside the Git checkout; defaults to <out-root>/population.json)",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="External output directory; must be outside the Git checkout",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-run and assert byte-identical output against existing out/population.json",
    )
    args = parser.parse_args()

    try:
        out_root = external_out_root(args.out_root)
    except ValueError as exc:
        parser.error(str(exc))
    out_path = Path(args.out).expanduser().resolve() if args.out else out_root / "population.json"
    try:
        out_path.relative_to(out_root)
    except ValueError:
        parser.error(f"--out must be inside --out-root ({out_root})")

    if args.verify:
        if not out_path.exists():
            print(f"ERROR: {out_path} does not exist; cannot verify.", file=sys.stderr)
            sys.exit(1)
        existing = out_path.read_bytes()
        # Write to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp_path = Path(tmp.name)
        try:
            new_bytes = run(args.repo, tmp_path)
            if new_bytes == existing:
                print("\n✓ VERIFY PASSED — output is byte-identical.")
            else:
                print("\n✗ VERIFY FAILED — output differs.", file=sys.stderr)
                sys.exit(2)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        run(args.repo, out_path)


if __name__ == "__main__":
    main()
