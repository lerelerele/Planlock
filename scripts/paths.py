#!/usr/bin/env python3
"""Safe output-path handling for the preregistration instrumentation."""

import subprocess
from pathlib import Path


def external_out_root(raw_path: str) -> Path:
    """Return an absolute output root that is outside the current Git checkout.

    The sealed maps contain the deblinding information, so putting them below
    the repository is unsafe even when the directory is ignored by Git.
    """
    root = Path(raw_path).expanduser().resolve()
    cwd = Path.cwd().resolve()

    try:
        git_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError):
        git_root = cwd

    for checkout in (cwd, git_root):
        try:
            root.relative_to(checkout)
        except ValueError:
            continue
        raise ValueError(
            f"Unsafe out-root {root}: it is inside checkout {checkout}. "
            "Use a sibling directory or another path outside the repository."
        )

    return root
