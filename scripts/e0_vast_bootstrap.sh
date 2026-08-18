#!/usr/bin/env bash
set -euo pipefail

REFERENCE_SHA="9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"
PLANLOCK_REPO="${PLANLOCK_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME/planlock-e0-rental}"
REFERENCE_REPO="${REFERENCE_REPO:-$WORK_ROOT/torchtitan}"
VENV="${VENV:-$WORK_ROOT/.venv}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$WORK_ROOT/evidence}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu130}"

command -v git >/dev/null
command -v python3 >/dev/null
command -v nvidia-smi >/dev/null

mkdir -p "$WORK_ROOT" "$EVIDENCE_DIR"
nvidia-smi -L

if [[ ! -d "$REFERENCE_REPO/.git" ]]; then
  git clone https://github.com/pytorch/torchtitan.git "$REFERENCE_REPO"
  git -C "$REFERENCE_REPO" checkout --detach "$REFERENCE_SHA"
fi

test "$(git -C "$REFERENCE_REPO" rev-parse HEAD)" = "$REFERENCE_SHA"
test -z "$(git -C "$REFERENCE_REPO" status --porcelain)"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --pre torch --index-url "$TORCH_INDEX_URL"
"$VENV/bin/python" -m pip install -e "$REFERENCE_REPO"

set +e
"$VENV/bin/python" "$PLANLOCK_REPO/scripts/e0_nccl_validation.py" \
  --reference-repo "$REFERENCE_REPO" \
  --manifest "$PLANLOCK_REPO/e0-manifest-candidate.json" \
  --output "$EVIDENCE_DIR/e0-nccl-pe-moe.json"
validation_status=$?
set -e

if [[ -f "$EVIDENCE_DIR/e0-nccl-pe-moe.json" ]]; then
  sha256sum "$EVIDENCE_DIR/e0-nccl-pe-moe.json"
fi
exit "$validation_status"
