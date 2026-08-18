#!/usr/bin/env bash
set -euo pipefail

REFERENCE_SHA="9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"
PLANLOCK_REPO="${PLANLOCK_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:?NODE_RANK must be set to 0, 1, 2, or 3}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be the private address of node 0}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-}"
WORK_ROOT="${WORK_ROOT:-$HOME/planlock-e0-dense-node-$NODE_RANK}"
REFERENCE_REPO="${REFERENCE_REPO:-$WORK_ROOT/torchtitan}"
VENV="${VENV:-$WORK_ROOT/.venv}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$HOME/planlock-e0-dense-evidence}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu130}"

test "$NNODES" = 4
test "$NPROC_PER_NODE" = 8
test "$NODE_RANK" -ge 0
test "$NODE_RANK" -lt "$NNODES"
command -v git >/dev/null
command -v python3 >/dev/null
command -v nvidia-smi >/dev/null

mkdir -p "$WORK_ROOT"
if [[ "$NODE_RANK" = 0 ]]; then
  mkdir -p "$EVIDENCE_DIR"
fi
nvidia-smi -L
test "$(nvidia-smi -L | wc -l)" = 8

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

args=(
  --reference-repo "$REFERENCE_REPO"
  --manifest "$PLANLOCK_REPO/e0-manifest-candidate.json"
  --pe-moe-evidence "$PLANLOCK_REPO/e0-nccl-pe-moe-evidence.json"
  --output "$EVIDENCE_DIR/e0-nccl-pe-dense.json"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --nproc-per-node "$NPROC_PER_NODE"
  --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT"
)
if [[ -n "$NETWORK_INTERFACE" ]]; then
  args+=(--network-interface "$NETWORK_INTERFACE")
fi

set +e
"$VENV/bin/python" "$PLANLOCK_REPO/scripts/e0_dense_multinode_validation.py" "${args[@]}"
validation_status=$?
set -e

if [[ "$NODE_RANK" = 0 && -f "$EVIDENCE_DIR/e0-nccl-pe-dense.json" ]]; then
  sha256sum "$EVIDENCE_DIR/e0-nccl-pe-dense.json"
fi
exit "$validation_status"
