#!/usr/bin/env bash
# V12 Phase 4 — one-shot KIVI Exp 024 on RunPod (prep + inspect + markdown).
set -euo pipefail

EXACTKV_ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
WORKDIR="${KIVI_EXP024_WORKDIR:-/workspace/kivi_exp024}"

bash "$EXACTKV_ROOT/scripts/research/kivi_cuda_triton_exp024_prep.sh"

# shellcheck disable=SC1091
source "$WORKDIR/.venv-kivi/bin/activate"
export PYTHONPATH="$WORKDIR/KIVI:$WORKDIR/KIVI/quant:${PYTHONPATH:-}"

cd "$EXACTKV_ROOT"
python scripts/research/kivi_cuda_triton_exp024_inspect.py \
  --kivi-repo "$WORKDIR/KIVI" \
  --workdir "$WORKDIR" \
  --json-out "$WORKDIR/exp024_inspect.json" \
  --write-markdown "$EXACTKV_ROOT/docs/EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md"

echo "exp024_execute_complete"
