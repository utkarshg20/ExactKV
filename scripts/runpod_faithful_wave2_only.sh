#!/usr/bin/env bash
# Run wave-2 faithful smoke only (KnormPress + TurboQuant). Use when wave-1 is already
# complete locally and on the pod volume — e.g. after pod restart / volume recovery.
#
# On RunPod:
#   bash scripts/setup_faithful_compressor_env.sh   # once per fresh pod
#   bash scripts/runpod_faithful_wave2_only.sh
#
# From Mac after completion:
#   bash scripts/pull_faithful_from_runpod.sh
set -euo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

VENV="${FAITHFUL_VENV:-/workspace/.venv-faithful}"
PY="${PYTHON:-$VENV/bin/python3}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"
export EXACTKV_TURBOQUANT_ROOT="${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}"
export INSTALL_TURBOQUANT="${INSTALL_TURBOQUANT:-1}"

WAVE2="$ROOT/reports/external_panels/faithful/wave2"
mkdir -p "$WAVE2"

echo "==> Wave-2 only (skipping wave-1 Llama/Mistral grids)"
echo "==> Checking for recoverable wave-2 checkpoints on volume:"
ls -lah "$WAVE2" 2>/dev/null || echo "(empty wave2 dir)"

for f in "$WAVE2"/*_wave2_smoke_raw.json; do
  [[ -f "$f" ]] || continue
  n=$("$PY" -c "import json; d=json.load(open('$f')); print(sum(1 for c in d.get('cells',[]) if c.get('status')=='ok'))" 2>/dev/null || echo 0)
  echo "    $(basename "$f"): $n ok cells (will resume if incomplete)"
done

echo "==> Setup faithful env + TurboQuant if needed"
bash "$ROOT/scripts/setup_faithful_compressor_env.sh"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/reports/faithful_wave2_only_${STAMP}.log"
echo "==> Launching wave-2 in tmux 'faithful_wave2' (log: $LOG)"

tmux kill-session -t faithful_wave2 2>/dev/null || true
tmux new-session -d -s faithful_wave2 \
  "export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_KIVI_ROOT='$EXACTKV_KIVI_ROOT' \
   EXACTKV_TURBOQUANT_ROOT='$EXACTKV_TURBOQUANT_ROOT' INSTALL_TURBOQUANT='$INSTALL_TURBOQUANT' \
   HF_HOME='$HF_HOME' TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' TMPDIR='$TMPDIR' && \
   echo '=== wave2_only $STAMP ===' | tee '$LOG' && \
   bash '$ROOT/scripts/run_faithful_external_wave2_smoke.sh' 2>&1 | tee -a '$LOG'"

echo "==> Monitor: tmux attach -t faithful_wave2"
echo "==> Pull when done: bash scripts/pull_faithful_from_runpod.sh"
