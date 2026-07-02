#!/usr/bin/env bash
# Launch wave-3 TurboQuant faithful panel in tmux (~6–8 GPU hours).
#
# From Mac:
#   bash scripts/sync_to_runpod.sh
#   ssh root@HOST -p PORT -i ~/.ssh/runpod_exactkv \
#     'bash /workspace/ExactKV/scripts/runpod_faithful_wave3_launch.sh'
#
# Monitor:
#   tmux attach -t faithful_wave3
#   tail -f /workspace/ExactKV/reports/faithful_wave3_*.log
#
# Pull when done:
#   bash scripts/pull_faithful_wave3_from_runpod.sh
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

WAVE3="$ROOT/reports/external_panels/faithful/wave3"
mkdir -p "$WAVE3"

echo "==> Wave-3 TurboQuant panel launcher $(date -u -Iseconds)"
echo "==> Target: ~576 cells (int8 + turboquant, both models, 3 families)"
echo "==> Existing wave-3 checkpoints:"
ls -lah "$WAVE3" 2>/dev/null || echo "(empty)"

for f in "$WAVE3"/*_wave3_raw.json; do
  [[ -f "$f" ]] || continue
  n=$("$PY" -c "import json; d=json.load(open('$f')); print(sum(1 for c in d.get('cells',[]) if c.get('status')=='ok'))" 2>/dev/null || echo 0)
  echo "    $(basename "$f"): $n ok cells"
done

echo "==> Setup faithful env + TurboQuant"
INSTALL_TURBOQUANT=1 bash "$ROOT/scripts/setup_faithful_compressor_env.sh"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/reports/faithful_wave3_launch_${STAMP}.log"

tmux kill-session -t faithful_wave3 2>/dev/null || true
tmux new-session -d -s faithful_wave3 \
  "export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_KIVI_ROOT='$EXACTKV_KIVI_ROOT' \
   EXACTKV_TURBOQUANT_ROOT='$EXACTKV_TURBOQUANT_ROOT' HF_HOME='$HF_HOME' \
   TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' TMPDIR='$TMPDIR' && \
   rm -rf '$ROOT/platform' && \
   echo '=== wave3_launch $STAMP ===' | tee '$LOG' && \
   bash '$ROOT/scripts/run_faithful_turboquant_wave3_panel.sh' 2>&1 | tee -a '$LOG'"

echo "==> Launched tmux session: faithful_wave3"
echo "==> Log: $LOG"
echo "==> Monitor: tmux attach -t faithful_wave3"
echo "==> Pull: bash scripts/pull_faithful_wave3_from_runpod.sh"
