#!/usr/bin/env bash
# Resume faithful panel on RunPod after Mistral wave-1: sync fixes, start Llama + queue wave-2.
set -euo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

VENV="${FAITHFUL_VENV:-/workspace/.venv-faithful}"
PY="${PYTHON:-$VENV/bin/python3}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"

chmod +x "$ROOT/scripts/wait_for_external_panel_idle.sh" \
  "$ROOT/scripts/run_faithful_llama_after_mistral.sh" \
  "$ROOT/scripts/runpod_wave2_launch.sh" \
  "$ROOT/scripts/pull_faithful_from_runpod.sh" 2>/dev/null || true

echo "==> Killing stale faithful tmux sessions (keep faithful_wave2 if queued)"
tmux kill-session -t faithful 2>/dev/null || true
tmux kill-session -t faithful_llama 2>/dev/null || true

# Drop empty Llama checkpoints from failed import/resume attempts
for f in "$ROOT/reports/external_panels/faithful/"*Llama*_raw.json; do
  [[ -f "$f" ]] || continue
  n=$("$PY" -c "import json; d=json.load(open('$f')); print(sum(1 for c in d.get('cells',[]) if c.get('status')=='ok'))" 2>/dev/null || echo 0)
  if [[ "$n" -eq 0 ]]; then
    echo "==> Removing empty checkpoint $f"
    rm -f "$f"
  fi
done

# Mistral wave-1 complete if full grid exists
if [[ -f "$ROOT/reports/external_panels/faithful/mbpp_Mistral_7B_Instruct_v0_3_raw.json" ]]; then
  echo "==> Mistral faithful grid present, launching Llama"
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s faithful_llama \
    "export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_RUNDIR='/tmp/exactkv_panel_run_llama' \
     EXACTKV_KIVI_ROOT='${KIVI_DIR:-/tmp/kivi_research}' HF_HOME='${HF_HOME}' \
     TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' TMPDIR='${TMPDIR}' && \
     echo '=== faithful_llama resume $STAMP ===' >> '$ROOT/reports/faithful_llama_panel.log' && \
     bash '$ROOT/scripts/run_faithful_llama_after_mistral.sh' 2>&1 | tee -a '$ROOT/reports/faithful_llama_panel.log'"
  echo "==> Llama: tmux attach -t faithful_llama"
else
  echo "WARN: Mistral artifacts missing, not starting Llama"
fi

echo "==> Queue wave-2 smoke (runs after GPU idle)"
if ! tmux has-session -t faithful_wave2 2>/dev/null; then
  tmux new-session -d -s faithful_wave2 \
    "bash '$ROOT/scripts/wait_for_external_panel_idle.sh' 120 && \
     export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_KIVI_ROOT='${KIVI_DIR:-/tmp/kivi_research}' \
     EXACTKV_TURBOQUANT_ROOT='${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}' \
     INSTALL_TURBOQUANT=1 && \
     bash '$ROOT/scripts/runpod_wave2_launch.sh'"
  echo "==> Wave-2 queued: tmux attach -t faithful_wave2"
else
  echo "==> Wave-2 session already exists"
fi

echo "==> Sessions:"
tmux ls 2>/dev/null || echo "(no tmux)"
