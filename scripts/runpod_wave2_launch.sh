#!/usr/bin/env bash
# Launch wave-2 external smoke after wave-1 panel releases GPU (or immediately if idle).
set -euo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"

external_panel_running() {
  pgrep -f '[p]ython.*scripts/run_external_panel\.py' >/dev/null 2>&1
}

if external_panel_running; then
  echo "==> GPU panel still running, queue wave2 in tmux 'faithful_wave2'"
  tmux kill-session -t faithful_wave2 2>/dev/null || true
  tmux new-session -d -s faithful_wave2 \
    "bash '$ROOT/scripts/wait_for_external_panel_idle.sh' 120 && \
     export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_KIVI_ROOT='${KIVI_DIR:-/tmp/kivi_research}' \
     EXACTKV_TURBOQUANT_ROOT='${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}' \
     INSTALL_TURBOQUANT=1 && \
     bash '$ROOT/scripts/runpod_wave2_launch.sh'"
  echo "==> Queued: tmux attach -t faithful_wave2"
  exit 0
fi

export PYTHON="$PY"
export EXACTKV_ROOT="$ROOT"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"
export INSTALL_TURBOQUANT="${INSTALL_TURBOQUANT:-1}"
export EXACTKV_TURBOQUANT_ROOT="${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}"

bash "$ROOT/scripts/setup_faithful_compressor_env.sh"
bash "$ROOT/scripts/run_faithful_external_wave2_smoke.sh"
