#!/usr/bin/env bash
# Launch wave-2 external smoke after wave-1 panel releases GPU (or immediately if idle).
set -euo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"

if pgrep -f "run_external_panel.py" >/dev/null 2>&1; then
  echo "==> GPU panel still running — queue wave2 in tmux 'faithful_wave2'"
  tmux kill-session -t faithful_wave2 2>/dev/null || true
  tmux new-session -d -s faithful_wave2 \
    "while pgrep -f run_external_panel.py >/dev/null 2>&1; do sleep 120; done; \
     export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_KIVI_ROOT='${KIVI_DIR:-/tmp/kivi_research}' \
     EXACTKV_TURBOQUANT_ROOT='${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}' \
     INSTALL_TURBOQUANT=1 && bash '$ROOT/scripts/setup_faithful_compressor_env.sh' && \
     bash '$ROOT/scripts/run_faithful_external_wave2_smoke.sh'"
  echo "==> Queued: tmux attach -t faithful_wave2"
  exit 0
fi

bash "$ROOT/scripts/setup_faithful_compressor_env.sh"
bash "$ROOT/scripts/run_faithful_external_wave2_smoke.sh"
