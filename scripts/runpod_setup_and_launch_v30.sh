#!/usr/bin/env bash
# Setup script to run ONCE on a fresh RunPod pod after rsync.
# Installs dependencies, then launches the v3.0 panel in a tmux session.
#
# Usage (from local machine after rsync completes):
#   ssh runpod-a5000 'bash /workspace/ExactKV/scripts/runpod_setup_and_launch_v30.sh'
set -uo pipefail

ROOT="/workspace/ExactKV"
cd "$ROOT"

# --- 1. Python environment ---
PY=""
for candidate in /workspace/.venv-runpod/bin/python3 /opt/conda/bin/python3 python3; do
  if command -v "$candidate" &>/dev/null; then PY="$candidate"; break; fi
done
echo "==> Using Python: $PY"

# Install/upgrade dependencies if pyproject.toml present
if [[ -f pyproject.toml ]]; then
  echo "==> Installing package..."
  "$PY" -m pip install -e ".[dev]" --quiet 2>&1 | tail -3
fi

# Ensure datasets is installed (needed for HF LongBench)
"$PY" -m pip install datasets --quiet 2>&1 | tail -2

# --- 2. Smoke test new compressors ---
echo "==> Smoke-testing new compressors..."
"$PY" -c "
from exactkv.compressors import get_compressor
for name in ['int8', 'int6_sim', 'int4_per_vec_sim', 'int4_sim']:
    c = get_compressor(name)
    print(f'  {c.name}: OK')
import torch
print(f'  CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"n/a\"}')"

# --- 3. Create/attach tmux session and run panel ---
SESSION="v30panel"

# Kill stale session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo ""
echo "==> Launching v3.0 panel in tmux session '$SESSION'..."
echo "    Monitor with: ssh runpod-a5000 'tmux attach -t $SESSION'"
echo "    Or read logs: ssh runpod-a5000 'tail -f /workspace/v30_panel.log'"
echo ""

tmux new-session -d -s "$SESSION" \
  "bash $ROOT/scripts/run_v30_new_compressors.sh 2>&1 | tee /workspace/v30_panel.log; echo 'PANEL COMPLETE'; bash"

echo "==> tmux session '$SESSION' started. Panel running in background."
echo "    It will keep running even if you disconnect."
tmux list-sessions
