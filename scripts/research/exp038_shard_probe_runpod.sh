#!/usr/bin/env bash
# Experiment 038: Shard external-drafter probe on RunPod GPU.
# Clone Shard outside ExactKV; do not vendor into repo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SHARD_DIR="${SHARD_REPO_PATH:-$HOME/shard}"
LOG="${EXP038_LOG:-/workspace/exp038_shard_probe.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== Exp 038 Shard external-drafter probe ==="
echo "date: $(date -Iseconds)"
echo "HF_TOKEN present: $(test -n "${HF_TOKEN:-}" && echo yes || echo no)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true

if [[ ! -d "$SHARD_DIR/.git" ]]; then
  echo "Cloning Shard to $SHARD_DIR"
  git clone --depth 1 https://github.com/krish1905/shard.git "$SHARD_DIR"
else
  echo "Reusing Shard clone at $SHARD_DIR"
fi

export SHARD_REPO_PATH="$SHARD_DIR"

PYTHON="${EXP038_PYTHON:-python3}"
if [[ -x "${EXP038_VENV:-}/bin/python" ]]; then
  PYTHON="${EXP038_VENV}/bin/python"
fi

echo "Installing Shard (editable) if needed..."
"$PYTHON" -m pip install -q -e "$SHARD_DIR" 2>/dev/null || "$PYTHON" -m pip install -q -e "$SHARD_DIR"

echo "--- import check ---"
"$PYTHON" scripts/probe_shard_external_drafter.py

echo "--- try-run ---"
"$PYTHON" scripts/probe_shard_external_drafter.py --try-run \
  --device cuda \
  --dtype float16 \
  --max-new-tokens 16 \
  --draft-len 4

echo "exp038_complete exit=0"
echo "Report: reports/experiment_038_shard_external_drafter_probe.json"
