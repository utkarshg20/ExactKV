#!/usr/bin/env bash
# Run Experiment 033 on RunPod GPU (Llama-3.1-8B-Instruct exactness panel).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="${EXP033_VENV:-/workspace/kivi_exp024/.venv-kivi}"
PYTHON="${EXP033_PYTHON:-$VENV/bin/python}"
LOG="${EXP033_LOG:-/workspace/exp033_llama31_8b.log}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
exec > >(tee -a "$LOG") 2>&1
echo "=== Exp 033 Llama-3.1-8B small suite ==="
echo "date: $(date -Iseconds)"
echo "python: $PYTHON"
echo "HF_TOKEN present: $(test -n "${HF_TOKEN:-}" && echo yes || echo no)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
"$PYTHON" scripts/run_experiment_033_llama31_8b_small_suite.py \
  --device cuda \
  --try-snapkv
echo "exp033_complete exit=$?"
