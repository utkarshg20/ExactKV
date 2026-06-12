#!/usr/bin/env bash
# Run Experiment 030 on RunPod GPU (float16 CUDA).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="${EXP030_VENV:-/workspace/kivi_exp024/.venv-kivi}"
PYTHON="${EXP030_PYTHON:-$VENV/bin/python}"
export TRANSFORMERS_OFFLINE=1
LOG="${EXP030_LOG:-/workspace/exp030_diagnostic_timing.log}"
exec > >(tee -a "$LOG") 2>&1
echo "=== Exp 030 GPU run ==="
echo "date: $(date -Iseconds)"
echo "python: $PYTHON"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
if ! "$PYTHON" -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)" 2>/dev/null; then
  echo "caching Qwen/Qwen2.5-0.5B (one-time download)..."
  unset TRANSFORMERS_OFFLINE
  "$PYTHON" -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m='Qwen/Qwen2.5-0.5B'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto')"
  export TRANSFORMERS_OFFLINE=1
fi
"$PYTHON" scripts/run_experiment_030_diagnostic_timing.py \
  --device cuda \
  --dtype float16 \
  --num-warmup 2 \
  --num-trials 3
echo "exp030_complete exit=$?"
