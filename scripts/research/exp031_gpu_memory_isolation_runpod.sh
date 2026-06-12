#!/usr/bin/env bash
# Run Experiment 031 on RunPod GPU (float16 CUDA).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="${EXP031_VENV:-/workspace/kivi_exp024/.venv-kivi}"
PYTHON="${EXP031_PYTHON:-$VENV/bin/python}"
export TRANSFORMERS_OFFLINE=1
LOG="${EXP031_LOG:-/workspace/exp031_gpu_memory_isolation.log}"
exec > >(tee -a "$LOG") 2>&1
echo "=== Exp 031 GPU memory isolation ==="
echo "date: $(date -Iseconds)"
echo "python: $PYTHON"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
if ! "$PYTHON" -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)" 2>/dev/null; then
  echo "caching Qwen/Qwen2.5-0.5B (one-time download)..."
  unset TRANSFORMERS_OFFLINE
  "$PYTHON" -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m='Qwen/Qwen2.5-0.5B'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto')"
  export TRANSFORMERS_OFFLINE=1
fi
"$PYTHON" scripts/run_experiment_031_gpu_memory_isolation.py \
  --device cuda \
  --dtype float16 \
  --num-trials 2
echo "exp031_complete exit=$?"
