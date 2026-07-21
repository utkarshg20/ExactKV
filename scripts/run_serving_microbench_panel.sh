#!/usr/bin/env bash
# HF multi-request serving microbench (TTFT-like / RPS / peak CUDA).
#
# Usage on RunPod (tmux recommended):
#   export HF_TOKEN=hf_...   # Llama gated
#   bash scripts/run_serving_microbench_panel.sh 2>&1 | tee /workspace/serving_microbench.log
#
# Prefer the torch image (NOT the auto-serve vLLM template).
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

PY="${PYTHON:-/workspace/.venv-runpod/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

OUT_DIR="${OUT_DIR:-reports/external_panels/serving_microbench}"
DTYPE="${DTYPE:-float16}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

echo "==> Serving microbench $STAMP"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

run_model() {
  local model="$1"
  local tag
  tag="$(echo "${model##*/}" | tr '.-' '_')"
  local out="$OUT_DIR/${tag}_raw.json"
  local log="$OUT_DIR/logs/${tag}_${STAMP}.log"
  echo "==> $model -> $out"
  "$PY" scripts/run_serving_microbench_panel.py \
      --device cuda \
      --dtype "$DTYPE" \
      --models "$model" \
      --output-json "$out" \
      --checkpoint-json "$out" \
      --resume-json "$out" \
      2>&1 | tee "$log"
}

run_model "mistralai/Mistral-7B-Instruct-v0.3"
run_model "meta-llama/Llama-3.1-8B"

"$PY" scripts/build_serving_microbench_pack.py \
    --input-dir "$OUT_DIR" \
    --output-json reports/systems/serving_microbench.json \
    --output-md reports/systems/serving_microbench.md

echo DONE > "$OUT_DIR/_DONE"
echo "==> Serving microbench complete $(date -u +%Y%m%dT%H%M%SZ)"
