#!/usr/bin/env bash
# Backfill wave-3 LongBench panels (HF loader may fail on newer datasets; uses export fallback).
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"
OUTDIR="$ROOT/reports/external_panels/faithful/wave3"
COMPRESSORS="int8,turboquant_experimental"
LOG="$ROOT/reports/faithful_wave3_longbench_backfill_$(date -u +%Y%m%dT%H%M%SZ).log"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_HOME/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

run_lb() {
  local model="$1"
  local tag="${model##*/}"
  tag="${tag//./_}"
  tag="${tag//-/_}"
  local out="$OUTDIR/longbench_${tag}_wave3_raw.json"
  local resume=()
  [[ -f "$out" ]] && resume=(--resume-json "$out" --checkpoint-json "$out")
  echo "==> LongBench backfill $model -> $out" | tee -a "$LOG"
  "$PY" "$ROOT/scripts/run_external_panel.py" \
    --family longbench \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 12 \
    --context-buckets 2048,4096,8192 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$model" \
    "${resume[@]}" \
    --checkpoint-json "$out" \
    --output-json "$out" 2>&1 | tee -a "$LOG"
}

for MODEL in \
  "mistralai/Mistral-7B-Instruct-v0.3" \
  "meta-llama/Llama-3.1-8B"; do
  run_lb "$MODEL"
done

echo "WAVE3_LONGBENCH_BACKFILL_DONE $(date -u -Iseconds)" | tee -a "$LOG"
