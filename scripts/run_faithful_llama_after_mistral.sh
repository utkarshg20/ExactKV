#!/usr/bin/env bash
# Wait for Mistral faithful panel, then run Llama faithful grid with resume/api resume.
set -euo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"
RUNDIR="${EXACTKV_RUNDIR:-/tmp/exactkv_panel_run_llama}"
LOG="$ROOT/reports/faithful_llama_panel.log"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "$ROOT/platform" 2>/dev/null || true

mkdir -p "$RUNDIR" "$ROOT/reports/external_panels/faithful" "$ROOT/reports/external_panels/logs"
cd "$RUNDIR"

echo "==> Waiting for Mistral faithful panel to finish..." | tee -a "$LOG"
while ! grep -q "FAITHFUL_PANEL_DONE" "$ROOT/reports/faithful_panel.log" 2>/dev/null; do
  sleep 120
  grep -E "^\[longbench\]|^\[bfcl\]|^\[mbpp\]|cell " "$ROOT/reports/faithful_panel.log" 2>/dev/null | tail -1 | tee -a "$LOG" || true
done
while pgrep -f "run_external_panel.py" >/dev/null 2>&1; do sleep 30; done

echo "==> Starting Llama faithful panel $(date -u -Iseconds)" | tee -a "$LOG"

COMPRESSORS="int8,kivi_offline_r32,snapkv_experimental"
MODEL="meta-llama/Llama-3.1-8B"
TAG="Llama_3_1_8B"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PLOG="$ROOT/reports/external_panels/logs/faithful_llama_${STAMP}.log"

run_panel() {
  local name="$1"
  local out_json="$2"
  shift 2
  local resume_args=()
  if [[ -f "$out_json" ]]; then
    resume_args=(--resume-json "$out_json" --checkpoint-json "$out_json")
    echo "==> [$name] resuming from $out_json" | tee -a "$LOG"
  else
    resume_args=(--checkpoint-json "$out_json")
  fi
  echo "==> [$name] $(date -u -Iseconds)" | tee -a "$LOG"
  "$PY" "$ROOT/scripts/run_external_panel.py" "$@" "${resume_args[@]}" --output-json "$out_json" 2>&1 | tee -a "$PLOG" >> "$LOG"
}

run_panel longbench "$ROOT/reports/external_panels/faithful/longbench_${TAG}_raw.json" \
  --family longbench --prompt-source hf --device cuda --dtype float16 \
  --max-prompts 12 --context-buckets 2048,4096,8192 --max-new-tokens 16,32 \
  --compressors "$COMPRESSORS" --models "$MODEL"

run_panel bfcl "$ROOT/reports/external_panels/faithful/bfcl_${TAG}_raw.json" \
  --family bfcl --device cuda --dtype float16 \
  --max-prompts 10 --context-buckets 512,1024 --max-new-tokens 16,32,64,128,256 \
  --compressors "$COMPRESSORS" --models "$MODEL"

run_panel mbpp "$ROOT/reports/external_panels/faithful/mbpp_${TAG}_raw.json" \
  --family mbpp --prompt-source hf --device cuda --dtype float16 \
  --max-prompts 8 --context-buckets 512,1024 --max-new-tokens 16,32 \
  --compressors "$COMPRESSORS" --models "$MODEL"

echo "FAITHFUL_LLAMA_DONE $(date -u -Iseconds)" | tee -a "$LOG"
