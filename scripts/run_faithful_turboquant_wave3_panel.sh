#!/usr/bin/env bash
# Wave-3 faithful TurboQuant panel — full wave-1 grid, int8 + turboquant only.
#
# Target: ~576 cells (288/model × 2 models), ~6–8 GPU hours on L40S/A5000.
# Families: HF LongBench, BFCL, MBPP (same prompts/contexts as wave-1 faithful panel).
#
# Usage (RunPod):
#   INSTALL_TURBOQUANT=1 bash scripts/setup_faithful_compressor_env.sh
#   bash scripts/run_faithful_turboquant_wave3_panel.sh
#
# Outputs: reports/external_panels/faithful/wave3/
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
RUNDIR="${EXACTKV_RUNDIR:-/tmp/exactkv_wave3_run}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_HOME/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"
export EXACTKV_TURBOQUANT_ROOT="${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}"
if [[ -n "${EXACTKV_TURBOQUANT_ROOT:-}" ]]; then
  export PYTHONPATH="${EXACTKV_TURBOQUANT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
fi
rm -rf "$ROOT/platform" 2>/dev/null || true

OUTDIR="$ROOT/reports/external_panels/faithful/wave3"
LOG_DIR="$ROOT/reports/external_panels/logs"
mkdir -p "$OUTDIR" "$LOG_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MAIN_LOG="$ROOT/reports/faithful_wave3_${STAMP}.log"
COMPRESSORS="int8,turboquant_experimental"

echo "==> Wave-3 TurboQuant faithful panel $STAMP" | tee "$MAIN_LOG"
echo "==> compressors=$COMPRESSORS OUTDIR=$OUTDIR" | tee -a "$MAIN_LOG"

"$PY" -c "
from exactkv.benchmarks.evidence_plus_panel import resolve_evidence_plus_compressor
for c in '$COMPRESSORS'.split(','):
    r = resolve_evidence_plus_compressor(c)
    print(c, r.backend_tier, r.adapter_available)
" 2>&1 | tee -a "$MAIN_LOG"

run_panel() {
  local name="$1"
  local out_json="$2"
  shift 2
  local logfile="$LOG_DIR/${name}_${STAMP}.log"
  local resume_args=()
  if [[ -f "$out_json" ]]; then
    resume_args=(--resume-json "$out_json" --checkpoint-json "$out_json")
    echo "==> [$name] resuming $out_json" | tee -a "$MAIN_LOG"
  else
    resume_args=(--checkpoint-json "$out_json")
  fi
  echo "" | tee -a "$MAIN_LOG"
  echo "========================================" | tee -a "$MAIN_LOG"
  echo "==> [$name] $(date -u -Iseconds)" | tee -a "$MAIN_LOG"
  echo "========================================" | tee -a "$MAIN_LOG"
  if "$PY" "$ROOT/scripts/run_external_panel.py" "$@" \
      "${resume_args[@]}" --output-json "$out_json" \
      2>&1 | tee -a "$logfile" >> "$MAIN_LOG"; then
    echo "==> [$name] SUCCESS" | tee -a "$MAIN_LOG"
    return 0
  fi
  local rc=$?
  echo "==> [$name] FAILED (exit $rc), continuing" | tee -a "$MAIN_LOG"
  return $rc
}

if [[ -n "${FAITHFUL_WAVE3_MODELS:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$FAITHFUL_WAVE3_MODELS"
else
  MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.1-8B"
  )
fi

for MODEL in "${MODELS[@]}"; do
  MODEL="${MODEL#"${MODEL%%[![:space:]]*}"}"
  MODEL="${MODEL%"${MODEL##*[![:space:]]}"}"
  [[ -z "$MODEL" ]] && continue
  MODEL_TAG="${MODEL##*/}"
  MODEL_TAG="${MODEL_TAG//./_}"
  MODEL_TAG="${MODEL_TAG//-/_}"

  echo "" | tee -a "$MAIN_LOG"
  echo "############################################" | tee -a "$MAIN_LOG"
  echo "## WAVE-3 MODEL: $MODEL" | tee -a "$MAIN_LOG"
  echo "############################################" | tee -a "$MAIN_LOG"

  run_panel "wave3_longbench_${MODEL_TAG}" \
    "$OUTDIR/longbench_${MODEL_TAG}_wave3_raw.json" \
    --family longbench \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 12 \
    --context-buckets 2048,4096,8192 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL"

  run_panel "wave3_bfcl_${MODEL_TAG}" \
    "$OUTDIR/bfcl_${MODEL_TAG}_wave3_raw.json" \
    --family bfcl \
    --device cuda --dtype float16 \
    --max-prompts 10 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32,64,128,256 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL"

  run_panel "wave3_mbpp_${MODEL_TAG}" \
    "$OUTDIR/mbpp_${MODEL_TAG}_wave3_raw.json" \
    --family mbpp \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 8 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL"
done

echo "" | tee -a "$MAIN_LOG"
echo "WAVE3_TURBOQUANT_DONE $STAMP" | tee -a "$MAIN_LOG"

cd "$ROOT"
"$PY" "$ROOT/scripts/integrate_faithful_panel_results.py" --dir "$OUTDIR" --write \
  2>&1 | tee -a "$MAIN_LOG" || true

ls -lah "$OUTDIR/" | tee -a "$MAIN_LOG"
