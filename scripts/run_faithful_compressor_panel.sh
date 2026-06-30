#!/usr/bin/env bash
# Faithful external compressor panel — KIVI r32 + SnapKV (kvpress) on v3.0-style grid.
#
# Run inside tmux on RunPod:
#   bash scripts/runpod_faithful_panel_launch.sh
#
# Optional: limit models
#   FAITHFUL_MODELS='mistralai/Mistral-7B-Instruct-v0.3' bash scripts/run_faithful_compressor_panel.sh
#
# Outputs: reports/external_panels/faithful/
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
RUNDIR="${EXACTKV_RUNDIR:-/tmp/exactkv_panel_run}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

PY="${PYTHON:-/usr/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"
rm -rf "$ROOT/platform" 2>/dev/null || true

OUTDIR="$ROOT/reports/external_panels/faithful"
mkdir -p "$OUTDIR" "$ROOT/reports/external_panels/logs"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT/reports/external_panels/logs"
PANEL_LOG="${FAITHFUL_PANEL_LOG:-$ROOT/reports/faithful_panel.log}"

echo "==> Faithful compressor panel start $STAMP" | tee -a "$PANEL_LOG"
echo "==> EXACTKV_ROOT=$ROOT RUNDIR=$RUNDIR" | tee -a "$PANEL_LOG"

echo "" | tee -a "$PANEL_LOG"
echo "==> [check] CUDA" | tee -a "$PANEL_LOG"
"$PY" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')" | tee -a "$PANEL_LOG"

echo "==> [check] exactkv import" | tee -a "$PANEL_LOG"
"$PY" -c "import exactkv.benchmarks.external_panel as ep; print('exactkv:', ep.EXTERNAL_PANEL_ID)" | tee -a "$PANEL_LOG"

echo "==> [check] KIVI models.utils_quant" | tee -a "$PANEL_LOG"
if ! EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}" "$PY" -c "from exactkv.compressors.kivi_adapter import _import_kivi_utils; _import_kivi_utils(); print('kivi: ok')" 2>/dev/null | tee -a "$PANEL_LOG"; then
  echo "ERROR: KIVI not importable. Set EXACTKV_KIVI_ROOT to jy-yuan/KIVI repo root." | tee -a "$PANEL_LOG"
  exit 1
fi

echo "==> [check] kvpress (SnapKV)" | tee -a "$PANEL_LOG"
if ! "$PY" -c "import kvpress; print('kvpress: ok')" 2>/dev/null | tee -a "$PANEL_LOG"; then
  echo "WARN: kvpress not installed — SnapKV cells will be skipped." | tee -a "$PANEL_LOG"
fi

COMPRESSORS="int8,kivi_offline_r32,snapkv_experimental"

run_panel() {
  local name="$1"
  shift
  local out_json="$1"
  shift
  local logfile="$LOG_DIR/${name}_${STAMP}.log"
  local resume_args=()
  if [[ -f "$out_json" ]]; then
    resume_args=(--resume-json "$out_json" --checkpoint-json "$out_json")
    echo "==> [$name] resuming from $out_json" | tee -a "$PANEL_LOG"
  else
    resume_args=(--checkpoint-json "$out_json")
  fi
  echo "" | tee -a "$PANEL_LOG"
  echo "========================================" | tee -a "$PANEL_LOG"
  echo "==> [$name] $(date -u -Iseconds)" | tee -a "$PANEL_LOG"
  echo "CMD: $PY $ROOT/scripts/run_external_panel.py $* --output-json $out_json" | tee -a "$PANEL_LOG"
  echo "========================================" | tee -a "$PANEL_LOG"
  if "$PY" "$ROOT/scripts/run_external_panel.py" "$@" "${resume_args[@]}" --output-json "$out_json" 2>&1 | tee -a "$logfile" >> "$PANEL_LOG"; then
    echo "==> [$name] SUCCESS" | tee -a "$PANEL_LOG"
    return 0
  else
    local rc=$?
    echo "==> [$name] FAILED (exit $rc), continuing" | tee -a "$PANEL_LOG"
    return $rc
  fi
}

if [[ -n "${FAITHFUL_MODELS:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$FAITHFUL_MODELS"
else
  MODELS=(
    "meta-llama/Llama-3.1-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
  )
fi

for MODEL in "${MODELS[@]}"; do
  MODEL="${MODEL#"${MODEL%%[![:space:]]*}"}"
  MODEL="${MODEL%"${MODEL##*[![:space:]]}"}"
  [[ -z "$MODEL" ]] && continue
  MODEL_TAG="${MODEL##*/}"
  MODEL_TAG="${MODEL_TAG//./_}"
  MODEL_TAG="${MODEL_TAG//-/_}"

  echo "" | tee -a "$PANEL_LOG"
  echo "############################################" | tee -a "$PANEL_LOG"
  echo "## MODEL: $MODEL" | tee -a "$PANEL_LOG"
  echo "############################################" | tee -a "$PANEL_LOG"

  run_panel "faithful_longbench_${MODEL_TAG}" \
    "$OUTDIR/longbench_${MODEL_TAG}_raw.json" \
    --family longbench \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 12 \
    --context-buckets 2048,4096,8192 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL"

  run_panel "faithful_bfcl_${MODEL_TAG}" \
    "$OUTDIR/bfcl_${MODEL_TAG}_raw.json" \
    --family bfcl \
    --device cuda --dtype float16 \
    --max-prompts 10 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32,64,128,256 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL"

  run_panel "faithful_mbpp_${MODEL_TAG}" \
    "$OUTDIR/mbpp_${MODEL_TAG}_raw.json" \
    --family mbpp \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 8 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL"
done

echo "" | tee -a "$PANEL_LOG"
echo "==> Faithful panel complete. Artifacts in $OUTDIR/" | tee -a "$PANEL_LOG"
ls -la "$OUTDIR/" 2>/dev/null | tee -a "$PANEL_LOG" || true
echo "FAITHFUL_PANEL_DONE $STAMP" | tee -a "$PANEL_LOG"
