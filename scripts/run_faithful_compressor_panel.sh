#!/usr/bin/env bash
# Faithful external compressor panel — KIVI r32 + SnapKV (kvpress) on v3.0-style grid.
#
# PREREQUISITES (RunPod or local GPU):
#   1. KIVI upstream (simulate quant math):
#        git clone https://github.com/jy-yuan/KIVI.git /tmp/kivi_research
#        export PYTHONPATH=/tmp/kivi_research
#   2. kvpress for SnapKV experimental adapter:
#        pip install kvpress
#   3. HuggingFace auth for gated models (Llama):
#        hf auth login
#
# Run inside tmux on RunPod:
#   tmux new-session -s faithful
#   bash scripts/run_faithful_compressor_panel.sh 2>&1 | tee reports/faithful_panel.log
#
# Outputs: reports/external_panels/faithful/
#
# Claim boundary:
#   - kivi_offline_r32: real KIVI quantizer math + residual fp16 window (r=32).
#     NOT KIVI production CUDA/Triton. supports_real_bytes_claim=False.
#   - snapkv_experimental: kvpress SnapKVPress replay prefill. NOT paper-exact SnapKV.
#   - int8: built-in control compressor.
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
# Avoid repo-root `platform/` shadowing stdlib when cwd is ExactKV (RunPod artifact).
RUNDIR="${EXACTKV_RUNDIR:-/tmp/exactkv_panel_run}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

PY="${PYTHON:-/usr/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"

OUTDIR="$ROOT/reports/external_panels/faithful"
mkdir -p "$OUTDIR" "$ROOT/reports/external_panels/logs"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT/reports/external_panels/logs"

echo "==> Faithful compressor panel start $STAMP"
echo "==> EXACTKV_ROOT=$ROOT"

echo ""
echo "==> [check] CUDA"
"$PY" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo "==> [check] KIVI models.utils_quant"
if ! EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}" "$PY" -c "from exactkv.compressors.kivi_adapter import _import_kivi_utils; _import_kivi_utils(); print('kivi: ok')" 2>/dev/null; then
  echo "ERROR: KIVI not importable. Set EXACTKV_KIVI_ROOT to jy-yuan/KIVI repo root."
  exit 1
fi

echo "==> [check] kvpress (SnapKV)"
if ! "$PY" -c "import kvpress; print('kvpress: ok')" 2>/dev/null; then
  echo "WARN: kvpress not installed — SnapKV cells will be skipped."
fi

# int8 control + faithful external compressors
COMPRESSORS="int8,kivi_offline_r32,snapkv_experimental"

run_panel() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/${name}_${STAMP}.log"
  echo ""
  echo "========================================"
  echo "==> [$name] $(date -u -Iseconds)"
  echo "CMD: $PY scripts/run_external_panel.py $*"
  echo "========================================"
  if "$PY" "$ROOT/scripts/run_external_panel.py" "$@" 2>&1 | tee -a "$logfile"; then
    echo "==> [$name] SUCCESS"
    return 0
  else
    local rc=$?
    echo "==> [$name] FAILED (exit $rc), continuing" | tee -a "$logfile"
    return $rc
  fi
}

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

for MODEL in "${MODELS[@]}"; do
  MODEL_TAG="${MODEL##*/}"
  MODEL_TAG="${MODEL_TAG//./_}"
  MODEL_TAG="${MODEL_TAG//-/_}"

  echo ""
  echo "############################################"
  echo "## MODEL: $MODEL"
  echo "############################################"

  # LongBench — main stress test (matches v3.0 grid)
  run_panel "faithful_longbench_${MODEL_TAG}" \
    --family longbench \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 12 \
    --context-buckets 2048,4096,8192 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL" \
    --output-json "$OUTDIR/longbench_${MODEL_TAG}_raw.json"

  # BFCL short
  run_panel "faithful_bfcl_${MODEL_TAG}" \
    --family bfcl \
    --device cuda --dtype float16 \
    --max-prompts 10 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32,64,128,256 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL" \
    --output-json "$OUTDIR/bfcl_${MODEL_TAG}_raw.json"

  # MBPP code
  run_panel "faithful_mbpp_${MODEL_TAG}" \
    --family mbpp \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 8 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL" \
    --output-json "$OUTDIR/mbpp_${MODEL_TAG}_raw.json"
done

echo ""
echo "==> Faithful panel complete. Artifacts in $OUTDIR/"
ls -la "$OUTDIR/" 2>/dev/null || true
