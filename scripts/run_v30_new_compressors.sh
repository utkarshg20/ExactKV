#!/usr/bin/env bash
# v3.0 panel: validate int6_sim and int4_per_vec_sim against int8/int4_sim baselines.
#
# Run this INSIDE a tmux session on RunPod:
#   tmux new-session -s v30
#   bash scripts/run_v30_new_compressors.sh 2>&1 | tee reports/v30_panel.log
#
# Expected runtime on a single A5000: ~3–5 hours (720 cells per model, 2 models = ~1440 cells).
# Writes per-compressor per-model JSON into reports/external_panels/v30/
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

PY="${PYTHON:-/workspace/.venv-runpod/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

# Verify new compressors are importable before spending GPU time
echo "==> Smoke-testing new compressors..."
"$PY" -c "
from exactkv.compressors import get_compressor
c6 = get_compressor('int6_sim')
c4v = get_compressor('int4_per_vec_sim')
print('int6_sim:', c6.name, '-- OK')
print('int4_per_vec_sim:', c4v.name, '-- OK')
"
echo ""

"$PY" -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

OUTDIR="reports/external_panels/v30"
mkdir -p "$OUTDIR" reports/external_panels/logs

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="reports/external_panels/logs"

# New compressors to validate + int8/int4_sim as controls
COMPRESSORS="int8,int6_sim,int4_per_vec_sim,int4_sim"

run_panel() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/${name}_${STAMP}.log"
  echo ""
  echo "========================================"
  echo "==> [$name] $(date -u -Iseconds)"
  echo "CMD: $PY scripts/run_external_panel.py $*"
  echo "========================================"
  if "$PY" scripts/run_external_panel.py "$@" 2>&1 | tee -a "$logfile"; then
    echo "==> [$name] SUCCESS"
    return 0
  else
    local rc=$?
    echo "==> [$name] FAILED (exit $rc), continuing" | tee -a "$logfile"
    return $rc
  fi
}

MODELS_LLAMA="meta-llama/Llama-3.1-8B"
MODELS_MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

for MODEL in "$MODELS_LLAMA" "$MODELS_MISTRAL"; do
  MODEL_TAG="${MODEL##*/}"
  MODEL_TAG="${MODEL_TAG//./_}"
  MODEL_TAG="${MODEL_TAG//-/_}"

  echo ""
  echo "############################################"
  echo "## MODEL: $MODEL"
  echo "############################################"

  # --- LongBench v3.0 (main stress test: 90% divergence for int4_sim baseline) ---
  # 12 prompts × 3 ctx buckets × 2 mnt × 4 compressors = 288 cells per model
  run_panel "v30_longbench_${MODEL_TAG}" \
    --family longbench \
    --prompt-source hf \
    --device cuda --dtype float16 \
    --max-prompts 12 \
    --context-buckets 2048,4096,8192 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL" \
    --output-json "$OUTDIR/longbench_${MODEL_TAG}_raw.json" || true

  # --- BFCL validity v3.0 (structured-output drift, 50% divergence for int4_sim) ---
  # 25 prompts × 2 ctx buckets × 2 mnt × 4 compressors = 400 cells per model
  run_panel "v30_bfcl_${MODEL_TAG}" \
    --family bfcl \
    --prompt-source export \
    --device cuda --dtype float16 \
    --max-prompts 25 \
    --context-buckets 1024,2048 \
    --max-new-tokens 128,256 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL" \
    --output-json "$OUTDIR/bfcl_${MODEL_TAG}_raw.json" || true

  # --- MBPP v3.0 (code stability, ~6% divergence for int4_sim) ---
  # 6 prompts × 2 ctx buckets × 2 mnt × 4 compressors = 96 cells per model
  run_panel "v30_mbpp_${MODEL_TAG}" \
    --family mbpp \
    --device cuda --dtype float16 \
    --max-prompts 6 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL" \
    --output-json "$OUTDIR/mbpp_${MODEL_TAG}_raw.json" || true

done

# Total expected: (288 + 400 + 96) × 2 models = 1,568 cells
echo ""
echo "==> All v3.0 panels complete. Results in $OUTDIR/"
echo "==> Run: python3 scripts/summarize_v30_panel.py to build the paper table."

# Quick summary of all outputs
"$PY" -c "
import json, glob, pathlib
total = ok = fail = 0
for f in sorted(glob.glob('$OUTDIR/*_raw.json')):
    try:
        d = json.loads(pathlib.Path(f).read_text())
        n = d.get('cells_run', 0)
        f_ok = d.get('cells_ok', n)
        total += n
        ok += f_ok
        fail += d.get('exactkv_failures', 0)
        print(f'  {pathlib.Path(f).name}: {n} cells, {f_ok} ok, exactkv_failures={d.get(\"exactkv_failures\",0)}')
    except Exception as e:
        print(f'  {pathlib.Path(f).name}: parse error {e}')
print(f'TOTAL: {total} cells, exactkv_failures={fail}')
" 2>/dev/null || true
