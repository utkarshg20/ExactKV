#!/usr/bin/env bash
# KIVI external panel runbook — RunPod A5000 (targeted: real HF LongBench + MBPP HF).
#
# PREREQUISITES (read carefully before running):
#   1. Clone jy-yuan/KIVI and export PYTHONPATH:
#        git clone https://github.com/jy-yuan/KIVI.git /tmp/kivi_research
#        export PYTHONPATH=/tmp/kivi_research
#   2. Ensure datasets==3.6.0 is installed:
#        pip install "datasets==3.6.0"
#   3. At least 40 GB free on the volume (Llama-3.1-8B ~16 GB weights).
#   4. Run from the ExactKV repo root, or set EXACTKV_ROOT.
#
# Panel design:
#   - LongBench: real HF subset (THUDM/LongBench), 10 subsets × 2 examples × 3 compressors
#     (noop, int8, int4_sim) + kivi_offline × 2 buckets × 2 mnt → ~240 cells
#   - MBPP: real HF subset (google-research-datasets/mbpp), 20 prompts × 4 compressors
#     × 2 buckets × 2 mnt → ~320 cells
#   Both Llama-3.1-8B only (one model to stay within 50 GB disk).
#
# Claim boundary: kivi_offline uses real KIVI quantizer math (models.utils_quant
# simulate path, is_simulated=False) but supports_real_bytes_claim=False.
# NOT KIVI production CUDA/Triton serving. Token-drift measurement only.
#
# Outputs:
#   reports/external_panels/kivi_longbench_hf_raw.json
#   reports/external_panels/kivi_mbpp_hf_raw.json

set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

PY="${PYTHON:-/workspace/.venv-runpod/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

LOG_DIR="reports/external_panels/logs"
mkdir -p "$LOG_DIR" reports/external_panels

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="$LOG_DIR/kivi_workflow_manifest_${STAMP}.jsonl"
: > "$MANIFEST"

log_step() {
  local name="$1" status="$2" detail="$3"
  "$PY" -c "
import json, datetime
print(json.dumps({
  'step': '$name', 'status': '$status', 'detail': '$detail',
  'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()
}))
" >> "$MANIFEST"
}

run_panel() {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}_${STAMP}.log"
  echo "==> [$name] $*"
  { echo "==> [$name] $(date -u -Iseconds)"; echo "CMD: $PY scripts/run_external_panel.py $*"; } | tee "$logfile"
  if "$PY" scripts/run_external_panel.py "$@" 2>&1 | tee -a "$logfile"; then
    log_step "$name" "ok" "log=$logfile"
  else
    local rc=$?
    log_step "$name" "failed" "exit=$rc log=$logfile"
    echo "==> [$name] FAILED (exit $rc), continuing" | tee -a "$logfile"
  fi
}

echo "==> KIVI external panel workflow start $STAMP"
echo "==> EXACTKV_ROOT=$ROOT"

# --------------------------------------------------------------------------
# Step 0: Environment checks — fail fast before any GPU work
# --------------------------------------------------------------------------
echo ""
echo "==> [check] CUDA availability"
"$PY" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo "==> [check] KIVI models.utils_quant"
if ! "$PY" -c "import models.utils_quant; print('kivi: ok, path:', models.utils_quant.__file__)" 2>/dev/null; then
  echo "ERROR: KIVI models.utils_quant not importable."
  echo "  Fix: git clone https://github.com/jy-yuan/KIVI.git /tmp/kivi_research"
  echo "       export PYTHONPATH=/tmp/kivi_research:\$PYTHONPATH"
  echo "  Then re-run this script."
  exit 1
fi
log_step "check_kivi" "ok" "models.utils_quant importable"

echo "==> [check] datasets availability"
if ! "$PY" -c "import datasets; print('datasets:', datasets.__version__)" 2>/dev/null; then
  echo "ERROR: datasets not installed."
  echo "  Fix: pip install 'datasets==3.6.0'"
  exit 1
fi
DATASETS_VER=$("$PY" -c "import datasets; print(datasets.__version__)")
echo "==> datasets version: $DATASETS_VER"
log_step "check_datasets" "ok" "version=$DATASETS_VER"

echo "==> [check] huggingface_hub"
"$PY" -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)" || {
  echo "WARNING: huggingface_hub not installed; install if needed: pip install huggingface_hub"
}

# --------------------------------------------------------------------------
# Step 1: Export real HF LongBench subset to JSONL
# --------------------------------------------------------------------------
LB_EXPORT="benchmarks/prompts/longbench_hf_export.jsonl"
echo ""
echo "==> [longbench_hf_export] Exporting real HF LongBench subset → $LB_EXPORT"
LB_LOG="$LOG_DIR/longbench_hf_export_${STAMP}.log"

if "$PY" scripts/export_longbench_subset.py \
    --max-per-subset 2 \
    --output "$LB_EXPORT" \
    2>&1 | tee "$LB_LOG"; then
  NLINES=$(wc -l < "$LB_EXPORT" || echo 0)
  echo "==> Exported $NLINES prompts to $LB_EXPORT"
  log_step "longbench_hf_export" "ok" "lines=$NLINES log=$LB_LOG"
else
  echo "==> [longbench_hf_export] FAILED — check $LB_LOG"
  log_step "longbench_hf_export" "failed" "log=$LB_LOG"
  echo "Cannot proceed without real LongBench prompts. Exiting."
  exit 1
fi

# --------------------------------------------------------------------------
# Step 2: LongBench HF panel — noop + int8 + int4_sim + kivi_offline
# --------------------------------------------------------------------------
echo ""
echo "==> [kivi_longbench_hf] Running real HF LongBench panel with kivi_offline"
run_panel "kivi_longbench_hf" \
  --family longbench \
  --prompt-source hf \
  --device cuda --dtype float16 \
  --models "meta-llama/Llama-3.1-8B" \
  --compressors "noop,int8,int4_sim,kivi_offline" \
  --context-buckets 2048,4096 \
  --max-new-tokens 16,32 \
  --max-prompts 20 \
  --output-json "reports/external_panels/kivi_longbench_hf_raw.json"

# --------------------------------------------------------------------------
# Step 3: MBPP HF panel — real subset with kivi_offline
# --------------------------------------------------------------------------
echo ""
echo "==> [kivi_mbpp_hf] Running real HF MBPP panel with kivi_offline"
run_panel "kivi_mbpp_hf" \
  --family mbpp \
  --prompt-source hf \
  --device cuda --dtype float16 \
  --models "meta-llama/Llama-3.1-8B" \
  --compressors "noop,int8,int4_sim,kivi_offline" \
  --context-buckets 512,1024 \
  --max-new-tokens 16,32 \
  --max-prompts 20 \
  --output-json "reports/external_panels/kivi_mbpp_hf_raw.json"

# --------------------------------------------------------------------------
# Step 4: Validate all artifacts
# --------------------------------------------------------------------------
echo ""
echo "==> [validate] Running artifact validator"
VAL_LOG="$LOG_DIR/kivi_validate_${STAMP}.log"
if "$PY" scripts/validate_external_panel_artifacts.py \
    --input reports/external_panels \
    2>&1 | tee "$VAL_LOG"; then
  log_step "validate" "ok" "log=$VAL_LOG"
else
  log_step "validate" "failed" "log=$VAL_LOG"
  echo "==> Validation reported failures — review $VAL_LOG"
fi

# --------------------------------------------------------------------------
# Step 5: Summary report
# --------------------------------------------------------------------------
echo ""
echo "==> [summary] Building external panel summary"
"$PY" scripts/build_external_panel_summary.py --write-readme 2>&1 || \
  echo "==> build_external_panel_summary failed (non-fatal)"

echo ""
echo "==> KIVI workflow complete. Manifest: $MANIFEST"
echo ""
echo "==> Key output files:"
echo "    reports/external_panels/kivi_longbench_hf_raw.json"
echo "    reports/external_panels/kivi_mbpp_hf_raw.json"
echo "    reports/external_panels/validation_report.json"
echo ""
echo "==> Next steps:"
echo "    1. Paste kivi_longbench_hf_raw.json and kivi_mbpp_hf_raw.json back to local ExactKV."
echo "    2. Run locally: python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels"
echo "    3. Run locally: python3 scripts/build_external_analysis_pack.py"
echo "    4. The assistant will integrate numbers into the paper (v2.5 bump)."
