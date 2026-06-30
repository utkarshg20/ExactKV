#!/usr/bin/env bash
# Conservative GPU external benchmark workflow (RunPod A5000).
# Logs to reports/external_panels/logs/. Continues on per-step failure.
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

PY="${PYTHON:-/workspace/.venv-runpod/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

LOG_DIR="reports/external_panels/logs"
mkdir -p "$LOG_DIR" reports/external_panels

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="$LOG_DIR/workflow_manifest_${STAMP}.jsonl"
: > "$MANIFEST"

log_step() {
  local name="$1" status="$2" detail="$3"
  "$PY" -c "import json,datetime; print(json.dumps({'step':'''$name''','status':'''$status''','detail':'''$detail''','ts':datetime.datetime.now(datetime.timezone.utc).isoformat()}))" >> "$MANIFEST"
}

run_panel() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/${name}_${STAMP}.log"
  echo "==> [$name] $*"
  {
    echo "==> [$name] $(date -u -Iseconds)"
    echo "CMD: $PY scripts/run_external_panel.py $*"
  } | tee "$logfile"
  if "$PY" scripts/run_external_panel.py "$@" 2>&1 | tee -a "$logfile"; then
    log_step "$name" "ok" "log=$logfile"
    return 0
  else
    local rc=$?
    log_step "$name" "failed" "exit=$rc log=$logfile"
    echo "==> [$name] FAILED (exit $rc), continuing" | tee -a "$logfile"
    return $rc
  fi
}

echo "==> External GPU workflow start $STAMP"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

MODELS_LLAMA="meta-llama/Llama-3.1-8B"
MODELS_MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

for MODEL in "$MODELS_LLAMA" "$MODELS_MISTRAL"; do
  MODEL_TAG="${MODEL##*/}"
  MODEL_TAG="${MODEL_TAG//./_}"
  MODEL_TAG="${MODEL_TAG//-/_}"

  run_panel "longbench_pilot_${MODEL_TAG}" \
    --family longbench --device cuda --dtype float16 \
    --max-prompts 6 --context-buckets 2048,4096 --max-new-tokens 16,32 \
    --models "$MODEL" \
    --output-json "reports/external_panels/longbench_pilot_${MODEL_TAG}_raw.json" || true

  if "$PY" -c "import datasets" 2>/dev/null; then
    EXPORT_LOG="$LOG_DIR/longbench_hf_export_${MODEL_TAG}_${STAMP}.log"
    if "$PY" scripts/export_longbench_subset.py --max-per-subset 2 \
      --output "benchmarks/prompts/longbench_export.jsonl" >"$EXPORT_LOG" 2>&1; then
      log_step "longbench_hf_export_${MODEL_TAG}" "ok" "log=$EXPORT_LOG"
      run_panel "longbench_hf_${MODEL_TAG}" \
        --family longbench --prompt-source hf --device cuda --dtype float16 \
        --max-prompts 12 --context-buckets 2048,4096 --max-new-tokens 16,32 \
        --models "$MODEL" \
        --output-json "reports/external_panels/longbench_hf_${MODEL_TAG}_raw.json" || true
    else
      log_step "longbench_hf_export_${MODEL_TAG}" "skipped" "export failed log=$EXPORT_LOG"
    fi
  else
    log_step "longbench_hf_${MODEL_TAG}" "skipped" "datasets not installed"
  fi

  run_panel "ruler_2048_4096_${MODEL_TAG}" \
    --family ruler --device cuda --dtype float16 \
    --context-buckets 2048,4096 --max-new-tokens 16,32 \
    --models "$MODEL" \
    --output-json "reports/external_panels/ruler_2048_4096_${MODEL_TAG}_raw.json" || true

  RULER_PREV="reports/external_panels/ruler_2048_4096_${MODEL_TAG}_raw.json"
  if [[ -f "$RULER_PREV" ]]; then
    if "$PY" -c "import json; r=json.load(open('$RULER_PREV')); exit(0 if r.get('cells_run',0)>0 else 1)"; then
      run_panel "ruler_8192_${MODEL_TAG}" \
        --family ruler --device cuda --dtype float16 \
        --context-buckets 8192 --max-new-tokens 16,32 \
        --models "$MODEL" \
        --output-json "reports/external_panels/ruler_8192_${MODEL_TAG}_raw.json" || true
    else
      log_step "ruler_8192_${MODEL_TAG}" "skipped" "prior ruler panel had zero ok cells"
    fi
  fi

  run_panel "bfcl_${MODEL_TAG}" \
    --family bfcl --device cuda --dtype float16 \
    --max-prompts 25 --context-buckets 1024,2048 --max-new-tokens 16,32 \
    --models "$MODEL" \
    --output-json "reports/external_panels/bfcl_${MODEL_TAG}_raw.json" || true

  run_panel "humaneval_${MODEL_TAG}" \
    --family humaneval --device cuda --dtype float16 \
    --max-prompts 20 --context-buckets 1024,2048 --max-new-tokens 32 \
    --models "$MODEL" \
    --output-json "reports/external_panels/humaneval_${MODEL_TAG}_raw.json" || true
done

echo "==> Merging per-model artifacts"
"$PY" scripts/build_external_panel_summary.py --write-readme || true

echo "==> Workflow complete. Manifest: $MANIFEST"
