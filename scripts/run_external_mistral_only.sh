#!/usr/bin/env bash
# Mistral-only external GPU panels (Option A: clear Llama cache first).
# Use when RunPod workspace quota cannot hold Llama + Mistral weights together.
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

PY="${PYTHON:-/workspace/.venv-runpod/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

MODEL="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG//./_}"
MODEL_TAG="${MODEL_TAG//-/_}"

LOG_DIR="reports/external_panels/logs"
mkdir -p "$LOG_DIR" reports/external_panels

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="$LOG_DIR/mistral_only_manifest_${STAMP}.jsonl"
MASTER_LOG="$LOG_DIR/mistral_only_${STAMP}.log"
: > "$MANIFEST"

exec > >(tee -a "$MASTER_LOG") 2>&1

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
    echo "==> [$name] FAILED (exit $rc), continuing"
    return $rc
  fi
}

echo "==> Mistral-only external workflow start $STAMP"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo "==> Clearing Llama HF cache (Option A)"
rm -rf /workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B
echo "==> HF cache after cleanup:"
du -sh /workspace/.cache/huggingface 2>/dev/null || echo "(empty)"
df -h /workspace 2>/dev/null || true

run_panel "longbench_pilot_${MODEL_TAG}" \
  --family longbench --device cuda --dtype float16 \
  --max-prompts 6 --context-buckets 2048,4096 --max-new-tokens 16,32 \
  --models "$MODEL" \
  --output-json "reports/external_panels/longbench_pilot_${MODEL_TAG}_raw.json" || true

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

echo "==> Merging artifacts (includes Llama + Mistral if present)"
"$PY" scripts/build_external_panel_summary.py --write-readme || true

echo "==> Mistral-only workflow complete. Manifest: $MANIFEST"
echo "==> Master log: $MASTER_LOG"
