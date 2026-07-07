#!/usr/bin/env bash
# Backfill Mistral LongBench int8 cells only — preserves existing turboquant cells.
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"
OUTDIR="$ROOT/reports/external_panels/faithful/wave3"
OUT="$OUTDIR/longbench_Mistral_7B_Instruct_v0_3_wave3_raw.json"
LOG="$ROOT/reports/faithful_wave3_mistral_int8_backfill_$(date -u +%Y%m%dT%H%M%SZ).log"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_HOME/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ ! -f "$OUT" ]]; then
  echo "ERROR: missing $OUT — sync turboquant-complete file before int8 backfill" | tee -a "$LOG"
  exit 1
fi

tq_ok=$("$PY" -c "
import json
d=json.load(open('$OUT'))
print(sum(1 for c in d.get('cells',[]) if c.get('status')=='ok' and c.get('compressor_name')=='turboquant_experimental'))
" 2>/dev/null || echo 0)

if [[ "$tq_ok" != "72" ]]; then
  echo "WARN: expected 72 turboquant ok cells, found $tq_ok" | tee -a "$LOG"
fi

resume=(--resume-json "$OUT" --checkpoint-json "$OUT")

echo "==> Mistral LongBench int8-only backfill -> $OUT (turboquant ok=$tq_ok)" | tee -a "$LOG"
"$PY" "$ROOT/scripts/run_external_panel.py" \
  --family longbench \
  --prompt-source hf \
  --device cuda --dtype float16 \
  --max-prompts 12 \
  --context-buckets 2048,4096,8192 \
  --max-new-tokens 16,32 \
  --compressors int8 \
  --models mistralai/Mistral-7B-Instruct-v0.3 \
  "${resume[@]}" \
  --output-json "$OUT" 2>&1 | tee -a "$LOG"

echo "WAVE3_MISTRAL_INT8_BACKFILL_DONE $(date -u -Iseconds)" | tee -a "$LOG"
