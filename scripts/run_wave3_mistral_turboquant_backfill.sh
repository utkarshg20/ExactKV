#!/usr/bin/env bash
# Backfill Mistral LongBench turboquant cells only — do not touch Llama panel.
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"
OUTDIR="$ROOT/reports/external_panels/faithful/wave3"
OUT="$OUTDIR/longbench_Mistral_7B_Instruct_v0_3_wave3_raw.json"
LOG="$ROOT/reports/faithful_wave3_mistral_tq_backfill_$(date -u +%Y%m%dT%H%M%SZ).log"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export EXACTKV_TURBOQUANT_ROOT="${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}"
export PYTHONPATH="${EXACTKV_TURBOQUANT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_HOME/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ "${INSTALL_TURBOQUANT:-1}" == "1" ]]; then
  INSTALL_TURBOQUANT=1 bash "$ROOT/scripts/setup_faithful_compressor_env.sh" >>"$LOG" 2>&1 || true
fi

if ! "$PY" -c "import turboquant" 2>/dev/null; then
  echo "ERROR: turboquant not importable — set EXACTKV_TURBOQUANT_ROOT and INSTALL_TURBOQUANT=1" | tee -a "$LOG"
  exit 1
fi

resume=()
[[ -f "$OUT" ]] && resume=(--resume-json "$OUT" --checkpoint-json "$OUT")

echo "==> Mistral LongBench turboquant-only backfill -> $OUT" | tee -a "$LOG"
"$PY" "$ROOT/scripts/run_external_panel.py" \
  --family longbench \
  --prompt-source hf \
  --device cuda --dtype float16 \
  --max-prompts 12 \
  --context-buckets 2048,4096,8192 \
  --max-new-tokens 16,32 \
  --compressors turboquant_experimental \
  --models mistralai/Mistral-7B-Instruct-v0.3 \
  "${resume[@]}" \
  --checkpoint-json "$OUT" \
  --output-json "$OUT" 2>&1 | tee -a "$LOG"

echo "WAVE3_MISTRAL_TQ_BACKFILL_DONE $(date -u -Iseconds)" | tee -a "$LOG"
