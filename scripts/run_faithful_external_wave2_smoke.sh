#!/usr/bin/env bash
# Wave-2 faithful external smoke: compare int8 vs KnormPress vs TurboQuant vs SnapKV.
#
# Goal: find a faithful external compressor with non-catastrophic drift on structured tasks.
# KnormPress (Exp 005) had ~0.97 acceptance on core suite — best external candidate.
#
# Prerequisites (RunPod):
#   bash scripts/setup_faithful_compressor_env.sh
#   # TurboQuant optional:
#   git clone --depth 1 https://github.com/TheTom/turboquant_plus /tmp/turboquant_plus
#   export EXACTKV_TURBOQUANT_ROOT=/tmp/turboquant_plus
#
# Usage:
#   FAITHFUL_WAVE2_MODEL='mistralai/Mistral-7B-Instruct-v0.3' \
#     bash scripts/run_faithful_external_wave2_smoke.sh
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
RUNDIR="${EXACTKV_RUNDIR:-/tmp/exactkv_wave2_run}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

PY="${PYTHON:-/workspace/.venv-faithful/bin/python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"
export EXACTKV_SNAPKV_ISOLATE="${EXACTKV_SNAPKV_ISOLATE:-0}"
export EXACTKV_KVPRESS_ISOLATE="${EXACTKV_KVPRESS_ISOLATE:-0}"
if [[ -n "${EXACTKV_TURBOQUANT_ROOT:-}" ]]; then
  export PYTHONPATH="${EXACTKV_TURBOQUANT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
fi
rm -rf "$ROOT/platform" 2>/dev/null || true

MODEL="${FAITHFUL_WAVE2_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG//./_}"
MODEL_TAG="${MODEL_TAG//-/_}"

OUTDIR="$ROOT/reports/external_panels/faithful/wave2"
mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUTDIR/wave2_smoke_${MODEL_TAG}_${STAMP}.log"

COMPRESSORS="int8,kvpress_knorm_experimental,turboquant_experimental,snapkv_experimental"

echo "==> Wave-2 external smoke $STAMP model=$MODEL" | tee "$LOG"
echo "==> compressors=$COMPRESSORS" | tee -a "$LOG"

"$PY" -c "
from exactkv.benchmarks.evidence_plus_panel import resolve_evidence_plus_compressor
for c in '$COMPRESSORS'.split(','):
    r = resolve_evidence_plus_compressor(c)
    print(c, r.backend_tier, r.adapter_available)
" 2>&1 | tee -a "$LOG"

for FAMILY in mbpp bfcl; do
  EXTRA=()
  if [[ "$FAMILY" == "mbpp" ]]; then
    EXTRA=(--prompt-source hf --max-prompts 4 --context-buckets 512,1024 --max-new-tokens 16,32)
  else
    EXTRA=(--max-prompts 4 --context-buckets 512,1024 --max-new-tokens 16,32)
  fi
  OUT="$OUTDIR/${FAMILY}_${MODEL_TAG}_wave2_smoke_raw.json"
  echo "==> [$FAMILY] -> $OUT" | tee -a "$LOG"
  RESUME=()
  [[ -f "$OUT" ]] && RESUME=(--resume-json "$OUT")
  "$PY" "$ROOT/scripts/run_external_panel.py" \
    --family "$FAMILY" \
    --device cuda --dtype float16 \
    --compressors "$COMPRESSORS" \
    --models "$MODEL" \
    --output-json "$OUT" \
    --checkpoint-json "$OUT" \
    "${RESUME[@]}" \
    "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
done

echo "WAVE2_SMOKE_DONE $STAMP" | tee -a "$LOG"
"$PY" "$ROOT/scripts/integrate_faithful_panel_results.py" --dir "$OUTDIR" --write 2>&1 | tee -a "$LOG" || true
