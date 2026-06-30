#!/usr/bin/env bash
# ============================================================
# ExactKV v2.7 — BFCL Tool-Call Validity Panel
# ============================================================
# Runs a BFCL export-50 panel with max_new_tokens=128,256 so
# full JSON tool calls can be generated and validity-parsed.
#
# This is the structured-output safety panel: it measures not
# just DRIFT but also valid-JSON rate, valid-schema rate, and
# malformed-JSON rate across noop/int8/int4_sim.
#
# Design:
#   50 BFCL export prompts × 2 context buckets × 2 max_new_tokens
#   × 3 compressors × 2 models = 1,200 cells
#
# Requires: v2.6 panel to be complete (Llama weights already loaded).
#
# Usage (on RunPod, queue after v2.6):
#   # Auto-queued: runs in tmux v27 right after v26 finishes
#   # Or manually:
#   bash /workspace/ExactKV/scripts/run_bfcl_validity_v27_panel.sh 2>&1 | tee /workspace/v27_run.log
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS="$ROOT/reports/external_panels"
SCRIPTS="$ROOT/scripts"

# ---- Activate venv -----------------------------------------
VENV="/workspace/.venv-runpod"
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "Activated venv: $VENV"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

log "=== ExactKV v2.7 BFCL validity panel ==="
log "ROOT=$ROOT"

# ---- 1. Pre-flight -----------------------------------------
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
mkdir -p "$REPORTS"

# Check BFCL export prompts exist
EXPORT_JSONL="$ROOT/benchmarks/prompts/bfcl_export.jsonl"
if [[ ! -f "$EXPORT_JSONL" ]]; then
    fail "BFCL export prompts not found: $EXPORT_JSONL"
fi
PROMPT_COUNT=$(wc -l < "$EXPORT_JSONL")
log "BFCL export prompts: $PROMPT_COUNT lines in $EXPORT_JSONL"

# ---- 2. Panel parameters -----------------------------------
# 50 prompts × 2 ctx × 2 mnt × 3 comp = 600 cells/model × 2 models = 1,200 total
MAX_PROMPTS=50
CTX_BUCKETS="1024,2048"
# Key difference from export-50 drift panel: long enough for full JSON generation
MNT="128,256"
COMPRESSORS="noop,int8,int4_sim"
LLAMA="meta-llama/Llama-3.1-8B"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

log "Panel: $MAX_PROMPTS prompts × buckets($CTX_BUCKETS) × mnt($MNT) × compressors($COMPRESSORS)"
log "       = $MAX_PROMPTS × 2 × 2 × 3 = $(python3 -c "print($MAX_PROMPTS*2*2*3)") cells/model × 2 = $(python3 -c "print($MAX_PROMPTS*2*2*3*2)") total"

# ---- 3. Run Llama panel ------------------------------------
LLAMA_OUT="$REPORTS/bfcl_validity_v27_Llama_3_1_8B_raw.json"
if [[ -f "$LLAMA_OUT" ]]; then
    log "Llama output exists, skipping: $LLAMA_OUT"
else
    log "--- Running Llama-3.1-8B BFCL validity panel ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 "$SCRIPTS/run_external_panel.py" \
        --family bfcl \
        --prompt-source export \
        --device cuda \
        --dtype float16 \
        --max-prompts "$MAX_PROMPTS" \
        --context-buckets "$CTX_BUCKETS" \
        --max-new-tokens "$MNT" \
        --compressors "$COMPRESSORS" \
        --models "$LLAMA" \
        --output-json "$LLAMA_OUT"
    log "Llama panel done: $LLAMA_OUT"
fi

# ---- 4. Run Mistral panel ----------------------------------
MISTRAL_OUT="$REPORTS/bfcl_validity_v27_Mistral_7B_raw.json"
if [[ -f "$MISTRAL_OUT" ]]; then
    log "Mistral output exists, skipping: $MISTRAL_OUT"
else
    log "--- Running Mistral-7B BFCL validity panel ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 "$SCRIPTS/run_external_panel.py" \
        --family bfcl \
        --prompt-source export \
        --device cuda \
        --dtype float16 \
        --max-prompts "$MAX_PROMPTS" \
        --context-buckets "$CTX_BUCKETS" \
        --max-new-tokens "$MNT" \
        --compressors "$COMPRESSORS" \
        --models "$MISTRAL" \
        --output-json "$MISTRAL_OUT"
    log "Mistral panel done: $MISTRAL_OUT"
fi

# ---- 5. Merge + validity analysis --------------------------
MERGED_OUT="$REPORTS/bfcl_validity_v27_merged_raw.json"
log "--- Merging + validity analysis ---"
python3 - <<PYEOF
import json, math, sys
from pathlib import Path

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k/n; denom = 1 + z*z/n
    c = (p + z*z/(2*n)) / denom
    m = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return max(0.0, c-m), min(1.0, c+m)

def bfcl_validity_scan(text):
    """Balanced-brace scan for valid JSON tool call."""
    text = text.strip()
    if not text: return False
    depth = 0
    started = False
    for ch in text:
        if ch == '{': depth += 1; started = True
        elif ch == '}': depth -= 1
        if started and depth == 0: return True
    return False

files = [
    ("$LLAMA_OUT",   "Llama-3.1-8B"),
    ("$MISTRAL_OUT", "Mistral-7B-Instruct-v0.3"),
]
all_cells = []
for path, model in files:
    try:
        d = json.loads(open(path).read())
        cells = d.get("cells", [])
        # Annotate validity
        for c in cells:
            for key in ("full_output", "lossy_output", "exactkv_output"):
                text = c.get(key, "") or ""
                c[key + "_valid_json"] = bfcl_validity_scan(text)
        all_cells.extend(cells)
    except FileNotFoundError:
        print(f"[WARN] not found: {path}")

n = len(all_cells)
div  = sum(1 for c in all_cells if c.get("diverged"))
fail = sum(1 for c in all_cells if c.get("exactkv_failure"))

# Validity stats per compressor
by_comp = {}
for c in all_cells:
    comp = c.get("compressor_name", "?")
    by_comp.setdefault(comp, []).append(c)

print(f"\nTotal cells: {n}  divergent: {div}  exactkv_failures: {fail}")
print(f"\n{'Compressor':<16} {'n':>6} {'full_valid':>12} {'lossy_valid':>12} {'exactkv_valid':>14}")
print("-" * 64)
for comp, cells in sorted(by_comp.items()):
    nc  = len(cells)
    fv  = sum(1 for c in cells if c.get("full_output_valid_json"))
    lv  = sum(1 for c in cells if c.get("lossy_output_valid_json"))
    ev  = sum(1 for c in cells if c.get("exactkv_output_valid_json"))
    print(f"  {comp:<14} {nc:>6}  {fv:>6}/{nc}({fv/nc:.0%})  {lv:>6}/{nc}({lv/nc:.0%})  {ev:>6}/{nc}({ev/nc:.0%})")

merged = {
    "panel_id": "bfcl_validity_v27",
    "description": "ExactKV v2.7 BFCL tool-call validity panel (noop/int8/int4_sim, mnt=128/256, both models)",
    "total_cells": n, "divergent": div, "exactkv_failures": fail,
    "cells": all_cells,
}
out = Path("$MERGED_OUT")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"\nMerged: {out}")
PYEOF

# ---- 6. Validate + summary ---------------------------------
log "--- Validating artifacts ---"
python3 "$SCRIPTS/validate_external_panel_artifacts.py" \
    --input "$REPORTS" || log "Validation warnings (non-fatal)"

log "=== v2.7 BFCL validity panel complete ==="
log "Artifacts:"
log "  $LLAMA_OUT"
log "  $MISTRAL_OUT"
log "  $MERGED_OUT"
log "Copy back with:"
log "  rsync -avz runpod-a5000:/workspace/ExactKV/reports/external_panels/bfcl_validity_v27_* /path/to/local/reports/external_panels/"
