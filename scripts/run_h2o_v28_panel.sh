#!/usr/bin/env bash
# ============================================================
# ExactKV v2.8 — H2O Token-Eviction Compressor Panel
# ============================================================
# Runs a drift panel comparing H2O-style token eviction vs.
# quantization compressors (noop, int8, int4_sim) on both models.
#
# H2O-sim uses the attention sink + recency window approximation
# (keep_ratio=0.5 by default; also runs keep_ratio=0.25 and 0.75).
#
# Design:
#   20 HF LongBench prompts × 2 ctx × 2 mnt × 5 comp × 2 models = 800 cells
#
# Expected runtime: ~2 hours. Requires v2.6 artifacts (uses same model weights).
#
# Usage (on RunPod, queue after v2.7):
#   bash /workspace/ExactKV/scripts/run_h2o_v28_panel.sh 2>&1 | tee /workspace/v28_run.log
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

log "=== ExactKV v2.8 H2O token-eviction panel ==="
log "ROOT=$ROOT"

# ---- 1. Pre-flight -----------------------------------------
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from exactkv.compressors import get_compressor, list_compressors
assert 'h2o_sim' in list_compressors(), 'h2o_sim not in registry — sync scripts first'
c = get_compressor('h2o_sim')
print('h2o_sim OK: keep_ratio=' + str(c.keep_ratio if hasattr(c,'keep_ratio') else c().keep_ratio))
"
mkdir -p "$REPORTS"
log "Pre-flight complete."

# ---- 2. Panel parameters -----------------------------------
#  h2o_sim (0.5), h2o_sim_25 (0.25), h2o_sim_75 (0.75) + noop + int4_sim for comparison
SUBSETS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,trec,samsum,lcc,passage_retrieval_en"
MAX_PROMPTS=20
CTX_BUCKETS="2048,4096"
MNT="32,64"
COMPRESSORS="noop,int4_sim,h2o_sim,h2o_sim_75,h2o_sim_25"
LLAMA="meta-llama/Llama-3.1-8B"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

log "Panel: $MAX_PROMPTS prompts × buckets($CTX_BUCKETS) × mnt($MNT) × 5 compressors × 2 models"
log "       = $(python3 -c "print($MAX_PROMPTS*2*2*5*2)") cells total"

# ---- 3. Run Llama panel ------------------------------------
LLAMA_OUT="$REPORTS/h2o_v28_Llama_3_1_8B_raw.json"
if [[ -f "$LLAMA_OUT" ]]; then
    log "Llama output exists, skipping: $LLAMA_OUT"
else
    log "--- Running Llama-3.1-8B H2O panel ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 "$SCRIPTS/run_external_panel.py" \
        --family longbench \
        --prompt-source hf \
        --longbench-subsets "$SUBSETS" \
        --device cuda \
        --dtype float16 \
        --max-prompts "$MAX_PROMPTS" \
        --context-buckets "$CTX_BUCKETS" \
        --max-new-tokens "$MNT" \
        --compressors "$COMPRESSORS" \
        --models "$LLAMA" \
        --store-top-k-logits \
        --output-json "$LLAMA_OUT"
    log "Llama panel done: $LLAMA_OUT"
fi

# ---- 4. Run Mistral panel ----------------------------------
MISTRAL_OUT="$REPORTS/h2o_v28_Mistral_7B_raw.json"
if [[ -f "$MISTRAL_OUT" ]]; then
    log "Mistral output exists, skipping: $MISTRAL_OUT"
else
    log "--- Running Mistral-7B H2O panel ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 "$SCRIPTS/run_external_panel.py" \
        --family longbench \
        --prompt-source hf \
        --longbench-subsets "$SUBSETS" \
        --device cuda \
        --dtype float16 \
        --max-prompts "$MAX_PROMPTS" \
        --context-buckets "$CTX_BUCKETS" \
        --max-new-tokens "$MNT" \
        --compressors "$COMPRESSORS" \
        --models "$MISTRAL" \
        --store-top-k-logits \
        --output-json "$MISTRAL_OUT"
    log "Mistral panel done: $MISTRAL_OUT"
fi

# ---- 5. Print summary --------------------------------------
python3 - <<'EOF'
import json, math
from pathlib import Path

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k/n; denom = 1+z*z/n
    c = (p+z*z/(2*n))/denom
    m = (z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/denom
    return max(0.0,c-m), min(1.0,c+m)

for label, path in [("Llama","reports/external_panels/h2o_v28_Llama_3_1_8B_raw.json"),
                    ("Mistral","reports/external_panels/h2o_v28_Mistral_7B_raw.json")]:
    try:
        d = json.loads(open(path).read())
    except FileNotFoundError:
        print(f"  {label}: NOT FOUND"); continue
    cells = d.get("cells", [])
    by_comp = {}
    for c in cells:
        comp = c.get("compressor_name","?")
        by_comp.setdefault(comp,[]).append(c)
    print(f"\n{label}:")
    print(f"  {'Compressor':<16} {'n':>5}  {'Div':>4}  {'Rate':>7}  CI95")
    for comp in ("noop","int4_sim","h2o_sim_75","h2o_sim","h2o_sim_25"):
        cs = by_comp.get(comp, [])
        if not cs: continue
        div = sum(1 for c in cs if c.get("diverged"))
        n = len(cs)
        lo, hi = wilson_ci(div, n)
        print(f"  {comp:<16} {n:>5}  {div:>4}  {div/n:>6.1%}  [{lo:.1%}, {hi:.1%}]")
EOF

log "=== v2.8 H2O panel complete ==="
log "Copy back with:"
log "  rsync -avz runpod-a5000:/workspace/ExactKV/reports/external_panels/h2o_v28_* /path/to/local/reports/external_panels/"
