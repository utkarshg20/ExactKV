#!/usr/bin/env bash
# ============================================================
# ExactKV v2.6 — Real HF LongBench Drift Panel
# ============================================================
# Runs a 720-cell drift panel using REAL Hugging Face LongBench
# examples (not bundled pilot prompts), both models, 2K/4K/8K
# context buckets, max_new_tokens 32/64, noop/int8/int4_sim.
#
# Stores top-k logits on divergent cells for v2.7 forensic analysis.
#
# Expected runtime: ~2–3 hours total (tmux recommended).
# GPU memory: tested on RTX A5000 24 GB; 8K context uses ~20 GB.
#
# Usage (on RunPod):
#   tmux new -s v26
#   bash /workspace/ExactKV/scripts/run_hf_longbench_v26_panel.sh 2>&1 | tee /workspace/v26_run.log
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS="$ROOT/reports/external_panels"
SCRIPTS="$ROOT/scripts"

# ---- Activate venv with transformers/datasets/torch -----------
VENV="/workspace/.venv-runpod"
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "Activated venv: $VENV (python=$(which python3))"
else
    echo "[WARN] $VENV not found — using system Python. Install datasets/transformers if needed."
fi

# ---- Helpers -----------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

# ---- 1. Pre-flight checks ----------------------------------
log "=== ExactKV v2.6 HF LongBench drift panel ==="
log "ROOT=$ROOT"

command -v python3 >/dev/null 2>&1 || fail "python3 not found"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
    print(f'CUDA OK: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory//1024**3} GB')"

python3 -c "
import importlib, importlib.util
missing=[]
for pkg in ['datasets','transformers','torch']:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)
if missing:
    print('MISSING:', missing)
    exit(1)
print('All required packages present.')
"

log "Checking HF LongBench connectivity (small probe)..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('THUDM/LongBench', 'narrativeqa', split='test', trust_remote_code=True)
print(f'HF LongBench probe OK: {len(ds)} examples in narrativeqa/test')
" || fail "Cannot reach THUDM/LongBench. Check network/HF_TOKEN."

mkdir -p "$REPORTS"
log "Pre-flight complete."

# ---- 2. Panel parameters -----------------------------------
#  10 LongBench subsets x 2 examples each = 20 prompts
#  3 context buckets x 2 max_new_tokens x 3 compressors = 18 cells/prompt
#  2 models = 720 cells total
SUBSETS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,trec,samsum,lcc,passage_retrieval_en"
MAX_PROMPTS=20
CTX_BUCKETS="2048,4096,8192"
MNT="32,64"
COMPRESSORS="noop,int8,int4_sim"
LLAMA="meta-llama/Llama-3.1-8B"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

log "Panel: $MAX_PROMPTS prompts × buckets($CTX_BUCKETS) × mnt($MNT) × compressors($COMPRESSORS)"
log "       = $MAX_PROMPTS × 3 × 2 × 3 = $(python3 -c "print($MAX_PROMPTS*3*2*3)") cells/model"
log "       × 2 models = $(python3 -c "print($MAX_PROMPTS*3*2*3*2)") cells total"

# ---- 3. Run Llama panel ------------------------------------
LLAMA_OUT="$REPORTS/hf_longbench_v26_Llama_3_1_8B_raw.json"
if [[ -f "$LLAMA_OUT" ]]; then
    log "Llama output already exists, skipping. Delete to rerun: $LLAMA_OUT"
else
    log "--- Running Llama-3.1-8B panel ---"
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
MISTRAL_OUT="$REPORTS/hf_longbench_v26_Mistral_7B_raw.json"
if [[ -f "$MISTRAL_OUT" ]]; then
    log "Mistral output already exists, skipping. Delete to rerun: $MISTRAL_OUT"
else
    log "--- Running Mistral-7B panel ---"
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

# ---- 5. Merge Llama + Mistral results ----------------------
MERGED_OUT="$REPORTS/hf_longbench_v26_merged_raw.json"
log "--- Merging Llama + Mistral results ---"
python3 "$SCRIPTS/merge_hf_longbench_v26.py" \
    --llama "$LLAMA_OUT" \
    --mistral "$MISTRAL_OUT" \
    --output "$MERGED_OUT"
log "Merged: $MERGED_OUT"

# ---- 6. Validate artifacts ---------------------------------
log "--- Validating artifacts ---"
python3 "$SCRIPTS/validate_external_panel_artifacts.py" \
    --input "$REPORTS" || log "Validation warnings (non-fatal)"

# ---- 7. Build analysis pack --------------------------------
log "--- Building analysis pack ---"
python3 "$SCRIPTS/build_external_analysis_pack.py" \
    > "$REPORTS/analysis_pack_v26.json" || log "Analysis pack warning (non-fatal)"

# ---- 8. Print summary --------------------------------------
log "=== v2.6 panel complete ==="
python3 - <<'EOF'
import json, sys

files = [
    ("Llama",   "reports/external_panels/hf_longbench_v26_Llama_3_1_8B_raw.json"),
    ("Mistral", "reports/external_panels/hf_longbench_v26_Mistral_7B_raw.json"),
]
total_cells = 0
total_div   = 0
total_fail  = 0
for label, path in files:
    try:
        d = json.loads(open(path).read())
    except FileNotFoundError:
        print(f"  {label}: NOT FOUND")
        continue
    cells   = d.get("total_cells", "?")
    div     = d.get("total_divergent_cells", "?")
    fail    = d.get("exactkv_failures", "?")
    accept  = d.get("full_acceptance_rate", "?")
    if isinstance(cells, int): total_cells += cells
    if isinstance(div,  int): total_div   += div
    if isinstance(fail, int): total_fail  += fail
    print(f"  {label}: cells={cells}  divergent={div}  exactkv_failures={fail}  accept={accept}")
print(f"  TOTAL:  cells={total_cells}  divergent={total_div}  exactkv_failures={total_fail}")
EOF

log "Artifacts ready — copy back with:"
log "  rsync -avz runpod-a5000:/workspace/ExactKV/reports/external_panels/hf_longbench_v26_* /path/to/local/reports/external_panels/"
