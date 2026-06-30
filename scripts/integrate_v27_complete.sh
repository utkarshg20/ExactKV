#!/usr/bin/env bash
# integrate_v27_complete.sh — Run after both v2.7 BFCL artifacts are on disk.
# Merges, post-processes, generates paper tables, and rebuilds PDF.
#
# Usage: bash scripts/integrate_v27_complete.sh [--write]
#
# Prerequisites:
#   reports/external_panels/bfcl_validity_v27_Llama_3_1_8B_raw.json   (already done)
#   reports/external_panels/bfcl_validity_v27_Mistral_7B_raw.json     (from RunPod)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WRITE_FLAG="${1:-}"

LOG() { echo "[$(date +'%H:%M:%S')] $*"; }

LLAMA="$REPO/reports/external_panels/bfcl_validity_v27_Llama_3_1_8B_raw.json"
MISTRAL="$REPO/reports/external_panels/bfcl_validity_v27_Mistral_7B_raw.json"
MERGED="$REPO/reports/external_panels/bfcl_validity_v27_merged_raw.json"

if [[ ! -f "$LLAMA" ]]; then
  echo "[error] Llama artifact not found: $LLAMA" >&2; exit 1
fi
if [[ ! -f "$MISTRAL" ]]; then
  echo "[error] Mistral artifact not found: $MISTRAL" >&2
  echo "  Pull with: rsync -avz runpod-a5000:/workspace/ExactKV/reports/external_panels/bfcl_validity_v27_Mistral_7B_raw.json $REPO/reports/external_panels/" >&2
  exit 1
fi

LOG "Step 1: Post-process validity + merge Llama + Mistral"
cd "$REPO"
python3 scripts/postprocess_merge_v27_bfcl.py \
    --llama  "$LLAMA" \
    --mistral "$MISTRAL" \
    --merged  "$MERGED"

LOG "Step 2: Run integrate_v27_results.py (dry-run preview)"
python3 scripts/integrate_v27_results.py \
    --merged "$MERGED" \
    --md paper/ExactKV_Technical_Report.md

if [[ "$WRITE_FLAG" == "--write" ]]; then
  LOG "Step 3: Applying patches to paper files"
  python3 scripts/integrate_v27_results.py \
      --merged "$MERGED" \
      --md paper/ExactKV_Technical_Report.md \
      --write
  
  LOG "Step 4: Rebuilding PDF"
  bash scripts/build_paper_pdf.sh
  LOG "PDF rebuilt."
else
  LOG "Step 3 (dry-run): Run with --write to patch paper + rebuild PDF"
fi

LOG "Done. Merged artifact: $MERGED"
LOG "Next: bash scripts/integrate_v27_complete.sh --write"
