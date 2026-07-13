#!/usr/bin/env bash
# Matched task × context × max_new factorial panel (friend residual ask).
#
# Same model, compressors, context buckets, and max_new across MBPP / BFCL / LongBench
# so cross-family divergence is not confounded by mismatched budgets.
#
# Default thin design (Llama-only):
#   3 families × 10 prompts × 3 ctx × 3 mnt × 3 compressors ≈ 810 cells
# Expected wall time on L40S: ~2–4 h (A5000 ~3–5 h).
#
# Usage (on RunPod, inside tmux):
#   export HF_TOKEN=hf_...   # required for gated Llama-3.1-8B
#   bash scripts/run_matched_factorial_panel.sh 2>&1 | tee /workspace/matched_factorial.log
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

PY="${PYTHON:-/workspace/.venv-runpod/bin/python3}"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
CTX="${CTX:-2048,4096,8192}"
MNT="${MNT:-32,64,128}"
COMPRESSORS="${COMPRESSORS:-noop,int8,int4_sim}"
MAX_PROMPTS="${MAX_PROMPTS:-10}"
DTYPE="${DTYPE:-float16}"
OUT_DIR="${OUT_DIR:-reports/external_panels/matched_factorial}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG//./_}"
MODEL_TAG="${MODEL_TAG//-/_}"

echo "==> Matched factorial panel $STAMP"
echo "    model=$MODEL ctx=$CTX mnt=$MNT compressors=$COMPRESSORS max_prompts=$MAX_PROMPTS"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

run_one() {
  local family="$1"
  local prompt_source="$2"
  local out="$OUT_DIR/${family}_${MODEL_TAG}_raw.json"
  local log="$LOG_DIR/${family}_${MODEL_TAG}_${STAMP}.log"
  echo ""
  echo "==> [$family] -> $out"
  {
    echo "CMD: $PY scripts/run_external_panel.py --family $family --prompt-source $prompt_source \\"
    echo "  --device cuda --dtype $DTYPE --models $MODEL --compressors $COMPRESSORS \\"
    echo "  --context-buckets $CTX --max-new-tokens $MNT --max-prompts $MAX_PROMPTS \\"
    echo "  --output-json $out --checkpoint-json $out --resume-json $out"
  } | tee "$log"
  if "$PY" scripts/run_external_panel.py \
      --family "$family" \
      --prompt-source "$prompt_source" \
      --device cuda \
      --dtype "$DTYPE" \
      --models "$MODEL" \
      --compressors "$COMPRESSORS" \
      --context-buckets "$CTX" \
      --max-new-tokens "$MNT" \
      --max-prompts "$MAX_PROMPTS" \
      --output-json "$out" \
      --checkpoint-json "$out" \
      --resume-json "$out" \
      2>&1 | tee -a "$log"; then
    echo "==> [$family] OK"
  else
    local rc=$?
    echo "==> [$family] FAILED exit=$rc (continuing)"
    return "$rc"
  fi
}

# Prefer HF LongBench prompts when datasets is available; else bundled pilot.
LB_SOURCE="pilot"
if "$PY" -c "import datasets" 2>/dev/null; then
  LB_SOURCE="hf"
fi

run_one mbpp pilot || true
run_one bfcl export || true
# BFCL export may need HF; fall back to pilot if export fails early — runner handles source.
if [[ ! -f "$OUT_DIR/bfcl_${MODEL_TAG}_raw.json" ]] || \
   ! "$PY" -c "import json,sys; p=json.load(open('$OUT_DIR/bfcl_${MODEL_TAG}_raw.json')); sys.exit(0 if p.get('total_cells',0)>0 else 1)" 2>/dev/null; then
  echo "==> BFCL export empty/missing; retrying with pilot prompts"
  run_one bfcl pilot || true
fi
run_one longbench "$LB_SOURCE" || true

echo ""
echo "==> Building summary pack"
"$PY" - <<'PY' || true
import json
from pathlib import Path
from collections import defaultdict

out_dir = Path("reports/external_panels/matched_factorial")
rows = []
for path in sorted(out_dir.glob("*_raw.json")):
    data = json.loads(path.read_text())
    family = data.get("dataset_family") or path.name.split("_")[0]
    for cell in data.get("cells") or []:
        if cell.get("status") not in (None, "ok", "success", "completed"):
            # keep cells that look complete even without status
            if not cell.get("metrics"):
                continue
        comp = cell.get("compressor_name")
        ctx = cell.get("context_bucket")
        mnt = cell.get("max_new_tokens")
        div = bool((cell.get("metrics") or {}).get("token_level_divergence"))
        if not div:
            lossy = cell.get("lossy") or {}
            div = lossy.get("first_divergence_idx") is not None or bool(lossy.get("diverged"))
        rows.append((family, comp, ctx, mnt, div))

by = defaultdict(lambda: [0, 0])  # divergent, n
for family, comp, ctx, mnt, div in rows:
    key = (family, comp)
    by[key][1] += 1
    by[key][0] += int(div)

summary = {
    "schema": "exactkv.matched_factorial.summary.v1",
    "claim_boundary": (
        "Matched context and max_new across MBPP/BFCL/LongBench on one model. "
        "Isolates task-family effects better than cross-panel observational spans; "
        "prompt content still differs by family."
    ),
    "n_cells": len(rows),
    "by_family_compressor": {
        f"{fam}|{comp}": {
            "divergent": d,
            "n": n,
            "divergence_rate": (d / n) if n else None,
        }
        for (fam, comp), (d, n) in sorted(by.items())
    },
}
out = out_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(f"Wrote {out} ({summary['n_cells']} cells)")
for k, v in summary["by_family_compressor"].items():
    if v["n"] and "int4" in k:
        print(f"  {k}: {v['divergence_rate']:.1%} ({v['divergent']}/{v['n']})")
PY

echo "==> Matched factorial complete $STAMP"
echo "    Outputs under $OUT_DIR"
