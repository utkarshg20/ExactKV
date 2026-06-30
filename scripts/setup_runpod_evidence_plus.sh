#!/usr/bin/env bash
# Bootstrap ExactKV evidence-plus panel on RunPod (torch v2.8 template, A5000).
# Mount repo at /workspace/ExactKV or set EXACTKV_ROOT.
set -euo pipefail

EXACTKV_ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$EXACTKV_ROOT"

echo "==> ExactKV root: $EXACTKV_ROOT"
python3 -V
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> Installing ExactKV (editable)"
pip install -q -e ".[dev]" 2>/dev/null || pip install -q -e .

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

echo "==> Deterministic smoke (no GPU weights)"
python3 scripts/run_evidence_plus_panel.py --deterministic-mode --smoke \
  --output-json reports/evidence_plus/smoke_deterministic.json

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
  echo "==> GPU smoke (Qwen 0.5B, 512 ctx)"
  python3 scripts/run_evidence_plus_panel.py --smoke --device cuda --dtype float16 \
    --output-json reports/evidence_plus/smoke_gpu.json \
    --output-md reports/evidence_plus/smoke_gpu.md

  echo "==> Pilot panel (builtins; external adapters if env provides them)"
  python3 scripts/run_evidence_plus_panel.py \
    --device cuda --dtype float16 \
    --max-prompts 6 \
    --context-buckets 512,1024 \
    --max-new-tokens 16,32 \
    --models "meta-llama/Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.3" \
    --output-json reports/evidence_plus/raw.json \
    --output-md reports/evidence_plus/summary.md
else
  echo "CUDA not available — skipping GPU panel"
fi

echo "==> Done. Artifacts under reports/evidence_plus/"
