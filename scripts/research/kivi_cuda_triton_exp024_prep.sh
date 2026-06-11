#!/usr/bin/env bash
# V12 Phase 4 — KIVI CUDA/Triton Experiment 024 RunPod prep.
set -euo pipefail

WORKDIR="${KIVI_EXP024_WORKDIR:-/workspace/kivi_exp024}"
KIVI_REPO="$WORKDIR/KIVI"
VENV="$WORKDIR/.venv-kivi"
MANIFEST="$WORKDIR/prep_manifest.txt"
LOG="$WORKDIR/prep.log"
PYTHON="${PYTHON:-/usr/bin/python3}"

mkdir -p "$WORKDIR"
exec > >(tee -a "$LOG") 2>&1

echo "=== KIVI Exp 024 prep ==="
echo "date: $(date -Iseconds)"
echo "hostname: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

if [[ ! -d "$KIVI_REPO/.git" ]]; then
  git clone https://github.com/jy-yuan/KIVI.git "$KIVI_REPO"
fi
cd "$KIVI_REPO"
KIVI_SHA=$(git rev-parse HEAD)
echo "kivi_sha: $KIVI_SHA"
echo "$KIVI_SHA" > "$WORKDIR/kivi_repo_sha.txt"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"
pip install -U pip wheel -q

# Prefer pod torch; avoid hard pin conflicts where possible.
pip install "transformers>=4.43,<4.44" -q || pip install "transformers==4.43.1" -q || true
pip install numpy scipy datasets accelerate -q || true

KIVI_PIP_OK=false
if pip install -e "$KIVI_REPO" --no-deps -q 2>"$WORKDIR/kivi_pip_install.log"; then
  KIVI_PIP_OK=true
fi
echo "kivi_pip_install: $KIVI_PIP_OK"

# Patch llama_kivi flash if needed (optional for tensor smoke).
LLAMA_KIVI="$KIVI_REPO/models/llama_kivi.py"
if [[ -f "$LLAMA_KIVI" ]] && grep -q "use_flash_attention_2=True" "$LLAMA_KIVI"; then
  cp "$LLAMA_KIVI" "$WORKDIR/llama_kivi.py.bak"
  sed -i.bak "s/use_flash_attention_2=True/attn_implementation='sdpa'/" "$LLAMA_KIVI" || true
  echo "patched_llama_kivi_sdpa: attempted"
fi

# quant/ CUDA extension (kivi_gemv)
QUANT_EXT_OK=false
if pip install -e "$KIVI_REPO/quant" -q 2>"$WORKDIR/quant_pip_install.log"; then
  QUANT_EXT_OK=true
fi
echo "quant_ext_install: $QUANT_EXT_OK"

FLASH_OK=false
if pip install flash-attn --no-build-isolation -q 2>"$WORKDIR/flash_attn_install.log"; then
  FLASH_OK=true
fi
echo "flash_attn_install: $FLASH_OK"

export PYTHONPATH="$KIVI_REPO:$KIVI_REPO/quant:${PYTHONPATH:-}"
$PY - <<'PY' || true
import json, os, platform
import torch
info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
try:
    import transformers
    info["transformers"] = transformers.__version__
except Exception as e:
    info["transformers_error"] = str(e)
try:
    import triton
    info["triton"] = triton.__version__
except Exception as e:
    info["triton_error"] = str(e)
print(json.dumps(info, indent=2))
PY

{
  echo "kivi_sha=$KIVI_SHA"
  echo "kivi_pip_ok=$KIVI_PIP_OK"
  echo "quant_ext_ok=$QUANT_EXT_OK"
  echo "flash_attn_ok=$FLASH_OK"
  echo "venv=$VENV"
  echo "pythonpath=$KIVI_REPO:$KIVI_REPO/quant"
} > "$MANIFEST"

echo "prep_complete workdir=$WORKDIR"
