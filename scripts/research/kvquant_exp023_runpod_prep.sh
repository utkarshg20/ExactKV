#!/usr/bin/env bash
# V12 Phase 3 — KVQuant Experiment 023 RunPod prep (env + 1.5B quantizer).
set -euo pipefail

WORKDIR="/workspace/kvquant_d4"
KVQUANT_REPO="$WORKDIR/KVQuant"
VENV="$WORKDIR/.venv-kvquant"
PICKLE_15B="$WORKDIR/quantizers_qwen15b.pickle"
PYTHON="${PYTHON:-/usr/bin/python3}"

mkdir -p "$WORKDIR"
exec > >(tee -a "$WORKDIR/exp023_prep.log") 2>&1

echo "=== KVQuant Exp 023 prep ==="
echo "date: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -d "$KVQUANT_REPO/.git" ]]; then
  git clone https://github.com/SqueezeAILab/KVQuant.git "$KVQUANT_REPO"
fi
cd "$KVQUANT_REPO"
echo "kvquant_sha: $(git rev-parse HEAD)"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"
pip install -U pip wheel -q
pip install "transformers>=4.44,<4.45" -q
pip install -e "$KVQUANT_REPO/quant" --no-deps -q
pip install datasets accelerate safetensors scikit-learn numpy scipy -q

PATCH_FILE="$KVQUANT_REPO/quant/llama_simquant.py"
if grep -q "use_flash_attention_2=True" "$PATCH_FILE"; then
  sed -i.bak "s/use_flash_attention_2=True, torch_dtype=torch.half/attn_implementation='sdpa', torch_dtype=torch.half/" "$PATCH_FILE"
  echo "patched sdpa in llama_simquant.py"
fi

$PY -c "import torch,kvquant; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); import transformers; print('transformers', transformers.__version__)"

if [[ ! -f "$PICKLE_15B" ]]; then
  echo "generating quantizers_qwen15b.pickle ..."
  export PYTHONPATH="$KVQUANT_REPO/quant:${PYTHONPATH:-}"
  CUDA_VISIBLE_DEVICES=0 $PY /workspace/ExactKV/scripts/research/kvquant_runpod_synthetic_calib_15b.py
else
  echo "quantizer exists: $PICKLE_15B ($(stat -c%s "$PICKLE_15B") bytes)"
fi

echo "prep_complete pickle=$PICKLE_15B"
