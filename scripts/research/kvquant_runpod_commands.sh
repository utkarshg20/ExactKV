#!/usr/bin/env bash
# V9 Phase D4 — KVQuant RunPod validation commands (scratch / not ExactKV adapter code).
#
# Run on a RunPod GPU pod (A100 40GB, L40S 48GB, or A40). Do NOT commit model weights,
# quantizer pickles, or large logs from this script.
#
# Usage on pod:
#   bash scripts/research/kvquant_runpod_commands.sh 2>&1 | tee /tmp/kvquant_d4_run.log
#
# Environment variables (optional overrides):
#   KVQUANT_REPO   — path to cloned KVQuant repo (default: ~/KVQuant)
#   MODEL          — HF model id (default: Qwen/Qwen2.5-0.5B)
#   NSAMPLES       — calibration samples (default: 4)
#   SEQLEN         — calibration sequence length (default: 128)
#   WORKDIR        — scratch output dir (default: /workspace/kvquant_d4 on RunPod)

set -euo pipefail

KVQUANT_REPO="${KVQUANT_REPO:-$HOME/KVQuant}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
NSAMPLES="${NSAMPLES:-4}"
SEQLEN="${SEQLEN:-128}"
WORKDIR="${WORKDIR:-/workspace/kvquant_d4}"
PYTHON="${PYTHON:-python3}"

echo "=== KVQuant Phase D4 RunPod validation ==="
echo "date: $(date -Iseconds)"
echo "hostname: $(hostname)"
nvidia-smi || true
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
"$PYTHON" -c "import transformers; print('transformers', transformers.__version__)"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ ! -d "$KVQUANT_REPO/.git" ]]; then
  git clone https://github.com/SqueezeAILab/KVQuant.git "$KVQUANT_REPO"
fi
cd "$KVQUANT_REPO"
git rev-parse HEAD | tee "$WORKDIR/kvquant_repo_sha.txt"

# --- 1. Environment: quant package only (skip gradients Fisher on first pass) ---
"$PYTHON" -m venv "$WORKDIR/.venv-kvquant"
# shellcheck disable=SC1091
source "$WORKDIR/.venv-kvquant/bin/activate"
pip install -U pip wheel
pip install -e "$KVQUANT_REPO/quant"
pip install datasets accelerate safetensors

# flash-attn: try install; continue if build fails (patch simquant below)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARN: flash-attn install failed — will patch get_model"

# --- 2. Import checks ---
"$PYTHON" - <<'PY'
from kvquant.simquant_module_quantizer import QuantLinearSim, SimQuant, make_quant_sim
print("import_ok", QuantLinearSim, SimQuant, make_quant_sim)
PY

# --- 3. Patch llama_simquant for Qwen feasibility (flash-attn off) ---
PATCH_FILE="$KVQUANT_REPO/quant/llama_simquant.py"
cp "$PATCH_FILE" "$WORKDIR/llama_simquant.py.bak"
export PATCH_FILE
"$PYTHON" - <<'PY'
from pathlib import Path
import os
p = Path(os.environ["PATCH_FILE"])
text = p.read_text()
text = text.replace(
    "use_flash_attention_2=True, torch_dtype=torch.half",
    "attn_implementation='sdpa', torch_dtype=torch.half",
)
p.write_text(text)
print("patched", p)
PY

# --- 4. Tiny calibration (no Fisher) → quantizers.pickle ---
cd "$KVQUANT_REPO/quant"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" llama_simquant.py "$MODEL" \
  --abits 4 \
  --nsamples "$NSAMPLES" \
  --seqlen "$SEQLEN" \
  --maxseqlen "$SEQLEN" \
  --dataset wikitext2 \
  --quantize \
  --quantizer-path "$WORKDIR/quantizers_qwen05b.pickle"

ls -la "$WORKDIR/quantizers_qwen05b.pickle"
export PICKLE="$WORKDIR/quantizers_qwen05b.pickle"
"$PYTHON" - <<'PY'
import os, pickle
path = os.environ["PICKLE"]
with open(path, "rb") as f:
    q = pickle.load(f)
print("quantizer_keys_count", len(q))
print("sample_keys", list(q.keys())[:6])
PY

# --- 5. Load quantizers + make_quant_sim + one forward pass ---
CUDA_VISIBLE_DEVICES=0 "$PYTHON" - <<'PY'
import copy, pickle, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvquant.simquant_module_quantizer import make_quant_sim

model_id = "Qwen/Qwen2.5-0.5B"
pickle_path = "/workspace/kvquant_d4/quantizers_qwen05b.pickle"

with open(pickle_path, "rb") as f:
    quantizers = pickle.load(f)

perchannel = {k: v for k, v in quantizers.items() if "k_proj" in k}
pertoken = {k: v for k, v in quantizers.items() if "v_proj" in k}

draft = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa"
)
verify = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cpu"
)

make_quant_sim(draft, perchannel, 4, perchannel=True, include_sparse=False)
make_quant_sim(draft, pertoken, 4, perchannel=False, dynamicquantization=True)

tok = AutoTokenizer.from_pretrained(model_id)
inp = tok("The capital of France is", return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out = draft(input_ids=inp, use_cache=True)
print("draft_forward_ok", out.logits.shape, "past", out.past_key_values is not None)

# verifier untouched
assert not any(
    type(m).__name__ == "QuantLinearSim"
    for m in verify.modules()
), "verify model must stay unmodified"
print("verify_model_clean", True)
PY

echo "=== Phase D4 RunPod validation complete ==="
echo "Artifacts (gitignored): $WORKDIR/quantizers_qwen05b.pickle"
echo "Restore llama_simquant.py from $WORKDIR/llama_simquant.py.bak if needed"
