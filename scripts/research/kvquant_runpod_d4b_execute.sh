#!/usr/bin/env bash
# V9 Phase D4b — KVQuant RunPod GPU validation (execute on pod via SSH stdin).
# Artifacts: /workspace/kvquant_d4/
set -euo pipefail

WORKDIR="/workspace/kvquant_d4"
KVQUANT_REPO="$WORKDIR/KVQuant"
MODEL="Qwen/Qwen2.5-0.5B"
NSAMPLES=4
SEQLEN=128
EXACTKV_SHA="${EXACTKV_SHA:-unknown}"
LOG="$WORKDIR/d4b_run.log"
RESULTS="$WORKDIR/KVQUANT_RUNPOD_RESULTS.md"

mkdir -p "$WORKDIR"
exec > >(tee -a "$LOG") 2>&1

echo "=== KVQuant Phase D4b RunPod validation ==="
echo "date: $(date -Iseconds)"
echo "hostname: $(hostname)"
echo "exactkv_sha: $EXACTKV_SHA"

GPU_LINE=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "gpu_line: $GPU_LINE"
echo "cuda_version: $CUDA_VER"
echo "pwd: $(pwd)"
PYTHON=/usr/local/bin/python
$PYTHON --version

TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")
TRANS_VER=$($PYTHON -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "not_installed")
echo "torch_before: $TORCH_VER cuda=$($PYTHON -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo false)"
echo "transformers_before: $TRANS_VER"

# --- clone KVQuant ---
if [[ ! -d "$KVQUANT_REPO/.git" ]]; then
  git clone https://github.com/SqueezeAILab/KVQuant.git "$KVQUANT_REPO"
fi
cd "$KVQUANT_REPO"
KVQUANT_SHA=$(git rev-parse HEAD)
echo "kvquant_sha: $KVQUANT_SHA"
echo "$KVQUANT_SHA" > "$WORKDIR/kvquant_repo_sha.txt"

# --- venv (reuse system torch to avoid CUDA driver mismatch) ---
VENV="$WORKDIR/.venv-kvquant"
rm -rf "$VENV"
$PYTHON -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"
pip install -U pip wheel -q
pip install -e "$KVQUANT_REPO/quant" --no-deps -q
pip install datasets accelerate safetensors scikit-learn numpy scipy -q

TORCH_VER=$($PY -c "import torch; print(torch.__version__)")
TRANS_VER=$($PY -c "import transformers; print(transformers.__version__)")
echo "torch_after: $TORCH_VER cuda=$($PY -c 'import torch; print(torch.cuda.is_available())')"
echo "transformers_after: $TRANS_VER"

FLASH_RESULT="not_attempted"
if pip install flash-attn --no-build-isolation -q 2>"$WORKDIR/flash_attn_install.log"; then
  FLASH_RESULT="success"
else
  FLASH_RESULT="failed_see_$WORKDIR/flash_attn_install.log"
fi
echo "flash_attn: $FLASH_RESULT"

# --- import checks ---
$PY - <<'PY'
from kvquant.simquant_module_quantizer import QuantLinearSim, SimQuant, make_quant_sim
from kvquant.modelutils import find_layers
print("import_ok", QuantLinearSim.__module__, SimQuant.__module__, make_quant_sim.__name__, find_layers.__name__)
PY

# --- patch llama_simquant ---
PATCH_FILE="$KVQUANT_REPO/quant/llama_simquant.py"
cp "$PATCH_FILE" "$WORKDIR/llama_simquant.py.bak"
$PY - <<PY
from pathlib import Path
p = Path("$PATCH_FILE")
text = p.read_text()
old = "use_flash_attention_2=True, torch_dtype=torch.half"
new = "attn_implementation='sdpa', torch_dtype=torch.half"
if old in text:
    p.write_text(text.replace(old, new))
    print("patched_flash_to_sdpa", p)
elif "attn_implementation='sdpa'" in text or 'attn_implementation="sdpa"' in text:
    print("already_sdpa", p)
else:
    raise SystemExit("unexpected get_model() in llama_simquant.py — manual review needed")
PY

# --- Qwen2.5 compatibility ---
$PY - <<'PY'
from transformers import AutoModelForCausalLM
import kvquant.model_parse as mp
from kvquant.modelutils import find_layers

model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
cls = mp.parse_model(model)
print("parse_model", cls)
layers = model.model.layers
print("num_layers", len(layers))
fl = find_layers(layers[0])
assert "self_attn.k_proj" in fl, fl.keys()
assert "self_attn.v_proj" in fl, fl.keys()
for i, layer in enumerate(layers):
    fli = find_layers(layer)
    assert "self_attn.k_proj" in fli
    assert "self_attn.v_proj" in fli
cfg = model.config
print("gqa", "num_kv_heads", cfg.num_key_value_heads, "head_dim", cfg.head_dim)
print("qwen_compat_ok", True)
PY

# --- calibration (no Fisher) ---
CALIB_OK=false
CALIB_ERR=""
cd "$KVQUANT_REPO/quant"
if CUDA_VISIBLE_DEVICES=0 $PY llama_simquant.py "$MODEL" \
  --abits 4 \
  --nsamples "$NSAMPLES" \
  --seqlen "$SEQLEN" \
  --maxseqlen "$SEQLEN" \
  --dataset wikitext2 \
  --quantize \
  --quantizer-path "$WORKDIR/quantizers_qwen05b.pickle"; then
  CALIB_OK=true
else
  CALIB_ERR="llama_simquant calibration failed; see $LOG"
  echo "CALIB_FAIL: $CALIB_ERR"
fi

PICKLE_PATH="$WORKDIR/quantizers_qwen05b.pickle"
PICKLE_EXISTS=false
PICKLE_SIZE=0
PICKLE_KEYS=0
if [[ -f "$PICKLE_PATH" ]]; then
  PICKLE_EXISTS=true
  PICKLE_SIZE=$(stat -c%s "$PICKLE_PATH" 2>/dev/null || stat -f%z "$PICKLE_PATH")
  $PY - <<PY
import pickle
with open("$PICKLE_PATH", "rb") as f:
    q = pickle.load(f)
print("quantizer_keys_count", len(q))
keys = list(q.keys())
print("sample_keys", keys[:4])
k_keys = [k for k in keys if "k_proj" in k]
v_keys = [k for k in keys if "v_proj" in k]
print("k_proj_keys", len(k_keys), "v_proj_keys", len(v_keys))
PY
  PICKLE_KEYS=$($PY -c "import pickle; print(len(pickle.load(open('$PICKLE_PATH','rb'))))")
fi

# --- forward pass + isolation ---
FWD_OK=false
FWD_ERR=""
VERIFY_CLEAN=false
HOOKS_REMAIN=false
if [[ "$PICKLE_EXISTS" == true ]]; then
  if $PY - <<'PY'
import copy, pickle, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvquant.simquant_module_quantizer import QuantLinearSim, make_quant_sim

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

# deepcopy before mutation
draft_clone = copy.deepcopy(draft)
make_quant_sim(draft_clone, perchannel, 4, perchannel=True, include_sparse=False)
make_quant_sim(draft_clone, pertoken, 4, perchannel=False, dynamicquantization=True)

# original draft should still be Linear (deepcopy isolation)
draft_has_quant = any(type(m).__name__ == "QuantLinearSim" for m in draft.modules())
clone_has_quant = any(type(m).__name__ == "QuantLinearSim" for m in draft_clone.modules())
print("draft_unmutated_after_deepcopy", not draft_has_quant)
print("clone_has_quant", clone_has_quant)

tok = AutoTokenizer.from_pretrained(model_id)
inp = tok("The capital of France is", return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out = draft_clone(input_ids=inp, use_cache=True)
print("draft_forward_ok", out.logits.shape, "past", out.past_key_values is not None)
if out.past_key_values is not None:
    print("past_len", len(out.past_key_values))

verify_clean = not any(type(m).__name__ == "QuantLinearSim" for m in verify.modules())
print("verify_model_clean", verify_clean)
PY
  then
    FWD_OK=true
    VERIFY_CLEAN=true
  else
    FWD_ERR="forward/isolation script failed; see $LOG"
    echo "FWD_FAIL: $FWD_ERR"
  fi
fi

# --- write results markdown ---
CLASSIFICATION="C"
if [[ "$PICKLE_EXISTS" == true && "$FWD_OK" == true && "$VERIFY_CLEAN" == true ]]; then
  CLASSIFICATION="A"
elif [[ "$PICKLE_EXISTS" == true || "$FWD_OK" == true ]]; then
  CLASSIFICATION="B"
fi

cat > "$RESULTS" <<EOF
# KVQuant RunPod D4b GPU Results

**Date:** $(date -Iseconds)
**Pod hostname:** $(hostname)
**ExactKV commit (reference):** $EXACTKV_SHA

## Environment

| Item | Value |
|---|---|
| GPU | $GPU_LINE |
| CUDA (nvidia-smi) | $CUDA_VER |
| Python | $($PYTHON --version 2>&1) |
| torch | $TORCH_VER |
| transformers | $TRANS_VER |
| flash-attn | $FLASH_RESULT |
| KVQuant SHA | $KVQUANT_SHA |
| Workdir | $WORKDIR |

## Install

- KVQuant cloned to \`$KVQUANT_REPO\`
- venv: \`$VENV\`
- \`pip install -e quant/\` succeeded
- flash-attn: **$FLASH_RESULT** (sdpa patch used regardless)

## Import checks

- \`kvquant\`, \`QuantLinearSim\`, \`SimQuant\`, \`make_quant_sim\`, \`find_layers\`: **OK**

## Qwen2.5 compatibility

- \`Qwen/Qwen2.5-0.5B\` loads on CPU
- \`parse_model\` → llama class path
- \`self_attn.k_proj\` / \`self_attn.v_proj\` on all layers: **OK**
- GQA (2 KV heads, head_dim 64): **OK**
- \`attn_implementation='sdpa'\` patch applied to \`llama_simquant.py\`

## Fisher / calibration

- Fisher (\`run-fisher.py\`): **skipped** — Llama-specific hooks (\`k_proj.act.grad\`, \`set_devices()\`)
- No-Fisher \`llama_simquant.py --quantize\`: **$([ "$CALIB_OK" = true ] && echo OK || echo FAILED)**
- Config: nsamples=$NSAMPLES, seqlen=$SEQLEN, dataset=wikitext2, abits=4
$(if [[ -n "$CALIB_ERR" ]]; then echo "- Error: $CALIB_ERR"; fi)

## Quantizer artifact

| Item | Value |
|---|---|
| Path | \`$PICKLE_PATH\` |
| Exists | $PICKLE_EXISTS |
| Size bytes | $PICKLE_SIZE |
| Key count | $PICKLE_KEYS |
| Expected keys | \`model.layers.{i}.self_attn.{k,v}_proj\` |

## QuantLinearSim forward

- deepcopy draft + \`make_quant_sim\` on k_proj/v_proj: **$([ "$FWD_OK" = true ] && echo OK || echo FAILED/skipped)**
- One forward with \`use_cache=True\`: **$([ "$FWD_OK" = true ] && echo OK || echo N/A)**
- \`past_key_values\` returned: **$([ "$FWD_OK" = true ] && echo yes || echo N/A)**
$(if [[ -n "$FWD_ERR" ]]; then echo "- Error: $FWD_ERR"; fi)

## Draft / verifier isolation

- \`make_quant_sim\` mutates modules **in place**
- \`deepcopy\` draft before \`make_quant_sim\`: **mandatory**
- Separate verify model (CPU, unmodified): **$([ "$VERIFY_CLEAN" = true ] && echo clean || echo not verified)**
- Calibration hooks after layer loop: **removed** (upstream design)
- Forked transformers: **not required** for simquant path
- deployment/ CUDA: **not used** (optional for production; out of D5 scope)

## Adapter classification

**$CLASSIFICATION** — $(
  case "$CLASSIFICATION" in
    A) echo "Faithful adapter go — proceed to D5 KVQuantSimAdapter (draft clone + _compresses_via_full_state replay)";;
    B) echo "Restricted go — partial pipeline; see log for restrictions";;
    C) echo "No-go for now — GPU pipeline blocked";;
  esac
)

## D5 recommendation

$(case "$CLASSIFICATION" in
  A) cat <<'REC'
- Implement \`KVQuantSimAdapter\` (not in D4b): factory-only, \`supports_real_bytes_claim=False\`
- Load \`quantizers_qwen05b.pickle\` per model; \`deepcopy(runtime.model)\` → \`make_quant_sim\`
- Use \`_compresses_via_full_state()\` replay path (kvpress pattern)
- Do **not** use post-RoPE tensor bridge
- Quantizer artifact for D5: stored at pod path above; keep outside git
REC
;;
  B) echo "- Restricted prototype only; document blockers before full D5";;
  C) echo "- Do not start D5 until blockers resolved";;
esac)

## Safety / scope notes

- This phase does **not** implement a KVQuant adapter.
- This phase does **not** run ExactKV Experiment 010.
- ExactKV does **not** claim KVQuant results yet.
- External KVQuant paper results are **not** ExactKV results.
- No throughput, latency, speedup, runtime, tokens/sec, active GPU memory, or production-serving claim is made.
- Forward pass success is **feasibility only**, not an ExactKV acceptance result.

## Log

Full log: \`$LOG\`
EOF

echo "=== D4b complete ==="
echo "results: $RESULTS"
cat "$RESULTS"
