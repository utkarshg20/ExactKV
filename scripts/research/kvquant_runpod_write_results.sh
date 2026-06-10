#!/usr/bin/env bash
# Write D4b results markdown on RunPod (run via SSH stdin).
set -euo pipefail
RESULTS="/workspace/kvquant_d4/KVQUANT_RUNPOD_RESULTS.md"
source /workspace/kvquant_d4/.venv-kvquant/bin/activate
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
TRANS_VER=$(python -c "import transformers; print(transformers.__version__)")
KVQUANT_SHA=$(cat /workspace/kvquant_d4/kvquant_repo_sha.txt)
PICKLE_SIZE=$(stat -c%s /workspace/kvquant_d4/quantizers_qwen05b.pickle)
PICKLE_KEYS=$(python -c "import pickle; print(len(pickle.load(open('/workspace/kvquant_d4/quantizers_qwen05b.pickle','rb'))))")
GPU_LINE=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

cat > "$RESULTS" <<EOF
# KVQuant RunPod D4b GPU Results

**Date:** $(date -Iseconds)
**Pod hostname:** $(hostname)
**ExactKV commit (reference):** ${EXACTKV_SHA:-unknown}

## Environment

| Item | Value |
|---|---|
| GPU | $GPU_LINE |
| CUDA (nvidia-smi) | $CUDA_VER |
| Python | $(python --version 2>&1) |
| torch | $TORCH_VER (system-site-packages venv) |
| transformers | $TRANS_VER (pinned 4.44.2 for KVQuant compat) |
| flash-attn | failed (not required; upstream already uses sdpa) |
| KVQuant SHA | $KVQUANT_SHA |
| Workdir | /workspace/kvquant_d4 |

## Install

- KVQuant cloned to \`/workspace/kvquant_d4/KVQuant\`
- venv: \`/workspace/kvquant_d4/.venv-kvquant\` with \`--system-site-packages\` (avoids torch CUDA mismatch)
- \`pip install -e quant/ --no-deps\` + explicit deps succeeded
- **Fix applied:** venv must not reinstall torch (first attempt got cu130 + cuda=False)

## Import checks

- \`kvquant\`, \`QuantLinearSim\`, \`SimQuant\`, \`make_quant_sim\`, \`find_layers\`: **OK**

## Qwen2.5 compatibility

- \`Qwen/Qwen2.5-0.5B\` loads
- \`parse_model\` → **llama** class path
- \`self_attn.k_proj\` / \`self_attn.v_proj\` on all 24 layers: **OK**
- GQA: 2 KV heads (head_dim via hidden/num_heads)
- \`attn_implementation='sdpa'\` already in upstream \`llama_simquant.py\` (no flash-attn patch needed)
- **Note:** Qwen2.5 k_proj/v_proj have **bias** (Llama-class scripts assume optional bias handling)

## Fisher / calibration

- Fisher (\`run-fisher.py\`): **skipped** — Llama-specific hooks
- \`llama_simquant.py --dataset wikitext2\`: **FAILED** — \`HfUriError\` loading wikitext (datasets/huggingface_hub URI parsing)
- **Workaround:** synthetic local calibration via \`synthetic_calib.py\` (4 samples, seqlen 128): **OK**
- **Additional fix:** pin \`transformers==4.44.2\` (5.x breaks Qwen2 calibration forward: \`position_embeddings\` None)

## Quantizer artifact

| Item | Value |
|---|---|
| Path | \`/workspace/kvquant_d4/quantizers_qwen05b.pickle\` |
| Exists | true |
| Size bytes | $PICKLE_SIZE |
| Key count | $PICKLE_KEYS |
| Sample keys | \`model.layers.0.self_attn.k_proj\`, \`...v_proj\`, ... |

## QuantLinearSim forward

- \`deepcopy\` draft + \`make_quant_sim\` on k_proj/v_proj: **OK** (after bias patch)
- One forward with \`use_cache=True\`: **OK** — logits \`(1, 5, 151936)\`, \`past_key_values\` returned (24 layers)
- **Patch required for Qwen bias:** \`make_quant_sim\` must pass \`tmp.bias\` not \`tmp.bias is not None\`; \`QuantLinearSim\` uses \`if bias is not None:\`

## Draft / verifier isolation

- \`make_quant_sim\` mutates modules **in place**
- \`deepcopy\` draft before \`make_quant_sim\`: **mandatory** (\`draft_unmutated_after_deepcopy=True\`)
- Separate verify model (CPU): **clean** (\`verify_model_clean=True\`)
- Calibration hooks: **removed** after layer loop (upstream)
- Forked transformers: **not required** for simquant
- deployment/ CUDA: **not used**

## Adapter classification

**A — Faithful adapter go** (with documented D5 patches)

Proceed to D5 \`KVQuantSimAdapter\`: draft clone + \`_compresses_via_full_state()\` replay.

**D5 restrictions from D4b:**
1. Pin \`transformers~=4.44\` in isolated KVQuant venv
2. Ship bias-handling patch for \`make_quant_sim\` / \`QuantLinearSim\` (Qwen2.5 has proj bias)
3. Calibration artifact: per-model pickle; synthetic or fixed dataset loader acceptable for prototype
4. \`supports_real_bytes_claim=False\`; factory-only; no registry

## D5 recommendation

- Implement \`KVQuantSimAdapter\` (not in D4b): factory-only, \`supports_real_bytes_claim=False\`
- Load quantizers pickle per model; \`deepcopy(runtime.model)\` → patched \`make_quant_sim\`
- Use \`_compresses_via_full_state()\` replay (kvpress pattern)
- Do **not** use post-RoPE tensor bridge
- Quantizer for D5 prototype: \`/workspace/kvquant_d4/quantizers_qwen05b.pickle\` (keep outside git)

## Safety / scope notes

- This phase does **not** implement a KVQuant adapter.
- This phase does **not** run ExactKV Experiment 010.
- ExactKV does **not** claim KVQuant results yet.
- External KVQuant paper results are **not** ExactKV results.
- No throughput, latency, speedup, runtime, tokens/sec, active GPU memory, or production-serving claim is made.
- Forward pass success is **feasibility only**, not an ExactKV acceptance result.

## Log

- Full D4b log: \`/workspace/kvquant_d4/d4b_run.log\` (partial; stepwise SSH execution)
- Synthetic calib script: \`/workspace/kvquant_d4/synthetic_calib.py\`
- Forward check: \`/workspace/kvquant_d4/forward_check.py\`
EOF

echo "wrote $RESULTS"
cat "$RESULTS"
