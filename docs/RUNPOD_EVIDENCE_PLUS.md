# RunPod evidence-plus panel (8.5+ GPU upgrade)

Run this on your **RTX A5000** pod (`possible_tan_silverfish`) after syncing the repo to `/workspace/ExactKV`.

## What you need to provide (agent cannot do from local machine)

1. **SSH access** — add your public key to the pod, or run commands yourself in the web terminal.
2. **`HF_TOKEN`** (for gated **Llama-3.1-8B**):
   ```bash
   export HF_TOKEN=hf_...
   ```
3. **Optional external adapters** (cells skip gracefully if missing):
   - **KIVI**: clone [MIT-HAN-LAB/kivi](https://github.com/MIT-HAN-LAB/kivi) and `export PYTHONPATH=/workspace/kivi:$PYTHONPATH`
   - **KVQuant**: install `kvquant` in the env
   - **SnapKV**: `pip install kvpress`

Nothing else is required for the **pilot** (builtins + long-context buckets + timing).

## Quick start

```bash
cd /workspace
git clone <your-exactkv-remote> ExactKV   # or rsync from laptop
cd ExactKV
chmod +x scripts/setup_runpod_evidence_plus.sh
export HF_TOKEN=hf_...   # if running Llama-3.1-8B
bash scripts/setup_runpod_evidence_plus.sh
```

## Manual run (full control)

```bash
# Smoke on GPU
python3 scripts/run_evidence_plus_panel.py --smoke --device cuda --dtype float16

# Pilot (recommended for 8.5+ paper upgrade)
python3 scripts/run_evidence_plus_panel.py \
  --device cuda --dtype float16 \
  --max-prompts 6 \
  --context-buckets 512,1024,2048 \
  --max-new-tokens 16,32,64 \
  --models meta-llama/Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.3
```

Outputs:

| File | Purpose |
|------|---------|
| `reports/evidence_plus/raw.json` | Primary artifact (cells + timing + bucket summary) |
| `reports/evidence_plus/summary.md` | Human-readable rollup |

## Claim boundaries

- `timing_ms` is **diagnostic wall-clock per benchmark cell**, not production serving throughput.
- External compressors are labeled `RESTRICTED_ADAPTER` in cell metadata.
- Skipped adapters are recorded with `status: skipped` — not silent omission.

## Copy results back

```bash
# From laptop (direct TCP SSH)
scp -P 10113 -i ~/.ssh/id_ed25519 \
  root@203.57.40.169:/workspace/ExactKV/reports/evidence_plus/raw.json \
  reports/evidence_plus/raw.json
```

After `raw.json` exists locally, regenerate the paper §6/§8/§14 tables from real numbers.
