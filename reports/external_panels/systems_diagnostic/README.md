# Systems diagnostic panel

**Status:** runner shipped; **GPU execution pending a live RunPod** (all previously used
SSH endpoints were down when this note was written).

## Design (96 cells)

| Axis | Values |
|------|--------|
| Models | Llama-3.1-8B, Mistral-7B-Instruct-v0.3 |
| Compressors | noop, int8, int4_sim |
| Context | 2048, 4096 |
| max_new | 64, 128 |
| Prompts | 4 fixed (2 LongBench-style + 2 BFCL-style) |
| Arms | full / lossy / ExactKV (timed + peak CUDA each) |

## Claim boundary

Diagnostic peak CUDA allocation and harness path wall-clock only.
**Not** serving RPS, TTFT, or unqualified production VRAM savings.

## Run (RunPod)

```bash
export HF_TOKEN=hf_...   # Llama gated
bash scripts/run_systems_diagnostic_panel.sh 2>&1 | tee /workspace/systems_diagnostic.log
```

Then locally:

```bash
rsync -avz -e "ssh -p PORT -i ~/.ssh/runpod_exactkv" \
  root@HOST:/workspace/ExactKV/reports/external_panels/systems_diagnostic/ \
  reports/external_panels/systems_diagnostic/
python3 scripts/build_systems_diagnostic_pack.py
```
