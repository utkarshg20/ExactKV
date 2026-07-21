# Systems diagnostic panel

**Status:** complete on NVIDIA RTX PRO 4000 Blackwell (torch 2.8.0+cu128, float16).
96 cells, `exactkv_failures=0`. Pack: `reports/systems/systems_diagnostic.{json,md}`.

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

## Headline numbers (mean)

| Model | Comp | Peak GiB full / lossy / ExactKV | Wall ms full / lossy / ExactKV |
|-------|------|--------------------------------:|-------------------------------:|
| Llama | int4_sim | 16.10 / 16.67 / 16.72 | 3660 / 3675 / 8572 |
| Mistral | int4_sim | 14.23 / 15.22 / 15.26 | 3535 / 3141 / 8345 |

ExactKV is ~2.3× slower than full on this harness (verify cost) and peaks slightly
above lossy-only because full + compressed state coexist.

## Re-run (RunPod)

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
