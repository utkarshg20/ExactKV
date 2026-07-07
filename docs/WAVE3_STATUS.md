# Wave-3 faithful panel — status and narrative

**Last updated:** 2026-07-07 (complete)  
**Target grid:** 576 cells · `int8` + `turboquant_experimental` · Llama-3.1-8B + Mistral-7B · HF LongBench + BFCL + MBPP  
**Artifact path:** `reports/external_panels/faithful/wave3/`  
**Status:** **576/576 ok · exactkv_failures: 0** (locked locally after int8+turboquant merge)

---

## Completion status

| Panel | Expected | Status |
|-------|----------|--------|
| BFCL Llama + Mistral | 80 each | ✅ complete |
| MBPP Llama + Mistral | 64 each | ✅ complete |
| LongBench Llama | 144 | ✅ complete (72 int8 + 72 turboquant) |
| LongBench Mistral | 144 | ✅ complete (72 int8 + 72 turboquant) |

**Local total:** **576/576 ok · exactkv_failures: 0**

---

## Claim-safe narrative (504-cell snapshot)

Wave-3 is the first **full wave-1 grid** rerun with only two compressors: production-style **int8** and faithful **TurboQuant** (`turboquant_experimental`, offline NumPy adapter). It is appendix evidence — separate from the 8,132-cell headline set.

### Headline findings (stable across completed cells)

1. **Zero exactness failures.** Every completed cell reports `exactkv_failures: 0`. The verifier never silently accepted a wrong token.

2. **Task type dominates drift — again.** On short structured tasks (BFCL, MBPP), both compressors stay near-clean:
   - int8: 0% divergence on code/tool panels
   - turboquant: 0–10% on BFCL, 0–3% on MBPP

3. **Long context is where TurboQuant breaks.** On HF LongBench (Llama, complete):
   - **int8:** 18.1% divergence, 99.2% mean acceptance
   - **turboquant_experimental:** **62.5%** divergence, 86.1% mean acceptance  
   This is the wave-3 story: TurboQuant looks fine on short tasks, fails on reading/summarization at 2K–8K context.

4. **int8 remains the only non-catastrophic real compressor in this grid.** TurboQuant is informative as an upstream adapter diagnostic, not a deployment recommendation.

### Pending (Mistral LongBench turboquant)

Mistral int8 LongBench is complete (15.3% divergence, 98.5% acceptance). The remaining ~72 turboquant cells run on CPU-heavy offline TurboQuant (~15–20 min/cell on 8K context). Final combined stats will update the “both models” column in §6.17.2.

---

## What wave-3 is not

- Not a replacement for the 8,132-cell headline leaderboard
- Not official LongBench/BFCL/MBPP benchmark scores (token-path drift only)
- Not a TurboQuant production benchmark (restricted Python adapter, CPU path)
- Not publishable until 576/576

---

## When backfill completes

```bash
bash scripts/pull_faithful_wave3_from_runpod.sh
python3 scripts/rebuild_wave3_panels.py --dir reports/external_panels/faithful/wave3 --write
python3 scripts/integrate_faithful_panel_results.py --dir reports/external_panels/faithful/wave3 --write
python3 scripts/rebuild_wave3_panels.py --dir reports/external_panels/faithful/wave3  # expect 576/576
```

Then follow `docs/WAVE3_RELEASE_CHECKLIST.md`.
