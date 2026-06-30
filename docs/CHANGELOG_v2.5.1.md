# ExactKV v2.5.1 Changelog — Consistency + Claim-Boundary Cleanup

**Date:** June 27, 2026  
**Score target:** 8.55/10 strict research-paper draft  
**Previous version:** v2.5 (8.30/10 as-is)

---

## Changes

### 1. Abstract typo fixed (`.tex`)

**Before:** "compressor-agnostic **evaluation evaluation** crash-test framework"  
**After:** "compressor-agnostic **crash-test framework** with leaderboard-style reporting"

### 2. KIVI contradiction resolved — abstract + §14 + §15

The abstract, §14.2.1, and §15 all said the KIVI panel was "pre-registered and pending RunPod execution."
v2.5.1 corrects this throughout: the panel is **completed** (640 cells, June 27 2026).

Abstract now reads:
> "A separate 640-cell `kivi_offline` panel over real HF LongBench and MBPP subsets tests a real
> upstream KIVI quantizer path as an offline adapter diagnostic..."

§14.3 future work table updated:  
`Pre-registered; script ready` → `Completed (§6.4.6); 640 cells, exactkv_failures=0`

§15 items 3/4/9 rewritten to reflect real HF datasets completed and KIVI as adapter diagnostic.

### 3. int8 divergence contradiction fixed — leaderboard table

**Root cause:** the leaderboard table used `divergence_score = 1 − stability_score = 0.173` for
`int8` Mistral, while Table 3 (compressor_summary) correctly showed `divergence_rate = 0.000`.
These are different metrics with the same column header "Div."

**Fix:** Leaderboard table now shows:
- `Div. rate†` = raw lossy-path token divergence fraction (0.000 for int8 Mistral)
- `Stab.` = stability_score (0.827 for int8 Mistral)

Footnote added explaining: "int8 Mistral: divergence_rate=0.000 (zero cells diverge)
but stability_score=0.827 because some cells have late-window first-divergence..."

Both tables now consistently show `int8` with **0% raw token divergence**.

### 4. "Built-in compressors only" wording fixed

**Before:** "These results validate the measurement harness on built-in compressors only."  
**After:** "The headline and smoke panels validate the harness on built-in compressors;
a separate 640-cell `kivi_offline` panel tests a real upstream KIVI quantizer path as
an offline adapter diagnostic."

### 5. KIVI wording made safer (§6.4.6 + table caption)

**Before:** "catastrophic KV corruption... Not a claim about the KIVI algorithm."  
**After:** "`kivi_offline` produced catastrophic **adapter-level** drift in this ExactKV
integration... Because this is an offline simulate-path adapter rather than KIVI production
CUDA/Triton serving, these results diagnose the adapter/integration path, not the KIVI
algorithm as deployed."

Table caption updated to match.

### 6. §13 novelty framing updated

Added mention of crash-test coverage extending to "real HF benchmarks and external adapter
integrations (including `kivi_offline` as an offline adapter diagnostic)."

### 7. Reproducibility appendix (§17) — KIVI commands added

New block:
```bash
export PYTHONPATH=/tmp/kivi_research
bash scripts/run_kivi_external_panel.sh
# Note: kivi_offline uses simulate path (is_simulated=False,
#       supports_real_bytes_claim=False). Not production KIVI serving.
```

Source-of-truth list updated to include both KIVI artifact files.

### 8. `supports_real_bytes_claim=False` note inline

The adapter claim boundary (no real byte savings, no CUDA/Triton) is now explicit in
both the §17 command block and the KIVI table caption.

---

## Score impact

| Issue fixed | Score delta |
|-------------|-------------|
| Abstract typo | +0.05 (paper quality) |
| KIVI pending→completed contradiction | +0.10 (claim safety, consistency) |
| int8 Div contradiction | +0.10 (trust, claim safety) |
| Built-in only wording | +0.05 (claim safety) |
| KIVI wording safety | +0.05 (claim safety) |
| Reproducibility completeness | +0.05 (reproducibility) |

**Estimated strict score: 8.55/10** (up from 8.30/10 as-is v2.5)
