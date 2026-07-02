# Paper roadmap — strict review lens → 9.0+

Internal planning doc. Not a public claim.

## Current strict scores (June 2026)

| Lens | Score |
|------|------:|
| Technical artifact / benchmark harness | 9.1–9.3 |
| Public technical report | 8.9–9.1 |
| Workshop paper | 8.5–8.7 |
| Main conference paper (as-is) | 7.8–8.1 |
| **Overall strict** | **~8.6** |

Wave-2 faithful pull moved the needle from ~8.45 → ~8.6. TurboQuant smoke is real
progress but still diagnostic, not decisive.

## Scientific center (what reviewers should weight)

1. **Task-type sensitivity** — `int4_sim` 6% (MBPP) → 90% (HF LongBench); Table 6.16.
2. **Mechanistic autopsy** — 1,103 divergent cells; three failure modes + forensic cases.
3. **Compressor design curve** — int8 → int6_sim → int4_per_vec_sim → int4_sim / H2O.

**Not the center:** `exactkv_failures=0` (expected when verify/commit is correct).

## Remaining weaknesses

| Gap | Why it matters | Priority |
|-----|----------------|----------|
| Faithful compressors diagnostic only | Strongest results are built-in sims + int8 | P0 for 9.0+ |
| Phase D3 reads like experiment log | Interrupts narrative after core curve | P1 (structure) |
| Downstream metrics BFCL-only | “Validity” too narrow for conference | P1 |
| Version labels (v2.6/v2.7/Phase D3) | Timeline smell, not claim organization | P2 |
| TurboQuant smoke too small | 128 cells, Mistral, MBPP+BFCL only | P0 for faithful story |

## Path to 9.0+ (ordered)

### Tier A — GPU evidence (biggest lift)

1. **Full TurboQuant faithful panel** — both models, MBPP + BFCL validity + HF LongBench,
   same grid depth as Table 6.16 (~720–864 cells per model if feasible).
2. **One non-BFCL downstream metric** — pick one:
   - MBPP syntax-validity or pass@1 (sandboxed), or
   - LongBench answer-overlap / ROUGE-style diagnostic (not official F1 claim).

### Tier B — structural cleanup (no GPU)

3. **Reorganize §6 by claim**, not version: e.g. “Task sensitivity”, “Length scaling”,
   “Failure modes”, “Compressor curve”, “Downstream validity”, “Appendix: faithful adapters”.
4. **Consolidate faithful adapters** — master Table 6.17 + wave detail in appendix;
   shell commands only in Appendix E / `FAITHFUL_COMPRESSOR_INTEGRATION.md`.
5. **Demote `exactkv_failures=0`** in abstract, contributions, site hero — one sentence
   as engineering invariant; lead with drift hierarchy.

### Tier C — conference fork

6. Extract **15–20 page conference paper** from technical report:
   - Drop version history tables from main text
   - Move Phase D3 entirely to appendix
   - Single unified compressor comparison table (int8, int6, int4_per_vec, turboquant, int4_sim)

## Estimated score after tiers

| Milestone | Strict score |
|-----------|-------------:|
| Current (wave-2 complete) | 8.6 |
| Tier B only (structure) | 8.7–8.8 |
| Tier A + B (full TurboQuant + downstream metric) | 9.0+ |
| Tier C polish | 9.0–9.2 conference-ready |

## Next RunPod job (when funded)

```bash
bash scripts/runpod_faithful_wave3_launch.sh   # ~576 cells, ~6–8 GPU hours
bash scripts/pull_faithful_wave3_from_runpod.sh # after completion
```

Target artifact tree: `reports/external_panels/faithful/wave3/`
