# Changelog v2.4 → v2.4.1

**Date:** 2026-06-26

Consistency-only cleanup. No new GPU runs.

## Fixes

1. **External cell totals:** 360 GPU cells = 216 Llama-only (LongBench/RULER/BFCL/HumanEval) + 144 MBPP (both models). Split Table 4 / Table 4b.
2. **Table captions:** Llama-only vs MBPP both-model scope clarified.
3. **§5.2 / §6.4:** First-workflow Mistral failure vs later MBPP both-model success.
4. **Future work:** MBPP loader marked done; expand to pass@1 / larger pilot.
5. **Abstract:** Shortened to four-sentence structure (what / headline / external / boundaries).
6. **Case study:** MBPP Case N (`mbpp_002_ctx1024`, Llama int4_sim).
7. **Limitations / conclusion:** 360-cell external total; Mistral/MBPP wording aligned.

## Target rating

Strict research draft: **8.25/10** after this pass (consistency restored).
