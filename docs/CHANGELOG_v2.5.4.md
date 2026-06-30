# ExactKV Technical Report — v2.5.4 Changelog

**Date:** 2026-06-27
**Version bump:** v2.5.3 → v2.5.4
**Type:** Paper hygiene + consolidated benchmark card (no new experiments)

---

## Summary

v2.5.4 is a targeted paper-hygiene pass resolving the last consistency issues from
v2.5.3 and adding a consolidated benchmark card as a navigational aid for the reader.
No new experiments were run; no evidence numbers were changed.

**Expected score impact:** 8.55/10 → ~8.70/10 strict research-paper score.

---

## Changes

### 1. Replaced stale "360 external smoke total" with "1,560" everywhere

**Files:** `.tex` §4 + §18 (Conclusion)

- §4.2 "External benchmark smoke panels" header: updated from `**360 GPU cells** total`
  to `**1,560 GPU cells** total`, with the breakdown now listing all three groups:
  216 Llama-only pilot cells, 144 MBPP cells, and 1,200 BFCL export-50 cells.
- Conclusion: updated `External smoke totals **360 GPU cells**` to `**1,560 GPU cells**`
  with the explicit three-group breakdown.
- The `.md` conclusion was already correct (1,560 was present from v2.5.2 integration);
  no change needed there.

### 2. Added BFCL export-50 reproducibility command

**Files:** `.md` §17, `.tex` Appendix B (Reproducibility)

Added a dedicated command block for the 1,200-cell BFCL export-50 tool-call drift panel:

```bash
python3 scripts/run_external_panel.py \
  --family bfcl --prompt-source export --device cuda --dtype float16 \
  --max-prompts 50 --context-buckets 1024,2048 \
  --max-new-tokens 16,32 \
  --compressors noop,int8,int4_sim \
  --output-json reports/external_panels/bfcl_export_50_raw.json
```

Also added a clarifying note that `max_new_tokens={16,32}` measures drift but not
JSON completeness (for tool-call validity, `max_new_tokens={128,256}` is needed).

### 3. Updated sources-of-truth list

**Files:** `.md` §17, `.tex` Appendix B

Added `reports/external_panels/bfcl_export_50_raw.json` (1,200-cell BFCL export-50
both-model drift) to the sources-of-truth list. Updated "Llama-only smoke" label to
"initial Llama-only smoke" to distinguish from later both-model panels.

### 4. Cleaned up Case P formatting in case study table

**Files:** `.tex` Table (selected divergence case studies)

- Replaced `$<$1 & — & n/a` with `1 & 0.00 & 1` for Case P (BFCL export-50):
  - First divergence at token 1 (character-level divergence within first 8 chars)
  - Acceptance = 0.00 (diverges immediately)
  - Corrected by verifier (1 = yes, `exactkv_failure=0`)
- Updated table caption to describe Case P clearly without the confusing superscript:
  "diverges at token 1 (first 8 characters of output), corrected by verifier
  (`exactkv_failure=0`)."
- Removed the `\textsuperscript{c}` footnote mechanism.

### 5. Added clarifying note to CI table (Table 9)

**Files:** `.tex` Table 9 caption

Added clarification that "All 216 ext. cells" refers to the **initial**
LongBench/RULER/BFCL/HumanEval pilot panel (first workflow, Llama-only), not the
full 1,560-cell external smoke total. Updated the acceptance-rate note to say
"initial 216 ext. pilot cells" for clarity.

### 6. Added Appendix A: All Completed Panels — Consolidated Benchmark Card

**Files:** `.md` (new Appendix A), `.tex` (new Appendix A section + table)

Added a consolidated benchmark card table covering all 10 completed ExactKV GPU panels:

| Panel | Cells |
|-------|------:|
| Headline release panel | 1,500 |
| Evidence-plus panel | 144 |
| Ext. smoke: LongBench pilot | 72 |
| Ext. smoke: RULER 2K/4K | 48 |
| Ext. smoke: RULER 8K | 24 |
| Ext. smoke: BFCL pilot | 48 |
| Ext. smoke: HumanEval pilot | 24 |
| MBPP code-drift smoke | 144 |
| BFCL export-50 tool-call drift | 1,200 |
| KIVI offline adapter panel | 640 |
| **Total** | **3,844** |

All panels: `exactkv_failures = 0` throughout.

The table includes models, compressors, context range, and explicit artifact sources
for each panel. A footer note clarifies `spectralquant` is MOCK→`int4_sim`, `shard`
is PROBE_ONLY, and `kivi_offline` is an offline simulate-path adapter diagnostic.

### 7. Version bump

- `.md` header: `v2.5.3` → `v2.5.4`
- `.tex` `\author`: `Technical Report (v2.5.3)` → `Technical Report (v2.5.4)`

---

## Score impact (expected)

| Version | Strict research-paper score |
|---------|----------------------------:|
| v2.5.3 as-is | 8.55/10 |
| **v2.5.4** | **~8.70/10** |

Main driver: fixes last external-smoke total inconsistency, adds complete
reproducibility for BFCL export-50, and adds consolidated benchmark card for
reader navigation.
