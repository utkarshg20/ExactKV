# ExactKV v2.5.2 Changelog — BFCL Export-50 Integration

**Date:** June 27, 2026
**Previous version:** v2.5.1 (8.55/10 estimated)

---

## New evidence

### BFCL export-50 drift panel (1,200 cells, both models)

**Artifact:** `reports/external_panels/bfcl_export_50_raw.json` (13.5 MB)

**Setup:** 50 exported BFCL tool-call prompts × 2 models (Llama-3.1-8B + Mistral-7B) ×
2 context buckets (1024, 2048) × 3 compressors (`noop`, `int8`, `int4_sim`) ×
2 `max_new_tokens` (16, 32) = **1,200 cells total**.

**Results:**

| Compressor | Model | Cells | Div. rate | Wilson 95% CI |
|------------|-------|------:|----------:|---------------|
| `noop` | Llama-3.1-8B | 200 | 0.000 | [0.000, 0.019] |
| `noop` | Mistral-7B | 200 | 0.000 | [0.000, 0.019] |
| `int8` | Llama-3.1-8B | 200 | 0.000 | [0.000, 0.019] |
| `int8` | Mistral-7B | 200 | 0.000 | [0.000, 0.019] |
| `int4_sim` | Llama-3.1-8B | 200 | 0.055 | [0.031, 0.096] |
| `int4_sim` | Mistral-7B | 200 | 0.170 | [0.124, 0.228] |

`exactkv_failures = 0` across all 1,200 cells.
Mistral-7B `int4_sim` divergence (17.0%) is ~3× Llama (5.5%).

---

## Paper changes

- **Abstract:** Updated external smoke count from 360 to **1,560 cells** (adds 1,200-cell BFCL export-50)
- **§6.4 external tables:** Added Table 4c (BFCL export-50) with per-model divergence breakdown
- **§6.4.2:** Replaced "BFCL validity rerun queued" note with completed BFCL export-50 results + framework finding
- **Wilson CI table (§6.4):** Added 3 new rows for BFCL export-50 (Llama `int4_sim`, Mistral `int4_sim`, `int8`/`noop` combined)
- **§14.2:** Updated from "360 GPU cells" to "1,560 GPU cells"; BFCL expand marked Completed
- **§15 limitation 6:** Updated from "BFCL 4-prompt pilot only" to BFCL expanded to 1,200-cell panel
- **§18 conclusion:** Updated external cell counts
- **Version bumped:** v2.5.1 → v2.5.2

---

## External cell count summary (v2.5.2)

| Panel | Cells | Models | Status |
|-------|------:|--------|--------|
| LongBench/RULER/BFCL pilot/HumanEval bundled | 216 | Llama only | Completed |
| MBPP both-model smoke | 144 | Llama + Mistral | Completed |
| **BFCL export-50 drift panel** | **1,200** | **Llama + Mistral** | **Completed** |
| KIVI offline adapter panel | 640 | Llama only | Completed (separate category) |
| RULER 12K | 24 | Llama only | Running now |

**Total external smoke (excl. KIVI): 1,560 cells**
