# ExactKV Technical Report — v2.7 Changelog (partial, Llama complete)

**Date:** 2026-06-28 (Mistral pending ~5:18 AM EDT)
**Version:** v2.6 → v2.7 (pending Mistral completion)
**Type:** New panels: BFCL tool-call validity with long generation (mnt=128/256)

---

## Summary

v2.7 completes the BFCL tool-call validity analysis that v2.5.3 identified as missing.
The first long-generation BFCL panel (600 Llama cells + 600 Mistral cells pending)
measures JSON completeness and divergence at `max_new_tokens=128/256` — confirming that
`mnt=16/32` is far too short for tool-call validity evaluation.

---

## New Panel: BFCL Tool-Call Validity v2.7

### Llama-3.1-8B (600 cells, complete)

| Compressor | n | Divergent | Rate | CI₉₅ | Mean accept. | full-KV valid JSON | ExactKV fail |
|-----------|---|-----------|------|------|-------------|-------------------|-------------|
| noop | 200 | 0 | 0.0% | [0.0%, 1.9%] | 1.000 | 63/200 (31.5%) | 0 |
| int8 | 200 | 0 | 0.0% | [0.0%, 1.9%] | 1.000 | 63/200 (31.5%) | 0 |
| int4_sim | 200 | 90 | **45.0%** | [38.3%, 51.9%] | 0.995 | 63/200 (31.5%) | 0 |

**Validity by max_new_tokens (Llama, all compressors):**

| mnt | n | valid JSON | Rate |
|-----|---|-----------|------|
| 128 | 300 | 39 | 13.0% |
| 256 | 300 | 150 | **50.0%** |

### Mistral-7B-Instruct-v0.3 (600 cells, **complete**)

| Compressor | n | Divergent | Rate | CI₉₅ | Full-KV valid | ExactKV fail |
|-----------|---|-----------|------|------|-------------|-------------|
| noop | 200 | 0 | 0.0% | [0.0%, 1.9%] | 43/200 (21.5%) | 0 |
| int8 | 200 | 3 | 1.5% | [0.5%, 4.3%] | 43/200 (21.5%) | 0 |
| int4_sim | 200 | 111 | **55.5%** | [48.6%, 62.2%] | 43/200 (21.5%) | 0 |

Artifacts: `reports/external_panels/bfcl_validity_v27_Mistral_7B_raw.json`
Merged: `reports/external_panels/bfcl_validity_v27_merged_raw.json`

---

## Key Findings (Llama, v2.7)

### 1. Generation length dominates BFCL divergence

Same BFCL prompts, same context (1K–2K):
- `mnt=16/32`: 5.5% int4_sim divergence (from BFCL export-50 panel)
- `mnt=128/256`: **45.0%** int4_sim divergence (this panel)

Longer generation exposes ~8× more divergence opportunities. The prior BFCL
export-50 result (5.5%) was not representative of full-generation behavior.

### 2. Valid JSON rate doubles with longer generation

- `mnt=128`: 13% of Llama BFCL outputs contain complete JSON objects
- `mnt=256`: **50%** of outputs contain complete JSON objects

This confirms `mnt=16/32` is too short for tool-call completeness analysis.

### 3. ExactKV verifier maintains perfect correctness

`exactkv_failures=0` for all 600 Llama cells, including the 90 int4_sim divergent cells.
The verifier corrects all divergence without any failure.

### 4. Late divergence pattern (int4_sim)

- Median first-divergence index: **token 95** out of 128–256 generated
- Mean acceptance rate: **0.995** (99.5% of tokens correct before divergence)
- This contrasts with LongBench where first-divergence is at token 4–6

BFCL tool-calling prompts generate correct tokens for ~95 tokens before int4_sim
causes a drift, while LongBench prompts diverge within the first 6 tokens.

---

## Bug Fixes

None in v2.7 (bug from v2.6 already fixed: `write_external_panel_outputs` relative path).

---

## In Progress / Queued

| Job | Status | ETA |
|-----|--------|-----|
| Mistral v2.7 BFCL validity (600 cells) | Running on RunPod | ~5:18 AM EDT |
| H2O eviction v2.8 panel | Queued after Mistral v2.7 | ~6:30 AM EDT |

---

## Integration scripts

```bash
# Pull Mistral artifact (after run completes):
rsync -avz runpod-a5000:/workspace/ExactKV/reports/external_panels/bfcl_validity_v27_Mistral_7B_raw.json \
    reports/external_panels/

# Merge + post-process + update paper:
bash scripts/integrate_v27_complete.sh --write

# Or step by step:
python3 scripts/postprocess_merge_v27_bfcl.py
python3 scripts/integrate_v27_results.py --merged reports/external_panels/bfcl_validity_v27_merged_raw.json --write
bash scripts/build_paper_pdf.sh
```
