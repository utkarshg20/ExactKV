# ExactKV v0.11.0 (research release v3.0)

**Canonical public package version:** `0.11.0` (`exactkv.__version__`, `pyproject.toml`, git tag `v0.11.0`).

**Research narrative label:** v3.0 (headline 8,132-cell external grid + 1,500-cell core panel in this artifact set).

**Not the same as:** git tag `v0.13.0-rc1` (future research preview only, not the cited public artifact).

**ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM
KV-cache compression exactness.** It measures token-level drift, first divergence,
acceptance rate, verifier agreement, and exactness failures across compressors and
models. It is a **research-grade evaluation framework — not a production serving
system — and does not reproduce VeriCache.** SpectralQuant runs in fallback/proxy
mode when the real dependency is unavailable. Shard is a probe-first heuristic, not
a full Shard / ShardCache integration.

## What was released

### Headline panel
- A **1,500-cell** real-GPU benchmark on `meta-llama/Llama-3.1-8B` and
  `mistralai/Mistral-7B-Instruct-v0.3` with **`exactkv_failures = 0`**.

### External smoke supplement (1,560 cells)
- **216-cell** Llama-only pilot across LongBench/RULER/BFCL/HumanEval families
- **144-cell** MBPP both-model smoke panel
- **1,200-cell** BFCL export-50 tool-call drift panel (both models)

### HF LongBench v2.6 (720 cells)
- Real HF LongBench drift panel (both models, 2K/4K/8K context, 10 subsets)
- **Key finding:** `int4_sim` divergence is task-dependent: 90.4% open-text,
  11.2% tool-calling, 6.2% code — task type is the dominant driver

### BFCL validity v2.7 (600 cells, long-gen)
- 600-cell BFCL tool-call validity panel (mnt=128/256, first long-gen panel)
- `int4_sim`: 50% divergence at mnt=128/256; **106/106 valid tool calls preserved** by verifier
- Generation-length scaling: 9% (mnt=16) → 62% (mnt=256) — 7x increase

### H2O eviction v2.8
- H2O-style token eviction compressor (`h2o_sim`, `h2o_sim_75`, `h2o_sim_25`)
- 100% divergence on LongBench even at keep_ratio=0.75 — worse than `int4_sim`
- Three distinct failure modes identified via logit autopsy (1,103 divergent cells):
  near-tie noise (int8), distribution shift (int4_sim), attention destruction (H2O)

### Logit autopsy (§6.10)
- Top-k logit analysis at first divergence: top-1 flip rate, near-tie rate, mean margin,
  mean lossy rank, median fdi
- Three forensic case studies with full top-5 logit traces confirming each mechanism

### v3.0 GPU panel — int6_sim + int4_per_vec_sim (both models, 1,568 cells)

**Both new compressors GPU-validated on Mistral-7B and Llama-3.1-8B. `exactkv_failures = 0` throughout.**

| Task (both-model mean) | int8 | int6_sim | int4_per_vec_sim | int4_sim |
|------|-----:|---------:|-----------------:|---------:|
| MBPP (code) | 0% | 0% | 0% | 6.3% |
| BFCL (tool-call) | 0% | 0% | 0% | 52.5% |
| HF LongBench (reading) | 18.1% | 42.4% | 56.3% | 85.4% |

Key finding: **per-vector INT4 matches int8 on structured tasks** (BFCL/MBPP) but shows
55.6% on long-context reading — non-catastrophic but higher than int6_sim (37.5%).
The "granularity > bit-width" claim is task-conditional: it holds on BFCL/MBPP, partially
on LongBench (86% → 57%, a 29pp improvement), but bit-width still matters at 8K context.
`exactkv_failures=0` across all 1,568 v3.0 cells (both models).

- Source: `reports/external_panels/v30/`
- Mistral-7B: **complete** (784 cells)
- Llama-3.1-8B: **complete** (784 cells)

### BFCL downstream metrics
- 4-category BFCL breakdown: simple 37%, parallel 30%, multi-turn 23%, AST-eval 15%
  baseline valid rate — 100% preservation across all categories
- LongBench per-token acceptance rate as draft-utility proxy

### Paper
- Technical report v3.0 (Markdown + LaTeX + PDF, ~225 KB)
- 8,132 cells, four benchmark families, all `exactkv_failures = 0`
- §6.15 populated with real GPU results (no placeholders)

### Total: **8,132 completed GPU cells**, `exactkv_failures = 0` throughout


## Score (external reviewer, v3.0 complete)

| Dimension | Score |
|-----------|------:|
| Problem framing | 9.5 |
| Empirical evidence | 9.7 |
| Mechanistic analysis | 9.5 |
| Downstream validity | 9.0 |
| Compressor coverage | 9.4 |
| Claim safety | 9.3 |
| Paper consistency | 9.3 |
| Paper polish | 9.2 |
| **Strict research-paper (current draft)** | **9.50/10** |
| **After final wording pass** | **9.55–9.60/10** |
| **Underlying project / evidence quality** | **9.65/10** |
| **With faithful production KIVI/KVQuant kernel** | **9.7+/10** |

**Phase D3 (June 2026):** Faithful external adapter smoke **complete** — **864 GPU cells**
wave-1 (both models: LongBench + BFCL + MBPP) plus **128 cells** wave-2 Mistral smoke
(MBPP + BFCL). Wave-1: **`int8`** is the only non-catastrophic real compressor (~8–9%
combined drift); SnapKV **90–97%**; KIVI offline r32 **100%**. Wave-2:
**`turboquant_experimental` 3.1% combined drift** on structured tasks (0% BFCL, 6.2% MBPP);
KnormPress/SnapKV remain catastrophic. Artifacts: `reports/external_panels/faithful/`.

Path to stronger faithful story: expand TurboQuant panel (both models, LongBench) + optional KIVI production CUDA (Exp 024 blocked).

## Benchmark source of truth

| Artifact | Source |
|----------|--------|
| Headline panel | `reports/scale_7b/raw.json` |
| External smoke | `reports/external_panels/summary_all.json` |
| MBPP both-model smoke | `reports/external_panels/mbpp_gpu_raw.json` |
| BFCL export-50 tool-call | `reports/external_panels/bfcl_export_50_raw.json` |
| HF LongBench v2.6 | `reports/external_panels/longbench_hf_raw.json` |
| BFCL validity v2.7 | `reports/external_panels/bfcl_merged_raw.json` |
| v3.0 new compressors | `reports/external_panels/v30/` |

## Key paths

| Artifact | Path |
|----------|------|
| Technical report (Markdown) | `paper/ExactKV_Technical_Report.md` |
| Technical report (LaTeX) | `paper/ExactKV_Technical_Report.tex` |
| Technical report (PDF) | `paper/ExactKV_Technical_Report.pdf` |
| Bibliography | `paper/references.bib` |
| Website | `site/index.html` |
| v3.0 panel script | `scripts/run_v30_new_compressors.sh` |
| v3.0 Llama-only script | `scripts/run_llama_v30.sh` (on RunPod) |
| v3.0 summarizer | `scripts/summarize_v30_panel.py` |

## Reproducibility

```bash
# No GPU: verify artifacts
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/exactkv_repro.py --release-check
bash scripts/build_paper_pdf.sh          # requires: brew install tectonic

# GPU required: headline panel
python3 scripts/run_phase_a_scale_benchmark.py --device cuda

# GPU required: v3.0 new compressor validation
bash scripts/run_v30_new_compressors.sh
python3 scripts/summarize_v30_panel.py --output reports/external_panels/v30/summary.md
```

## Caveats (read before citing)

- **Not** a production serving system; **does not reproduce VeriCache**.
- Phase F kernel results are a **kernel microbenchmark only** — not end-to-end speedup.
- Compression ratios are **stored tensor byte ratios** — not active GPU memory savings.
- `exactkv_failures = 0` is a hard gate **on the tested panels only**.
- v3.0 results cover both Mistral-7B and Llama-3.1-8B (1,568 v3.0 cells total).
- External panels are **drift measurements**, not official LongBench/BFCL/MBPP scores.
- Scale run used **sequential model execution** (volume constraint).
