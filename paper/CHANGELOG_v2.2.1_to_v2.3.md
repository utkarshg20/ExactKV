# Changelog: v2.2.1 → v2.3

**Date:** June 2026  
**Artifacts:** `reports/external_panels/summary_all.json`, `reports/external_panels/*_merged_raw.json`

## Added

- **§6.4 External Benchmark Smoke Panels** with summary table (216 GPU cells, Llama-3.1-8B only).
- External case-study table (Cases J–M) and three expanded forensics:
  - BFCL `ast_eval` tool-call JSON truncation (Case J).
  - LongBench `passage_retrieval_en` segment drift (Case K).
  - RULER `niah_single` @ 8192 needle drift (Case L).
- HumanEval benign baseline note (no divergent cells in artifact).
- **§14.2** external panels marked completed with reproduce commands.
- **§14.3** structured future-work table (LongBench HF, RULER 16K/32K, BFCL validity, MBPP, HELMET, InfiniteBench, real compressors).
- PDF (`.tex`) external smoke subsection and updated abstract/limitations/conclusion.

## Updated

- Version header v2.2.1 → **v2.3**.
- Abstract and conclusion cite 216-cell external smoke supplement.
- **§15 Limitations** expanded for smoke-panel scope, official-score boundaries, MBPP/HELMET/InfiniteBench future work, logits gap, Mistral disk failure.
- **§17 Reproducibility** cites `reports/external_panels/summary_all.json`.
- Kernel table renumbered to Table 5 (external summary is Table 4).

## Not claimed (unchanged boundaries)

- Official LongBench/RULER/BFCL/HumanEval scores.
- VeriCache reproduction or compressed-KV draft novelty.
- Real KIVI/KVQuant/SnapKV/SpectralQuant/Shard head-to-head results.
- Mistral-7B external panels (failed: disk quota).
- LongBench HF export (skipped: `datasets` not installed).

## Self-review (strict)

| Dimension | v2.3 score | Notes |
|-----------|------------|-------|
| Research paper | **8.1 / 10** | Real external smoke data, claim-safe framing, still single-model external GPU |
| Technical report | **8.7 / 10** | Artifact-backed tables, forensic examples, reproducibility paths |
| Benchmark strength | **7.4 / 10** | Up from harness-only, still pilot JSONL not full LongBench |
| Case studies | **7.6 / 10** | Three external forensics + scale/evidence-plus cases |
| Systems/runtime | **6.8 / 10** | External timing 4.7–12.2 s/cell documented |

### What still blocks 8.5+ research

1. **Second model** on external panels (Mistral failed, disk).
2. **Real LongBench HF** or NVIDIA RULER generated prompts (not pilot JSONL).
3. **Real external compressors** (KIVI/KVQuant/SnapKV) in any headline or external panel.
4. **BFCL executability** and **HumanEval/MBPP pass@1** downstream analysis.
5. **Logits at divergence** and **confidence intervals**.
6. **RULER 16K/32K** and multi-model scaling curves.
