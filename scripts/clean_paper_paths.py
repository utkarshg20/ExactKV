#!/usr/bin/env python3
"""Strip inline artifact paths from paper main text; insert Appendix E."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "ExactKV_Technical_Report.md"

APPENDIX_E = """
## Appendix E: Artifact path index

Flat index for reproduction and case-study lookup. Panel summaries and cell counts
are in **Appendix A**; version labels in **Appendix D**.

### E.1 Primary panel paths

| Panel name | Path | Cells |
|------------|------|------:|
| Headline release panel | `reports/scale_7b/raw.json` | 1,500 |
| Headline leaderboard | `reports/public_release/leaderboard_final.json` | — |
| Evidence-plus panel | `reports/evidence_plus/raw.json` | 144 |
| External smoke summary | `reports/external_panels/summary_all.json` | 216 |
| External analysis pack | `reports/external_panels/analysis_pack.json` | — |
| MBPP code-drift smoke | `reports/external_panels/mbpp_gpu_raw.json` | 144 |
| BFCL export-50 drift | `reports/external_panels/bfcl_export_50_raw.json` | 1,200 |
| BFCL pilot merged | `reports/external_panels/bfcl_merged_raw.json` | 48 |
| LongBench pilot merged | `reports/external_panels/longbench_pilot_merged_raw.json` | 72 |
| RULER 8192 merged | `reports/external_panels/ruler_8192_merged_raw.json` | 24 |
| HumanEval pilot merged | `reports/external_panels/humaneval_merged_raw.json` | 24 |
| KIVI offline LongBench | `reports/external_panels/kivi_longbench_hf_raw.json` | 320 |
| KIVI offline MBPP | `reports/external_panels/kivi_mbpp_hf_raw.json` | 320 |
| HF LongBench v2.6 merged | `reports/external_panels/hf_longbench_v26_merged_raw.json` | 720 |
| BFCL validity v2.7 merged | `reports/external_panels/bfcl_validity_v27_merged_raw.json` | 1,200 |
| H2O eviction v2.8 merged | `reports/external_panels/h2o_v28_merged_raw.json` | 800 |
| v3.0 validation (both models) | `reports/external_panels/v30/` | 1,568 |
| Faithful adapter panel | `reports/external_panels/faithful/` | partial |
| External case-study pack | `reports/external_panels/case_studies_extracted.json` | 15+1 |
| LongBench overlap pack | `reports/external_panels/longbench_overlap_pack.{json,md}` | 720 |
| Phase-F kernel microbenchmark | `reports/phaseF_kernel_benchmark.json` | — |
| Logit autopsy summary | `reports/external_panels/logit_autopsy_summary.json` | 1,103 |
| Historical Phase-A demo | `reports/phaseA_benchmark.json` | — |

### E.2 Analysis and build scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_external_analysis_pack.py` | Wilson CIs, cross-panel aggregates |
| `scripts/build_longbench_overlap_pack.py` | LongBench reference-answer overlap |
| `scripts/analyze_logit_margins.py` | Top-k logit autopsy (§6.6, §6.10) |
| `scripts/build_downstream_validity_pack.py` | BFCL tool-call validity preservation |
| `scripts/run_external_panel.py` | External GPU panel runner |
| `scripts/run_evidence_plus_panel.py` | Evidence-plus panel runner |
| `scripts/run_phase_a_scale_benchmark.py` | Headline 1,500-cell panel |

### E.3 Case study index

| Case | Panel | Cell ID | Compressor | Notes |
|------|-------|---------|------------|-------|
| A–E, C, D | Headline release | `p00_p0_capital_france`, `p01_p1_simple_math`, `p02_p2_json_tool`, … | `int4_sim` | §8 release-panel table |
| F† | Historical Phase-A | `p2_json_tool` (Qwen 0.5B) | `int4_sim` | Illustrative only |
| G–I | Evidence-plus | `lc_001_ctx1024`, `lc_002_ctx512`, `p0_capital_france_ctx512` | `int4_sim` | Mistral long-context |
| J | BFCL pilot | `bfcl_ast_001_ctx2048` | `int4_sim` | Tool-call truncation |
| K | LongBench pilot | `lb_passage_retrieval_001_ctx4096` | `int4_sim` | Retrieval segment flip |
| L | RULER 8192 | `ruler_niah_single_4k_ctx8192` | `int4_sim` | Needle at 8K |
| M | HumanEval pilot | *(no divergent cells)* | — | Benign baseline |
| N | MBPP smoke | `mbpp_002_ctx1024` | `int4_sim` | Code body truncation |
| O | KIVI offline | `lb_narrativeqa_000_ctx2048` | `kivi_offline` | Catastrophic corruption |
| P | BFCL export-50 | `bfcl_parallel_parallel_6_ctx2048` | `int4_sim` | Mistral structural drift |
| Forensic A | HF LongBench v2.6 | NarrativeQA, 4K, Llama | `int8` | Near-tie noise (§7.1) |
| Forensic B | HF LongBench v2.6 | NarrativeQA, 2K, Llama | `int4_sim` | Distribution shift (§7.2) |
| Forensic C | H2O v2.8 | HotpotQA, 2K, Llama | `h2o_sim` | Attention destruction (§7.3) |

"""

REPLACEMENTS: list[tuple[str, str]] = [
    (
        "*All quantitative values are read from on-disk artifacts, primarily\n"
        "`reports/scale_7b/raw.json`, `reports/evidence_plus/raw.json`,\n"
        "`reports/external_panels/summary_all.json`, `reports/external_panels/*_merged_raw.json`,\n"
        "`reports/external_panels/v30/`, `reports/public_release/leaderboard_final.json`,\n"
        "`reports/phaseF_kernel_benchmark.json`, and `docs/METRIC_DEFINITIONS.md`. No\n"
        "results are invented. Claim boundaries follow `docs/CLAIM_BOUNDARIES.md` and the\n"
        "[`claim decision table`](../release_synthesis/claim_decision_table.md).*",
        "*All quantitative values are read from on-disk release artifacts "
        "(Appendix A, Appendix E). Metric definitions follow "
        "`docs/METRIC_DEFINITIONS.md`. Claim boundaries follow "
        "`docs/CLAIM_BOUNDARIES.md` and the "
        "[claim decision table](../release_synthesis/claim_decision_table.md). "
        "No results are invented.*",
    ),
    (
        "§6 is organized by **claim type** (drift panels, downstream validity, scaling,\n"
        "compressor curve) — not by internal release version. Version labels (v2.6–v3.0)\n"
        "appear only in Appendix D and artifact paths.",
        "§6 is organized by **claim type** (drift panels, downstream validity, scaling,\n"
        "compressor curve) — not by internal release version. Version labels (v2.6–v3.0)\n"
        "appear only in Appendix D.\n\n"
        "**Provenance convention:** Main-text tables cite panel names only. File paths,\n"
        "cell IDs, and reproduction commands are consolidated in **Appendix A** (panel\n"
        "inventory) and **Appendix E** (artifact path index).",
    ),
    (
        "Source: `reports/public_release/leaderboard_final.json`. Scores for validated built-in",
        "Source: headline release leaderboard (Appendix E). Scores for validated built-in",
    ),
    (
        "Source: `reports/scale_7b/raw.json` manifest + cell enumeration.",
        "Source: headline release panel (Appendix A).",
    ),
    (
        "Aggregates from `compressor_summary` in `reports/scale_7b/raw.json`",
        "Aggregates from `compressor_summary` in the headline release panel (Appendix A)",
    ),
    (
        "Full table: `reports/public_release/leaderboard_final.json`.",
        "Full table: headline release leaderboard (Appendix E).",
    ),
    (
        "Source: `reports/evidence_plus/raw.json`. RunPod **RTX A5000**",
        "Source: evidence-plus panel (Appendix A). RunPod **RTX A5000**",
    ),
    (
        "Source: `reports/external_panels/summary_all.json`,\n"
        "`reports/external_panels/*_merged_raw.json`, `reports/external_panels/analysis_pack.json`.",
        "Source: Llama-only external smoke panels and external analysis pack (Appendix E).",
    ),
    (
        "**Source:** `reports/external_panels/summary_all.json`,\n"
        "`reports/external_panels/*_merged_raw.json`, `reports/external_panels/mbpp_gpu_raw.json`,\n"
        "`reports/external_panels/analysis_pack.json`.",
        "**Source:** external smoke panels and analysis pack (Appendix E).",
    ),
    (
        "Source: `scripts/build_external_analysis_pack.py` (`wilson_ci` function, 95% two-sided).",
        "Source: external analysis pack — Wilson 95% CIs (Appendix E).",
    ),
    (
        "Source: `analysis_pack.json` → `totals.acceptance_full_rate_ci95`.",
        "Source: external analysis pack → `totals.acceptance_full_rate_ci95` (Appendix E).",
    ),
    (
        "Extracted from `reports/external_panels/case_studies_extracted.json` (15 divergent",
        "Extracted from external case-study pack (Appendix E; 15 divergent",
    ),
    (
        "**Source:** `reports/external_panels/bfcl_merged_raw.json`, cell",
        "**Source:** BFCL pilot panel (Appendix E), cell",
    ),
    (
        "Source: `reports/external_panels/bfcl_export_50_raw.json`.",
        "Source: BFCL export-50 panel (Appendix E).",
    ),
    (
        "**Source:** `reports/external_panels/longbench_pilot_merged_raw.json`, cell",
        "**Source:** LongBench pilot panel (Appendix E), cell",
    ),
    (
        "**Source:** `reports/external_panels/ruler_8192_merged_raw.json`, cell",
        "**Source:** RULER 8192 panel (Appendix E), cell",
    ),
    (
        "No divergent cells appear in `reports/external_panels/humaneval_merged_raw.json`",
        "No divergent cells appear in the HumanEval pilot panel (Appendix E)",
    ),
    (
        "**Source:** `reports/external_panels/mbpp_gpu_raw.json`, cell",
        "**Source:** MBPP smoke panel (Appendix E), cell",
    ),
    (
        "**Source:** `reports/external_panels/bfcl_export_50_raw.json`, cell",
        "**Source:** BFCL export-50 panel (Appendix E), cell",
    ),
    (
        "**Source:** `reports/external_panels/kivi_longbench_hf_raw.json` (320 cells),\n"
        "`reports/external_panels/kivi_mbpp_hf_raw.json` (320 cells).",
        "**Source:** KIVI offline adapter panel (Appendix E) — 320 LongBench + 320 MBPP cells.",
    ),
    (
        "Artifacts: `reports/external_panels/hf_longbench_v26_{Llama_3_1_8B,Mistral_7B}_raw.json`,\n"
        "`reports/external_panels/hf_longbench_v26_merged_raw.json`.",
        "Artifacts: HF LongBench v2.6 panel (Appendix E).",
    ),
    (
        "**Artifact paths (post-run):**\n"
        "- `reports/external_panels/hf_longbench_v26_Llama_3_1_8B_raw.json`\n"
        "- `reports/external_panels/hf_longbench_v26_Mistral_7B_raw.json`\n"
        "- `reports/external_panels/hf_longbench_v26_merged_raw.json`\n\n",
        "",
    ),
    (
        "Analysis: `scripts/analyze_logit_margins.py`. See §6.10 for full mechanistic analysis.",
        "Analysis: logit margin autopsy script (Appendix E). See §6.10 for full mechanistic analysis.",
    ),
    (
        "Script: `python3 scripts/build_longbench_overlap_pack.py --write`.",
        "Script: LongBench overlap pack builder (Appendix E).",
    ),
    (
        "Artifact: `reports/external_panels/longbench_overlap_pack.md`.",
        "Artifact: LongBench overlap pack (Appendix E).",
    ),
    (
        "**Status: Complete (both models). Mistral-7B-Instruct-v0.3: 784 cells. Llama-3.1-8B: 784 cells. Total: 1,568 v3.0 cells.** Source: `reports/external_panels/v30/`.",
        "**Status: Complete (both models). Mistral-7B-Instruct-v0.3: 784 cells. Llama-3.1-8B: 784 cells. Total: 1,568 v3.0 cells.** Source: v3.0 validation panel (Appendix A).",
    ),
    (
        "Artifacts: `reports/external_panels/faithful/*_raw.json`,\n"
        "`reports/external_panels/faithful/summary.md`. **`exactkv_failures=0`** on all",
        "Artifacts: faithful adapter panel (Appendix E). **`exactkv_failures=0`** on all",
    ),
    (
        "The table below uses fields saved in `reports/scale_7b/raw.json` (release panel,",
        "The table below uses fields from the headline release panel (Appendix A;",
    ),
    (
        "†Case F from `reports/phaseA_benchmark.json` / demo cards, **historical only**.",
        "†Case F from historical Phase-A demo cards (Appendix E), **historical only**.",
    ),
    (
        "Cases **G–I** from `reports/evidence_plus/raw.json` (512/1024 prefill buckets).",
        "Cases **G–I** from the evidence-plus panel (Appendix A; 512/1024 prefill buckets).",
    ),
    (
        "**Source:** `reports/external_panels/kivi_longbench_hf_raw.json`, cell",
        "**Source:** KIVI offline adapter panel (Appendix E), cell",
    ),
    (
        "**Source:** `reports/scale_7b/raw.json`, cell",
        "**Source:** headline release panel (Appendix A), cell",
    ),
    (
        "**Source:** `reports/evidence_plus/raw.json`, cell",
        "**Source:** evidence-plus panel (Appendix A), cell",
    ),
    (
        "Cases from `reports/external_panels/case_studies_extracted.json`. See also **§6.4.1**",
        "Cases from external case-study pack (Appendix E). See also **§6.4.1**",
    ),
    (
        "Source: `reports/phaseF_kernel_benchmark.json`. **Kernel microbenchmark only, NOT",
        "Source: Phase-F kernel microbenchmark (Appendix E). **Kernel microbenchmark only, NOT",
    ),
    (
        "**Verifier diagnostic timing** (evidence-plus panel, `reports/evidence_plus/raw.json`):",
        "**Verifier diagnostic timing** (evidence-plus panel, Appendix A):",
    ),
    (
        "and reproducible artifacts (`reports/scale_7b/raw.json`) for cross-compressor",
        "and reproducible headline release artifacts (Appendix A) for cross-compressor",
    ),
    (
        "in later June 2026 runs; see validated artifacts under `reports/external_panels/`.",
        "in later June 2026 runs; see validated external panel artifacts (Appendix A/E).",
    ),
    (
        "§6.4.6 and `reports/external_panels/kivi_longbench_hf_raw.json` (320 cells) +\n"
        "`reports/external_panels/kivi_mbpp_hf_raw.json` (320 cells).",
        "§6.4.6 and the KIVI offline adapter panel (640 cells, Appendix E).",
    ),
    (
        "(`reports/evidence_plus/raw.json`). Reproduce:",
        "(evidence-plus panel, Appendix A). Reproduce:",
    ),
    (
        "| **Later MBPP run** | Both Llama-3.1-8B and Mistral-7B (`mbpp_gpu_raw.json`, 144 cells) |",
        "| **Later MBPP run** | Both Llama-3.1-8B and Mistral-7B (MBPP smoke panel, 144 cells) |",
    ),
    (
        "| **Later BFCL export-50 run** | Both Llama-3.1-8B and Mistral-7B (`bfcl_export_50_raw.json`, 1,200 cells) |",
        "| **Later BFCL export-50 run** | Both Llama-3.1-8B and Mistral-7B (BFCL export-50 panel, 1,200 cells) |",
    ),
    (
        "validated). **`exactkv_failures = 0`**. **`noop` and `int8`: 0% divergence.**\n"
        "  Three divergent cells, all **Llama `int4_sim` at 1024 prefill** (`mbpp_002`,\n"
        "  `mbpp_004`). **Mistral: 0% divergence** (72/72). Generated code is **not**\n"
        "  executed against `test_list` (token drift only).\n\n"
        "8192-token RULER cells roughly double mean diagnostic wall-clock (12.2 s vs 6.4 s\n"
        "at 2K–4K). HumanEval shows a benign baseline on this panel, not evidence that code\n"
        "generation is immune under all compressors or longer generations.",
        "validated). **`exactkv_failures = 0`**. **`noop` and `int8`: 0% divergence.**\n"
        "  Three divergent cells, all **Llama `int4_sim` at 1024 prefill** (`mbpp_002`,\n"
        "  `mbpp_004`). **Mistral: 0% divergence** (72/72). Generated code is **not**\n"
        "  executed against `test_list` (token drift only).\n\n"
        "8192-token RULER cells roughly double mean diagnostic wall-clock (12.2 s vs 6.4 s\n"
        "at 2K–4K). HumanEval shows a benign baseline on this panel, not evidence that code\n"
        "generation is immune under all compressors or longer generations.",
    ),
]

# Fix MBPP line with mbpp_gpu_raw.json inline
MBPP_LINE_OLD = "- **MBPP:** **144 GPU cells** on bundled 6-prompt pilot (`mbpp_gpu_raw.json`,"
MBPP_LINE_NEW = "- **MBPP:** **144 GPU cells** on bundled 6-prompt pilot (MBPP smoke panel,"

REPRO_TABLE_OLD = """Source of truth (all artifacts):

| Artifact | Description | Cells |
|---|---|---|
| `reports/scale_7b/raw.json` | Headline panel | 1,500 |
| `reports/evidence_plus/raw.json` | Evidence-plus | 144 |
| `reports/external_panels/summary_all.json` | Initial 216-cell Llama-only smoke | 216 |
| `reports/external_panels/mbpp_gpu_raw.json` | MBPP code-drift smoke (both models) | 144 |
| `reports/external_panels/bfcl_export_50_raw.json` | BFCL export-50 tool-call drift | 1,200 |
| `reports/external_panels/kivi_*_hf_raw.json` | KIVI offline adapter diagnostic | 640 |
| `reports/external_panels/hf_longbench_v26_merged_raw.json` | HF LongBench v2.6 (both models) | 720 |
| `reports/external_panels/bfcl_validity_v27_merged_raw.json` | BFCL validity v2.7 (both models) | 1,200 |
| `reports/external_panels/h2o_v28_merged_raw.json` | H2O-style eviction v2.8 (both models) | 800 |
| `reports/external_panels/v30/` | v3.0 int6_sim + int4_per_vec_sim (both models) | 1,568 |
| **Total** | | **8,132** |"""

REPRO_TABLE_NEW = """Complete artifact path index: **Appendix A** (panel inventory) and
**Appendix E** (flat path index). Grand total: **8,132 cells**,
`exactkv_failures = 0`."""


def strip_paths_in_prose(text: str) -> str:
    """Remove remaining bare reports/ paths outside code fences."""
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)

    def clean_prose(chunk: str) -> str:
        # Leave docs/ and release_synthesis/ references; strip reports/ inline backticks
        chunk = re.sub(
            r"`reports/[^`]+`",
            lambda m: _panel_name_for_path(m.group(0)),
            chunk,
        )
        return chunk

    out: list[str] = []
    for i, part in enumerate(parts):
        out.append(part if i % 2 == 1 else clean_prose(part))
    return "".join(out)


def _panel_name_for_path(backtick_path: str) -> str:
    path = backtick_path.strip("`")
    names = {
        "reports/scale_7b/raw.json": "headline release panel (Appendix A)",
        "reports/evidence_plus/raw.json": "evidence-plus panel (Appendix A)",
        "reports/public_release/leaderboard_final.json": "headline leaderboard (Appendix E)",
        "reports/phaseF_kernel_benchmark.json": "Phase-F kernel microbenchmark (Appendix E)",
        "reports/phaseA_benchmark.json": "historical Phase-A demo (Appendix E)",
    }
    if path in names:
        return names[path]
    if "external_panels" in path:
        return "external panel artifact (Appendix E)"
    return backtick_path


def main() -> None:
    text = PAPER.read_text()
    marker = "## Appendix A:"
    if marker not in text:
        raise SystemExit("Appendix A marker not found")
    main_text, appendices = text.split(marker, 1)
    appendices = marker + appendices

    for old, new in REPLACEMENTS:
        if old not in main_text:
            continue
        main_text = main_text.replace(old, new)

    main_text = main_text.replace(MBPP_LINE_OLD, MBPP_LINE_NEW)
    main_text = main_text.replace(REPRO_TABLE_OLD, REPRO_TABLE_NEW)
    main_text = strip_paths_in_prose(main_text)

    # Insert Appendix E before Appendix B
    insert_at = "## Appendix B:"
    if insert_at not in appendices:
        raise SystemExit("Appendix B marker not found")
    appendices = appendices.replace(
        insert_at,
        APPENDIX_E.strip() + "\n\n" + insert_at,
    )
    appendices = appendices.replace(
        "main narrative. Use this table when tracing provenance in `reports/external_panels/`.",
        "main narrative. See **Appendix E** for flat artifact paths.",
    )

    PAPER.write_text(main_text + appendices)
    remaining = len(re.findall(r"`reports/", main_text))
    print(f"Wrote {PAPER}. Remaining `reports/` in main text (excl. code blocks): {remaining}")


if __name__ == "__main__":
    main()
