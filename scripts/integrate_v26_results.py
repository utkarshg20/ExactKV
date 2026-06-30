#!/usr/bin/env python3
"""One-command integration of v2.6 HF LongBench results into the paper.

Reads the merged v2.6 artifact, computes Wilson CIs, generates:
  1. Paper table markdown (replaces placeholder Table 4d in the .md)
  2. Abstract update snippet
  3. Case study candidates (top divergent cells for §6.5 narrative)
  4. Reproducibility command block

Usage (after RunPod artifacts are copied back):
  python3 scripts/integrate_v26_results.py \
      --merged reports/external_panels/hf_longbench_v26_merged_raw.json \
      --md     paper/ExactKV_Technical_Report.md \
      --tex    paper/ExactKV_Technical_Report.tex \
      --write   # actually patch the files (default: dry-run only)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.1%}, {hi:.1%}]"


def build_md_table(merged: dict) -> str:
    """Build the replacement Table 4d markdown."""
    # Support both breakdown and summary_by_model_compressor field names
    summary = merged.get("summary_by_model_compressor", merged.get("breakdown", {}))
    rows: list[str] = []

    for model_key in sorted(summary):
        by_comp = summary[model_key]
        for comp in ("noop", "int8", "int4_sim"):
            # Support both _aggregate sub-key and flat dict
            agg = by_comp.get(comp) or {}
            if isinstance(agg, dict) and "_aggregate" in agg:
                agg = agg["_aggregate"]
            if not agg:
                continue
            n = agg.get("cells", 0)
            div = agg.get("divergent", 0)
            rate = agg.get("divergence_rate", div / n if n else 0)
            fail = agg.get("exactkv_failures", 0)
            lo, hi = wilson_ci(div, n)
            rows.append(
                f"| {model_key} | {comp} | {n} | {div} | {rate:.1%} | "
                f"{_fmt_ci(lo, hi)} | {fail} |"
            )

    header = (
        "| Model | Compressor | n | Divergent | Rate | CI₉₅ | ExactKV fail |\n"
        "|-------|-----------|---|-----------|------|------|-------------|"
    )
    return header + "\n" + "\n".join(rows)


def build_subset_breakdown(merged: dict) -> str:
    """Build per-subset int4_sim divergence table."""
    breakdown = merged.get("breakdown", {})
    all_subsets: set[str] = set()
    for by_comp in breakdown.values():
        for comp_data in by_comp.values():
            all_subsets.update(k for k in comp_data if k != "_aggregate")

    header = (
        "| Subset | Llama int4_sim | Mistral int4_sim |\n"
        "|--------|---------------|----------------|"
    )
    rows: list[str] = []

    llama_key   = next((k for k in breakdown if "llama"   in k.lower() or "Llama"   in k), None)
    mistral_key = next((k for k in breakdown if "mistral" in k.lower() or "Mistral" in k), None)

    def _cell(mkey: str | None, comp: str, sub: str) -> str:
        if not mkey:
            return "—"
        d = breakdown.get(mkey, {}).get(comp, {}).get(sub)
        if not d:
            return "—"
        return f"{d['divergent']}/{d['cells']} ({d['divergence_rate']:.0%})"

    for sub in sorted(all_subsets):
        rows.append(f"| {sub} | {_cell(llama_key, 'int4_sim', sub)} | {_cell(mistral_key, 'int4_sim', sub)} |")

    return header + "\n" + "\n".join(rows)


def find_interesting_cases(merged: dict, n: int = 3) -> list[dict]:
    """Find the most paper-worthy divergent cells."""
    cells = merged.get("cells", [])

    def _diverged(c: dict) -> bool:
        m = c.get("metrics") or {}
        if m.get("token_level_divergence"):
            return True
        return bool(c.get("diverged"))

    def _fdi(c: dict) -> int:
        m = c.get("metrics") or {}
        v = m.get("first_divergence_index") or c.get("first_divergence_index")
        if v is None:
            return 999
        try:
            return int(v)
        except (TypeError, ValueError):
            return 999

    divergent = [c for c in cells if _diverged(c)]
    # Prefer int4_sim cells at medium context with early first_divergence
    scored = []
    for c in divergent:
        comp = c.get("compressor_name", "")
        ctx = c.get("context_tokens", 0) or 0
        fd = _fdi(c)
        score = (3 if comp == "int4_sim" else 0) + (1 if 2000 <= ctx <= 5000 else 0) + max(0, 10 - fd)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:n]]


def build_abstract_update(merged: dict) -> str:
    """Generate the abstract sentence to add for v2.6."""
    total   = merged.get("total_cells", 0)
    div     = merged.get("divergent", 0)
    fail    = merged.get("exactkv_failures", 0)
    rate    = merged.get("divergence_rate", 0)
    lo, hi  = wilson_ci(div, total)
    subsets = 10  # fixed from design
    return (
        f"A **{total}-cell real HF LongBench drift panel** (§6.5) across {subsets} subsets, "
        f"both models, 2K/4K/8K context, reports `exactkv_failures={fail}` with "
        f"`int4_sim` divergence rate {rate:.1%} (95% CI {_fmt_ci(lo, hi)})."
    )


def build_tex_table(merged: dict) -> str:
    """Build LaTeX tabular for Table 4d."""
    breakdown = merged.get("breakdown", {})
    rows: list[str] = []

    for model_key in sorted(breakdown):
        by_comp = breakdown[model_key]
        model_short = model_key.replace("-", "").replace(".", "")[:18]
        for comp in ("noop", "int8", "int4_sim"):
            agg = by_comp.get(comp, {}).get("_aggregate", {})
            if not agg:
                continue
            n    = agg.get("cells", 0)
            div  = agg.get("divergent", 0)
            rate = agg.get("divergence_rate", 0)
            fail = agg.get("exactkv_failures", 0)
            lo, hi = wilson_ci(div, n)
            rows.append(
                f"\\texttt{{{model_key[:24]}}} & \\texttt{{{comp}}} & {n} & {div} & "
                f"{rate:.1%} & [{lo:.1%}, {hi:.1%}] & {fail} \\\\"
            )

    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate v2.6 HF LongBench results into paper")
    parser.add_argument("--merged", required=True, type=Path,
                        help="Merged v2.6 JSON (hf_longbench_v26_merged_raw.json)")
    parser.add_argument("--md",  default="paper/ExactKV_Technical_Report.md",  type=Path)
    parser.add_argument("--tex", default="paper/ExactKV_Technical_Report.tex", type=Path)
    parser.add_argument("--write", action="store_true",
                        help="Actually patch the .md and .tex files (default: dry-run)")
    args = parser.parse_args()

    if not args.merged.is_file():
        print(f"[ERROR] Merged artifact not found: {args.merged}", file=sys.stderr)
        print("  Run the v2.6 panel first and copy artifacts back:", file=sys.stderr)
        print("  rsync -avz runpod-a5000:/workspace/ExactKV/reports/external_panels/hf_longbench_v26_* \\", file=sys.stderr)
        print("      reports/external_panels/", file=sys.stderr)
        return 1

    merged = json.loads(args.merged.read_text(encoding="utf-8"))

    # Normalize field names — merged artifact uses total_divergent/overall_divergence_rate
    total = merged.get("total_cells", 0)
    div = merged.get("total_divergent", merged.get("divergent", 0))
    fail = merged.get("exactkv_failures", 0)
    rate = merged.get("overall_divergence_rate", merged.get("divergence_rate", 0))
    # Inject normalized fields for downstream functions
    merged.setdefault("divergent", div)
    merged.setdefault("divergence_rate", rate)

    # Build outputs
    md_table          = build_md_table(merged)
    subset_table      = build_subset_breakdown(merged)
    abstract_sentence = build_abstract_update(merged)
    tex_rows          = build_tex_table(merged)
    cases             = find_interesting_cases(merged, n=3)

    total  = merged.get("total_cells", 0)
    div    = merged.get("divergent", 0)
    fail   = merged.get("exactkv_failures", 0)
    rate   = merged.get("divergence_rate", 0)

    print("=" * 65)
    print("v2.6 HF LongBench integration preview")
    print("=" * 65)
    print(f"  Total cells:        {total}")
    print(f"  Divergent:          {div}  ({rate:.1%})")
    print(f"  exactkv_failures:   {fail}")
    print()
    print("── Table 4d replacement (markdown) ──")
    print(md_table)
    print()
    print("── Per-subset int4_sim breakdown ──")
    print(subset_table)
    print()
    print("── Abstract sentence to add ──")
    print(abstract_sentence)
    print()
    print("── Top-3 case study candidates ──")
    for i, c in enumerate(cases, 1):
        print(f"  {i}. {c.get('prompt_id','?')}  comp={c.get('compressor_name','?')}  "
              f"ctx={c.get('context_tokens','?')}  fd={c.get('first_divergence_index','?')}")
    print()

    if not args.write:
        print("[DRY-RUN] Pass --write to patch the .md and .tex files.")
        print("Suggested next steps:")
        print("  1. Review the table and case study candidates above.")
        print("  2. Run with --write to apply the patch.")
        print("  3. Manually add a case-study narrative for the best candidate.")
        print("  4. Bump version to v2.6 in both files.")
        print("  5. Run: bash scripts/build_paper_pdf.sh")
        return 0

    # ── Patch .md ────────────────────────────────────────────────────────────
    md_text = args.md.read_text(encoding="utf-8")

    # Replace placeholder table inside §6.5
    placeholder = (
        "| Subset | Model | Compressor | n | Divergent | Rate | CI₉₅ | ExactKV fail |\n"
        "|--------|-------|-----------|---|-----------|------|------|-------------|\n"
        "| narrativeqa | Llama | int4_sim | [pending] | [pending] | [pending] | [pending] | [pending] |\n"
        "| qasper | Llama | int4_sim | [pending] | [pending] | [pending] | [pending] | [pending] |\n"
        "| *(10 subsets × 2 models × noop/int8/int4_sim)* | | | | | | | |"
    )
    if placeholder in md_text:
        md_text = md_text.replace(placeholder, md_table + "\n\n**Per-subset int4_sim breakdown:**\n\n" + subset_table)
        print("[md] Replaced placeholder table.")
    else:
        print("[WARN] Could not find placeholder table in .md — manual edit needed.")

    # Update §6.5 status from "pending" to completed
    md_text = md_text.replace(
        "**Status:** Panel queued. Runbook: `scripts/run_hf_longbench_v26_panel.sh`.\nResults will be integrated as paper v2.6 once artifacts are returned from RunPod.",
        f"**Status:** Completed. {total} cells, `exactkv_failures={fail}`, `int4_sim` divergence {rate:.1%}.",
    )

    # Bump version header
    md_text = md_text.replace(
        "**ExactKV Technical Report (v2.5.4, paper hygiene + consolidated benchmark card)**",
        "**ExactKV Technical Report (v2.6, real HF LongBench drift panel)**",
    )

    args.md.write_text(md_text, encoding="utf-8")
    print(f"[md] Wrote {args.md}")

    # ── Patch .tex ───────────────────────────────────────────────────────────
    tex_text = args.tex.read_text(encoding="utf-8")

    # Replace placeholder description in \caption of tab:hf-longbench-v26
    tex_text = tex_text.replace(
        "Table will be replaced with real numbers in v2.6.",
        f"Total: {total} cells, \\texttt{{exactkv\\_failures={fail}}}, "
        f"\\texttt{{int4\\_sim}} divergence {rate:.1%}.",
    )

    # Bump version
    tex_text = tex_text.replace(
        r"Technical Report (v2.5.4)",
        r"Technical Report (v2.6)",
    )

    args.tex.write_text(tex_text, encoding="utf-8")
    print(f"[tex] Wrote {args.tex}")

    print()
    print("Integration complete. Next steps:")
    print(f"  1. Add case study narrative for top candidate in §6.5.")
    print(f"  2. Update abstract with: '{abstract_sentence[:80]}...'")
    print(f"  3. Update Appendix A 'All Completed Panels' row for v2.6.")
    print(f"  4. Run: bash scripts/build_paper_pdf.sh")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
