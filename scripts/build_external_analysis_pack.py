#!/usr/bin/env python3
"""Build external panel analysis pack from on-disk artifacts only."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "reports/external_panels"

MERGED: dict[str, Path] = {
    "longbench_pilot": EXT / "longbench_pilot_merged_raw.json",
    "ruler_2048_4096": EXT / "ruler_2048_4096_merged_raw.json",
    "ruler_8192": EXT / "ruler_8192_merged_raw.json",
    "bfcl": EXT / "bfcl_merged_raw.json",
    "humaneval": EXT / "humaneval_merged_raw.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bfcl_tool_call_validity(c: dict[str, Any]) -> dict[str, Any]:
    """Check whether a BFCL cell's output contains a valid JSON tool call.

    Three dimensions are checked on the ExactKV (verifier-corrected) output:
      - has_json_open: output contains at least one '{' or '['
      - json_parseable: we can extract and parse a JSON object or array
      - has_tool_name: the parsed JSON has a 'name' key (BFCL v3 format)

    Returns a dict with keys: has_json_open, json_parseable, has_tool_name, validity_tier.
    validity_tier: "valid_tool_call" | "partial_json" | "no_json"
    """
    import re

    text = str((c.get("exactkv") or {}).get("output_text") or "")
    result: dict[str, Any] = {
        "has_json_open": False,
        "json_parseable": False,
        "has_tool_name": False,
        "validity_tier": "no_json",
    }

    if not text.strip():
        return result

    result["has_json_open"] = bool(re.search(r"[\[{]", text))
    if not result["has_json_open"]:
        return result

    # Try to parse JSON fragments from the output.
    # Use a position-scanning approach to handle nested braces/brackets.
    result["validity_tier"] = "partial_json"

    def extract_json_fragments(s: str) -> list[str]:
        """Yield all top-level JSON objects/arrays by balanced-brace scanning."""
        fragments = []
        for start_ch, end_ch in (("{", "}"), ("[", "]")):
            i = 0
            while i < len(s):
                idx = s.find(start_ch, i)
                if idx == -1:
                    break
                depth = 0
                j = idx
                while j < len(s):
                    if s[j] == start_ch:
                        depth += 1
                    elif s[j] == end_ch:
                        depth -= 1
                        if depth == 0:
                            fragments.append(s[idx : j + 1])
                            break
                    j += 1
                i = idx + 1
        return fragments

    for fragment in extract_json_fragments(text):
        try:
            parsed = json.loads(fragment)
            result["json_parseable"] = True
            if isinstance(parsed, dict) and "name" in parsed:
                result["has_tool_name"] = True
                result["validity_tier"] = "valid_tool_call"
                break
            elif isinstance(parsed, list) and any(isinstance(x, dict) and "name" in x for x in parsed):
                result["has_tool_name"] = True
                result["validity_tier"] = "valid_tool_call"
                break
        except (json.JSONDecodeError, ValueError):
            continue

    return result


def bfcl_validity_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate BFCL validity across a list of ok cells."""
    n = len(cells)
    if n == 0:
        return {"cells": 0, "valid_tool_call_rate": None, "partial_json_rate": None,
                "no_json_rate": None, "valid_tool_call_ci95": wilson_ci(0, 0)}
    results = [bfcl_tool_call_validity(c) for c in cells]
    valid = sum(1 for r in results if r["validity_tier"] == "valid_tool_call")
    partial = sum(1 for r in results if r["validity_tier"] == "partial_json")
    no_json = sum(1 for r in results if r["validity_tier"] == "no_json")
    return {
        "cells": n,
        "valid_tool_call_rate": round(valid / n, 4),
        "valid_tool_call_ci95": wilson_ci(valid, n),
        "partial_json_rate": round(partial / n, 4),
        "no_json_rate": round(no_json / n, 4),
    }


def acceptance_rate_ci(cells: list[dict[str, Any]]) -> dict[str, Any | None]:
    """Wilson 95% CI for the fraction of draft tokens accepted (acceptance_rate == 1.0).

    Uses binary classification: did the cell achieve full acceptance (1.0) or not.
    For a more precise estimate, compute CIs on the mean acceptance rate via bootstrap,
    but Wilson on full-acceptance binary is a safe, conservative lower bound.
    """
    n = len(cells)
    full_accept = sum(1 for c in cells if metrics(c).get("acceptance_rate", 0.0) >= 1.0)
    return wilson_ci(full_accept, n)


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> dict[str, float | None]:
    """Wilson score interval for a proportion (two-sided, default 95%).

    Returns {"lower": ..., "upper": ..., "mid": ...} rounded to 4 decimal places.
    Returns None values when n == 0.

    Reference: Wilson (1927); Agresti & Coull (1998).
    """
    if n == 0:
        return {"lower": None, "upper": None, "mid": None}
    import math
    p_hat = successes / n
    z2 = z * z
    centre = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half = (z / (1 + z2 / n)) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return {"lower": round(lo, 4), "upper": round(hi, 4), "mid": round(centre, 4)}


def quantiles(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"mean": None, "median": None, "min": None, "p90": None, "p95": None, "max": None, "count": 0}
    s = sorted(vals)
    n = len(s)

    def qi(p: float) -> float:
        if n == 1:
            return s[0]
        idx = max(0, min(n - 1, round((n - 1) * p)))
        return s[idx]

    return {
        "mean": round(statistics.mean(s), 6),
        "median": round(statistics.median(s), 6),
        "min": round(s[0], 6),
        "p90": round(qi(0.9), 6),
        "p95": round(qi(0.95), 6),
        "max": round(s[-1], 6),
        "count": n,
    }


def hist(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[str(int(v))] = counts.get(str(int(v)), 0) + 1
    return dict(sorted(counts.items(), key=lambda x: int(x[0])))


def metrics(c: dict[str, Any]) -> dict[str, Any]:
    return c.get("metrics") or {}


def divergence_rate(cells: list[dict[str, Any]]) -> float:
    if not cells:
        return 0.0
    div = sum(1 for c in cells if metrics(c).get("token_level_divergence"))
    return div / len(cells)


def acceptance_vals(cells: list[dict[str, Any]]) -> list[float]:
    return [float(metrics(c).get("acceptance_rate", 0.0)) for c in cells]


def timing_vals(cells: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for c in cells:
        t = (c.get("timing_ms") or {}).get("total_cell")
        if t is not None and float(t) > 0:
            out.append(float(t))
    return out


def first_div_vals(cells: list[dict[str, Any]], *, divergent_only: bool = False) -> list[int]:
    out: list[int] = []
    for c in cells:
        m = metrics(c)
        if divergent_only and not m.get("token_level_divergence"):
            continue
        v = m.get("first_divergence_index")
        if v is not None:
            out.append(int(v))
    return out


def interpret_cell(c: dict[str, Any]) -> str:
    if not metrics(c).get("token_level_divergence"):
        return "benign"
    fam = str(c.get("dataset_family") or "")
    cat = str(c.get("task_category") or c.get("category") or "").lower()
    if fam == "bfcl" or "tool" in cat or "ast" in cat:
        return "tool-risk"
    if fam == "humaneval" or "code" in cat:
        return "code-risk"
    lossy = (c.get("lossy") or {}).get("output_text") or ""
    exact = (c.get("exactkv") or {}).get("output_text") or ""
    if ('{"' in lossy or '{"' in exact) and lossy != exact:
        return "structural"
    if fam in ("longbench", "ruler") or "retrieval" in cat or "niah" in cat or "qa" in cat or "gov" in cat:
        return "semantic"
    return "unknown"


def snippet_fields(c: dict[str, Any]) -> dict[str, Any]:
    lossy = c.get("lossy") or {}
    exactkv = c.get("exactkv") or {}
    lossy_snip = lossy.get("output_text")
    exact_snip = exactkv.get("output_text")
    avail = bool(lossy_snip) and bool(exact_snip)
    tail = 300
    return {
        "full_snippet": exact_snip[-tail:] if exact_snip else None,
        "lossy_snippet": lossy_snip[-tail:] if lossy_snip else None,
        "exactkv_snippet": exact_snip[-tail:] if exact_snip else None,
        "snippets_available": avail,
    }


def group_stats(cells: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], Any]) -> dict[Any, list[dict[str, Any]]]:
    g: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        g[key_fn(c)].append(c)
    return g


def fmt_pct(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def fmt_num(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> int:
    panels: dict[str, dict[str, Any]] = {}
    all_gpu_cells: list[dict[str, Any]] = []
    per_file_meta: dict[str, Any] = {}

    for name, path in MERGED.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        r = load(path)
        panels[name] = r
        per_file_meta[str(path)] = {
            "panel": name,
            "deterministic_mode": r.get("deterministic_mode"),
            "cells_total": len(r.get("cells", [])),
            "cells_ok": sum(1 for c in r.get("cells", []) if c.get("status") == "ok"),
            "exactkv_failures_reported": r.get("exactkv_failures"),
            "generated_at": r.get("generated_at"),
        }
        if r.get("deterministic_mode"):
            raise ValueError(f"Expected GPU artifact, got deterministic_mode=true: {path}")
        for c in r.get("cells", []):
            if c.get("status") != "ok":
                continue
            c2 = dict(c)
            c2["_panel"] = name
            c2["_source_file"] = str(path)
            all_gpu_cells.append(c2)

    offline_extra = []
    for p in EXT.glob("*_raw.json"):
        if "_merged" in p.name or "Llama" in p.name or "Mistral" in p.name:
            continue
        r = load(p)
        if r.get("deterministic_mode"):
            offline_extra.append({
                "file": str(p.relative_to(ROOT)),
                "cells_run": r.get("cells_run"),
                "family": r.get("dataset_family"),
                "smoke": r.get("smoke"),
            })

    by_panel: dict[str, Any] = {}
    for name, r in panels.items():
        cells = [c for c in r.get("cells", []) if c.get("status") == "ok"]
        n = len(cells)
        n_div = sum(1 for c in cells if metrics(c).get("token_level_divergence"))
        n_acc = sum(1 for c in cells if metrics(c).get("acceptance_rate", 0.0) == 1.0)
        by_panel[name] = {
            "cells_ok": n,
            "dataset_family": r.get("dataset_family"),
            "models_evaluated": r.get("models_evaluated"),
            "context_buckets": r.get("context_buckets"),
            "max_new_tokens_values": r.get("max_new_tokens_values"),
            "compressors": r.get("compressors"),
            "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
            "divergence_rate": divergence_rate(cells),
            "divergence_rate_ci95": wilson_ci(n_div, n),
            "acceptance": quantiles(acceptance_vals(cells)),
            "acceptance_full_rate_ci95": acceptance_rate_ci(cells),
            "timing_ms": quantiles(timing_vals(cells)),
            "first_divergence_histogram": hist(first_div_vals(cells, divergent_only=True)),
        }

    def agg_groups(key_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, grp in group_stats(all_gpu_cells, key_fn).items():
            n = len(grp)
            n_div = sum(1 for c in grp if metrics(c).get("token_level_divergence"))
            out[str(k)] = {
                "cells": n,
                "divergence_rate": divergence_rate(grp),
                "divergence_rate_ci95": wilson_ci(n_div, n),
                "acceptance": quantiles(acceptance_vals(grp)),
                "timing_ms": quantiles(timing_vals(grp)),
                "exactkv_failures": sum(1 for c in grp if c.get("exactkv_failure")),
                "first_divergence_histogram": hist(first_div_vals(grp, divergent_only=True)),
            }
        return out

    by_panel_compressor: dict[str, Any] = {}
    bfcl_validity_by_panel: dict[str, Any] = {}
    for name, r in panels.items():
        cells = [c for c in r.get("cells", []) if c.get("status") == "ok"]
        pc: dict[str, Any] = {}
        for comp, grp in group_stats(cells, lambda c: c.get("compressor_name")).items():
            divs = first_div_vals(grp, divergent_only=True)
            n = len(grp)
            n_div = sum(1 for c in grp if metrics(c).get("token_level_divergence"))
            pc[str(comp)] = {
                "cells": n,
                "divergence_rate": divergence_rate(grp),
                "divergence_rate_ci95": wilson_ci(n_div, n),
                "acceptance": quantiles(acceptance_vals(grp)),
                "acceptance_full_rate_ci95": acceptance_rate_ci(grp),
                "mean_first_divergence": round(statistics.mean(divs), 3) if divs else None,
                "exactkv_failures": sum(1 for c in grp if c.get("exactkv_failure")),
            }
        by_panel_compressor[name] = pc
        # BFCL-specific: tool-call validity summary per compressor
        if r.get("dataset_family") == "bfcl":
            bfcl_validity_by_panel[name] = {}
            for comp, grp in group_stats(cells, lambda c: c.get("compressor_name")).items():
                bfcl_validity_by_panel[name][str(comp)] = bfcl_validity_summary(grp)
            bfcl_validity_by_panel[name]["__all__"] = bfcl_validity_summary(cells)
            # Record the max_new_tokens used so reviewers know why valid_tool_call_rate may be 0
            mnts = sorted({c.get("max_new_tokens") for c in cells if c.get("max_new_tokens") is not None})
            bfcl_validity_by_panel[name]["__diagnostic__"] = {
                "max_new_tokens_used": mnts,
                "note": (
                    "BFCL smoke panels used short max_new_tokens for drift measurement. "
                    "Completeness of JSON tool calls requires max_new_tokens >= 128. "
                    "partial_json_rate = fraction that began but did not complete JSON. "
                    "Re-run with --max-new-tokens 128,256 for validity scoring."
                ),
            }

    by_panel_bucket: dict[str, Any] = {}
    for name, r in panels.items():
        cells = [c for c in r.get("cells", []) if c.get("status") == "ok"]
        pb: dict[str, Any] = {}
        for b, grp in group_stats(cells, lambda c: c.get("context_bucket")).items():
            pb[str(b)] = {
                "cells": len(grp),
                "divergence_rate": divergence_rate(grp),
                "timing_ms": quantiles(timing_vals(grp)),
            }
        by_panel_bucket[name] = pb

    by_category: dict[str, Any] = {}
    for k, grp in group_stats(
        all_gpu_cells,
        lambda c: f"{c.get('dataset_family')}/{c.get('task_category') or c.get('category')}",
    ).items():
        by_category[str(k)] = {
            "cells": len(grp),
            "divergence_rate": divergence_rate(grp),
            "acceptance": quantiles(acceptance_vals(grp)),
            "exactkv_failures": sum(1 for c in grp if c.get("exactkv_failure")),
        }

    divergent = [c for c in all_gpu_cells if metrics(c).get("token_level_divergence")]

    def rank_ids(mode: str, n: int = 15) -> list[str]:
        def key(c: dict[str, Any]) -> tuple:
            m = metrics(c)
            if mode == "earliest":
                return (m.get("first_divergence_index") if m.get("first_divergence_index") is not None else 9999,)
            if mode == "lowest_acceptance":
                return (m.get("acceptance_rate", 1.0),)
            if mode == "highest_context":
                return (-(c.get("context_bucket") or 0),)
            if mode == "tool_code_risk":
                label = interpret_cell(c)
                risk = 0 if label in ("tool-risk", "code-risk", "structural") else 1
                return (risk, m.get("first_divergence_index") or 9999)
            return (0,)

        return [str(c.get("prompt_id")) for c in sorted(divergent, key=key)[:n]]

    case_studies: list[dict[str, Any]] = []
    for c in divergent:
        m = metrics(c)
        case_studies.append({
            "dataset_family": c.get("dataset_family"),
            "task_category": c.get("task_category") or c.get("category"),
            "prompt_id": c.get("prompt_id"),
            "model_name": c.get("model_name"),
            "compressor_name": c.get("compressor_name"),
            "context_bucket": c.get("context_bucket"),
            "max_new_tokens": c.get("max_new_tokens"),
            "first_divergence_index": m.get("first_divergence_index"),
            "acceptance_rate": m.get("acceptance_rate"),
            "exactkv_failure": c.get("exactkv_failure"),
            "timing_ms": c.get("timing_ms"),
            "panel": c.get("_panel"),
            "source_file": c.get("_source_file"),
            **snippet_fields(c),
            "interpretation": interpret_cell(c),
        })

    # benign humaneval baseline
    for c in all_gpu_cells:
        if c.get("dataset_family") == "humaneval" and c.get("compressor_name") == "int4_sim":
            m = metrics(c)
            case_studies.append({
                "dataset_family": c.get("dataset_family"),
                "task_category": c.get("task_category") or c.get("category"),
                "prompt_id": c.get("prompt_id"),
                "model_name": c.get("model_name"),
                "compressor_name": c.get("compressor_name"),
                "context_bucket": c.get("context_bucket"),
                "max_new_tokens": c.get("max_new_tokens"),
                "first_divergence_index": m.get("first_divergence_index"),
                "acceptance_rate": m.get("acceptance_rate"),
                "exactkv_failure": c.get("exactkv_failure"),
                "timing_ms": c.get("timing_ms"),
                "panel": c.get("_panel"),
                "source_file": c.get("_source_file"),
                **snippet_fields(c),
                "interpretation": "benign",
                "note": "non-divergent HumanEval baseline (included for completeness)",
            })
            break

    summary_cross: dict[str, Any] = {}
    summary_path = EXT / "summary_all.json"
    if summary_path.is_file():
        sa = load(summary_path)
        for g, info in (sa.get("merged_groups") or {}).items():
            if g in by_panel:
                summary_cross[g] = {
                    "summary_cells_run": info.get("cells_run"),
                    "computed_cells_ok": by_panel[g]["cells_ok"],
                    "match": info.get("cells_run") == by_panel[g]["cells_ok"],
                    "summary_exactkv_failures": info.get("exactkv_failures"),
                    "computed_exactkv_failures": by_panel[g]["exactkv_failures"],
                    "summary_divergence_rate": info.get("divergence_rate"),
                    "computed_divergence_rate": by_panel[g]["divergence_rate"],
                }

    non_ok = [c for r in panels.values() for c in r.get("cells", []) if c.get("status") != "ok"]
    noop_int8_div = sum(
        1 for c in all_gpu_cells
        if c.get("compressor_name") in ("noop", "int8") and metrics(c).get("token_level_divergence")
    )

    contradictions: list[str] = []
    total_ok = len(all_gpu_cells)
    if total_ok != sum(by_panel[p]["cells_ok"] for p in by_panel):
        contradictions.append("Sum of per-panel cells does not match total GPU ok cells")
    for g, x in summary_cross.items():
        if not x.get("match"):
            contradictions.append(f"summary_all cell count mismatch for {g}")
        if x.get("summary_exactkv_failures") != x.get("computed_exactkv_failures"):
            contradictions.append(f"exactkv_failures mismatch in summary_all for {g}")
    if noop_int8_div:
        contradictions.append(f"noop/int8 have {noop_int8_div} divergent cells (expected 0)")
    if sum(1 for c in all_gpu_cells if c.get("exactkv_failure")):
        contradictions.append("Per-cell exactkv_failure > 0")
    for name, r in panels.items():
        if r.get("exactkv_failures") != 0:
            contradictions.append(f"{name} reports exactkv_failures={r.get('exactkv_failures')}")

    int4_cells = [c for c in all_gpu_cells if c.get("compressor_name") == "int4_sim"]
    non_int4_div = [c for c in all_gpu_cells if c.get("compressor_name") != "int4_sim" and metrics(c).get("token_level_divergence")]

    analysis: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "ExactKV drift panels only. Not official LongBench/RULER/BFCL/HumanEval scores.",
        "artifact_sources": {k: str(v) for k, v in MERGED.items()},
        "totals": {
            "external_gpu_cells_ok": total_ok,
            "external_gpu_cells_all_statuses": sum(len(r.get("cells", [])) for r in panels.values()),
            "cells_non_ok": len(non_ok),
            "divergent_cells": len(divergent),
            "exactkv_failures_gpu": sum(1 for c in all_gpu_cells if c.get("exactkv_failure")),
            "divergence_rate_overall": divergence_rate(all_gpu_cells),
            "divergence_rate_overall_ci95": wilson_ci(
                sum(1 for c in all_gpu_cells if metrics(c).get("token_level_divergence")),
                len(all_gpu_cells),
            ),
            "acceptance_overall": quantiles(acceptance_vals(all_gpu_cells)),
            "acceptance_full_rate_ci95": acceptance_rate_ci(all_gpu_cells),
            "timing_overall_ms": quantiles(timing_vals(all_gpu_cells)),
            "first_divergence_histogram_overall": hist(first_div_vals(all_gpu_cells, divergent_only=True)),
        },
        "by_panel": by_panel,
        "by_model": agg_groups(lambda c: c.get("model_name")),
        "by_compressor": agg_groups(lambda c: c.get("compressor_name")),
        "by_context_bucket": agg_groups(lambda c: c.get("context_bucket")),
        "by_max_new_tokens": agg_groups(lambda c: c.get("max_new_tokens")),
        "by_category": by_category,
        "by_panel_compressor": by_panel_compressor,
        "by_panel_bucket_timing": by_panel_bucket,
        "bfcl_tool_call_validity": bfcl_validity_by_panel,
        "divergent_rankings_prompt_ids": {
            "by_earliest_first_divergence": rank_ids("earliest"),
            "by_lowest_acceptance": rank_ids("lowest_acceptance"),
            "by_highest_context_bucket": rank_ids("highest_context"),
            "by_tool_code_risk": rank_ids("tool_code_risk"),
        },
        "summary_all_crosscheck": summary_cross,
        "offline_artifacts_excluded": offline_extra,
        "validation": {
            "all_merged_deterministic_mode_false": True,
            "exactkv_failures_zero_all_merged_reports": all(r.get("exactkv_failures") == 0 for r in panels.values()),
            "exactkv_failures_zero_per_cell": sum(1 for c in all_gpu_cells if c.get("exactkv_failure")) == 0,
            "divergence_only_int4_sim": len(non_int4_div) == 0,
            "non_int4_divergent_cells": len(non_int4_div),
            "noop_int8_divergence_count": noop_int8_div,
            "int4_sim_cells": len(int4_cells),
            "int4_sim_divergent_cells": sum(1 for c in int4_cells if metrics(c).get("token_level_divergence")),
            "int4_sim_divergence_rate": divergence_rate(int4_cells),
            "non_ok_cell_statuses": list({c.get("status") for c in non_ok}) if non_ok else [],
        },
        "contradictions": contradictions,
        "per_file_meta": per_file_meta,
    }

    (EXT / "analysis_pack.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (EXT / "case_studies_extracted.json").write_text(
        json.dumps({
            "generated_at": analysis["generated_at"],
            "divergent_count": len(divergent),
            "total_case_entries": len(case_studies),
            "case_studies": case_studies,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    # --- Markdown outputs ---
    write_analysis_pack_md(analysis, divergent, case_studies)
    write_paper_tables(analysis, case_studies)
    write_run_quality(analysis, contradictions, offline_extra, non_ok)

    print(json.dumps({
        "gpu_cells": total_ok,
        "divergent": len(divergent),
        "contradictions": contradictions,
        "files": [
            "reports/external_panels/analysis_pack.json",
            "reports/external_panels/analysis_pack.md",
            "reports/external_panels/case_studies_extracted.json",
            "reports/external_panels/paper_tables_external.md",
            "reports/external_panels/run_quality_report.md",
        ],
    }, indent=2))
    return 0


def write_analysis_pack_md(analysis: dict[str, Any], divergent: list, case_studies: list) -> None:
    t = analysis["totals"]
    lines = [
        "# External Panel Analysis Pack",
        "",
        f"Generated: {analysis['generated_at']}",
        "",
        analysis["claim_boundary"],
        "",
        "## Executive summary",
        "",
        f"- **Total GPU cells (ok):** {t['external_gpu_cells_ok']}",
        f"- **Divergent cells:** {t['divergent_cells']} ({fmt_pct(t['divergence_rate_overall'])} panel-wide rate)",
        f"- **exactkv_failures:** {t['exactkv_failures_gpu']}",
        f"- **Model:** meta-llama/Llama-3.1-8B only (all merged GPU artifacts)",
        f"- **Prompt source:** bundled pilot JSONL (not HF LongBench, not official scores)",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Cells ok | {t['external_gpu_cells_ok']} |",
        f"| Divergence rate | {fmt_pct(t['divergence_rate_overall'])} |",
        f"| Acceptance mean | {fmt_num(t['acceptance_overall']['mean'])} |",
        f"| Acceptance median | {fmt_num(t['acceptance_overall']['median'])} |",
        f"| Acceptance p90 | {fmt_num(t['acceptance_overall']['p90'])} |",
        f"| Timing mean ms | {fmt_num(t['timing_overall_ms']['mean'])} |",
        f"| Timing p50 ms | {fmt_num(t['timing_overall_ms']['median'])} |",
        f"| Timing p90 ms | {fmt_num(t['timing_overall_ms']['p90'])} |",
        f"| Timing p95 ms | {fmt_num(t['timing_overall_ms']['p95'])} |",
        f"| Timing max ms | {fmt_num(t['timing_overall_ms']['max'])} |",
        "",
        "## By panel",
        "",
        "| Panel | Cells | Div rate | Accept mean | Accept p90 | Failures | Mean ms | P90 ms |",
        "|-------|------:|---------:|------------:|-----------:|---------:|--------:|-------:|",
    ]
    for p, s in analysis["by_panel"].items():
        acc = s["acceptance"]
        tim = s["timing_ms"]
        lines.append(
            f"| {p} | {s['cells_ok']} | {fmt_pct(s['divergence_rate'])} | "
            f"{fmt_num(acc['mean'])} | {fmt_num(acc['p90'])} | {s['exactkv_failures']} | "
            f"{fmt_num(tim['mean'])} | {fmt_num(tim['p90'])} |"
        )

    lines.extend(["", "## By compressor (all panels)", ""])
    lines.append("| Compressor | Cells | Div rate | Accept mean | Accept min | Accept p90 | Mean 1st div |")
    lines.append("|------------|------:|---------:|------------:|-----------:|-----------:|-------------:|")
    for comp, s in analysis["by_compressor"].items():
        acc = s["acceptance"]
        divs = []
        for grp in analysis["by_panel_compressor"].values():
            if comp in grp and grp[comp].get("mean_first_divergence") is not None:
                divs.append(grp[comp]["mean_first_divergence"])
        mean_div = round(statistics.mean(divs), 2) if divs else None
        lines.append(
            f"| {comp} | {s['cells']} | {fmt_pct(s['divergence_rate'])} | "
            f"{fmt_num(acc['mean'])} | {fmt_num(acc['min'])} | {fmt_num(acc['p90'])} | {fmt_num(mean_div)} |"
        )

    lines.extend(["", "## By context bucket (all panels)", ""])
    lines.append("| Bucket | Cells | Div rate | Mean ms | P90 ms |")
    lines.append("|-------:|------:|---------:|--------:|-------:|")
    for b, s in sorted(analysis["by_context_bucket"].items(), key=lambda x: int(x[0])):
        tim = s["timing_ms"]
        lines.append(
            f"| {b} | {s['cells']} | {fmt_pct(s['divergence_rate'])} | "
            f"{fmt_num(tim['mean'])} | {fmt_num(tim['p90'])} |"
        )

    lines.extend(["", "## By category", ""])
    lines.append("| Category | Cells | Div rate | Accept mean |")
    lines.append("|----------|------:|---------:|------------:|")
    for cat, s in sorted(analysis["by_category"].items()):
        lines.append(
            f"| {cat} | {s['cells']} | {fmt_pct(s['divergence_rate'])} | {fmt_num(s['acceptance']['mean'])} |"
        )

    lines.extend(["", "## First-divergence histogram (divergent cells only)", ""])
    for p, s in analysis["by_panel"].items():
        h = s.get("first_divergence_histogram") or {}
        if h:
            lines.append(f"**{p}:** " + ", ".join(f"idx {k}→{v}" for k, v in h.items()))

    lines.extend(["", "## Divergent cell rankings", ""])
    for title, key in [
        ("Earliest first divergence", "by_earliest_first_divergence"),
        ("Lowest acceptance", "by_lowest_acceptance"),
        ("Highest context bucket", "by_highest_context_bucket"),
        ("Tool/code risk priority", "by_tool_code_risk"),
    ]:
        lines.append(f"### {title}")
        for pid in analysis["divergent_rankings_prompt_ids"][key][:10]:
            lines.append(f"- `{pid}`")
        lines.append("")

    lines.extend(["", "## Validation", ""])
    v = analysis["validation"]
    for k, val in v.items():
        lines.append(f"- **{k}:** {val}")
    if analysis["contradictions"]:
        lines.append("")
        lines.append("**Contradictions:**")
        for c in analysis["contradictions"]:
            lines.append(f"- {c}")
    else:
        lines.append("")
        lines.append("**No contradictions found.**")

    (EXT / "analysis_pack.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_tables(analysis: dict[str, Any], case_studies: list) -> None:
    lines = [
        "# Paper-ready tables: external smoke panels",
        "",
        "Copy into technical report. All values from merged GPU artifacts.",
        "",
        "## Table 1. External smoke panel summary",
        "",
        "| Dataset family | Prompt source | Categories | Context buckets | max_new_tokens | Cells | Compressors | Acceptance | Div rate | Mean 1st div† | exactkv_failure | Mean/p90 ms |",
        "|----------------|---------------|------------|-----------------|----------------|------:|-------------|----------:|---------:|--------------:|----------------:|------------:|",
    ]
    panel_meta = {
        "longbench_pilot": ("LongBench", "bundled pilot", "6 tasks (6 prompts)", "2048, 4096", "16, 32"),
        "ruler_2048_4096": ("RULER", "bundled pilot", "4 task types", "2048, 4096", "16, 32"),
        "ruler_8192": ("RULER", "bundled pilot", "4 task types", "8192", "16, 32"),
        "bfcl": ("BFCL", "bundled pilot (4 prompts)", "simple, parallel, multi_turn, ast_eval", "1024, 2048", "16, 32"),
        "humaneval": ("HumanEval", "bundled pilot (4 prompts)", "code", "1024, 2048", "32"),
    }
    for p, s in analysis["by_panel"].items():
        fam, src, cats, buckets, mnt = panel_meta.get(p, (p, "pilot", "", "", ""))
        tim = s["timing_ms"]
        int4 = analysis["by_panel_compressor"][p].get("int4_sim", {})
        mean_div = int4.get("mean_first_divergence")
        lines.append(
            f"| {fam} | {src} | {cats} | {buckets} | {mnt} | {s['cells_ok']} | "
            f"noop, int8, int4_sim | {fmt_num(s['acceptance']['mean'])} | {fmt_pct(s['divergence_rate'])} | "
            f"{fmt_num(mean_div)} | {s['exactkv_failures']} | {fmt_num(tim['mean'])}/{fmt_num(tim['p90'])} |"
        )
    lines.extend([
        "",
        "† Mean first-divergence index over divergent `int4_sim` cells only.",
        "",
        "## Table 2. External smoke findings by compressor",
        "",
        "| Compressor | Cells | Div rate | Accept mean | Accept median | Accept min | Accept p90 | Divergent cells |",
        "|------------|------:|---------:|------------:|--------------:|-----------:|-----------:|----------------:|",
    ])
    for comp, s in analysis["by_compressor"].items():
        acc = s["acceptance"]
        div_n = int(round(s["divergence_rate"] * s["cells"]))
        lines.append(
            f"| `{comp}` | {s['cells']} | {fmt_pct(s['divergence_rate'])} | "
            f"{fmt_num(acc['mean'])} | {fmt_num(acc['median'])} | {fmt_num(acc['min'])} | "
            f"{fmt_num(acc['p90'])} | {div_n} |"
        )

    lines.extend([
        "",
        "## Table 3. Context bucket summary",
        "",
        "| Bucket | Cells | Div rate | Mean ms | P50 ms | P90 ms | P95 ms | Max ms |",
        "|-------:|------:|---------:|--------:|-------:|-------:|-------:|-------:|",
    ])
    for b, s in sorted(analysis["by_context_bucket"].items(), key=lambda x: int(x[0])):
        tim = s["timing_ms"]
        lines.append(
            f"| {b} | {s['cells']} | {fmt_pct(s['divergence_rate'])} | "
            f"{fmt_num(tim['mean'])} | {fmt_num(tim['median'])} | {fmt_num(tim['p90'])} | "
            f"{fmt_num(tim['p95'])} | {fmt_num(tim['max'])} |"
        )

    lines.extend([
        "",
        "## Table 4. Notable divergent case studies",
        "",
        "| Family | Category | Model | Compressor | Ctx | mnt | 1st div | Accept | ExactKV | Interpretation | Snippets |",
        "|--------|----------|-------|------------|----:|----:|--------:|-------:|---------|----------------|----------|",
    ])
    div_cases = [c for c in case_studies if c.get("interpretation") != "benign"]
    div_cases.sort(key=lambda c: (c.get("first_divergence_index") or 9999, c.get("acceptance_rate", 1)))
    for c in div_cases[:12]:
        sn = "yes" if c.get("snippets_available") else "no"
        lines.append(
            f"| {c.get('dataset_family')} | {c.get('task_category')} | Llama-3.1-8B | "
            f"`{c.get('compressor_name')}` | {c.get('context_bucket')} | {c.get('max_new_tokens')} | "
            f"{c.get('first_divergence_index')} | {fmt_num(c.get('acceptance_rate'))} | "
            f"{'fail' if c.get('exactkv_failure') else 'ok'} | {c.get('interpretation')} | {sn} |"
        )

    lines.extend([
        "",
        "## Table 5. Limitations and skipped runs",
        "",
        "| Item | Status |",
        "|------|--------|",
        "| LongBench HF export | skipped (`datasets` not installed on pod) |",
        "| Mistral-7B external panels | failed (disk quota exceeded) |",
        "| RULER 16K / 32K | not run |",
        "| Official benchmark scores | not computed (drift panels only) |",
        "| Deterministic offline smoke JSON | excluded from GPU totals |",
        "| Real KIVI / KVQuant / SnapKV | not in external smoke runs |",
        "",
    ])
    (EXT / "paper_tables_external.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_quality(
    analysis: dict[str, Any],
    contradictions: list[str],
    offline_extra: list,
    non_ok: list,
) -> None:
    v = analysis["validation"]
    lines = [
        "# External panel run quality report",
        "",
        f"Generated: {analysis['generated_at']}",
        "",
        "## Artifact inventory",
        "",
    ]
    for path, meta in analysis["per_file_meta"].items():
        lines.append(f"- `{path}`: ok={meta['cells_ok']}, deterministic={meta['deterministic_mode']}, failures={meta['exactkv_failures_reported']}")

    lines.extend([
        "",
        "## GPU vs offline separation",
        "",
        f"- **GPU merged cells counted:** {analysis['totals']['external_gpu_cells_ok']}",
        f"- **Non-ok cells in merged files:** {analysis['totals']['cells_non_ok']}",
        "",
        "Offline deterministic smoke files (excluded from GPU analysis):",
    ])
    for o in offline_extra:
        lines.append(f"- `{o['file']}`: {o['cells_run']} cells, family={o['family']}, smoke={o.get('smoke')}")

    lines.extend([
        "",
        "## Validation checklist",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| All merged artifacts `deterministic_mode=false` | {v['all_merged_deterministic_mode_false']} |",
        f"| Report-level `exactkv_failures=0` | {v['exactkv_failures_zero_all_merged_reports']} |",
        f"| Per-cell `exactkv_failure=0` | {v['exactkv_failures_zero_per_cell']} |",
        f"| Divergence only in int4_sim | {v['divergence_only_int4_sim']} |",
        f"| noop/int8 divergence count | {v['noop_int8_divergence_count']} |",
        f"| int4_sim divergent cells | {v['int4_sim_divergent_cells']} / {v['int4_sim_cells']} ({fmt_pct(v['int4_sim_divergence_rate'])}) |",
        f"| summary_all cross-check | {all(x.get('match') for x in analysis.get('summary_all_crosscheck', {}).values())} |",
        "",
        "## summary_all.json cross-check",
        "",
    ])
    for g, x in analysis.get("summary_all_crosscheck", {}).items():
        lines.append(
            f"- **{g}:** summary cells={x['summary_cells_run']}, computed={x['computed_cells_ok']}, "
            f"match={x['match']}, div summary={fmt_pct(x['summary_divergence_rate'])}, "
            f"computed={fmt_pct(x['computed_divergence_rate'])}"
        )

    lines.extend(["", "## Contradictions", ""])
    if contradictions:
        for c in contradictions:
            lines.append(f"- {c}")
    else:
        lines.append("None.")

    lines.extend([
        "",
        "## Paper update safety",
        "",
    ])
    safe = not contradictions and v["exactkv_failures_zero_per_cell"] and v["divergence_only_int4_sim"]
    if safe:
        lines.append(
            "Artifacts are **internally consistent** and safe for paper updates with claim boundaries: "
            "drift panels only, bundled pilot prompts, Llama-3.1-8B only, not official benchmark scores."
        )
    else:
        lines.append("**Resolve contradictions before paper updates.**")

    readme = EXT / "README.md"
    if readme.is_file():
        lines.extend(["", "## Workflow notes (from README)", ""])
        for line in readme.read_text(encoding="utf-8").splitlines():
            if "skipped" in line.lower() or "failed" in line.lower() or "GPU scope" in line:
                lines.append(f"- {line.strip()}")

    (EXT / "run_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
