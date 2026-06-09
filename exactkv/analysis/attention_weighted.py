"""Proxy divergence analysis for V7 Phase A.

Analyses *where* lossy drafts diverge and how that relates to acceptance,
rejection, and correction behaviour using **existing benchmark report fields**.

This module does **not** use raw attention weights.  Unless a report contains
logged attention tensors or entropy fields (none do in Experiments 003–005),
all outputs are **proxy divergence analysis** — not true attention-weighted
analysis.  No attention weights are fabricated.

Public API
----------
``has_attention_weights(report)``
    Return whether any result carries logged attention-weight data.

``proxy_analysis_metadata(report)``
    Metadata dict labelling the analysis type honestly.

``divergence_by_compressor(report)``
    Per-compressor divergence counts and rates.

``rejection_by_compressor(report)``
    Per-compressor rejection and correction aggregates.

``divergence_position_table(report)``
    First-divergence position distribution per compressor.

``acceptance_vs_divergence_summary(report)``
    Joint acceptance and divergence summary per compressor.

``compare_reports_for_divergence(reports_by_name)``
    Cross-report divergence comparison for overlapping compressors.

No timing, latency, throughput, or speedup fields are produced.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from exactkv.analysis.histograms import DEFAULT_DIVERGENCE_BUCKETS, _assign_bucket
from exactkv.analysis.mismatch import first_lossy_divergences, rejection_position_summary

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds",
})

# Keys that would indicate genuine attention-weight logging (none in V1–V6 reports).
_ATTENTION_WEIGHT_KEYS = frozenset({
    "attention_weights",
    "attention_entropy",
    "attention_scores",
    "per_head_attention",
    "layer_attention",
})


def _get_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report.get("results", [])


def _assert_no_forbidden(obj: Any, path: str = "output") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden performance fields at {path}: {sorted(hits)}")
        for k, v in obj.items():
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")


def has_attention_weights(report: dict[str, Any]) -> bool:
    """Return True only when a result contains logged attention-weight fields.

    Experiments 003–005 do not log attention weights; this returns False for
    those reports.
    """
    for r in _get_results(report):
        for key in r:
            if key in _ATTENTION_WEIGHT_KEYS:
                return True
        for section in (r.get("lossy", {}), r.get("exactkv", {})):
            for key in section:
                if key in _ATTENTION_WEIGHT_KEYS:
                    return True
    return False


def proxy_analysis_metadata(report: dict[str, Any]) -> dict[str, Any]:
    """Return honest metadata about what kind of analysis is possible."""
    attn = has_attention_weights(report)
    return {
        "has_attention_weights": attn,
        "analysis_type": (
            "attention_weighted" if attn else "proxy_divergence"
        ),
        "note": (
            "Report contains logged attention fields; true attention-weighted "
            "analysis is possible."
            if attn
            else (
                "Proxy divergence analysis only: uses first_divergence_idx, "
                "acceptance, rejection, and correction fields. No attention "
                "weights are present or fabricated."
            )
        ),
    }


def _compressor_groups(records: list[dict[str, Any]], field: str) -> dict[str, list]:
    groups: dict[str, list] = {}
    for r in records:
        key = str(r.get(field, ""))
        groups.setdefault(key, []).append(r)
    return groups


def divergence_by_compressor(report: dict[str, Any]) -> dict[str, Any]:
    """Summarise lossy divergence behaviour grouped by compressor.

    Returns:
        Dict with ``metadata`` (from :func:`proxy_analysis_metadata`),
        ``total_runs``, ``lossy_divergence_count``, and ``by_compressor`` mapping
        compressor name → stats dict.
    """
    records = first_lossy_divergences(report)
    by_compressor: dict[str, dict[str, Any]] = {}

    for name, group in _compressor_groups(records, "compressor_name").items():
        diverged = [r for r in group if r["lossy_diverged"]]
        indices = [
            r["first_divergence_idx"]
            for r in diverged
            if r["first_divergence_idx"] is not None
        ]
        by_compressor[name] = {
            "total_runs": len(group),
            "lossy_divergence_count": len(diverged),
            "no_divergence_count": len(group) - len(diverged),
            "divergence_rate": len(diverged) / max(len(group), 1),
            "mean_first_divergence_idx": (
                sum(indices) / len(indices) if indices else None
            ),
            "min_first_divergence_idx": min(indices) if indices else None,
            "max_first_divergence_idx": max(indices) if indices else None,
        }

    diverged_all = [r for r in records if r["lossy_diverged"]]
    result = {
        "metadata": proxy_analysis_metadata(report),
        "total_runs": len(records),
        "lossy_divergence_count": len(diverged_all),
        "by_compressor": by_compressor,
    }
    _assert_no_forbidden(result)
    return result


def rejection_by_compressor(report: dict[str, Any]) -> dict[str, Any]:
    """Summarise rejection and correction counts grouped by compressor.

    Rejection and correction totals reconcile with per-result acceptance data.
    """
    records = rejection_position_summary(report)
    by_compressor: dict[str, dict[str, Any]] = {}

    for name, group in _compressor_groups(records, "compressor_name").items():
        total_rejected = sum(r["total_rejected"] for r in group)
        total_corrections = sum(r["total_corrections"] for r in group)
        rates = [r["acceptance_rate"] for r in group]
        by_compressor[name] = {
            "total_runs": len(group),
            "total_rejected": total_rejected,
            "total_corrections": total_corrections,
            "mean_acceptance_rate": sum(rates) / max(len(rates), 1),
            "max_rejected_single_run": max(
                (r["total_rejected"] for r in group), default=0
            ),
        }

    result = {
        "metadata": proxy_analysis_metadata(report),
        "total_runs": len(records),
        "aggregate_rejected": sum(r["total_rejected"] for r in records),
        "aggregate_corrections": sum(r["total_corrections"] for r in records),
        "by_compressor": by_compressor,
    }
    _assert_no_forbidden(result)
    return result


def divergence_position_table(
    report: dict[str, Any],
    *,
    group_by: str = "compressor_name",
) -> dict[str, Any]:
    """Bucket first-divergence positions per group (default: compressor).

    Uses the same bucket definitions as :mod:`exactkv.analysis.histograms`.
    The ``no_divergence`` bucket counts runs where lossy output matched full.
    """
    records = first_lossy_divergences(report)
    table: dict[str, dict[str, int]] = {}

    for key, group in _compressor_groups(records, group_by).items():
        buckets: OrderedDict[str, int] = OrderedDict(
            (label, 0) for _, _, label in DEFAULT_DIVERGENCE_BUCKETS
        )
        for r in group:
            idx = r["first_divergence_idx"] if r["lossy_diverged"] else None
            if not r["lossy_diverged"]:
                idx = None
            label = _assign_bucket(idx, DEFAULT_DIVERGENCE_BUCKETS)
            buckets[label] += 1
        table[key] = dict(buckets)

    result = {
        "metadata": proxy_analysis_metadata(report),
        "group_by": group_by,
        "bucket_labels": [label for _, _, label in DEFAULT_DIVERGENCE_BUCKETS],
        "table": table,
    }
    _assert_no_forbidden(result)
    return result


def acceptance_vs_divergence_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Joint per-compressor acceptance and divergence summary.

    This is a **correlational** proxy summary.  It does not establish causal
    attention importance — see :func:`proxy_analysis_metadata`.
    """
    div = divergence_by_compressor(report)["by_compressor"]
    rej = rejection_by_compressor(report)["by_compressor"]

    compressors = sorted(set(div) | set(rej))
    rows: list[dict[str, Any]] = []
    for name in compressors:
        d = div.get(name, {})
        r = rej.get(name, {})
        rows.append({
            "compressor_name": name,
            "total_runs": d.get("total_runs", r.get("total_runs", 0)),
            "divergence_rate": d.get("divergence_rate", 0.0),
            "mean_first_divergence_idx": d.get("mean_first_divergence_idx"),
            "mean_acceptance_rate": r.get("mean_acceptance_rate", 0.0),
            "total_rejected": r.get("total_rejected", 0),
            "total_corrections": r.get("total_corrections", 0),
        })

    result = {
        "metadata": proxy_analysis_metadata(report),
        "rows": rows,
        "note": (
            "Lower acceptance_rate often co-occurs with higher divergence_rate "
            "and earlier mean_first_divergence_idx. This is proxy correlation "
            "only; ExactKV does not claim causal attention importance from "
            "this summary alone."
        ),
    }
    _assert_no_forbidden(result)
    return result


def compare_reports_for_divergence(
    reports_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare divergence behaviour across named reports.

    Args:
        reports_by_name: Mapping of report label → report dict, e.g.
            ``{"experiment_003": report_dict, ...}``.

    Returns:
        Dict with per-report summaries and ``overlapping_compressors`` showing
        side-by-side divergence rates for compressors present in multiple reports.
    """
    per_report: dict[str, dict[str, Any]] = {}
    compressor_reports: dict[str, dict[str, float]] = {}

    for label, report in reports_by_name.items():
        summary = divergence_by_compressor(report)
        per_report[label] = {
            "metadata": summary["metadata"],
            "total_runs": summary["total_runs"],
            "lossy_divergence_count": summary["lossy_divergence_count"],
            "by_compressor": summary["by_compressor"],
        }
        for cname, stats in summary["by_compressor"].items():
            compressor_reports.setdefault(cname, {})[label] = stats["divergence_rate"]

    overlapping: list[dict[str, Any]] = []
    for cname, rates in sorted(compressor_reports.items()):
        if len(rates) > 1:
            overlapping.append({
                "compressor_name": cname,
                "divergence_rate_by_report": rates,
            })

    result = {
        "metadata": {
            "analysis_type": "proxy_divergence",
            "has_attention_weights": any(
                has_attention_weights(r) for r in reports_by_name.values()
            ),
            "report_labels": list(reports_by_name.keys()),
            "note": (
                "Cross-report comparison uses proxy divergence rates only. "
                "Differences in max_new_tokens or draft_len between experiments "
                "must be considered when interpreting overlaps."
            ),
        },
        "per_report": per_report,
        "overlapping_compressors": overlapping,
    }
    _assert_no_forbidden(result)
    return result
