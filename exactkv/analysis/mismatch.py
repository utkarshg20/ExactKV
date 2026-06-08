"""Mismatch and divergence analysis for ExactKV benchmark reports.

Analyses *where* lossy divergences and ExactKV rejections occur, without
re-running the model.

Key distinction
---------------
*Lossy divergence* — ``lossy.token_exact_match == False``.
  Expected behaviour; the compressor changes the output.  Not a failure.

*ExactKV rejection* — the verifier rejected a drafted token and emitted a
  correction.  Tracked via acceptance traces; ExactKV still produces the
  correct output.

*ExactKV failure* — ``exactkv_failure == True``.
  A bug; reported separately in ``failure_report.py``, not here.

No timing, latency, throughput, or speedup fields are produced.

Public API
----------
``first_lossy_divergences(report)``
    One record per result with ``first_divergence_idx`` and divergence flag.

``mismatch_position_summary(report)``
    Aggregate summary: counts, mean/min/max divergence position, per-compressor
    breakdown.

``rejection_position_summary(report)``
    Per-result rejection and correction counts (from acceptance data).
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Per-result divergence records
# ---------------------------------------------------------------------------

def first_lossy_divergences(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one record per result with lossy-vs-full divergence information.

    A ``first_divergence_idx`` of ``None`` means the lossy output exactly
    matched the full output for that run (no divergence).

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        List of dicts with:
        ``prompt_id``, ``category``, ``compressor_name``, ``draft_len``,
        ``first_divergence_idx`` (int or None), ``lossy_diverged`` (bool).
    """
    results = report.get("results", [])
    records = []
    for r in results:
        lossy = r.get("lossy", {})
        records.append({
            "prompt_id": r.get("prompt_id", ""),
            "category": r.get("category", ""),
            "compressor_name": r.get("compressor_name", ""),
            "draft_len": r.get("draft_len", ""),
            "first_divergence_idx": lossy.get("first_divergence_idx"),
            "lossy_diverged": not lossy.get("token_exact_match", True),
        })
    return records


# ---------------------------------------------------------------------------
# Aggregate mismatch summaries
# ---------------------------------------------------------------------------

def _group_by_field(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    """Group divergence records by a string field and compute per-group stats."""
    groups: dict[str, dict[str, Any]] = {}
    for r in records:
        key = str(r.get(field, ""))
        if key not in groups:
            groups[key] = {"total_runs": 0, "divergence_count": 0}
        groups[key]["total_runs"] += 1
        if r.get("lossy_diverged", False):
            groups[key]["divergence_count"] += 1
    # Add divergence_rate to each group
    for stats in groups.values():
        n = max(stats["total_runs"], 1)
        stats["divergence_rate"] = stats["divergence_count"] / n
    return groups


def mismatch_position_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Summarise lossy divergence positions across all results.

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        Dict with:
        ``total_runs``, ``lossy_divergence_count``, ``no_divergence_count``,
        ``mean_first_divergence_idx`` (float or None if no divergences),
        ``min_first_divergence_idx`` (int or None),
        ``max_first_divergence_idx`` (int or None),
        ``by_compressor`` (per-compressor divergence stats dict),
        ``by_category`` (per-category divergence stats dict).

    Note:
        ExactKV failures are NOT counted here.  They are reported separately
        in ``failure_report.py``.
    """
    records = first_lossy_divergences(report)
    diverged = [r for r in records if r["lossy_diverged"]]
    not_diverged = [r for r in records if not r["lossy_diverged"]]

    div_indices = [
        r["first_divergence_idx"]
        for r in diverged
        if r["first_divergence_idx"] is not None
    ]

    mean_idx: float | None = (
        sum(div_indices) / len(div_indices) if div_indices else None
    )
    min_idx: int | None = min(div_indices) if div_indices else None
    max_idx: int | None = max(div_indices) if div_indices else None

    return {
        "total_runs": len(records),
        "lossy_divergence_count": len(diverged),
        "no_divergence_count": len(not_diverged),
        "mean_first_divergence_idx": mean_idx,
        "min_first_divergence_idx": min_idx,
        "max_first_divergence_idx": max_idx,
        "by_compressor": _group_by_field(records, "compressor_name"),
        "by_category": _group_by_field(records, "category"),
    }


def rejection_position_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-result rejection and correction counts from acceptance data.

    Rejections are counted from ``exactkv.acceptance.total_rejected``.
    Corrections are counted from ``exactkv.acceptance.total_corrections``.
    These reflect ExactKV's internal verification behaviour — a non-zero
    rejection count is expected for lossy compressors and does NOT mean
    the final output is wrong.

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        List of dicts (one per result) with:
        ``prompt_id``, ``category``, ``compressor_name``, ``draft_len``,
        ``total_rejected``, ``total_corrections``, ``acceptance_rate``.
    """
    results = report.get("results", [])
    records = []
    for r in results:
        acc = r.get("exactkv", {}).get("acceptance", {})
        records.append({
            "prompt_id": r.get("prompt_id", ""),
            "category": r.get("category", ""),
            "compressor_name": r.get("compressor_name", ""),
            "draft_len": r.get("draft_len", ""),
            "total_rejected": acc.get("total_rejected", 0),
            "total_corrections": acc.get("total_corrections", 0),
            "acceptance_rate": acc.get("acceptance_rate", 0.0),
        })
    return records
