"""Acceptance-rate table builder for ExactKV benchmark and sweep reports.

All functions operate on report dicts (output of ``runner.run_suite`` or
``sweeps.run_sweep``) and produce Python dicts / lists — no model re-runs.

No timing, latency, throughput, or speedup fields are produced.

Public API
----------
``build_acceptance_table(report)``
    One row per unique (compressor_name, draft_len) pair.

``group_acceptance_by_compressor(report)``
    One row per unique compressor_name (draft_len collapsed).

``group_acceptance_by_draft_len(report)``
    One row per unique draft_len (compressor_name collapsed).

``group_acceptance_by_category(report)``
    One row per unique prompt category.

``write_acceptance_table_csv(table, path)``
    Write any acceptance table to CSV (creates parent dirs automatically).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Internal aggregation helper
# ---------------------------------------------------------------------------

def _aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate acceptance metrics across a list of per-result dicts.

    The invariant ``total_drafted == total_accepted + total_rejected`` is
    preserved by summation; no reconciliation assertion is needed here
    because the runner already enforces it per-result.

    Forbidden fields (timing/speedup) are deliberately excluded.
    """
    n = len(results)
    acc_blocks = [r.get("exactkv", {}).get("acceptance", {}) for r in results]

    total_drafted = sum(a.get("total_drafted", 0) for a in acc_blocks)
    total_accepted = sum(a.get("total_accepted", 0) for a in acc_blocks)
    total_rejected = sum(a.get("total_rejected", 0) for a in acc_blocks)
    total_corrections = sum(a.get("total_corrections", 0) for a in acc_blocks)
    exactkv_failures = sum(1 for r in results if r.get("exactkv_failure", False))

    denom = max(n, 1)
    acceptance_rates = [a.get("acceptance_rate", 0.0) for a in acc_blocks]
    avg_lengths = [a.get("avg_accepted_per_round", 0.0) for a in acc_blocks]

    return {
        "num_runs": n,
        "mean_acceptance_rate": sum(acceptance_rates) / denom,
        "mean_average_accepted_length": sum(avg_lengths) / denom,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_corrections": total_corrections,
        "exactkv_failures": exactkv_failures,
    }


def _group_results(
    results: list[dict[str, Any]],
    key_fn,
) -> dict[Any, list[dict[str, Any]]]:
    """Partition results into groups using ``key_fn(result) → hashable key``."""
    groups: dict[Any, list] = {}
    for r in results:
        k = key_fn(r)
        groups.setdefault(k, []).append(r)
    return groups


# ---------------------------------------------------------------------------
# Public table builders
# ---------------------------------------------------------------------------

def build_acceptance_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Build an acceptance table with one row per (compressor_name, draft_len) pair.

    This is the primary sweep analysis table — it shows how acceptance rate
    varies across the compressor × draft-length grid.

    Args:
        report: Dict with a ``"results"`` list (from ``run_suite`` or ``run_sweep``).

    Returns:
        Sorted list of dicts with keys:
        ``compressor_name``, ``draft_len``, ``num_runs``,
        ``mean_acceptance_rate``, ``mean_average_accepted_length``,
        ``total_drafted``, ``total_accepted``, ``total_rejected``,
        ``total_corrections``, ``exactkv_failures``.
    """
    results = report.get("results", [])
    groups = _group_results(
        results,
        key_fn=lambda r: (r.get("compressor_name", ""), r.get("draft_len", "")),
    )
    table = []
    for (compressor_name, draft_len), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "compressor_name": compressor_name,
            "draft_len": draft_len,
        }
        row.update(_aggregate_group(group))
        table.append(row)
    return table


def group_acceptance_by_compressor(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse all draft lengths — one row per compressor_name.

    Useful for comparing compressors regardless of draft-length variation.

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        Sorted list of dicts keyed by ``compressor_name``.
    """
    results = report.get("results", [])
    groups = _group_results(results, key_fn=lambda r: r.get("compressor_name", ""))
    table = []
    for compressor_name, group in sorted(groups.items()):
        row: dict[str, Any] = {"compressor_name": compressor_name}
        row.update(_aggregate_group(group))
        table.append(row)
    return table


def group_acceptance_by_draft_len(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse all compressors — one row per draft_len.

    Useful for isolating the effect of draft length on acceptance rate.

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        Sorted list of dicts keyed by ``draft_len``.
    """
    results = report.get("results", [])
    groups = _group_results(results, key_fn=lambda r: r.get("draft_len", ""))
    table = []
    for draft_len, group in sorted(groups.items()):
        row: dict[str, Any] = {"draft_len": draft_len}
        row.update(_aggregate_group(group))
        table.append(row)
    return table


def group_acceptance_by_category(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse compressors and draft lengths — one row per prompt category.

    Useful for understanding how prompt type affects acceptance behaviour.

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        Sorted list of dicts keyed by ``category``.
    """
    results = report.get("results", [])
    groups = _group_results(results, key_fn=lambda r: r.get("category", "unknown"))
    table = []
    for category, group in sorted(groups.items()):
        row: dict[str, Any] = {"category": category}
        row.update(_aggregate_group(group))
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_acceptance_table_csv(
    table: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write an acceptance table to a CSV file.

    Parent directories are created automatically.  Columns are inferred from
    the first row's keys (all rows are expected to have identical keys).

    Args:
        table: Output of any ``build_acceptance_table`` / ``group_by_*`` call.
        path:  Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not table:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(table[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
