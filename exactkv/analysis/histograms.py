"""Histogram compute utilities for ExactKV benchmark and sweep reports.

All functions operate on existing report dicts and do NOT re-run the model.

No timing, latency, throughput, or speedup fields are produced or consumed.

Histogram structure
-------------------
Every histogram function returns a dict with two keys:

``"buckets"``
    OrderedDict mapping label → count, in bucket order.

``"total"``
    Total number of observations (sum of all bucket counts).

When ``group_by`` is supplied, the return value is instead a dict mapping
``group_key → {"buckets": ..., "total": ...}`` — one histogram per group.

Bucket types
------------
``Bucket``   A ``(low, high, label)`` tuple where ``low`` and ``high`` are
             inclusive integer bounds and ``high=None`` means "and above".
             ``low=None`` and ``high=None`` together mark a special sentinel
             bucket (e.g. ``"no_divergence"``).

Public API
----------
``accepted_length_histogram(report, buckets=None, group_by=None)``
    Bucket ``avg_accepted_per_round`` per result.

``first_divergence_histogram(report, buckets=None, group_by=None)``
    Bucket ``lossy.first_divergence_idx`` per result;
    ``None`` → ``"no_divergence"`` bucket.

``rejection_count_histogram(report, buckets=None, group_by=None)``
    Bucket ``exactkv.acceptance.total_rejected`` per result.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

# ---------------------------------------------------------------------------
# Bucket type: (low: int|None, high: int|None, label: str)
# low=None AND high=None → special sentinel bucket (e.g. "no_divergence").
# high=None with low >= 0  → "low+" open-ended upper bound.
# ---------------------------------------------------------------------------

Bucket = tuple[int | None, int | None, str]

# ---------------------------------------------------------------------------
# Default bucket definitions
# ---------------------------------------------------------------------------

#: Default buckets for ``accepted_length_histogram``.
#: Values are ``avg_accepted_per_round`` floored to int.
DEFAULT_ACCEPTED_BUCKETS: list[Bucket] = [
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 3, "2-3"),
    (4, 7, "4-7"),
    (8, 15, "8-15"),
    (16, None, "16+"),
]

#: Default buckets for ``first_divergence_histogram``.
#: The sentinel ``(None, None, "no_divergence")`` catches results where
#: ``lossy.first_divergence_idx`` is ``None`` (lossy output matched full).
DEFAULT_DIVERGENCE_BUCKETS: list[Bucket] = [
    (None, None, "no_divergence"),
    (0, 0, "0"),
    (1, 4, "1-4"),
    (5, 16, "5-16"),
    (17, 32, "17-32"),
    (33, None, "33+"),
]

#: Default buckets for ``rejection_count_histogram``.
#: Values are ``total_rejected`` (integer).
DEFAULT_REJECTION_BUCKETS: list[Bucket] = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, None, "11+"),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_histogram(buckets: list[Bucket]) -> dict[str, Any]:
    """Return an empty histogram dict with all bucket counts set to zero."""
    return {
        "buckets": OrderedDict((label, 0) for _, _, label in buckets),
        "total": 0,
    }


def _assign_bucket(value: int | None, buckets: list[Bucket]) -> str:
    """Return the label of the first bucket that contains ``value``.

    Sentinel bucket ``(None, None, label)`` matches only ``None`` values.
    Open-ended upper bucket ``(low, None, label)`` matches ``value >= low``.
    """
    for low, high, label in buckets:
        # Sentinel bucket
        if low is None and high is None:
            if value is None:
                return label
            continue
        # Normal numeric bucket — skip if value is None
        if value is None:
            continue
        v = int(value) if not isinstance(value, int) else value
        if high is None:
            if v >= low:
                return label
        else:
            if low <= v <= high:
                return label
    # Fallback: last label
    return buckets[-1][2]


def _get_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report.get("results", [])


def _build_histogram(
    values: list[tuple[str | None, int | None]],
    buckets: list[Bucket],
) -> dict[str, Any]:
    """Build a histogram from ``(group_key, value)`` pairs.

    When ``group_key`` is ``None``, produces a single flat histogram.
    When ``group_key`` values are non-None, produces a per-group dict.
    """
    uses_groups = any(gk is not None for gk, _ in values)

    if not uses_groups:
        h = _empty_histogram(buckets)
        for _, v in values:
            label = _assign_bucket(v, buckets)
            h["buckets"][label] += 1
            h["total"] += 1
        return h

    # Per-group histograms
    groups: dict[str, dict[str, Any]] = {}
    for gk, v in values:
        key = str(gk)
        if key not in groups:
            groups[key] = _empty_histogram(buckets)
        label = _assign_bucket(v, buckets)
        groups[key]["buckets"][label] += 1
        groups[key]["total"] += 1
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def accepted_length_histogram(
    report: dict[str, Any],
    buckets: list[Bucket] | None = None,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Bucket ``avg_accepted_per_round`` (floored) per result.

    Args:
        report:   Report dict (``run_suite`` or ``run_sweep`` output).
        buckets:  Custom bucket list.  Defaults to
                  :data:`DEFAULT_ACCEPTED_BUCKETS`.
        group_by: Optional field name to group by (e.g. ``"compressor_name"``,
                  ``"draft_len"``, ``"category"``).  If supplied, returns a
                  dict of per-group histograms instead of a single histogram.

    Returns:
        Flat histogram dict (``buckets``, ``total``) or per-group dict when
        ``group_by`` is specified.

    Note:
        No timing, throughput, latency, or speedup fields are produced.
    """
    _buckets = buckets if buckets is not None else DEFAULT_ACCEPTED_BUCKETS
    pairs: list[tuple[str | None, int | None]] = []

    for r in _get_results(report):
        acc = r.get("exactkv", {}).get("acceptance", {})
        raw = acc.get("avg_accepted_per_round")
        value = int(raw) if raw is not None else None
        group_key = str(r.get(group_by, "")) if group_by else None
        pairs.append((group_key, value))

    return _build_histogram(pairs, _buckets)


def first_divergence_histogram(
    report: dict[str, Any],
    buckets: list[Bucket] | None = None,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Bucket ``lossy.first_divergence_idx`` per result.

    Results where the lossy output matches full (``first_divergence_idx`` is
    ``None``) fall into the ``"no_divergence"`` bucket.

    Args:
        report:   Report dict.
        buckets:  Custom bucket list.  Defaults to
                  :data:`DEFAULT_DIVERGENCE_BUCKETS`.
        group_by: Optional grouping field.

    Returns:
        Flat histogram or per-group dict.

    Note:
        No timing, throughput, latency, or speedup fields are produced.
    """
    _buckets = buckets if buckets is not None else DEFAULT_DIVERGENCE_BUCKETS
    pairs: list[tuple[str | None, int | None]] = []

    for r in _get_results(report):
        lossy = r.get("lossy", {})
        value = lossy.get("first_divergence_idx")  # int or None
        group_key = str(r.get(group_by, "")) if group_by else None
        pairs.append((group_key, value))

    return _build_histogram(pairs, _buckets)


def rejection_count_histogram(
    report: dict[str, Any],
    buckets: list[Bucket] | None = None,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Bucket ``exactkv.acceptance.total_rejected`` per result.

    A non-zero rejection count means the ExactKV verification engine
    overrode one or more drafted tokens.  This is expected for lossy
    compressors and does NOT imply the final output is wrong.

    Args:
        report:   Report dict.
        buckets:  Custom bucket list.  Defaults to
                  :data:`DEFAULT_REJECTION_BUCKETS`.
        group_by: Optional grouping field.

    Returns:
        Flat histogram or per-group dict.

    Note:
        No timing, throughput, latency, or speedup fields are produced.
    """
    _buckets = buckets if buckets is not None else DEFAULT_REJECTION_BUCKETS
    pairs: list[tuple[str | None, int | None]] = []

    for r in _get_results(report):
        acc = r.get("exactkv", {}).get("acceptance", {})
        value = acc.get("total_rejected", 0)
        group_key = str(r.get(group_by, "")) if group_by else None
        pairs.append((group_key, value))

    return _build_histogram(pairs, _buckets)
