"""Text-table renderers for ExactKV histogram dicts.

All renderers accept pre-computed histogram dicts (output of
``exactkv.analysis.histograms``) and return Markdown strings.
No images.  No model re-runs.
No timing, latency, throughput, or speedup fields.

Public API
----------
``render_accepted_length_table(histogram)``
    Markdown table from an accepted-length histogram.

``render_first_divergence_table(histogram)``
    Markdown table from a first-divergence histogram.

``render_rejection_count_table(histogram)``
    Markdown table from a rejection-count histogram.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(count: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{100 * count / total:.1f}%"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([_row(headers), sep] + [_row(r) for r in rows])


def _render_flat_histogram(
    histogram: dict[str, Any],
    title: str,
    value_label: str,
) -> str:
    """Render a flat (non-grouped) histogram as a Markdown table."""
    buckets: dict[str, int] = dict(histogram.get("buckets", {}))
    total: int = histogram.get("total", 0)

    if not buckets:
        return f"_No data for {title}._"

    headers = [value_label, "count", "share"]
    rows = [
        [label, str(count), _pct(count, total)]
        for label, count in buckets.items()
    ]
    rows.append(["**Total**", f"**{total}**", ""])

    return _md_table(headers, rows)


def _render_grouped_histogram(
    histogram: dict[str, Any],
    title: str,
    group_label: str,
    value_label: str,
) -> str:
    """Render a grouped histogram as a Markdown section with sub-tables."""
    lines: list[str] = []
    for group_key, sub_hist in sorted(histogram.items()):
        lines.append(f"**{group_label}:** `{group_key}`\n")
        lines.append(_render_flat_histogram(sub_hist, title, value_label))
        lines.append("")
    return "\n".join(lines)


def _is_grouped(histogram: dict[str, Any]) -> bool:
    """Return True if the histogram is a grouped dict of sub-histograms."""
    if not histogram:
        return False
    first_value = next(iter(histogram.values()))
    return isinstance(first_value, dict) and "buckets" in first_value


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------

def render_accepted_length_table(
    histogram: dict[str, Any],
    group_label: str = "group",
) -> str:
    """Render an accepted-length histogram as a Markdown text table.

    Args:
        histogram:   Output of ``accepted_length_histogram``.
        group_label: Label for the grouping dimension when the histogram is
                     grouped (e.g. ``"compressor"``, ``"draft_len"``).

    Returns:
        Markdown string with a pipe table of bucket → count → share.
        If grouped, one table per group.

    Note:
        No timing, throughput, latency, or speedup columns are included.
    """
    if _is_grouped(histogram):
        return _render_grouped_histogram(
            histogram, "accepted length", group_label, "avg_accept_len_bucket"
        )
    return _render_flat_histogram(
        histogram, "accepted length", "avg_accept_len_bucket"
    )


def render_first_divergence_table(
    histogram: dict[str, Any],
    group_label: str = "group",
) -> str:
    """Render a first-divergence histogram as a Markdown text table.

    Args:
        histogram:   Output of ``first_divergence_histogram``.
        group_label: Label for grouping dimension when grouped.

    Returns:
        Markdown string.  The ``"no_divergence"`` bucket indicates results
        where lossy output exactly matched full-KV output.

    Note:
        No timing, throughput, latency, or speedup columns are included.
    """
    if _is_grouped(histogram):
        return _render_grouped_histogram(
            histogram, "first divergence", group_label, "first_div_idx_bucket"
        )
    return _render_flat_histogram(
        histogram, "first divergence", "first_div_idx_bucket"
    )


def render_rejection_count_table(
    histogram: dict[str, Any],
    group_label: str = "group",
) -> str:
    """Render a rejection-count histogram as a Markdown text table.

    Args:
        histogram:   Output of ``rejection_count_histogram``.
        group_label: Label for grouping dimension when grouped.

    Returns:
        Markdown string.

    Note:
        No timing, throughput, latency, or speedup columns are included.
    """
    if _is_grouped(histogram):
        return _render_grouped_histogram(
            histogram, "rejection count", group_label, "total_rejected_bucket"
        )
    return _render_flat_histogram(
        histogram, "rejection count", "total_rejected_bucket"
    )
