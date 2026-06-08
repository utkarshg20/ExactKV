"""Markdown leaderboard table renderers for ExactKV reports.

All renderers accept pre-computed acceptance table dicts (output of
``exactkv.analysis.acceptance_tables``) and return Markdown strings.
No model re-runs.  No timing, latency, throughput, or speedup fields.

Public API
----------
``render_compressor_leaderboard(table)``
    Markdown table with one row per compressor.

``render_draft_len_leaderboard(table)``
    Markdown table with one row per draft length.

``render_compressor_x_draft_leaderboard(table)``
    Markdown table with one row per (compressor, draft_len) pair
    (sweep reports only).
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Public utility: average effective bit width
# ---------------------------------------------------------------------------

def average_effective_bit_width(
    key_bit_width: int | None,
    value_bit_width: int | None,
    full_bit_width: int = 32,
) -> float:
    """Return the average of K and V effective bit-widths.

    ``None`` means full precision and is treated as ``full_bit_width`` (default 32).

    This is a **metadata comparison aid only** — it is not a real memory
    measurement and must not be presented as evidence of memory savings.

    Args:
        key_bit_width:   Key-side bit width, or ``None`` for full precision.
        value_bit_width: Value-side bit width, or ``None`` for full precision.
        full_bit_width:  Bit width to use for full-precision sides (default 32).

    Returns:
        ``(k_bits + v_bits) / 2`` as a float.

    Examples:
        >>> average_effective_bit_width(8, 8)    # int8   → 8.0
        8.0
        >>> average_effective_bit_width(4, 4)    # int4   → 4.0
        4.0
        >>> average_effective_bit_width(8, 4)    # k8_v4  → 6.0
        6.0
        >>> average_effective_bit_width(8, 2)    # k8_v2  → 5.0
        5.0
        >>> average_effective_bit_width(None, 4) # kfull_v4, full=32 → 18.0
        18.0
        >>> average_effective_bit_width(8, None) # k8_vfull, full=32 → 20.0
        20.0
    """
    k = key_bit_width if key_bit_width is not None else full_bit_width
    v = value_bit_width if value_bit_width is not None else full_bit_width
    return (k + v) / 2.0


def _fmt_bits(bits: int | None) -> str:
    """Render a bit-width for table display: integer or 'full'."""
    return "full" if bits is None else str(bits)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any, decimals: int = 3) -> str:
    """Format a value for table display."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value) if value is not None else "—"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown pipe table from header and row lists."""
    # Compute column widths (at least as wide as the header)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _row(cells: list[str]) -> str:
        padded = [c.ljust(widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [_row(headers), sep] + [_row(r) for r in rows]
    return "\n".join(lines)


def _acceptance_row(row: dict[str, Any], extra_keys: list[str]) -> list[str]:
    """Build a table row from an acceptance-table dict."""
    cells = extra_keys[:]
    values = [str(row.get(k, "—")) for k in cells]
    values += [
        _fmt(row.get("mean_acceptance_rate", 0.0)),
        _fmt(row.get("mean_average_accepted_length", 0.0), decimals=2),
        str(row.get("total_drafted", 0)),
        str(row.get("total_accepted", 0)),
        str(row.get("total_rejected", 0)),
        str(row.get("total_corrections", 0)),
        str(row.get("num_runs", 0)),
        str(row.get("exactkv_failures", 0)),
    ]
    return [str(row.get(k, "—")) for k in extra_keys] + [
        _fmt(row.get("mean_acceptance_rate", 0.0)),
        _fmt(row.get("mean_average_accepted_length", 0.0), decimals=2),
        str(row.get("total_drafted", 0)),
        str(row.get("total_accepted", 0)),
        str(row.get("total_rejected", 0)),
        str(row.get("total_corrections", 0)),
        str(row.get("num_runs", 0)),
        str(row.get("exactkv_failures", 0)),
    ]


# ---------------------------------------------------------------------------
# Compressor leaderboard
# ---------------------------------------------------------------------------

def render_compressor_leaderboard(
    table: list[dict[str, Any]],
    *,
    compressor_caps: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render an acceptance leaderboard table grouped by compressor.

    Args:
        table:           Output of ``group_acceptance_by_compressor``.
        compressor_caps: Optional mapping ``compressor_name → capabilities dict``
                         used to add ``is_simulated``, ``supports_real_bytes_claim``,
                         K/V bit-width, and average effective bit-width columns.

    Returns:
        Markdown table string.  Empty string if ``table`` is empty.

    Note:
        No timing, throughput, latency, or speedup columns are included.
        Average effective bits is a metadata comparison aid only.
    """
    if not table:
        return "_No data._"

    use_caps = compressor_caps is not None

    headers = ["compressor"]
    if use_caps:
        headers += ["simulated", "real-bytes", "K bits", "V bits", "avg eff bits"]
    headers += [
        "accept_rate", "avg_accept_len",
        "drafted", "accepted", "rejected", "corrections",
        "runs", "exactkv_fail",
    ]

    rows = []
    for row in table:
        comp = row.get("compressor_name", "—")
        cells = [comp]
        if use_caps:
            caps = (compressor_caps or {}).get(comp, {})
            cells.append("yes" if caps.get("is_simulated") else "no")
            cells.append("yes" if caps.get("supports_real_bytes_claim") else "no")
            k_bits = caps.get("key_bit_width")    # None or int
            v_bits = caps.get("value_bit_width")  # None or int
            cells.append(_fmt_bits(k_bits))
            cells.append(_fmt_bits(v_bits))
            avg = average_effective_bit_width(k_bits, v_bits)
            cells.append(f"{avg:.1f}")
        cells += [
            _fmt(row.get("mean_acceptance_rate", 0.0)),
            _fmt(row.get("mean_average_accepted_length", 0.0), decimals=2),
            str(row.get("total_drafted", 0)),
            str(row.get("total_accepted", 0)),
            str(row.get("total_rejected", 0)),
            str(row.get("total_corrections", 0)),
            str(row.get("num_runs", 0)),
            str(row.get("exactkv_failures", 0)),
        ]
        rows.append(cells)

    return _md_table(headers, rows)


# ---------------------------------------------------------------------------
# Draft-length leaderboard
# ---------------------------------------------------------------------------

def render_draft_len_leaderboard(table: list[dict[str, Any]]) -> str:
    """Render an acceptance leaderboard table grouped by draft length.

    Args:
        table: Output of ``group_acceptance_by_draft_len``.

    Returns:
        Markdown table string.  Empty string if ``table`` is empty.
    """
    if not table:
        return "_No data._"

    headers = [
        "draft_len", "accept_rate", "avg_accept_len",
        "drafted", "accepted", "rejected", "corrections",
        "runs", "exactkv_fail",
    ]
    rows = []
    for row in table:
        rows.append([
            str(row.get("draft_len", "—")),
            _fmt(row.get("mean_acceptance_rate", 0.0)),
            _fmt(row.get("mean_average_accepted_length", 0.0), decimals=2),
            str(row.get("total_drafted", 0)),
            str(row.get("total_accepted", 0)),
            str(row.get("total_rejected", 0)),
            str(row.get("total_corrections", 0)),
            str(row.get("num_runs", 0)),
            str(row.get("exactkv_failures", 0)),
        ])
    return _md_table(headers, rows)


# ---------------------------------------------------------------------------
# Compressor × draft-length grid leaderboard (sweep reports)
# ---------------------------------------------------------------------------

def render_compressor_x_draft_leaderboard(
    table: list[dict[str, Any]],
    *,
    compressor_caps: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render an acceptance leaderboard table with one row per
    (compressor, draft_len) combination.

    Args:
        table:           Output of ``build_acceptance_table`` (primary sweep table).
        compressor_caps: Optional mapping ``compressor_name → capabilities dict``
                         used to add K/V bit-width and average effective bit-width
                         columns.

    Returns:
        Markdown table string.  Empty string if ``table`` is empty.
    """
    if not table:
        return "_No data._"

    use_caps = compressor_caps is not None

    headers = ["compressor", "draft_len"]
    if use_caps:
        headers += ["K bits", "V bits", "avg eff bits"]
    headers += [
        "accept_rate", "avg_accept_len",
        "drafted", "accepted", "rejected", "corrections",
        "runs", "exactkv_fail",
    ]

    rows = []
    for row in table:
        comp = str(row.get("compressor_name", "—"))
        cells = [comp, str(row.get("draft_len", "—"))]
        if use_caps:
            caps = (compressor_caps or {}).get(comp, {})
            k_bits = caps.get("key_bit_width")
            v_bits = caps.get("value_bit_width")
            cells.append(_fmt_bits(k_bits))
            cells.append(_fmt_bits(v_bits))
            cells.append(f"{average_effective_bit_width(k_bits, v_bits):.1f}")
        cells += [
            _fmt(row.get("mean_acceptance_rate", 0.0)),
            _fmt(row.get("mean_average_accepted_length", 0.0), decimals=2),
            str(row.get("total_drafted", 0)),
            str(row.get("total_accepted", 0)),
            str(row.get("total_rejected", 0)),
            str(row.get("total_corrections", 0)),
            str(row.get("num_runs", 0)),
            str(row.get("exactkv_failures", 0)),
        ]
        rows.append(cells)
    return _md_table(headers, rows)
