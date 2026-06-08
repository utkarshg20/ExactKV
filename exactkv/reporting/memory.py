"""Workspace-aware memory rendering helpers for ExactKV reports (V5 Phase C).

Converts the V5 workspace-aware memory fields stored in report result dicts
into human-readable Markdown tables.

Public API
----------
``format_bytes(n)``
    Format a byte count as a human-readable string (e.g. "1.2 MiB").

``render_workspace_memory_table(report)``
    Render a Markdown table of workspace memory fields for each compressor
    present in the report.

Design constraints (V5)
-----------------------
* ``total_kv_footprint_bytes`` is a conservative accounting sum, NOT a
  measured peak GPU memory value.
* Active GPU memory measurement is deferred to a later CUDA-specific phase.
* No timing, latency, throughput, or speedup fields are produced.
* Simulated sub-INT8 compressors are clearly flagged; their
  ``stored_kv_bytes`` reflects int8 container reality, not packed-bit savings.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Byte formatter
# ---------------------------------------------------------------------------

def format_bytes(n: int | float) -> str:
    """Format a byte count as a concise human-readable string.

    Args:
        n: Non-negative integer or float byte count.

    Returns:
        Formatted string such as ``"1.2 MiB"``, ``"48.8 KiB"``, or ``"512 B"``.
        Returns ``"—"`` for negative, None, or non-numeric values.

    Examples:
        >>> format_bytes(0)
        '0 B'
        >>> format_bytes(1023)
        '1023 B'
        >>> format_bytes(1024)
        '1.0 KiB'
        >>> format_bytes(1_048_576)
        '1.0 MiB'
        >>> format_bytes(2_097_152)
        '2.0 MiB'
        >>> format_bytes(-1)
        '—'
    """
    if not isinstance(n, (int, float)) or n < 0:
        return "—"
    n = int(n)
    for unit, divisor in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= divisor:
            return f"{n / divisor:.1f} {unit}"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Workspace field aggregator
# ---------------------------------------------------------------------------

def _collect_workspace_per_compressor(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return one workspace memory snapshot per compressor (first result seen).

    Uses the first result found for each compressor name.  For typical
    ExactKV reports all prompts use the same model, so byte values are
    consistent across prompts for the same compressor.

    Looks for workspace fields in both the raw ``memory`` sub-dict (live
    run_one results) and the enriched form (loaded from JSON written by
    write_json_report).  Falls back to 0 for any missing field.
    """
    seen: dict[str, dict[str, Any]] = {}
    for r in report.get("results", []):
        comp = r.get("compressor_name", "")
        if not comp or comp in seen:
            continue
        mem: dict[str, Any] = r.get("memory", {})
        caps: dict[str, Any] = r.get("compressor_capabilities", {})
        seen[comp] = {
            "stored_kv_bytes": mem.get("stored_kv_bytes", 0),
            "materialized_working_kv_bytes": mem.get("materialized_working_kv_bytes", 0),
            "metadata_bytes": mem.get("metadata_bytes", 0),
            "temporary_workspace_bytes": mem.get("temporary_workspace_bytes", 0),
            "total_kv_footprint_bytes": mem.get("total_kv_footprint_bytes", 0),
            # Prefer caps (enriched), fall back to mem (Phase A loaded report)
            "supports_real_bytes_claim": caps.get(
                "supports_real_bytes_claim",
                mem.get("supports_real_bytes_claim", True),
            ),
            "is_simulated": caps.get(
                "is_simulated",
                mem.get("is_simulated", False),
            ),
        }
    return seen


# ---------------------------------------------------------------------------
# Markdown table renderer
# ---------------------------------------------------------------------------

def _simple_md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a minimal Markdown pipe table from header and row lists."""
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_workspace_memory_table(report: dict[str, Any]) -> str:
    """Render a Markdown table of workspace memory fields for each compressor.

    The table lists per-compressor estimates of:
    * stored_kv_bytes — the compressed/quantised tensor storage
    * materialized_working_kv_bytes — dequantised working KV for attention
    * metadata_bytes — per-tensor scales, zero-points, etc.
    * temporary_workspace_bytes — conservative transient scratch estimate
    * total_kv_footprint_bytes — accounting sum of all four fields above
    * whether real byte claims apply
    * whether the compressor is simulated

    For legacy reports where all workspace fields are zero, returns a brief
    explanatory note rather than a misleading all-zero table.

    Important: total_kv_footprint_bytes is a **conservative accounting sum**,
    NOT a measured peak GPU memory value.

    Args:
        report: Report dict from run_suite / run_sweep / load_json_report.

    Returns:
        Markdown string (table, or a brief note for legacy reports).
    """
    per_comp = _collect_workspace_per_compressor(report)
    if not per_comp:
        return "_No workspace memory data available._\n"

    has_data = any(
        v.get("total_kv_footprint_bytes", 0) > 0 for v in per_comp.values()
    )
    if not has_data:
        return (
            "_Workspace memory fields are all zero — "
            "this may be a legacy report generated before V5 Phase A. "
            "Re-run with a V5+ ExactKV build to populate these fields._\n"
        )

    headers = [
        "Compressor",
        "Stored KV",
        "Materialized KV",
        "Metadata",
        "Temp workspace",
        "Total footprint †",
        "Real bytes?",
        "Simulated?",
    ]
    rows = []
    for comp in sorted(per_comp):
        ws = per_comp[comp]
        real = ws.get("supports_real_bytes_claim", True)
        sim = ws.get("is_simulated", False)
        rows.append([
            f"`{comp}`",
            format_bytes(ws.get("stored_kv_bytes", 0)),
            format_bytes(ws.get("materialized_working_kv_bytes", 0)),
            format_bytes(ws.get("metadata_bytes", 0)),
            format_bytes(ws.get("temporary_workspace_bytes", 0)),
            format_bytes(ws.get("total_kv_footprint_bytes", 0)),
            "yes" if real else "no ⚠️",
            "yes ⚠️" if sim else "no",
        ])

    table = _simple_md_table(headers, rows)
    footnote = (
        "\n† Total footprint = stored + materialized + metadata + temp workspace. "
        "This is a **conservative accounting sum, NOT a measured peak GPU memory value**."
    )
    return table + footnote + "\n"
