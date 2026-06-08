"""Top-level Markdown report renderer for ExactKV benchmark/sweep reports.

Converts an existing report dict (output of ``runner.run_suite`` or
``sweeps.run_sweep``) into a docs-ready Markdown string.

No model re-runs.  No timing, latency, throughput, or speedup metrics.

Key wording (always present in every rendered report)
-----------------------------------------------------
- Lossy divergence is expected and is NOT an ExactKV failure.
- ExactKV failure means ExactKV output differs from full-KV output.
- int4_sim is simulated and does not claim real packed INT4 memory savings.
- This report does not claim speedup, throughput, latency, or production
  readiness.

Public API
----------
``render_markdown_report(report, title=None, include_examples=True)``
    Render a report dict to a Markdown string.

``write_markdown_report(report, path, title=None, include_examples=True)``
    Render and write to a file; creates parent directories automatically.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Required disclaimer text (must always appear in rendered output)
# ---------------------------------------------------------------------------

_DISCLAIMER = """\
> **Interpretation notes**
>
> * **Lossy divergence is expected and is not an ExactKV failure.** \
The compressor alters the KV cache, so the unverified lossy output may differ \
from full-KV greedy output. ExactKV corrects this via verification.
>
> * **ExactKV failure means ExactKV output differs from full-KV output.** \
This is a correctness bug. ExactKV failure count must be 0 in a correct \
implementation.
>
> * **`int4_sim` is simulated and does not claim real packed INT4 memory \
savings.** Values are quantized to the INT4 numeric range but stored in \
`int8` containers. Memory figures for `int4_sim` reflect `int8` storage.
>
> * **This report does not claim speedup, throughput, latency, or production \
readiness.** It documents exactness and acceptance behaviour only.
"""

_WHAT_PROVES = """\
* ExactKV produced token IDs that **exactly match** `generate_full_greedy` \
for every prompt in this report (ExactKV failure count = 0).
* Lossy divergence was detected and corrected by the verification engine.
* Acceptance rates, accepted lengths, and rejection counts reflect the \
draft-verify-commit loop behaviour for each compressor and draft length.
* The compressor registry and analysis pipeline function correctly.
"""

_WHAT_NOT_PROVES = """\
* **No speedup claim.** ExactKV does not measure tokens/second, latency, \
or wall-clock time.
* **No throughput claim.** Sequential verification is used in V1/V2; this \
is not a production-serving benchmark.
* **No production readiness.** ExactKV runs with locally cached model \
weights under a research/experimental framework.
* **`int4_sim` is simulated.** No real packed 4-bit storage is used; \
memory figures are conservative `int8` estimates.
* **No real compressor backends.** All compressors in V2/V3/V4 are implemented \
in PyTorch for research purposes.
* **Sub-INT8 asymmetric compressors are simulated.** V4 compressors with `_sim` \
suffix (e.g. `k8_v4_sim`, `k4_v8_sim`) quantise K or V to a sub-INT8 numeric \
range but store values in `int8` containers — no real bit-packing. Do not cite \
their `compressed_kv_bytes` as evidence of real packed memory savings. \
`k8_v_full` and `k_full_v8` use only real INT8 and full precision and carry \
`is_simulated=False`.
* **Average effective bit width is a comparison aid only.** It is defined as \
(K bits + V bits) / 2, where full precision counts as 32 bits. It is not a \
real memory measurement.
* **`total_kv_footprint_bytes` is a conservative accounting sum, not a measured \
peak GPU memory value.** It equals stored KV + materialized working KV + metadata \
+ temporary workspace — all derived from tensor shapes and dtype widths. \
Active GPU memory measurement (torch.cuda.memory_reserved, etc.) is deferred to a \
later CUDA-specific validation phase and is not performed in V5.
* **Current materializing compressors dequantise to full precision for attention.** \
This means `materialized_working_kv_bytes` equals `full_kv_bytes` for all current \
ExactKV compressors. Stored-byte savings and working-cache footprint are different \
concepts; the table in the Workspace-Aware Memory Accounting section makes both \
visible.
* **VeriCache attribution.** ExactKV is inspired by the VeriCache paper \
(Yao et al., arXiv:2605.17613, 2026) and does not claim to have invented \
the draft-then-verify algorithm. This report evaluates the current \
Hugging Face correctness and analysis framework, not the paper's system.
"""


# ---------------------------------------------------------------------------
# V5 workspace memory section header/prose
# ---------------------------------------------------------------------------

_WORKSPACE_PREAMBLE = """\
> **V5 accounting note:** `total_kv_footprint_bytes` is a **conservative \
accounting sum** (stored KV + materialized working KV + metadata + temporary \
workspace). It is **NOT** a measured peak GPU memory value. Active GPU memory \
measurement is deferred to a later CUDA-specific validation phase.

For all current ExactKV compressors, attention requires a full-precision \
dequantised working copy of the KV cache during each attention call, so \
`materialized_working_kv_bytes` equals `full_kv_bytes`. The practical peak KV \
memory footprint during attention is therefore dominated by this working copy, \
not the stored bytes alone.

For simulated sub-INT8 compressors (`_sim` suffix), `stored_kv_bytes` reflects \
**int8 container storage** — no real packed 4-bit or 2-bit bit-packing is used. \
Do not cite these figures as evidence of real packed-bit memory savings.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _h(level: int, text: str) -> str:
    return "#" * level + " " + text


def _safe(value: Any, fmt: str = "") -> str:
    if value is None:
        return "—"
    if fmt and isinstance(value, float):
        return format(value, fmt)
    return str(value)


def _has_int4_sim(report: dict[str, Any]) -> bool:
    """Return True if any result uses int4_sim compressor."""
    for r in report.get("results", []):
        if "int4_sim" in str(r.get("compressor_name", "")):
            return True
        caps = r.get("compressor_capabilities", {})
        if caps.get("is_simulated", False):
            return True
    return False


def _has_asymmetric(report: dict[str, Any]) -> bool:
    """Return True if any result uses an asymmetric compressor."""
    for r in report.get("results", []):
        caps = r.get("compressor_capabilities", {})
        if caps.get("asymmetric", False):
            return True
    return False


def _is_sweep(report: dict[str, Any]) -> bool:
    """Return True when the report appears to be a sweep (multiple compressors
    or multiple draft lengths)."""
    results = report.get("results", [])
    compressors = {r.get("compressor_name") for r in results}
    draft_lens = {r.get("draft_len") for r in results}
    return len(compressors) > 1 or len(draft_lens) > 1


def _collect_compressor_caps(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Collect compressor capabilities keyed by compressor name."""
    caps: dict[str, dict[str, Any]] = {}
    for r in report.get("results", []):
        name = r.get("compressor_name", "")
        if name and name not in caps:
            caps[name] = r.get("compressor_capabilities", {})
    return caps


def _render_manifest(report: dict[str, Any]) -> str:
    manifest = report.get("manifest", {})
    if not manifest:
        return "_Manifest not available._\n"
    lines = []
    for key, val in manifest.items():
        if val is not None:
            lines.append(f"* **{key}:** {val}")
    return "\n".join(lines) + "\n"


def _render_correctness_summary(
    failure_report: dict[str, Any],
    total_results: int,
) -> str:
    fail_count = failure_report.get("exactkv_failure_count", 0)
    div_count = failure_report.get("lossy_divergence_count", 0)
    status = failure_report.get("status", "unknown")

    status_icon = "✓" if status == "pass" else "✗"
    lines = [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total results | {total_results} |",
        f"| ExactKV failures | **{fail_count}** ({status_icon} {status.upper()}) |",
        f"| Lossy divergences | {div_count} _(expected for lossy compressors)_ |",
    ]
    return "\n".join(lines) + "\n"


def _render_kv_metadata_section(
    comp_caps: dict[str, dict[str, Any]],
) -> str:
    """Render a 'K/V Compression Metadata' table for all compressors in this report."""
    from exactkv.reporting.leaderboard import average_effective_bit_width, _fmt_bits

    lines = [
        "| Compressor | K bits | V bits | Avg eff bits | Simulated | Real bytes |",
        "|------------|--------|--------|--------------|-----------|------------|",
    ]
    for comp, caps in sorted(comp_caps.items()):
        k = caps.get("key_bit_width")
        v = caps.get("value_bit_width")
        avg = average_effective_bit_width(k, v)
        lines.append(
            f"| `{comp}` "
            f"| {_fmt_bits(k)} "
            f"| {_fmt_bits(v)} "
            f"| {avg:.1f} "
            f"| {'yes ⚠️' if caps.get('is_simulated') else 'no'} "
            f"| {'yes' if caps.get('supports_real_bytes_claim') else 'no'} |"
        )

    note = (
        "\n"
        "**Notes:**\n"
        "* `full` means full-precision passthrough; that side is not quantised.\n"
        "* **Average effective bits = (K bits + V bits) / 2**, treating full "
        "precision as 32 bits. This is a metadata comparison aid — not a real "
        "memory measurement.\n"
        "* Compressors marked **simulated** store sub-INT8 values in `int8` "
        "containers. Do not cite their `compressed_kv_bytes` as evidence of real "
        "packed memory savings.\n"
    )

    return "\n".join(lines) + note


_NOTE_MAX_CHARS: int = 90  # truncation limit for compact note column


def _truncate_note(note: str, max_chars: int = _NOTE_MAX_CHARS) -> str:
    """Return the first sentence of ``note``, capped at ``max_chars`` chars."""
    if not note:
        return "—"
    dot_pos = note.find(". ")
    first = note[:dot_pos + 1] if dot_pos != -1 else note
    if len(first) > max_chars:
        return first[:max_chars - 1] + "…"
    return first


def _render_memory_notes(report: dict[str, Any]) -> str:
    """Render a compact per-compressor memory honesty summary table.

    Shows one row per compressor with real-bytes flag, simulated flag, and a
    single-sentence key note.  The full workspace-aware accounting prose and
    per-compressor byte table are in the Workspace-Aware Memory Accounting
    section that immediately follows.
    """
    rows_data: list[tuple[str, bool, bool, str]] = []
    for r in report.get("results", []):
        comp = r.get("compressor_name", "")
        if not comp or any(row[0] == comp for row in rows_data):
            continue
        mem: dict[str, Any] = r.get("memory", {})
        caps: dict[str, Any] = r.get("compressor_capabilities", {})
        real = caps.get("supports_real_bytes_claim", mem.get("supports_real_bytes_claim", True))
        sim = caps.get("is_simulated", mem.get("is_simulated", False))
        note = mem.get("memory_claim_note", "")
        rows_data.append((comp, bool(real), bool(sim), note))

    if not rows_data:
        return "_No special memory honesty notes for this report._\n"

    lines = [
        "| Compressor | Real bytes? | Simulated? | Key note |",
        "|------------|-------------|------------|----------|",
    ]
    for comp, real, sim, note in sorted(rows_data):
        short = _truncate_note(note)
        real_cell = "yes" if real else "no ⚠️"
        sim_cell  = "yes ⚠️" if sim else "no"
        lines.append(f"| `{comp}` | {real_cell} | {sim_cell} | {short} |")

    footer = (
        "\n_For all compressors: `total_kv_footprint_bytes` is a conservative "
        "accounting sum, not a measured peak GPU memory value. "
        "Current materializing compressors dequantise to full working KV for attention. "
        "Active GPU measurement is deferred. "
        "See **Workspace-Aware Memory Accounting** below for the per-compressor table "
        "and full notes._\n"
    )
    return "\n".join(lines) + footer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_markdown_report(
    report: dict[str, Any],
    title: str | None = None,
    include_examples: bool = True,
    max_examples: int = 3,
) -> str:
    """Render a benchmark or sweep report dict to a Markdown string.

    The returned string contains the following sections:

    1. Title
    2. Experiment summary
    3. Manifest
    4. Correctness summary
    5. Acceptance leaderboard (by compressor and by draft length)
    6. Compressor × draft-length grid (sweep reports only)
    7. Lossy divergence examples (if ``include_examples=True``)
    8. ExactKV failure examples (always — should be empty)
    9. Histogram tables
    10. Memory honesty notes
    10b. K/V compression metadata (asymmetric reports only)
    10c. Workspace-Aware Memory Accounting (V5 — table + framing)
    11. What this report proves
    12. What this report does not prove
    13. Required disclaimers

    Args:
        report:           Report dict from ``run_suite`` or ``run_sweep``.
        title:            Report title.  Defaults to "ExactKV Benchmark Report".
        include_examples: Include lossy-divergence and rejection examples.
        max_examples:     Maximum examples per section.

    Returns:
        Markdown string.

    Note:
        This renderer never re-runs the model.
        No timing, throughput, latency, or speedup fields are rendered.
    """
    from exactkv.analysis.acceptance_tables import (
        build_acceptance_table,
        group_acceptance_by_compressor,
        group_acceptance_by_draft_len,
    )
    from exactkv.analysis.failure_report import build_failure_report
    from exactkv.analysis.histograms import (
        accepted_length_histogram,
        first_divergence_histogram,
        rejection_count_histogram,
    )
    from exactkv.analysis.examples import (
        extract_lossy_divergence_examples,
        extract_exactkv_failure_examples,
        extract_rejection_examples,
    )
    from exactkv.reporting.leaderboard import (
        render_compressor_leaderboard,
        render_draft_len_leaderboard,
        render_compressor_x_draft_leaderboard,
    )
    from exactkv.reporting.examples import (
        render_lossy_divergence_examples,
        render_exactkv_failure_examples,
        render_rejection_examples,
    )
    from exactkv.reporting.histograms import (
        render_accepted_length_table,
        render_first_divergence_table,
        render_rejection_count_table,
    )

    title = title or "ExactKV Benchmark Report"
    results = report.get("results", [])
    total = len(results)
    failure_report = build_failure_report(report)
    comp_caps = _collect_compressor_caps(report)
    sweep = _is_sweep(report)
    has_int4 = _has_int4_sim(report)
    has_asym = _has_asymmetric(report)

    sections: list[str] = []

    # ── 1. Title ─────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    sections.append(_h(1, title))
    sections.append(f"_Generated {ts} by ExactKV. See disclaimers below._\n")

    # ── 2. Experiment summary ─────────────────────────────────────────────────
    sections.append(_h(2, "Experiment Summary"))
    compressors = sorted({r.get("compressor_name", "?") for r in results})
    draft_lens = sorted({r.get("draft_len", "?") for r in results})
    prompt_ids = sorted({r.get("prompt_id", "?") for r in results})
    sections.append(
        f"* **Total results:** {total}\n"
        f"* **Compressors:** {', '.join(f'`{c}`' for c in compressors)}\n"
        f"* **Draft lengths:** {', '.join(str(d) for d in draft_lens)}\n"
        f"* **Prompts:** {len(prompt_ids)}\n"
        f"* **Report type:** {'sweep' if sweep else 'single-compressor benchmark'}\n"
    )

    # ── 3. Manifest ───────────────────────────────────────────────────────────
    sections.append(_h(2, "Manifest"))
    sections.append(_render_manifest(report))

    # ── 4. Correctness summary ────────────────────────────────────────────────
    sections.append(_h(2, "Correctness Summary"))
    sections.append(_render_correctness_summary(failure_report, total))

    # ── 5. Acceptance leaderboard by compressor ───────────────────────────────
    sections.append(_h(2, "Acceptance Leaderboard — by Compressor"))
    by_comp = group_acceptance_by_compressor(report)
    sections.append(render_compressor_leaderboard(by_comp, compressor_caps=comp_caps))
    sections.append("")

    # ── 5b. By draft length ───────────────────────────────────────────────────
    sections.append(_h(2, "Acceptance Leaderboard — by Draft Length"))
    by_dl = group_acceptance_by_draft_len(report)
    sections.append(render_draft_len_leaderboard(by_dl))
    sections.append("")

    # ── 6. Compressor × draft-length grid (sweep only) ───────────────────────
    if sweep:
        sections.append(_h(2, "Acceptance Grid — Compressor × Draft Length"))
        full_table = build_acceptance_table(report)
        sections.append(
            render_compressor_x_draft_leaderboard(
                full_table, compressor_caps=comp_caps if comp_caps else None
            )
        )
        sections.append("")

    # ── 7. Histogram tables ───────────────────────────────────────────────────
    sections.append(_h(2, "Histogram Tables"))

    sections.append(_h(3, "Accepted-Length Distribution"))
    sections.append(
        "_Bucket: avg accepted tokens per verification round (floored to int)._\n"
    )
    sections.append(render_accepted_length_table(accepted_length_histogram(report)))
    sections.append("")

    sections.append(_h(3, "First-Divergence Position Distribution"))
    sections.append(
        "_Bucket: first token index where lossy output diverged from full output. "
        "`no_divergence` = lossy matched full._\n"
    )
    sections.append(render_first_divergence_table(first_divergence_histogram(report)))
    sections.append("")

    sections.append(_h(3, "Rejection-Count Distribution"))
    sections.append(
        "_Bucket: total tokens rejected (overridden) by the ExactKV verifier. "
        "Non-zero is expected for lossy compressors._\n"
    )
    sections.append(render_rejection_count_table(rejection_count_histogram(report)))
    sections.append("")

    # ── 8. Examples ───────────────────────────────────────────────────────────
    if include_examples:
        sections.append(_h(2, "Lossy Divergence Examples"))
        sections.append(
            "_These examples show prompts where the unverified lossy output differed "
            "from full-KV greedy output.  Lossy divergence is **expected** and is "
            "**not** an ExactKV failure._\n"
        )
        lossy_exs = extract_lossy_divergence_examples(report, limit=max_examples)
        sections.append(render_lossy_divergence_examples(lossy_exs))

        sections.append(_h(2, "ExactKV Failure Examples"))
        sections.append(
            "_ExactKV failure means the verified output did NOT match "
            "`generate_full_greedy`. This is a correctness bug. "
            "Should always be empty._\n"
        )
        fail_exs = extract_exactkv_failure_examples(report, limit=max_examples)
        sections.append(render_exactkv_failure_examples(fail_exs))

        sections.append(_h(2, "Top Rejection Examples"))
        sections.append(
            "_Sorted by total rejected tokens descending. High rejection is expected "
            "for aggressively lossy compressors and does NOT mean the output is wrong._\n"
        )
        rej_exs = extract_rejection_examples(report, limit=max_examples)
        sections.append(render_rejection_examples(rej_exs))

    # ── 9. Memory honesty notes ───────────────────────────────────────────────
    sections.append(_h(2, "Memory Honesty Notes"))
    if has_int4:
        sections.append(
            "> **`int4_sim` memory note:** `int4_sim` is simulated and does **not** "
            "claim real packed INT4 memory savings. Values are quantized to the INT4 "
            "numeric range but stored in `int8` containers. Memory figures for "
            "`int4_sim` reflect `int8` storage only.\n"
        )
    sections.append(_render_memory_notes(report))

    # ── 9b. K/V compression metadata (only when asymmetric compressors present) ──
    if has_asym and comp_caps:
        sections.append(_h(2, "K/V Compression Metadata"))
        sections.append(
            "_This report includes asymmetric compressors that compress keys and values "
            "at different bit-widths.  K bits and V bits are declared by each "
            "compressor's capabilities; they are not derived from measured memory. "
            "Average effective bits is a comparison aid only._\n"
        )
        sections.append(_render_kv_metadata_section(comp_caps))
        sections.append("")

    # ── 9c. Workspace-aware memory accounting (V5) ───────────────────────────
    from exactkv.reporting.memory import render_workspace_memory_table
    sections.append(_h(2, "Workspace-Aware Memory Accounting"))
    sections.append(_WORKSPACE_PREAMBLE)
    sections.append(render_workspace_memory_table(report))
    sections.append("")

    # ── 10. What this proves ──────────────────────────────────────────────────
    sections.append(_h(2, "What This Report Proves"))
    sections.append(_WHAT_PROVES)

    # ── 11. What this does not prove ─────────────────────────────────────────
    sections.append(_h(2, "What This Report Does Not Prove"))
    sections.append(_WHAT_NOT_PROVES)

    # ── 12. Required disclaimer ───────────────────────────────────────────────
    sections.append(_h(2, "Disclaimers"))
    sections.append(_DISCLAIMER)

    return "\n".join(sections)


def write_markdown_report(
    report: dict[str, Any],
    path: str | Path,
    title: str | None = None,
    include_examples: bool = True,
    max_examples: int = 3,
) -> None:
    """Render a report to Markdown and write it to ``path``.

    Parent directories are created automatically.

    Args:
        report:           Report dict.
        path:             Destination file path.
        title:            Report title.
        include_examples: Include lossy-divergence and rejection examples.
        max_examples:     Maximum examples per section.

    Note:
        This function never re-runs the model.
        No timing, throughput, latency, or speedup fields are written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown_report(
        report,
        title=title,
        include_examples=include_examples,
        max_examples=max_examples,
    )
    path.write_text(md, encoding="utf-8")
