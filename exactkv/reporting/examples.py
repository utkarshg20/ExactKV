"""Markdown example renderers for ExactKV reports.

All renderers accept pre-extracted example lists (output of
``exactkv.analysis.examples``) and return Markdown strings.
No model re-runs.  No timing, latency, throughput, or speedup fields.

Key distinction (printed in every rendered example block)
---------------------------------------------------------
*Lossy divergence* is **expected**.  It shows that the compressor changes
the output and demonstrates why verification is necessary.

*ExactKV failure* means the verified output did NOT match
``generate_full_greedy``.  This is a correctness bug.

Public API
----------
``render_lossy_divergence_examples(examples)``
    Render up to N lossy-divergence examples as Markdown.

``render_exactkv_failure_examples(examples)``
    Render ExactKV failure examples (should be empty in a correct run).

``render_rejection_examples(examples)``
    Render examples sorted by total_rejected descending.
"""
from __future__ import annotations

from typing import Any


def _truncate(text: str, max_chars: int = 200) -> str:
    if not text:
        return "_(empty)_"
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _header3(title: str) -> str:
    return f"### {title}"


# ---------------------------------------------------------------------------
# Lossy divergence examples
# ---------------------------------------------------------------------------

def render_lossy_divergence_examples(examples: list[dict[str, Any]]) -> str:
    """Render a list of lossy divergence examples as Markdown.

    Args:
        examples: Output of ``extract_lossy_divergence_examples``.

    Returns:
        Markdown string.  Returns a "none found" note if list is empty.

    Note:
        Each example includes an explanation that lossy divergence is expected
        and that ExactKV failure (not lossy divergence) is the real failure.
        No timing, throughput, latency, or speedup fields are rendered.
    """
    if not examples:
        return (
            "> **No lossy divergences found.** "
            "All compressors produced outputs identical to full-KV greedy.\n"
        )

    lines: list[str] = []
    for i, ex in enumerate(examples, 1):
        div_idx = ex.get("first_divergence_idx")
        div_str = str(div_idx) if div_idx is not None else "—"
        exactkv_ok = ex.get("exactkv_matches_full", True)

        lines.append(_header3(
            f"Example {i} — `{ex.get('compressor_name', '?')}` "
            f"| draft_len={ex.get('draft_len', '?')} "
            f"| category={ex.get('category', '?')}"
        ))
        lines.append("")
        lines.append(f"**Prompt ID:** `{ex.get('prompt_id', '?')}`  ")
        lines.append(f"**First divergence token index:** {div_str}  ")
        lines.append(f"**ExactKV matches full:** {'✓ yes' if exactkv_ok else '✗ NO (failure)'}")
        lines.append("")
        lines.append("**Prompt excerpt:**")
        lines.append(f"> {_truncate(str(ex.get('prompt', '')), 200)}")
        lines.append("")
        lines.append("**Full-KV output:**")
        lines.append(f"> {_truncate(str(ex.get('full_text', '')), 200)}")
        lines.append("")
        lines.append("**Lossy output** _(diverges from full — expected)_**:**")
        lines.append(f"> {_truncate(str(ex.get('lossy_text', '')), 200)}")
        lines.append("")
        lines.append("**ExactKV output** _(must match full)_**:**")
        lines.append(f"> {_truncate(str(ex.get('exactkv_text', '')), 200)}")
        lines.append("")
        lines.append(
            "> _Note: Lossy divergence is expected. "
            "The compressor altered the KV cache, causing the unverified lossy "
            "output to differ from full-KV greedy. ExactKV corrects this via "
            "verification. A non-zero `exactkv_matches_full=False` would be a "
            "correctness bug, not a lossy divergence._"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ExactKV failure examples
# ---------------------------------------------------------------------------

def render_exactkv_failure_examples(examples: list[dict[str, Any]]) -> str:
    """Render ExactKV failure examples as Markdown.

    Args:
        examples: Output of ``extract_exactkv_failure_examples``.

    Returns:
        Markdown string.  Returns a "none found" success note if list is empty.

    Note:
        ExactKV failure means the verified output did NOT match
        ``generate_full_greedy``.  This is a correctness bug that must always
        be zero in a correct implementation.
        No timing, throughput, latency, or speedup fields are rendered.
    """
    if not examples:
        return (
            "> ✓ **ExactKV failure count: 0.** "
            "The ExactKV loop produced output matching full-KV greedy "
            "for every prompt in this report.\n"
        )

    lines: list[str] = [
        "> ⚠️ **ExactKV failures detected!** "
        "The following results show cases where ExactKV output did NOT "
        "match `generate_full_greedy`. This is a correctness bug.\n",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(_header3(
            f"Failure {i} — `{ex.get('compressor_name', '?')}` "
            f"| draft_len={ex.get('draft_len', '?')} "
            f"| category={ex.get('category', '?')}"
        ))
        lines.append("")
        lines.append(f"**Prompt ID:** `{ex.get('prompt_id', '?')}`  ")
        lines.append("**`exactkv_matches_full`: ✗ False — correctness bug**")
        lines.append("")
        lines.append("**Prompt excerpt:**")
        lines.append(f"> {_truncate(str(ex.get('prompt', '')), 200)}")
        lines.append("")
        lines.append("**Full-KV output (ground truth):**")
        lines.append(f"> {_truncate(str(ex.get('full_text', '')), 200)}")
        lines.append("")
        lines.append("**ExactKV output (incorrect):**")
        lines.append(f"> {_truncate(str(ex.get('exactkv_text', '')), 200)}")
        lines.append("")
        lines.append(f"> _{ex.get('note', '')}_")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rejection examples
# ---------------------------------------------------------------------------

def render_rejection_examples(examples: list[dict[str, Any]]) -> str:
    """Render high-rejection examples as Markdown (sorted by total_rejected desc).

    Args:
        examples: Output of ``extract_rejection_examples``.

    Returns:
        Markdown string.

    Note:
        High rejection count means the verifier frequently overrode drafted
        tokens.  This is expected for aggressively lossy compressors and does
        NOT imply the final output is wrong.
        No timing, throughput, latency, or speedup fields are rendered.
    """
    if not examples:
        return "> _No rejection examples available._\n"

    lines: list[str] = []
    for i, ex in enumerate(examples, 1):
        exactkv_ok = ex.get("exactkv_matches_full", True)
        lines.append(_header3(
            f"Top-rejection {i} — `{ex.get('compressor_name', '?')}` "
            f"| draft_len={ex.get('draft_len', '?')} "
            f"| category={ex.get('category', '?')}"
        ))
        lines.append("")
        lines.append(f"**Prompt ID:** `{ex.get('prompt_id', '?')}`  ")
        lines.append(
            f"**Acceptance rate:** {ex.get('acceptance_rate', 0.0):.3f}  "
        )
        lines.append(
            f"**Drafted / Accepted / Rejected / Corrections:** "
            f"{ex.get('total_drafted', 0)} / "
            f"{ex.get('total_accepted', 0)} / "
            f"{ex.get('total_rejected', 0)} / "
            f"{ex.get('total_corrections', 0)}"
        )
        lines.append(
            f"**ExactKV matches full:** {'✓ yes' if exactkv_ok else '✗ NO (failure)'}"
        )
        lines.append("")
        lines.append("**Prompt excerpt:**")
        lines.append(f"> {_truncate(str(ex.get('prompt', '')), 200)}")
        lines.append("")
        lines.append(f"> _{ex.get('note', '')}_")
        lines.append("")

    return "\n".join(lines)
