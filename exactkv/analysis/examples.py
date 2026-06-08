"""Example extraction utilities for ExactKV benchmark and sweep reports.

All functions operate on existing report dicts and do NOT re-run the model.

No timing, latency, throughput, or speedup fields are produced or consumed.

Key distinction (enforced by these functions)
---------------------------------------------
*Lossy divergence* (``lossy.token_exact_match == False``) is **expected**.
It proves the compressor changes the output and demonstrates why the
verification step is necessary.  It is explicitly NOT an ExactKV failure.

*ExactKV failure* (``exactkv_failure == True``) means the verified output
did NOT match ``generate_full_greedy``.  This is a correctness bug that
must always be zero in a correct implementation.

Public API
----------
``extract_lossy_divergence_examples(report, limit=5)``
    Return up to ``limit`` results where the lossy output diverged from full.

``extract_exactkv_failure_examples(report, limit=5)``
    Return up to ``limit`` results where ExactKV output did NOT match full.

``extract_rejection_examples(report, limit=5)``
    Return up to ``limit`` results with the highest ``total_rejected`` counts.
"""
from __future__ import annotations

from typing import Any

_EXPLANATION_LOSSY = (
    "Lossy divergence is expected: the compressor altered the KV cache, "
    "so the unverified lossy output differs from full-KV greedy output. "
    "ExactKV corrects this by verifying against full-KV predictions. "
    "A non-zero 'exactkv_matches_full=False' would be a correctness bug, "
    "not a lossy divergence."
)

_FORBIDDEN_FIELDS = {
    "tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds",
}


def _get_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report.get("results", [])


def _safe_str(value: Any, max_chars: int = 400) -> str:
    """Return a string representation, truncated if needed."""
    s = str(value) if value is not None else ""
    return s[:max_chars] + "…" if len(s) > max_chars else s


def _build_lossy_example(r: dict[str, Any]) -> dict[str, Any]:
    """Build a lossy-divergence example dict from a result record."""
    acc = r.get("exactkv", {}).get("acceptance", {})
    return {
        "prompt_id": r.get("prompt_id", ""),
        "category": r.get("category", ""),
        "compressor_name": r.get("compressor_name", ""),
        "draft_len": r.get("draft_len", ""),
        "prompt": _safe_str(r.get("prompt", ""), max_chars=300),
        "full_text": _safe_str(r.get("full", {}).get("output_text", ""), max_chars=400),
        "lossy_text": _safe_str(r.get("lossy", {}).get("output_text", ""), max_chars=400),
        "exactkv_text": _safe_str(r.get("exactkv", {}).get("output_text", ""), max_chars=400),
        "first_divergence_idx": r.get("lossy", {}).get("first_divergence_idx"),
        "exactkv_matches_full": r.get("exactkv", {}).get("token_exact_match", True),
        "lossy_matches_full": r.get("lossy", {}).get("token_exact_match", True),
        "total_rejected": acc.get("total_rejected", 0),
        "total_corrections": acc.get("total_corrections", 0),
        "acceptance_rate": acc.get("acceptance_rate", 0.0),
        "explanation": _EXPLANATION_LOSSY,
    }


def extract_lossy_divergence_examples(
    report: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` examples where lossy output diverged from full.

    Examples are taken in report order (first divergences first).
    Results where ``lossy.token_exact_match`` is ``True`` are excluded.

    The returned dicts include ``full_text``, ``lossy_text``, and
    ``exactkv_text`` for direct comparison, plus an ``explanation`` field
    that states lossy divergence is expected and ExactKV failure is the
    real failure.

    Args:
        report: Report dict (``run_suite`` or ``run_sweep`` output).
        limit:  Maximum number of examples to return.

    Returns:
        List of example dicts.  Empty when no lossy divergences exist.

    Note:
        No timing, throughput, latency, or speedup fields are produced.
    """
    examples = []
    for r in _get_results(report):
        if len(examples) >= limit:
            break
        lossy = r.get("lossy", {})
        if not lossy.get("token_exact_match", True):  # diverged
            examples.append(_build_lossy_example(r))
    return examples


def extract_exactkv_failure_examples(
    report: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` results where ExactKV output did NOT match full.

    In a correct ExactKV implementation this list is always empty.  It is
    provided to surface bugs quickly when they occur.

    Each example includes ``full_text`` and ``exactkv_text`` for comparison,
    and ``exactkv_matches_full = False`` to make the failure explicit.

    Args:
        report: Report dict.
        limit:  Maximum number of examples to return.

    Returns:
        List of failure example dicts.  Empty when ``exactkv_failures == 0``.

    Note:
        No timing, throughput, latency, or speedup fields are produced.
    """
    examples = []
    for r in _get_results(report):
        if len(examples) >= limit:
            break
        if r.get("exactkv_failure", False):
            acc = r.get("exactkv", {}).get("acceptance", {})
            examples.append({
                "prompt_id": r.get("prompt_id", ""),
                "category": r.get("category", ""),
                "compressor_name": r.get("compressor_name", ""),
                "draft_len": r.get("draft_len", ""),
                "prompt": _safe_str(r.get("prompt", ""), max_chars=300),
                "full_text": _safe_str(
                    r.get("full", {}).get("output_text", ""), max_chars=400
                ),
                "exactkv_text": _safe_str(
                    r.get("exactkv", {}).get("output_text", ""), max_chars=400
                ),
                "exactkv_matches_full": False,
                "total_rejected": acc.get("total_rejected", 0),
                "total_corrections": acc.get("total_corrections", 0),
                "acceptance_rate": acc.get("acceptance_rate", 0.0),
                "note": (
                    "ExactKV failure: the verified output did NOT match "
                    "generate_full_greedy. This is a correctness bug."
                ),
            })
    return examples


def extract_rejection_examples(
    report: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` results with the highest rejection counts.

    Sorted by ``total_rejected`` descending.  A high rejection count means
    the ExactKV verification engine frequently had to override drafted tokens,
    but this does NOT mean the final output is wrong — ExactKV corrects all
    rejections via the full-KV verifier.

    Args:
        report: Report dict.
        limit:  Maximum number of examples to return.

    Returns:
        List of rejection example dicts.  Each dict includes
        ``total_rejected``, ``total_corrections``, ``acceptance_rate``,
        and prompt context.

    Note:
        No timing, throughput, latency, or speedup fields are produced.
    """
    results = _get_results(report)

    def _rejected(r: dict[str, Any]) -> int:
        return r.get("exactkv", {}).get("acceptance", {}).get("total_rejected", 0)

    ranked = sorted(results, key=_rejected, reverse=True)

    examples = []
    for r in ranked[:limit]:
        acc = r.get("exactkv", {}).get("acceptance", {})
        examples.append({
            "prompt_id": r.get("prompt_id", ""),
            "category": r.get("category", ""),
            "compressor_name": r.get("compressor_name", ""),
            "draft_len": r.get("draft_len", ""),
            "prompt": _safe_str(r.get("prompt", ""), max_chars=300),
            "acceptance_rate": acc.get("acceptance_rate", 0.0),
            "total_drafted": acc.get("total_drafted", 0),
            "total_accepted": acc.get("total_accepted", 0),
            "total_rejected": acc.get("total_rejected", 0),
            "total_corrections": acc.get("total_corrections", 0),
            "lossy_diverged": not r.get("lossy", {}).get("token_exact_match", True),
            "exactkv_matches_full": r.get("exactkv", {}).get("token_exact_match", True),
            "note": (
                "High rejection count is expected for aggressively lossy "
                "compressors. ExactKV corrects all rejections; "
                "exactkv_matches_full must be True."
            ),
        })
    return examples
