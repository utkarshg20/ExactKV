"""ExactKV failure classification and reporting.

Operates on existing benchmark and sweep reports without re-running the model.

Key distinction (enforced by this module)
------------------------------------------
*Lossy divergence* (``lossy.token_exact_match == False``) is **expected** and
is **not** an ExactKV failure.  It shows that the compressor changes the output
and demonstrates why verification is necessary.  Lossy divergences are listed
separately from ExactKV failures.

*ExactKV failure* (``exactkv_failure == True``) means the ExactKV loop
produced output that did **not** match ``generate_full_greedy``.  This is a
correctness bug and the status field will be ``"fail"``.

No timing, latency, throughput, or speedup fields are produced.

Public API
----------
``build_failure_report(report)``
    Full failure analysis dict with counts, lists, and status.

``list_exactkv_failures(report)``
    Subset of results where ``exactkv_failure == True``.

``list_lossy_divergences(report)``
    Subset of results where lossy output diverged from full output.

``write_failure_report_json(failure_report, path)``
    Write to JSON; creates parent dirs automatically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Individual failure / divergence extractors
# ---------------------------------------------------------------------------

def list_exactkv_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one record for each result where ``exactkv_failure == True``.

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        List of dicts with:
        ``prompt_id``, ``compressor_name``, ``draft_len``, ``category``.
        Empty list when ExactKV produced correct output for all prompts.
    """
    results = report.get("results", [])
    return [
        {
            "prompt_id": r.get("prompt_id", ""),
            "compressor_name": r.get("compressor_name", ""),
            "draft_len": r.get("draft_len", ""),
            "category": r.get("category", ""),
        }
        for r in results
        if r.get("exactkv_failure", False)
    ]


def list_lossy_divergences(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one record for each result where lossy output diverged from full.

    Lossy divergence is **expected** for non-trivial compressors; it is not
    an ExactKV failure.

    Args:
        report: Dict with a ``"results"`` list.

    Returns:
        List of dicts with:
        ``prompt_id``, ``compressor_name``, ``draft_len``, ``category``,
        ``first_divergence_idx`` (int or None).
        Empty list when lossy always matched full (e.g. with NoOpCompressor).
    """
    results = report.get("results", [])
    return [
        {
            "prompt_id": r.get("prompt_id", ""),
            "compressor_name": r.get("compressor_name", ""),
            "draft_len": r.get("draft_len", ""),
            "category": r.get("category", ""),
            "first_divergence_idx": r.get("lossy", {}).get("first_divergence_idx"),
        }
        for r in results
        if not r.get("lossy", {}).get("token_exact_match", True)
    ]


# ---------------------------------------------------------------------------
# Full failure report
# ---------------------------------------------------------------------------

def build_failure_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build a structured failure report from a benchmark or sweep report.

    Args:
        report: Dict with a ``"results"`` list (from ``run_suite`` or
                ``run_sweep``).

    Returns:
        Dict with:

        ``exactkv_failure_count``
            Number of results where ExactKV output != full greedy output.
            Must be 0 for a correct implementation.

        ``lossy_divergence_count``
            Number of results where lossy output != full greedy output.
            Non-zero is expected and is **not** a failure.

        ``exactkv_failures``
            List of failing result records (empty if all passed).

        ``lossy_divergences``
            List of diverging result records (may be non-empty for real
            compressors — this is normal and expected).

        ``status``
            ``"pass"`` when ``exactkv_failure_count == 0``.
            ``"fail"`` otherwise.

    Note:
        No timing or speedup fields are included.
    """
    failures = list_exactkv_failures(report)
    divergences = list_lossy_divergences(report)

    return {
        "exactkv_failure_count": len(failures),
        "lossy_divergence_count": len(divergences),
        "exactkv_failures": failures,
        "lossy_divergences": divergences,
        "status": "pass" if len(failures) == 0 else "fail",
    }


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def write_failure_report_json(
    failure_report: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a failure report to a JSON file.

    Parent directories are created automatically.

    Args:
        failure_report: Output of ``build_failure_report``.
        path:           Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(failure_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
