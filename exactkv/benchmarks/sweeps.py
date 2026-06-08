"""ExactKV V2 benchmark sweep runner.

Runs one model over multiple compressors and draft lengths across a prompt
suite, collecting one result per (prompt × compressor × draft_len) cell.

Design constraints (V2)
-----------------------
* No timing, latency, throughput, or speedup metrics in any output.
* Single model load reused across the entire sweep.
* Delegates per-cell execution to ``runner.run_one`` so all compressor,
  generation, and verification logic stays in one place.
* Sweep reports are directly compatible with ``reports.write_json_report``
  and ``reports.write_csv_report``; ``flatten_report_to_rows`` produces
  one CSV row per result.

Public API
----------
``run_sweep(runtime, prompts, compressor_names, draft_lengths, ...)``
    → sweep report dict (manifest + results + aggregate)
"""
from __future__ import annotations

from typing import Any

from exactkv.benchmarks.reports import build_run_manifest
from exactkv.benchmarks.runner import RunConfig, run_one
from exactkv.runtime.model_runtime import ModelRuntime


# ---------------------------------------------------------------------------
# Aggregate computation (no timing fields)
# ---------------------------------------------------------------------------

def _compute_aggregate(
    results: list[dict[str, Any]],
    num_prompts: int,
    compressor_names: list[str],
    draft_lengths: list[int],
) -> dict[str, Any]:
    """Aggregate statistics across all sweep cells.

    Forbidden fields: runtime_seconds, tokens_per_second, throughput,
    latency, speedup.  None are included here.
    """
    total_runs = len(results)

    exactkv_failures = sum(1 for r in results if r.get("exactkv_failure", False))
    lossy_divergence_count = sum(
        1 for r in results
        if not r.get("lossy", {}).get("token_exact_match", True)
    )

    acc_blocks = [r.get("exactkv", {}).get("acceptance", {}) for r in results]

    acceptance_rates = [a.get("acceptance_rate", 0.0) for a in acc_blocks]
    avg_accepted_lengths = [a.get("avg_accepted_per_round", 0.0) for a in acc_blocks]

    n = max(total_runs, 1)
    mean_acceptance_rate = sum(acceptance_rates) / n
    mean_avg_accepted = sum(avg_accepted_lengths) / n

    total_drafted = sum(a.get("total_drafted", 0) for a in acc_blocks)
    total_accepted = sum(a.get("total_accepted", 0) for a in acc_blocks)
    total_rejected = sum(a.get("total_rejected", 0) for a in acc_blocks)
    total_corrections = sum(a.get("total_corrections", 0) for a in acc_blocks)

    return {
        "total_runs": total_runs,
        "total_prompts": num_prompts,
        "compressor_names": compressor_names,
        "draft_lengths": draft_lengths,
        "exactkv_failures": exactkv_failures,
        "lossy_divergence_count": lossy_divergence_count,
        "mean_acceptance_rate": mean_acceptance_rate,
        "mean_average_accepted_length": mean_avg_accepted,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_corrections": total_corrections,
    }


# ---------------------------------------------------------------------------
# Main sweep function
# ---------------------------------------------------------------------------

def run_sweep(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    compressor_names: list[str],
    draft_lengths: list[int],
    max_new_tokens: int = 32,
    prompt_suite: str = "custom",
) -> dict[str, Any]:
    """Run a multi-compressor, multi-draft-length benchmark sweep.

    For each combination of (prompt × compressor × draft_len) the function
    calls ``runner.run_one`` and collects the result.  A single
    ``ModelRuntime`` is reused across the entire sweep.

    Args:
        runtime:          Loaded ModelRuntime shared across all cells.
        prompts:          List of prompt entry dicts (``prompt_id``,
                          ``category``, ``prompt`` required).
        compressor_names: List of compressor names to sweep over.
        draft_lengths:    List of draft-token budgets to sweep over.
        max_new_tokens:   Token generation budget per prompt.
        prompt_suite:     Label for the manifest (e.g. ``"smoke"``).

    Returns:
        A sweep report dict with keys:

        * ``manifest``  — provenance metadata (no timing fields).
        * ``results``   — flat list of per-cell dicts, compatible with
                          ``reports.flatten_report_to_rows``.
        * ``aggregate`` — summary statistics across all cells.

    Note:
        Iteration order is prompt → compressor → draft_len.  This groups
        all compressor/draft-length combinations for each prompt together
        in the results list.

    Constraints:
        No timing, latency, throughput, or speedup fields are produced.
        All memory statistics for simulated compressors (``int4_sim``) carry
        the appropriate ``supports_real_bytes_claim=False`` flag via
        ``compressor_capabilities`` inherited from ``run_one``.
    """
    if not compressor_names:
        raise ValueError("compressor_names must not be empty")
    if not draft_lengths:
        raise ValueError("draft_lengths must not be empty")
    if any(d < 1 for d in draft_lengths):
        raise ValueError(f"All draft_lengths must be >= 1, got {draft_lengths}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    manifest = build_run_manifest(
        model_name=runtime.model_name,
        prompt_suite=prompt_suite,
        compressor_names=compressor_names,
        draft_lengths=draft_lengths,
        max_new_tokens=max_new_tokens,
        device=str(runtime.device),
        dtype=str(runtime.dtype).replace("torch.", ""),
    )

    results: list[dict[str, Any]] = []

    for prompt_entry in prompts:
        for compressor_name in compressor_names:
            for draft_len in draft_lengths:
                config = RunConfig(
                    compressor_name=compressor_name,
                    draft_len=draft_len,
                    max_new_tokens=max_new_tokens,
                )
                result = run_one(runtime, prompt_entry, config)
                results.append(result)

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=compressor_names,
        draft_lengths=draft_lengths,
    )

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }
