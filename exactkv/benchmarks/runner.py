"""ExactKV V1 benchmark runner.

Runs a single prompt under three modes and reports a JSON-compatible dict:
  1. ``full``    — generate_full_greedy (ground truth)
  2. ``lossy``   — generate_lossy_greedy (no verification)
  3. ``exactkv`` — ExactKVGenerator draft-verify-commit loop

The runner is single-threaded and correctness-first.  It does NOT claim any
throughput or latency numbers.

Supported compressor names: ``"noop"``, ``"int8"``, ``"debug_noise"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from exactkv.benchmarks.prompts import load_prompts
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime


@dataclass
class RunConfig:
    """Configuration for one benchmark run."""
    compressor_name: str = "int8"
    draft_len: int = 4
    max_new_tokens: int = 32


def run_one(
    runtime: ModelRuntime,
    prompt_entry: dict,
    config: RunConfig,
) -> dict:
    """Benchmark a single prompt and return a JSON-compatible report dict.

    Args:
        runtime:      Loaded ModelRuntime.
        prompt_entry: Dict with at least ``prompt_id``, ``category``, ``prompt``.
        config:       RunConfig specifying compressor, draft_len, max_new_tokens.

    Returns:
        Dict with keys: prompt_id, prompt, category, model_name,
        compressor_name, draft_len, max_new_tokens, full, lossy, exactkv,
        memory, exactkv_failure.
    """
    compressor = get_compressor(config.compressor_name)
    prompt = prompt_entry["prompt"]
    max_new = config.max_new_tokens

    # 1. Full greedy (ground truth)
    full_res = generate_full_greedy(runtime, prompt, max_new)
    full_ids = full_res.generated_ids.squeeze(0).tolist()

    # 2. Lossy greedy (no verification)
    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()

    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    # 3. ExactKV
    ekv_res = ExactKVGenerator(runtime, compressor, draft_len=config.draft_len).generate(
        prompt, max_new
    )
    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()

    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)

    # 4. Memory estimate (fresh prefill — independent of generation)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "prompt": prompt,
        "category": prompt_entry.get("category", "unknown"),
        "model_name": runtime.model_name,
        "compressor_name": config.compressor_name,
        "draft_len": config.draft_len,
        "max_new_tokens": max_new,
        "full": {
            "output_ids": full_ids,
            "output_text": full_res.output_text,
        },
        "lossy": {
            "output_ids": lossy_ids,
            "output_text": lossy_res.output_text,
            "token_exact_match": lossy_exact,
            "first_divergence_idx": lossy_div,
        },
        "exactkv": {
            "output_ids": ekv_ids,
            "output_text": ekv_res.output_text,
            "token_exact_match": ekv_exact,
            "acceptance": acceptance.to_dict(),
        },
        "memory": mem.to_dict(),
        "exactkv_failure": not ekv_exact,
    }


def run_suite(
    runtime: ModelRuntime,
    prompts: list[dict],
    config: RunConfig,
) -> dict:
    """Run the benchmark on a list of prompts and aggregate results.

    Returns:
        Dict with ``results`` (per-prompt dicts) and ``aggregate`` summary.
    """
    results: list[dict] = []
    failures = 0

    for entry in prompts:
        report = run_one(runtime, entry, config)
        results.append(report)
        if report["exactkv_failure"]:
            failures += 1

    total = len(results)
    return {
        "results": results,
        "aggregate": {
            "total_prompts": total,
            "compressor_name": config.compressor_name,
            "exactkv_failures": failures,
            "exactkv_pass_rate": (total - failures) / max(total, 1),
        },
    }
