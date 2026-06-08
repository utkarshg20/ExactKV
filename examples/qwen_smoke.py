#!/usr/bin/env python3
"""ExactKV V1 Demo — Qwen/Qwen2.5-0.5B

Runs one prompt under three modes and prints a side-by-side comparison:
  * Full KV greedy output (ground truth)
  * Lossy compressed output (no verification)
  * ExactKV output (draft-verify-commit, must match full)

Usage:
    python examples/qwen_smoke.py
    python examples/qwen_smoke.py --compressor int8 --max-new-tokens 40
    python examples/qwen_smoke.py --compressor debug_noise --draft-len 4

This script does NOT claim throughput or latency improvements.
V1 goal: prove that ExactKV output_ids == full_greedy_ids under greedy decoding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow running as a script from anywhere inside the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exactkv.benchmarks.runner import RunConfig, _make_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime

_DEMO_PROMPT = "The capital of France is"
_SEPARATOR = "─" * 70


def _print_section(title: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {title}")
    print(_SEPARATOR)


def main(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    prompt: str = _DEMO_PROMPT,
    max_new_tokens: int = 32,
    draft_len: int = 4,
    compressor_name: str = "int8",
    runtime: Optional[ModelRuntime] = None,
) -> dict:
    """Run the ExactKV demo and return a result summary dict.

    The ``runtime`` parameter can be provided by tests to avoid reloading
    the model.  When None, a fresh ModelRuntime is created.

    Returns:
        dict with keys: prompt, full_text, lossy_text, exactkv_text,
        exactkv_matches_full, lossy_matches_full, acceptance_rate,
        avg_accepted_per_round, correction_count, rejection_count,
        first_lossy_divergence_idx, compressor_name.
    """
    if runtime is None:
        print(f"Loading {model_name} …")
        runtime = ModelRuntime(model_name=model_name, device="auto", dtype="float32")

    compressor = _make_compressor(compressor_name)

    _print_section(f"Prompt")
    print(f"  {prompt!r}")
    print(f"  compressor={compressor_name}  draft_len={draft_len}  "
          f"max_new_tokens={max_new_tokens}")

    # 1. Full KV greedy
    full_res = generate_full_greedy(runtime, prompt, max_new_tokens)

    # 2. Lossy greedy
    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)

    # 3. ExactKV
    ekv_res = ExactKVGenerator(runtime, compressor, draft_len=draft_len).generate(
        prompt, max_new_tokens
    )

    # 4. Metrics
    acceptance = summarize_acceptance(ekv_res.traces)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    exactkv_ok = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    lossy_ok = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    # --- Print results -------------------------------------------------------
    _print_section("Full KV output (ground truth)")
    print(f"  {full_res.output_text!r}")

    _print_section(f"Lossy output ({compressor_name}, no verification)")
    print(f"  {lossy_res.output_text!r}")
    if lossy_div is not None:
        print(f"  First divergence from full at token index: {lossy_div}")
    else:
        print("  No divergence from full (same output)")

    _print_section("ExactKV output (draft-verify-commit)")
    print(f"  {ekv_res.output_text!r}")

    _print_section("Summary")
    print(f"  ExactKV matches full : {exactkv_ok}")
    print(f"  Lossy  matches full  : {lossy_ok}")
    print(f"  First lossy divergence idx : {lossy_div}")
    print()
    print(f"  Acceptance rate        : {acceptance.acceptance_rate:.3f}")
    print(f"  Avg accepted / round   : {acceptance.avg_accepted_per_round:.2f}")
    print(f"  Correction count       : {acceptance.total_corrections}")
    print(f"  Rejection count        : {acceptance.total_rejected}")
    print(f"  Rounds                 : {acceptance.total_rounds}")
    print()
    print(f"  KV full bytes     (prompt): {mem.full_bytes:,}")
    print(f"  KV compressed bytes      : {mem.compressed_bytes:,}")
    print(f"  Compression ratio        : {mem.compression_ratio:.3f}  (compressed/full; < 1 means smaller)")
    print(f"  Memory reduction factor  : {mem.memory_reduction_factor:.2f}x  (full/compressed; > 1 means savings)")
    print()

    if not exactkv_ok:
        print("  *** WARNING: ExactKV output does NOT match full output! ***")
        print("      This indicates a bug in the verification or commit logic.")

    # Trace snippet
    _print_section("Trace (first 3 rounds)")
    for trace in ekv_res.traces[:3]:
        acc = trace.acceptance
        print(
            f"  round {trace.round_idx}: drafted={len(trace.draft_tokens)}, "
            f"accepted={acc.num_accepted}, rejected={acc.num_rejected}, "
            f"correction={'yes' if acc.correction_token is not None else 'no'}"
        )

    print(_SEPARATOR)

    return {
        "prompt": prompt,
        "full_text": full_res.output_text,
        "lossy_text": lossy_res.output_text,
        "exactkv_text": ekv_res.output_text,
        "exactkv_matches_full": exactkv_ok,
        "lossy_matches_full": lossy_ok,
        "acceptance_rate": acceptance.acceptance_rate,
        "avg_accepted_per_round": acceptance.avg_accepted_per_round,
        "correction_count": acceptance.total_corrections,
        "rejection_count": acceptance.total_rejected,
        "first_lossy_divergence_idx": lossy_div,
        "compressor_name": compressor_name,
        "memory": mem.to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ExactKV V1 demo — Qwen/Qwen2.5-0.5B"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default=_DEMO_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32, dest="max_new_tokens")
    parser.add_argument("--draft-len", type=int, default=4, dest="draft_len")
    parser.add_argument(
        "--compressor",
        default="int8",
        choices=["noop", "int8", "debug_noise"],
    )
    args = parser.parse_args()
    main(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
        compressor_name=args.compressor,
    )
