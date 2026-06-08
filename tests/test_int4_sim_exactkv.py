"""ExactKV gate: Int4SimCompressor correctness.

Gate: int4_sim ExactKV gate.

Verifies:
  * ExactKVGenerator with Int4SimCompressor produces output_ids that exactly
    match generate_full_greedy output_ids (token ID equality).
  * Runs over at least 2 prompts × 2 max_new_token lengths.
  * Trace exists and per-round bookkeeping reconciles:
      drafted_count == accepted_count + rejected_count
  * Cache alignment holds after every round:
      full_state.seq_len == compressed_state.logical_seq_len
  * If lossy INT4-sim diverges from full on any prompt, ExactKV still
    corrects the divergence.

V1 behavior: greedy decoding only, fp32, sequential verification, no bonus token.
"""
from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# 2 prompts × 2 lengths = 4 combinations
_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]
_MAX_NEW_TOKENS = [8, 20]


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_exactkv_int4sim(runtime, prompt: str, max_new_tokens: int):
    """Return (full_result, lossy_result, ekv_result) for one combination."""
    from exactkv.compressors.int4_sim import Int4SimCompressor
    from exactkv.runtime.exactkv_generator import ExactKVGenerator
    from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy

    compressor = Int4SimCompressor()
    full_res = generate_full_greedy(runtime, prompt, max_new_tokens)
    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)
    ekv_res = ExactKVGenerator(runtime, compressor, draft_len=4).generate(
        prompt, max_new_tokens
    )
    return full_res, lossy_res, ekv_res


# ---------------------------------------------------------------------------
# Correctness gate: ExactKV == full for all prompt × length combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", _PROMPTS)
@pytest.mark.parametrize("max_new_tokens", _MAX_NEW_TOKENS)
def test_int4_sim_exactkv_matches_full(runtime, prompt, max_new_tokens):
    """INT4-sim ExactKV output_ids must exactly equal generate_full_greedy output_ids."""
    from exactkv.metrics.exactness import token_exact_match

    full_res, _, ekv_res = _run_exactkv_int4sim(runtime, prompt, max_new_tokens)
    match = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    assert match, (
        f"INT4-sim ExactKV output diverged from full greedy!\n"
        f"  prompt={prompt!r}  max_new_tokens={max_new_tokens}\n"
        f"  full   : {full_res.generated_ids.squeeze(0).tolist()}\n"
        f"  exactkv: {ekv_res.output_ids.squeeze(0).tolist()}"
    )


# ---------------------------------------------------------------------------
# Trace exists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", _PROMPTS)
def test_int4_sim_trace_exists(runtime, prompt):
    _, _, ekv_res = _run_exactkv_int4sim(runtime, prompt, max_new_tokens=8)
    assert ekv_res.traces is not None
    assert len(ekv_res.traces) > 0, "ExactKV should produce at least one round trace"


# ---------------------------------------------------------------------------
# Per-round bookkeeping: drafted == accepted + rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", _PROMPTS)
def test_int4_sim_bookkeeping_reconciles(runtime, prompt):
    _, _, ekv_res = _run_exactkv_int4sim(runtime, prompt, max_new_tokens=20)
    for i, trace in enumerate(ekv_res.traces):
        acc = trace.acceptance
        drafted = len(trace.draft_tokens)
        reconciled = acc.num_accepted + acc.num_rejected
        assert drafted == reconciled, (
            f"Round {i}: drafted={drafted} != accepted({acc.num_accepted}) "
            f"+ rejected({acc.num_rejected})"
        )


# ---------------------------------------------------------------------------
# Cache alignment: full_seq_len == compressed_seq_len after each round
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", _PROMPTS)
def test_int4_sim_cache_alignment(runtime, prompt):
    _, _, ekv_res = _run_exactkv_int4sim(runtime, prompt, max_new_tokens=20)
    for i, trace in enumerate(ekv_res.traces):
        assert trace.full_seq_len_after == trace.compressed_seq_len_after, (
            f"Round {i}: cache misaligned — "
            f"full_seq_len={trace.full_seq_len_after} "
            f"compressed_seq_len={trace.compressed_seq_len_after}"
        )


# ---------------------------------------------------------------------------
# Divergence + correction: if lossy diverges, ExactKV still matches full
# ---------------------------------------------------------------------------

def test_int4_sim_exactkv_corrects_lossy_divergence(runtime):
    """Even if INT4-sim lossy mode diverges, ExactKV must produce the full output."""
    from exactkv.metrics.exactness import first_divergence_idx, token_exact_match

    # Use a prompt and length likely to expose INT4 quantisation error.
    prompt = "def fibonacci(n):"
    max_new_tokens = 20

    full_res, lossy_res, ekv_res = _run_exactkv_int4sim(runtime, prompt, max_new_tokens)

    lossy_matches = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    ekv_matches = token_exact_match(full_res.generated_ids, ekv_res.output_ids)

    # ExactKV must always match full regardless of whether lossy diverged.
    assert ekv_matches, (
        f"INT4-sim ExactKV failed to match full output!\n"
        f"  lossy_matches_full={lossy_matches}\n"
        f"  full   : {full_res.generated_ids.squeeze(0).tolist()}\n"
        f"  lossy  : {lossy_res.generated_ids.squeeze(0).tolist()}\n"
        f"  exactkv: {ekv_res.output_ids.squeeze(0).tolist()}"
    )

    div_idx = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)
    if not lossy_matches:
        # When lossy diverged, confirm ExactKV still has trace showing corrections.
        correction_rounds = [
            t for t in ekv_res.traces if t.acceptance.correction_token is not None
        ]
        assert len(correction_rounds) > 0, (
            f"Lossy diverged at index {div_idx} but ExactKV trace shows no corrections — "
            "the correction logic may not be activating"
        )


# ---------------------------------------------------------------------------
# Acceptance rate is defined (0 to 1 inclusive)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", _PROMPTS)
def test_int4_sim_acceptance_rate_valid(runtime, prompt):
    _, _, ekv_res = _run_exactkv_int4sim(runtime, prompt, max_new_tokens=20)
    assert 0.0 <= ekv_res.acceptance_rate <= 1.0, (
        f"acceptance_rate out of range: {ekv_res.acceptance_rate}"
    )


# ---------------------------------------------------------------------------
# bonus_token is always None (V1 feature-flag)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", _PROMPTS)
def test_int4_sim_no_bonus_token(runtime, prompt):
    _, _, ekv_res = _run_exactkv_int4sim(runtime, prompt, max_new_tokens=8)
    for i, trace in enumerate(ekv_res.traces):
        assert trace.acceptance.bonus_token is None, (
            f"Round {i}: bonus_token should be None in V1"
        )
