"""NoOp ExactKV gate — requires the real model (Qwen/Qwen2.5-0.5B, fp32).

This is the primary correctness gate for Phase 1.  With NoOpCompressor every
draft token must equal the full model's prediction, so:

  exactkv_result.output_ids  ==  generate_full_greedy.generated_ids

ALL of the following must hold for the gate to pass:

1. output_ids exactly equals full-greedy generated_ids (token-for-token).
2. acceptance_rate == 1.0.
3. Every round has acceptance.all_matched == True.
4. correction_token is None in every round.
5. bonus_token is None in every round.
6. rejected_tokens == [] in every round.
7. Cache alignment: full_seq_len_after == compressed_seq_len_after in every trace.
"""
from __future__ import annotations

import pytest
import torch

from exactkv.compressors.noop import NoOpCompressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def generator(runtime: ModelRuntime) -> ExactKVGenerator:
    return ExactKVGenerator(runtime=runtime, compressor=NoOpCompressor(), draft_len=4)


# ---------------------------------------------------------------------------
# Test matrix: 2 prompts × 2 lengths × 2 draft_lengths implicitly
# ---------------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "Write a Python function that adds two numbers.",
]
LENGTHS = [8, 20]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
def test_noop_output_equals_full_greedy(
    runtime: ModelRuntime,
    generator: ExactKVGenerator,
    prompt: str,
    max_new_tokens: int,
) -> None:
    """Gate 1: ExactKV(NoOp) output_ids must exactly match generate_full_greedy."""

    full_result = generate_full_greedy(runtime, prompt, max_new_tokens)
    exactkv_result = generator.generate(prompt, max_new_tokens)

    expected = full_result.generated_ids    # [1, N]
    actual   = exactkv_result.output_ids    # [1, N]

    assert actual.shape == expected.shape, (
        f"Shape mismatch: expected {expected.shape}, got {actual.shape}\n"
        f"  expected text: {runtime.decode(expected)!r}\n"
        f"  actual text:   {exactkv_result.output_text!r}"
    )
    mismatch = (actual != expected).nonzero(as_tuple=True)
    assert len(mismatch[0]) == 0, (
        f"Token mismatch at positions {mismatch}\n"
        f"  expected ids: {expected.tolist()}\n"
        f"  actual ids:   {actual.tolist()}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
def test_noop_acceptance_rate_is_one(
    generator: ExactKVGenerator,
    prompt: str,
    max_new_tokens: int,
) -> None:
    """Gate 2: acceptance_rate must be exactly 1.0 for NoOp."""
    result = generator.generate(prompt, max_new_tokens)

    assert result.acceptance_rate == 1.0, (
        f"acceptance_rate={result.acceptance_rate} (expected 1.0)"
    )
    assert result.total_rejected == 0, (
        f"total_rejected={result.total_rejected} (expected 0)"
    )
    assert result.total_corrections == 0, (
        f"total_corrections={result.total_corrections} (expected 0)"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
def test_noop_per_round_invariants(
    generator: ExactKVGenerator,
    prompt: str,
    max_new_tokens: int,
) -> None:
    """Gate 3–7: per-round trace invariants for NoOp."""
    result = generator.generate(prompt, max_new_tokens)

    assert result.num_rounds > 0, "Expected at least one round"

    for trace in result.traces:
        acc = trace.acceptance

        # 3. all_matched must be True
        assert acc.all_matched is True, (
            f"Round {trace.round_idx}: all_matched=False\n"
            f"  draft:    {trace.draft_tokens}\n"
            f"  verifier: {acc.verifier_tokens}"
        )

        # 4. correction_token is always None
        assert acc.correction_token is None, (
            f"Round {trace.round_idx}: correction_token={acc.correction_token}"
        )

        # 5. bonus_token is always None
        assert acc.bonus_token is None, (
            f"Round {trace.round_idx}: bonus_token={acc.bonus_token}"
        )

        # 6. rejected_tokens is always empty
        assert acc.rejected_tokens == [], (
            f"Round {trace.round_idx}: rejected_tokens={acc.rejected_tokens}"
        )

        # 7. Cache alignment: full_seq_len_after == compressed_seq_len_after
        assert trace.full_seq_len_after == trace.compressed_seq_len_after, (
            f"Round {trace.round_idx}: cache misalignment: "
            f"full={trace.full_seq_len_after}, "
            f"compressed={trace.compressed_seq_len_after}"
        )
