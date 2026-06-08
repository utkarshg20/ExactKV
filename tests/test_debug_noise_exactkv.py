"""DebugNoiseCompressor tests — Step 12.

Behavioural invariants:
  * Lossy mode with DebugNoise diverges from full for at least one prompt.
  * ExactKV with DebugNoise still produces token-ID-exact output.
  * At least one round has rejected_tokens non-empty.
  * At least one round has correction_token not None
    OR acceptance_rate < 1.0 overall.
"""
from __future__ import annotations

from typing import Optional

import pytest
import torch

from exactkv.compressors.debug_noise import DebugNoiseCompressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import (
    generate_full_greedy,
    generate_lossy_greedy,
)
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network",
]
MAX_NEW_TOKENS = 16


def _first_divergence(a: torch.Tensor, b: torch.Tensor) -> Optional[int]:
    al, bl = a.squeeze(0).tolist(), b.squeeze(0).tolist()
    for i in range(min(len(al), len(bl))):
        if al[i] != bl[i]:
            return i
    if len(al) != len(bl):
        return min(len(al), len(bl))
    return None


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def noise_compressor() -> DebugNoiseCompressor:
    return DebugNoiseCompressor(noise_scale=10.0)


# ---------------------------------------------------------------------------
# Lossy divergence gate
# ---------------------------------------------------------------------------

def test_debug_noise_lossy_diverges_from_full(
    runtime: ModelRuntime, noise_compressor: DebugNoiseCompressor
) -> None:
    """Noisy lossy generation must diverge from full greedy for at least one prompt."""
    any_divergence = False
    for prompt in PROMPTS:
        full_res = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
        lossy_res = generate_lossy_greedy(runtime, prompt, noise_compressor, MAX_NEW_TOKENS)
        div = _first_divergence(full_res.generated_ids, lossy_res.generated_ids)
        if div is not None:
            any_divergence = True
            break

    assert any_divergence, (
        "DebugNoiseCompressor did not cause any divergence in lossy generation. "
        "noise_scale may be too small, or the test prompts may be unusual."
    )


# ---------------------------------------------------------------------------
# ExactKV correctness with DebugNoise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", PROMPTS)
def test_debug_noise_exactkv_matches_full_greedy(
    runtime: ModelRuntime, noise_compressor: DebugNoiseCompressor, prompt: str
) -> None:
    """ExactKV must still produce exact output despite noisy drafts."""
    full_result = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
    ekv_result = ExactKVGenerator(runtime, noise_compressor, draft_len=4).generate(
        prompt, MAX_NEW_TOKENS
    )

    full_ids = full_result.generated_ids.squeeze(0).tolist()
    ekv_ids = ekv_result.output_ids.squeeze(0).tolist()

    assert ekv_ids == full_ids, (
        f"[{prompt!r}] ExactKV(DebugNoise) does not match full greedy.\n"
        f"full:    {full_ids}\n"
        f"exactkv: {ekv_ids}"
    )


# ---------------------------------------------------------------------------
# Rejection gate — must reject at least sometimes
# ---------------------------------------------------------------------------

def test_debug_noise_has_rejection(
    runtime: ModelRuntime, noise_compressor: DebugNoiseCompressor
) -> None:
    """At least one round must have non-empty rejected_tokens."""
    found_rejection = False
    for prompt in PROMPTS:
        result = ExactKVGenerator(runtime, noise_compressor, draft_len=4).generate(
            prompt, MAX_NEW_TOKENS
        )
        for entry in result.traces:
            if len(entry.acceptance.rejected_tokens) > 0:
                found_rejection = True
                break
        if found_rejection:
            break

    assert found_rejection, "No rounds had rejected_tokens — rejection logic may be broken"


def test_debug_noise_has_correction_or_low_acceptance(
    runtime: ModelRuntime, noise_compressor: DebugNoiseCompressor
) -> None:
    """At least one round must have correction_token != None OR acceptance_rate < 1.0."""
    found = False
    for prompt in PROMPTS:
        result = ExactKVGenerator(runtime, noise_compressor, draft_len=4).generate(
            prompt, MAX_NEW_TOKENS
        )
        for entry in result.traces:
            if entry.acceptance.correction_token is not None:
                found = True
                break
        if found:
            break

    assert found, (
        "No rounds had correction_token != None. "
        "DebugNoiseCompressor may not be forcing mismatches."
    )
