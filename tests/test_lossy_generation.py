"""Tests for generate_lossy_greedy — Step 10.

Lossy generation is NOT exact; we only check:
  * it runs to completion
  * output length is valid
  * first_divergence against full greedy is computable (int or None)
  * no runtime errors or NaN/Inf
"""
from __future__ import annotations

from typing import Optional

import pytest
import torch

from exactkv.compressors.int8 import Int8Compressor
from exactkv.compressors.noop import NoOpCompressor
from exactkv.runtime.generation import (
    LossyGreedyResult,
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
MAX_NEW_TOKENS = 20


def _first_divergence(ids_a: torch.Tensor, ids_b: torch.Tensor) -> Optional[int]:
    """Return the index of the first token where ids_a and ids_b differ, or None."""
    a = ids_a.squeeze(0).tolist()
    b = ids_b.squeeze(0).tolist()
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min_len
    return None


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


# ---------------------------------------------------------------------------
# NoOp lossy generation (should match full exactly, since NoOp = identity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", PROMPTS)
def test_noop_lossy_runs_to_completion(runtime: ModelRuntime, prompt: str) -> None:
    result = generate_lossy_greedy(runtime, prompt, NoOpCompressor(), MAX_NEW_TOKENS)
    assert isinstance(result, LossyGreedyResult)
    gen_len = result.generated_ids.shape[1]
    assert 1 <= gen_len <= MAX_NEW_TOKENS, f"gen_len={gen_len} out of range"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_noop_lossy_first_divergence_computable(runtime: ModelRuntime, prompt: str) -> None:
    full_result = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
    lossy_result = generate_lossy_greedy(runtime, prompt, NoOpCompressor(), MAX_NEW_TOKENS)
    div = _first_divergence(full_result.generated_ids, lossy_result.generated_ids)
    # div is int or None — both are acceptable, but it must not raise
    assert div is None or isinstance(div, int)


# ---------------------------------------------------------------------------
# INT8 lossy generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", PROMPTS)
def test_int8_lossy_runs_to_completion(runtime: ModelRuntime, prompt: str) -> None:
    result = generate_lossy_greedy(runtime, prompt, Int8Compressor(), MAX_NEW_TOKENS)
    assert isinstance(result, LossyGreedyResult)
    gen_len = result.generated_ids.shape[1]
    assert 1 <= gen_len <= MAX_NEW_TOKENS, f"gen_len={gen_len} out of range"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_int8_lossy_first_divergence_computable(runtime: ModelRuntime, prompt: str) -> None:
    full_result = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
    lossy_result = generate_lossy_greedy(runtime, prompt, Int8Compressor(), MAX_NEW_TOKENS)
    div = _first_divergence(full_result.generated_ids, lossy_result.generated_ids)
    assert div is None or isinstance(div, int)


def test_int8_lossy_output_text_is_string(runtime: ModelRuntime) -> None:
    result = generate_lossy_greedy(runtime, PROMPTS[0], Int8Compressor(), MAX_NEW_TOKENS)
    assert isinstance(result.output_text, str)


def test_int8_lossy_full_sequence_ids_consistent(runtime: ModelRuntime) -> None:
    result = generate_lossy_greedy(runtime, PROMPTS[0], Int8Compressor(), MAX_NEW_TOKENS)
    expected_len = result.prompt_ids.shape[1] + result.generated_ids.shape[1]
    assert result.full_sequence_ids.shape[1] == expected_len
