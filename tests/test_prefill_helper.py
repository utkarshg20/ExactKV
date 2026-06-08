"""Tests for exactkv.runtime.prefill.prefill_to_full_state.

Verifies that the shared prefill helper:
  * returns a correctly-shaped FullKVState
  * stores the first greedy next_token_id in metadata
  * leaves generated_ids empty
  * full_sequence_ids == prompt_ids after prefill
  * is deterministic (same prompt → same next_token_id)
  * matches the token that generate_full_greedy would produce first
"""
from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


def test_prefill_returns_full_kv_state(runtime):
    from exactkv.cache.full_state import FullKVState
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    assert isinstance(state, FullKVState)


def test_prefill_prompt_ids_shape(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    assert state.prompt_ids.ndim == 2
    assert state.prompt_ids.shape[0] == 1
    assert state.prompt_ids.shape[1] > 0


def test_prefill_generated_ids_empty(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    assert state.generated_ids.shape == (1, 0), (
        f"generated_ids should be empty after prefill, got shape {state.generated_ids.shape}"
    )


def test_prefill_full_sequence_ids_equals_prompt_ids(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    assert torch.equal(state.full_sequence_ids, state.prompt_ids), (
        "full_sequence_ids should equal prompt_ids when nothing has been generated"
    )


def test_prefill_metadata_has_next_token_id(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    assert "next_token_id" in state.metadata, "metadata must contain 'next_token_id'"
    nxt = state.metadata["next_token_id"]
    assert isinstance(nxt, int) and nxt >= 0, f"next_token_id must be a non-negative int, got {nxt!r}"


def test_prefill_device_and_dtype(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    assert state.device == runtime.device
    assert state.dtype == runtime.dtype


def test_prefill_past_key_values_not_none(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    assert state.past_key_values is not None


def test_prefill_deterministic(runtime):
    """Same prompt must yield the same next_token_id on two calls."""
    from exactkv.runtime.prefill import prefill_to_full_state

    s1 = prefill_to_full_state(runtime, PROMPT)
    s2 = prefill_to_full_state(runtime, PROMPT)
    assert s1.metadata["next_token_id"] == s2.metadata["next_token_id"]


def test_prefill_next_token_matches_generate_full_greedy(runtime):
    """The first token from generate_full_greedy must match the helper's next_token_id."""
    from exactkv.runtime.generation import generate_full_greedy
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    helper_tok = state.metadata["next_token_id"]

    full_res = generate_full_greedy(runtime, PROMPT, max_new_tokens=1)
    first_generated_tok = int(full_res.generated_ids[0, 0].item())

    assert helper_tok == first_generated_tok, (
        f"prefill helper next_token_id={helper_tok} != "
        f"generate_full_greedy first token={first_generated_tok}"
    )
