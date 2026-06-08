"""Verification engine tests — require the real model (Qwen/Qwen2.5-0.5B, fp32).

What is tested
--------------
1. With NoOp-equivalent drafts (tokens produced by the full model), every
   token is accepted and all_matched=True.
2. When all match, correction_token is None.
3. The authoritative FullKVState's KV length is unchanged after verification.
4. Cache length assertions hold (seq_len matches prompt_len after prefill).
5. When an intentionally wrong draft is passed, only the correct prefix
   is accepted and correction_token is the verifier's predicted token.
"""
from __future__ import annotations

import copy

import pytest
import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len
from exactkv.compressors.noop import NoOpCompressor
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.generation import generate_full_greedy
from exactkv.verification.engine import VerificationEngine

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def engine(runtime: ModelRuntime) -> VerificationEngine:
    return VerificationEngine(runtime)


def _make_full_state_after_prefill(runtime: ModelRuntime, prompt: str) -> tuple[FullKVState, int]:
    """Run prefill and return (FullKVState, next_token_id) with no generated tokens yet."""
    prompt_ids = runtime.encode(prompt)  # [1, L]
    with torch.no_grad():
        out = runtime.forward(prompt_ids, past_key_values=None, use_cache=True)

    next_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
    empty_gen = torch.zeros(1, 0, dtype=torch.long, device=runtime.device)
    state = FullKVState(
        past_key_values=out.past_key_values,
        prompt_ids=prompt_ids,
        generated_ids=empty_gen,
        full_sequence_ids=prompt_ids,
        device=runtime.device,
        dtype=runtime.dtype,
        metadata={"next_token_id": next_tok},
    )
    return state, next_tok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PROMPT = "The capital of France is"


def test_noop_draft_all_accepted(runtime: ModelRuntime, engine: VerificationEngine) -> None:
    """Draft tokens produced by the full model must all be accepted."""
    full_state, first_tok = _make_full_state_after_prefill(runtime, PROMPT)

    # Build draft by running the full model 4 steps (identical to what NoOp would do).
    # Deep-copy the KV so that forward passes here do not mutate full_state.past_kv
    # (DynamicCache is mutated in-place in transformers >= 4.36).
    draft_tokens: list[int] = []
    past_kv = copy.deepcopy(full_state.past_key_values)
    next_tok = first_tok
    n = 4

    with torch.no_grad():
        for i in range(n):
            draft_tokens.append(next_tok)
            if i < n - 1:
                out = runtime.forward(
                    torch.tensor([[next_tok]], dtype=torch.long, device=runtime.device),
                    past_key_values=past_kv,
                )
                past_kv = out.past_key_values
                next_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())

    result = engine.verify_sequential(full_state, draft_tokens)

    assert result.all_matched is True, (
        f"Expected all matched, but verifier_tokens={result.verifier_tokens} "
        f"vs draft_tokens={result.draft_tokens}"
    )
    assert result.accepted_tokens == draft_tokens
    assert result.correction_token is None
    assert result.rejected_tokens == []
    assert result.bonus_token is None


def test_correction_token_none_on_full_match(runtime: ModelRuntime, engine: VerificationEngine) -> None:
    """correction_token is None when all draft tokens match."""
    full_state, first_tok = _make_full_state_after_prefill(runtime, PROMPT)
    draft = [first_tok]  # single token — definitely matches
    result = engine.verify_sequential(full_state, draft)

    assert result.correction_token is None
    assert result.all_matched is True


def test_authoritative_state_unchanged(runtime: ModelRuntime, engine: VerificationEngine) -> None:
    """Verification must not mutate the authoritative FullKVState."""
    full_state, first_tok = _make_full_state_after_prefill(runtime, PROMPT)

    kv_len_before = kv_seq_len(full_state.past_key_values)
    next_tok_before = full_state.next_token_id
    # Use a slightly longer draft to force internal temp_kv advancement
    draft = [first_tok, first_tok, first_tok]

    engine.verify_sequential(full_state, draft)

    kv_len_after = kv_seq_len(full_state.past_key_values)
    assert kv_len_after == kv_len_before, (
        f"KV seq_len changed: {kv_len_before} → {kv_len_after}"
    )
    assert full_state.next_token_id == next_tok_before, (
        "full_state.next_token_id was mutated by verify_sequential"
    )


def test_cache_length_after_prefill(runtime: ModelRuntime) -> None:
    """KV seq_len after prefill equals the prompt token count."""
    prompt = "Hello"
    full_state, _ = _make_full_state_after_prefill(runtime, prompt)
    prompt_len = full_state.prompt_len
    kv_len = kv_seq_len(full_state.past_key_values)

    assert kv_len == prompt_len, (
        f"Expected KV seq_len == prompt_len ({prompt_len}), got {kv_len}"
    )


def test_wrong_draft_partial_accept(runtime: ModelRuntime, engine: VerificationEngine) -> None:
    """When a wrong token is injected, only the correct prefix is accepted."""
    full_state, first_tok = _make_full_state_after_prefill(runtime, PROMPT)

    WRONG_TOKEN = 0  # token 0 is almost certainly not the model's prediction
    draft = [first_tok, WRONG_TOKEN, first_tok]  # second token is wrong
    result = engine.verify_sequential(full_state, draft)

    # First token must match (we used the actual prediction).
    assert result.accepted_tokens == [first_tok]
    assert result.correction_token is not None
    assert result.correction_token != WRONG_TOKEN
    assert result.all_matched is False
    assert result.bonus_token is None
