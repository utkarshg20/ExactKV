"""Baseline gate: generate_full_greedy must match model.generate exactly.

This is the hard gate for Phase 1. Nothing past Step 3 should be implemented
until every assertion in this file passes.

Comparison contract:
  - model.generate returns full_sequence_ids (prompt + generated).
  - generate_full_greedy.full_sequence_ids must equal that.
  - generate_full_greedy.generated_ids must equal model.generate[:, prompt_len:].

Settings used throughout:
  - do_sample=False, num_beams=1  (greedy deterministic)
  - dtype=float32  (deterministic across runs in same process)
  - Qwen/Qwen2.5-0.5B  (small, fast, confirmed fp32-stable)
"""
from __future__ import annotations

import pytest
import torch

from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.generation import generate_full_greedy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    """Load the model once for the whole module; reuse across tests."""
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "Write a Python function that adds two numbers.",
    "Translate 'hello' into Spanish:",
]

LENGTHS = [8, 24]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
def test_full_greedy_matches_model_generate(
    runtime: ModelRuntime,
    prompt: str,
    max_new_tokens: int,
) -> None:
    """generate_full_greedy token IDs must exactly equal model.generate under greedy."""

    prompt_ids = runtime.encode(prompt)  # [1, prompt_len]
    prompt_len = prompt_ids.shape[1]

    # --- Reference: model.generate ------------------------------------------
    with torch.no_grad():
        ref_ids: torch.Tensor = runtime.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=runtime.eos_token_id,
            use_cache=True,
        )
    # ref_ids: [1, prompt_len + generated_len]

    # --- Custom greedy loop -------------------------------------------------
    result = generate_full_greedy(runtime, prompt, max_new_tokens)

    # --- Compare full_sequence_ids ------------------------------------------
    # Trim ref_ids to the same length as result in case model.generate stops
    # at a different boundary (it should not under greedy, but be explicit).
    expected_full = ref_ids  # [1, prompt_len + N]
    actual_full = result.full_sequence_ids  # [1, prompt_len + M]

    assert actual_full.shape == expected_full.shape, (
        f"prompt={prompt!r}, max_new_tokens={max_new_tokens}\n"
        f"  expected shape: {expected_full.shape}\n"
        f"  actual shape:   {actual_full.shape}\n"
        f"  expected ids:   {expected_full.tolist()}\n"
        f"  actual ids:     {actual_full.tolist()}"
    )

    mismatch = (actual_full != expected_full).nonzero(as_tuple=True)
    assert len(mismatch[0]) == 0, (
        f"prompt={prompt!r}, max_new_tokens={max_new_tokens}\n"
        f"  first mismatch at positions {mismatch}\n"
        f"  expected ids: {expected_full.tolist()}\n"
        f"  actual ids:   {actual_full.tolist()}\n"
        f"  expected text: {runtime.decode(expected_full[:, prompt_len:])!r}\n"
        f"  actual text:   {result.output_text!r}"
    )

    # --- Compare generated_ids slice ----------------------------------------
    expected_gen = ref_ids[:, prompt_len:]
    actual_gen = result.generated_ids

    assert actual_gen.shape == expected_gen.shape, (
        f"generated_ids shape mismatch: {actual_gen.shape} vs {expected_gen.shape}"
    )
    gen_mismatch = (actual_gen != expected_gen).nonzero(as_tuple=True)
    assert len(gen_mismatch[0]) == 0, (
        f"generated_ids mismatch at {gen_mismatch}\n"
        f"  expected: {expected_gen.tolist()}\n"
        f"  actual:   {actual_gen.tolist()}"
    )

    # --- Sanity: prompt slice of full_sequence_ids is unchanged --------------
    actual_prompt_slice = result.full_sequence_ids[:, :prompt_len]
    assert torch.equal(actual_prompt_slice, prompt_ids), (
        "Prompt slice of full_sequence_ids does not match input prompt_ids."
    )
