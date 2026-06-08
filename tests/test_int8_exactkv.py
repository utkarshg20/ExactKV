"""INT8 ExactKV gate — Step 11.

Key invariant: ExactKVGenerator with Int8Compressor must produce exactly the
same generated_ids as generate_full_greedy, for every prompt and length.

Additional checks per round:
  * accepted_count + rejected_count == len(draft_tokens)
  * cache_alignment: full_state.seq_len == compressed.logical_seq_len
  * trace is non-empty
"""
from __future__ import annotations

import pytest
import torch

from exactkv.compressors.int8 import Int8Compressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network",
    "Once upon a time in a land far away",
]
LENGTHS = [8, 20]


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def compressor() -> Int8Compressor:
    return Int8Compressor()


# ---------------------------------------------------------------------------
# Parametrised correctness gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", PROMPTS[:2])   # at least 2 prompts
@pytest.mark.parametrize("max_new_tokens", LENGTHS)  # at least 2 lengths
def test_int8_exactkv_matches_full_greedy(
    runtime: ModelRuntime,
    compressor: Int8Compressor,
    prompt: str,
    max_new_tokens: int,
) -> None:
    full_result = generate_full_greedy(runtime, prompt, max_new_tokens)
    exactkv_result = ExactKVGenerator(runtime, compressor, draft_len=4).generate(
        prompt, max_new_tokens
    )

    full_ids = full_result.generated_ids.squeeze(0).tolist()
    ekv_ids = exactkv_result.output_ids.squeeze(0).tolist()

    assert ekv_ids == full_ids, (
        f"[{prompt!r}, max={max_new_tokens}] "
        f"ExactKV output does not match full-KV greedy.\n"
        f"full:    {full_ids}\n"
        f"exactkv: {ekv_ids}"
    )


# ---------------------------------------------------------------------------
# Trace checks
# ---------------------------------------------------------------------------

def test_int8_exactkv_trace_exists(
    runtime: ModelRuntime, compressor: Int8Compressor
) -> None:
    result = ExactKVGenerator(runtime, compressor, draft_len=4).generate(
        PROMPTS[0], max_new_tokens=16
    )
    assert result.traces is not None
    assert len(result.traces) > 0, "Traces list is empty"


@pytest.mark.parametrize("prompt", PROMPTS[:2])
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
def test_int8_accepted_plus_rejected_equals_drafted(
    runtime: ModelRuntime,
    compressor: Int8Compressor,
    prompt: str,
    max_new_tokens: int,
) -> None:
    result = ExactKVGenerator(runtime, compressor, draft_len=4).generate(
        prompt, max_new_tokens
    )
    for i, entry in enumerate(result.traces):
        drafted = len(entry.draft_tokens)
        acc = entry.acceptance.num_accepted
        rej = len(entry.acceptance.rejected_tokens)
        assert acc + rej == drafted, (
            f"Round {i}: accepted({acc}) + rejected({rej}) != drafted({drafted})"
        )


@pytest.mark.parametrize("prompt", PROMPTS[:2])
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
def test_int8_cache_alignment_per_round(
    runtime: ModelRuntime,
    compressor: Int8Compressor,
    prompt: str,
    max_new_tokens: int,
) -> None:
    result = ExactKVGenerator(runtime, compressor, draft_len=4).generate(
        prompt, max_new_tokens
    )
    for i, entry in enumerate(result.traces):
        assert entry.full_seq_len_after == entry.compressed_seq_len_after, (
            f"Round {i}: full_seq_len_after={entry.full_seq_len_after} != "
            f"compressed_seq_len_after={entry.compressed_seq_len_after}"
        )
