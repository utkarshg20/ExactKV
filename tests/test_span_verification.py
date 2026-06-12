"""Span verification tests — V13 Phase 2.

Requires Qwen/Qwen2.5-0.5B (offline cache).  Compares verify_span to verify_sequential
and generate_full_greedy; does not measure timing or throughput.
"""
from __future__ import annotations

import copy

import pytest
import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len
from exactkv.compressors import get_compressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.acceptance import compute_acceptance
from exactkv.verification.engine import VerificationEngine

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
PROMPT = "The capital of France is"
GENERATOR_PROMPT = "Write one short sentence about rivers."
COMPRESSORS = ("noop", "int8", "k8_v4_sim", "k8_v4_boundary4_v8_sim")
DRAFT_LEN = 4
MAX_NEW = 16


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def engine(runtime: ModelRuntime) -> VerificationEngine:
    return VerificationEngine(runtime)


def _make_full_state_after_prefill(runtime: ModelRuntime, prompt: str) -> FullKVState:
    prompt_ids = runtime.encode(prompt)
    with torch.no_grad():
        out = runtime.forward(prompt_ids, past_key_values=None, use_cache=True)
    next_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
    empty_gen = torch.zeros(1, 0, dtype=torch.long, device=runtime.device)
    return FullKVState(
        past_key_values=out.past_key_values,
        prompt_ids=prompt_ids,
        generated_ids=empty_gen,
        full_sequence_ids=prompt_ids,
        device=runtime.device,
        dtype=runtime.dtype,
        metadata={"next_token_id": next_tok},
    )


def _full_greedy_draft(runtime: ModelRuntime, full_state: FullKVState, n: int) -> list[int]:
    draft_tokens: list[int] = []
    past_kv = copy.deepcopy(full_state.past_key_values)
    next_tok = full_state.next_token_id
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
    return draft_tokens


def test_invalid_verification_method_raises(runtime: ModelRuntime) -> None:
    comp = get_compressor("noop")
    with pytest.raises(ValueError, match="Invalid verification_method"):
        ExactKVGenerator(runtime, comp, verification_method="parallel")  # type: ignore[arg-type]


def test_empty_draft(engine: VerificationEngine) -> None:
    full_state = _make_full_state_after_prefill(engine.runtime, PROMPT)
    seq = engine.verify_sequential(full_state, [])
    span = engine.verify_span(full_state, [])
    assert seq == span
    assert span.all_matched is True
    assert span.num_accepted == 0


def test_span_matches_sequential_all_match(
    runtime: ModelRuntime, engine: VerificationEngine
) -> None:
    full_state = _make_full_state_after_prefill(runtime, PROMPT)
    draft = _full_greedy_draft(runtime, full_state, 4)
    seq = engine.verify_sequential(full_state, draft)
    span = engine.verify_span(full_state, draft)
    assert span == seq
    assert span.all_matched is True
    assert span.accepted_tokens == draft


def test_golden_logits_shift(
    runtime: ModelRuntime, engine: VerificationEngine
) -> None:
    """v_0 from cache; v_1 from logits[:, 0, :] after teacher-forced span forward."""
    full_state = _make_full_state_after_prefill(runtime, PROMPT)
    draft = _full_greedy_draft(runtime, full_state, 4)
    assert len(draft) > 1

    span = engine.verify_span(full_state, draft)
    assert span.verifier_tokens[0] == full_state.next_token_id

    temp_kv = copy.deepcopy(full_state.past_key_values)
    input_ids = torch.tensor([draft[:-1]], dtype=torch.long, device=runtime.device)
    with torch.no_grad():
        out = runtime.forward(input_ids, past_key_values=temp_kv)
    expected_v1 = int(out.logits[:, 0, :].float().argmax(dim=-1).item())
    assert span.verifier_tokens[1] == expected_v1


def test_mismatch_first_token(runtime: ModelRuntime, engine: VerificationEngine) -> None:
    full_state = _make_full_state_after_prefill(runtime, PROMPT)
    draft = [0, 0, 0]
    seq = engine.verify_sequential(full_state, draft)
    span = engine.verify_span(full_state, draft)
    assert span == seq
    assert span.accepted_tokens == []
    assert span.correction_token is not None
    assert span.num_rejected == len(draft)


def test_mismatch_middle(runtime: ModelRuntime, engine: VerificationEngine) -> None:
    full_state = _make_full_state_after_prefill(runtime, PROMPT)
    first = full_state.next_token_id
    draft = [first, 0, 0]
    seq = engine.verify_sequential(full_state, draft)
    span = engine.verify_span(full_state, draft)
    assert span == seq
    assert span.accepted_tokens == [first]
    assert span.correction_token is not None
    assert span.all_matched is False


def test_single_token_draft(runtime: ModelRuntime, engine: VerificationEngine) -> None:
    full_state = _make_full_state_after_prefill(runtime, PROMPT)
    draft = [full_state.next_token_id]
    seq = engine.verify_sequential(full_state, draft)
    span = engine.verify_span(full_state, draft)
    assert span == seq
    assert span.all_matched is True


def test_authoritative_state_unchanged_span(
    runtime: ModelRuntime, engine: VerificationEngine
) -> None:
    full_state = _make_full_state_after_prefill(runtime, PROMPT)
    draft = _full_greedy_draft(runtime, full_state, 4)
    kv_before = kv_seq_len(full_state.past_key_values)
    next_before = full_state.next_token_id
    engine.verify_span(full_state, draft)
    assert kv_seq_len(full_state.past_key_values) == kv_before
    assert full_state.next_token_id == next_before


@pytest.mark.parametrize("compressor_name", COMPRESSORS)
def test_span_generator_matches_full_greedy(
    runtime: ModelRuntime, compressor_name: str
) -> None:
    comp = get_compressor(compressor_name)
    full = generate_full_greedy(runtime, GENERATOR_PROMPT, MAX_NEW)
    span = ExactKVGenerator(
        runtime, comp, draft_len=DRAFT_LEN, verification_method="span"
    ).generate(GENERATOR_PROMPT, MAX_NEW)
    assert bool((full.generated_ids == span.output_ids).all())


@pytest.mark.parametrize("compressor_name", COMPRESSORS)
def test_span_generator_matches_sequential(
    runtime: ModelRuntime, compressor_name: str
) -> None:
    comp = get_compressor(compressor_name)
    seq = ExactKVGenerator(
        runtime, comp, draft_len=DRAFT_LEN, verification_method="sequential"
    ).generate(GENERATOR_PROMPT, MAX_NEW)
    span = ExactKVGenerator(
        runtime, comp, draft_len=DRAFT_LEN, verification_method="span"
    ).generate(GENERATOR_PROMPT, MAX_NEW)
    assert bool((seq.output_ids == span.output_ids).all())
    assert seq.total_accepted == span.total_accepted
    assert seq.total_rejected == span.total_rejected
    assert seq.total_corrections == span.total_corrections


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lc003_long_context_fp16_span_matches_sequential() -> None:
    """Regression: Exp 030 blocker (lc_003, k8_v4_sim, draft_len=8, fp16)."""
    from exactkv.benchmarks.v10_prompts import load_v10_suite

    runtime = ModelRuntime(model_name=MODEL_NAME, device="cuda", dtype="float16")
    comp = get_compressor("k8_v4_sim")
    prompt = load_v10_suite("long_context")[2]["prompt"]
    seq = ExactKVGenerator(
        runtime, comp, draft_len=8, verification_method="sequential"
    ).generate(prompt, 32)
    span = ExactKVGenerator(
        runtime, comp, draft_len=8, verification_method="span"
    ).generate(prompt, 32)
    full = generate_full_greedy(runtime, prompt, 32)
    assert bool((full.generated_ids == seq.output_ids).all())
    assert bool((seq.output_ids == span.output_ids).all())


def test_default_generator_still_sequential(runtime: ModelRuntime) -> None:
    comp = get_compressor("noop")
    default = ExactKVGenerator(runtime, comp, draft_len=DRAFT_LEN).generate(
        GENERATOR_PROMPT, MAX_NEW
    )
    explicit = ExactKVGenerator(
        runtime, comp, draft_len=DRAFT_LEN, verification_method="sequential"
    ).generate(GENERATOR_PROMPT, MAX_NEW)
    assert bool((default.output_ids == explicit.output_ids).all())
    assert ExactKVGenerator(runtime, comp).verification_method == "sequential"


def test_compute_acceptance_empty_unchanged() -> None:
    r = compute_acceptance([], [])
    assert r.all_matched is True
