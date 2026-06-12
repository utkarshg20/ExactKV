"""GPU span parity tests (Exp 030b)."""
from __future__ import annotations

import pytest
import torch

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.engine import VerificationEngine

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lc003_fp16_batched_span_matches_sequential() -> None:
    """Exp 030b blocker: batched span verify must match sequential on fp16 GPU."""
    runtime = ModelRuntime(model_name=MODEL, device="cuda", dtype="float16")
    comp = get_compressor("k8_v4_sim")
    prompt = load_v10_suite("long_context")[2]["prompt"]
    engine = VerificationEngine(runtime)

    seq_gen = ExactKVGenerator(
        runtime, comp, draft_len=8, verification_method="sequential"
    )
    result = seq_gen.generate(prompt, 32)
    draft = result.traces[2].draft_tokens

    from exactkv.analysis.span_parity_debug import state_and_draft_at_round

    full_state, draft_at_round = state_and_draft_at_round(
        runtime, comp, prompt, draft_len=8, round_idx=2
    )
    assert draft_at_round == draft

    batched = engine._verify_span_batched(full_state, draft_at_round)
    sequential = engine.verify_sequential(full_state, draft_at_round)
    assert batched == sequential


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lc003_fp16_span_generator_matches_full_greedy() -> None:
    runtime = ModelRuntime(model_name=MODEL, device="cuda", dtype="float16")
    comp = get_compressor("k8_v4_sim")
    prompt = load_v10_suite("long_context")[2]["prompt"]
    full = generate_full_greedy(runtime, prompt, 32)
    span = ExactKVGenerator(
        runtime, comp, draft_len=8, verification_method="span"
    ).generate(prompt, 32)
    assert bool((full.generated_ids == span.output_ids).all())
