"""Tests for exactkv/metrics — Step 13.

Covers:
  * token_exact_match / text_exact_match
  * first_divergence_idx
  * summarize_acceptance (invariant, rate)
  * estimate_kv_memory (bytes positive, ratio positive)
  * Integration: works with traces from real ExactKV runs
"""
from __future__ import annotations

import pytest
import torch

from exactkv.compressors.debug_noise import DebugNoiseCompressor
from exactkv.compressors.int8 import Int8Compressor
from exactkv.compressors.noop import NoOpCompressor
from exactkv.metrics.acceptance import AcceptanceSummary, summarize_acceptance
from exactkv.metrics.exactness import (
    first_divergence_idx,
    text_exact_match,
    token_exact_match,
)
from exactkv.metrics.memory import MemorySummary, estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


# ---------------------------------------------------------------------------
# token_exact_match
# ---------------------------------------------------------------------------

def test_token_exact_match_true() -> None:
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[1, 2, 3]])
    assert token_exact_match(a, b) is True


def test_token_exact_match_false_content() -> None:
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[1, 2, 4]])
    assert token_exact_match(a, b) is False


def test_token_exact_match_false_length() -> None:
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[1, 2]])
    assert token_exact_match(a, b) is False


def test_token_exact_match_empty() -> None:
    a = torch.tensor([[]])
    b = torch.tensor([[]])
    assert token_exact_match(a, b) is True


# ---------------------------------------------------------------------------
# text_exact_match
# ---------------------------------------------------------------------------

def test_text_exact_match_true() -> None:
    assert text_exact_match("hello world", "hello world") is True


def test_text_exact_match_false() -> None:
    assert text_exact_match("hello world", "hello") is False


def test_text_exact_match_empty() -> None:
    assert text_exact_match("", "") is True


# ---------------------------------------------------------------------------
# first_divergence_idx
# ---------------------------------------------------------------------------

def test_first_divergence_identical_returns_none() -> None:
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[1, 2, 3]])
    assert first_divergence_idx(a, b) is None


def test_first_divergence_first_position() -> None:
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[9, 2, 3]])
    assert first_divergence_idx(a, b) == 0


def test_first_divergence_middle() -> None:
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[1, 9, 3]])
    assert first_divergence_idx(a, b) == 1


def test_first_divergence_different_lengths() -> None:
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[1, 2]])
    # All positions of shorter sequence match; divergence at position 2
    assert first_divergence_idx(a, b) == 2


def test_first_divergence_returns_int_or_none() -> None:
    a = torch.tensor([[1, 2]])
    b = torch.tensor([[1, 9]])
    result = first_divergence_idx(a, b)
    assert result is None or isinstance(result, int)


# ---------------------------------------------------------------------------
# summarize_acceptance
# ---------------------------------------------------------------------------

def test_summarize_acceptance_empty_trace() -> None:
    summary = summarize_acceptance([])
    assert summary.total_drafted == 0
    assert summary.total_accepted == 0
    assert summary.total_rejected == 0
    assert summary.acceptance_rate == 1.0


def test_summarize_acceptance_counts_reconcile(runtime: ModelRuntime) -> None:
    """drafted == accepted + rejected for every compressor."""
    for compressor in [NoOpCompressor(), Int8Compressor()]:
        result = ExactKVGenerator(runtime, compressor, draft_len=4).generate(
            PROMPT, max_new_tokens=16
        )
        summary = summarize_acceptance(result.traces)
        assert summary.total_drafted == summary.total_accepted + summary.total_rejected, (
            f"Bookkeeping broken for {compressor.name}: "
            f"drafted={summary.total_drafted}, "
            f"accepted={summary.total_accepted}, rejected={summary.total_rejected}"
        )


def test_summarize_acceptance_rate_correct(runtime: ModelRuntime) -> None:
    result = ExactKVGenerator(runtime, NoOpCompressor(), draft_len=4).generate(
        PROMPT, max_new_tokens=16
    )
    summary = summarize_acceptance(result.traces)
    expected_rate = (
        summary.total_accepted / max(summary.total_drafted, 1)
    )
    assert abs(summary.acceptance_rate - expected_rate) < 1e-9


def test_summarize_acceptance_noop_rate_is_one(runtime: ModelRuntime) -> None:
    result = ExactKVGenerator(runtime, NoOpCompressor(), draft_len=4).generate(
        PROMPT, max_new_tokens=16
    )
    summary = summarize_acceptance(result.traces)
    assert summary.acceptance_rate == 1.0


def test_summarize_acceptance_debug_noise_has_rejections(runtime: ModelRuntime) -> None:
    result = ExactKVGenerator(runtime, DebugNoiseCompressor(), draft_len=4).generate(
        PROMPT, max_new_tokens=16
    )
    summary = summarize_acceptance(result.traces)
    assert summary.total_rejected > 0, "Expected rejections with DebugNoise"
    assert summary.acceptance_rate < 1.0


def test_acceptance_summary_to_dict(runtime: ModelRuntime) -> None:
    result = ExactKVGenerator(runtime, NoOpCompressor(), draft_len=4).generate(
        PROMPT, max_new_tokens=8
    )
    summary = summarize_acceptance(result.traces)
    d = summary.to_dict()
    assert isinstance(d, dict)
    assert "total_drafted" in d
    assert "acceptance_rate" in d


# ---------------------------------------------------------------------------
# estimate_kv_memory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compressor_cls", [NoOpCompressor, Int8Compressor])
def test_memory_full_bytes_positive(runtime: ModelRuntime, compressor_cls) -> None:
    summary = estimate_kv_memory(runtime, PROMPT, compressor_cls())
    assert summary.full_bytes > 0, f"full_bytes={summary.full_bytes}"


@pytest.mark.parametrize("compressor_cls", [NoOpCompressor, Int8Compressor])
def test_memory_compressed_bytes_positive(runtime: ModelRuntime, compressor_cls) -> None:
    summary = estimate_kv_memory(runtime, PROMPT, compressor_cls())
    assert summary.compressed_bytes > 0, f"compressed_bytes={summary.compressed_bytes}"


@pytest.mark.parametrize("compressor_cls", [NoOpCompressor, Int8Compressor])
def test_memory_compression_ratio_positive(runtime: ModelRuntime, compressor_cls) -> None:
    summary = estimate_kv_memory(runtime, PROMPT, compressor_cls())
    assert summary.compression_ratio > 0.0


def test_int8_compression_ratio_less_than_one(runtime: ModelRuntime) -> None:
    """INT8 compression_ratio (compressed/full) must be < 1.0."""
    summary = estimate_kv_memory(runtime, PROMPT, Int8Compressor())
    assert 0.0 < summary.compression_ratio < 1.0, (
        f"INT8 compression_ratio={summary.compression_ratio} should be in (0, 1)"
    )


def test_int8_memory_reduction_factor_greater_than_one(runtime: ModelRuntime) -> None:
    """INT8 memory_reduction_factor (full/compressed) must exceed 1.0."""
    summary = estimate_kv_memory(runtime, PROMPT, Int8Compressor())
    assert summary.memory_reduction_factor > 1.0, (
        f"INT8 memory_reduction_factor={summary.memory_reduction_factor} should exceed 1.0"
    )


def test_memory_summary_to_dict(runtime: ModelRuntime) -> None:
    summary = estimate_kv_memory(runtime, PROMPT, Int8Compressor())
    d = summary.to_dict()
    assert isinstance(d, dict)
    for key in ("full_bytes", "compressed_bytes", "compression_ratio", "memory_reduction_factor"):
        assert key in d, f"Missing key {key!r} in MemorySummary.to_dict()"


def test_noop_compression_ratio_is_one(runtime: ModelRuntime) -> None:
    """NoOp: compression_ratio and memory_reduction_factor must both be 1.0."""
    summary = estimate_kv_memory(runtime, PROMPT, NoOpCompressor())
    assert abs(summary.compression_ratio - 1.0) < 1e-6
    assert abs(summary.memory_reduction_factor - 1.0) < 1e-6
