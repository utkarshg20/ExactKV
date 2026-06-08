"""V4 Phase B — AsymmetricQuantSimCompressor core tests.

Gate: asymmetric compressor core gate + asymmetric ExactKV gate.

Covers:
  * Constructor accepts all valid k_bits/v_bits combinations
  * Invalid widths raise ValueError
  * Capabilities are correct for each combination
  * Per-side quantisation ranges (8→[-128,127], 4→[-8,7], 2→[-2,1])
  * Full-precision side preserves values bit-identically
  * K and V can have different quantisation ranges in the same cache
  * compress() does not mutate the authoritative FullKVState
  * materialize_for_draft() does not mutate the authoritative FullKVState
  * Materialized cache is forward-usable (model.forward does not crash)
  * update_after_commit() refreshes from latest authoritative full KV
  * stats() returns non-negative, finite values
  * supports_real_bytes_claim correct per combination
  * No forbidden performance fields in any returned dict
  * Direct ExactKV smoke: output_ids == full greedy, bookkeeping reconciles
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
_SMOKE_PROMPT = "The capital of France is"

_FORBIDDEN = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from exactkv.compressors.asymmetric_sim import AsymmetricQuantSimCompressor


# ---------------------------------------------------------------------------
# Module-scoped model fixtures (shared across all model-dependent tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def full_state(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state
    return prefill_to_full_state(runtime, _SMOKE_PROMPT)


# ---------------------------------------------------------------------------
# Helper to produce a small synthetic FullKVState-like object for pure-unit
# tests (no model load required).
# ---------------------------------------------------------------------------

def _make_synthetic_full_state():
    """Return a minimal FullKVState using a tiny tuple cache."""
    from exactkv.cache.full_state import FullKVState

    device = torch.device("cpu")
    dtype = torch.float32
    # 2 layers, shape (1, 2, 4, 8) — batch=1, heads=2, seq=4, head_dim=8
    shape = (1, 2, 4, 8)
    k0 = torch.randn(shape)
    v0 = torch.randn(shape)
    k1 = torch.randn(shape)
    v1 = torch.randn(shape)
    past_kv = ((k0, v0), (k1, v1))

    prompt_ids = torch.zeros(1, 4, dtype=torch.long)
    return FullKVState(
        past_key_values=past_kv,
        prompt_ids=prompt_ids,
        generated_ids=torch.zeros(1, 0, dtype=torch.long),
        full_sequence_ids=prompt_ids,
        device=device,
        dtype=dtype,
        metadata={"next_token_id": 0},
    )


# ===========================================================================
# 1.  Constructor — valid combinations
# ===========================================================================

class TestConstructorValidCombinations:
    @pytest.mark.parametrize("k, v", [
        (8, 4),
        (8, 2),
        (4, 8),
        (None, 4),
        (4, None),
        (8, None),
        (None, 8),
        (None, None),
        ("full", 4),
        (4, "full"),
        (8, "full"),
        ("full", 8),
        ("full", "full"),
    ])
    def test_constructor_succeeds(self, k, v):
        comp = AsymmetricQuantSimCompressor(k, v)
        assert comp is not None

    def test_name_generated_from_widths_when_none_given(self):
        comp = AsymmetricQuantSimCompressor(8, 4)
        assert "8" in comp.name
        assert "4" in comp.name

    def test_custom_name_accepted(self):
        comp = AsymmetricQuantSimCompressor(8, 4, name="k8_v4_sim")
        assert comp.name == "k8_v4_sim"

    def test_full_string_normalised_same_as_none(self):
        c1 = AsymmetricQuantSimCompressor("full", 4)
        c2 = AsymmetricQuantSimCompressor(None, 4)
        assert c1.capabilities.key_bit_width == c2.capabilities.key_bit_width
        assert c1.capabilities.value_bit_width == c2.capabilities.value_bit_width


# ===========================================================================
# 2.  Constructor — invalid widths
# ===========================================================================

class TestConstructorInvalidWidths:
    @pytest.mark.parametrize("bad", [1, 3, 5, 6, 7, 16, 32, "int8", "q4", True])
    def test_invalid_k_bits_raises(self, bad):
        with pytest.raises(ValueError, match="k_bits"):
            AsymmetricQuantSimCompressor(bad, 8)

    @pytest.mark.parametrize("bad", [1, 3, 5, 6, 7, 16, 32, "int8", "q4", True])
    def test_invalid_v_bits_raises(self, bad):
        with pytest.raises(ValueError, match="v_bits"):
            AsymmetricQuantSimCompressor(8, bad)


# ===========================================================================
# 3.  Capabilities per combination
# ===========================================================================

class TestCapabilitiesPerCombination:

    # is_simulated
    @pytest.mark.parametrize("k, v, expected", [
        (8, 4,    True),   # V simulated
        (8, 2,    True),   # V simulated
        (4, 8,    True),   # K simulated
        (None, 4, True),   # V simulated
        (4, None, True),   # K simulated
        (8, None, False),  # both real / full
        (None, 8, False),  # both real / full
        (None, None, False),
        (8, 8,    False),
    ])
    def test_is_simulated(self, k, v, expected):
        assert AsymmetricQuantSimCompressor(k, v).capabilities.is_simulated is expected

    # supports_real_bytes_claim
    @pytest.mark.parametrize("k, v, expected", [
        (8, 4,    False),
        (4, 8,    False),
        (8, None, True),
        (None, 8, True),
        (None, None, True),
        (8, 8,    True),
    ])
    def test_supports_real_bytes_claim(self, k, v, expected):
        caps = AsymmetricQuantSimCompressor(k, v).capabilities
        assert caps.supports_real_bytes_claim is expected

    # asymmetric
    @pytest.mark.parametrize("k, v, expected_asym", [
        (8, 4,    True),
        (4, 8,    True),
        (None, 4, True),
        (4, None, True),
        (8, None, True),
        (None, 8, True),
        (8, 8,    False),
        (None, None, False),
        (4, 4,    False),
    ])
    def test_asymmetric_flag(self, k, v, expected_asym):
        assert AsymmetricQuantSimCompressor(k, v).capabilities.asymmetric is expected_asym

    # key_bit_width / value_bit_width
    @pytest.mark.parametrize("k, v", [(8, 4), (4, 8), (None, 4), (8, None)])
    def test_width_fields_match_inputs(self, k, v):
        caps = AsymmetricQuantSimCompressor(k, v).capabilities
        # "full" string normalised to None
        assert caps.key_bit_width == (None if k in (None, "full") else k)
        assert caps.value_bit_width == (None if v in (None, "full") else v)

    def test_compressor_type_is_quantization(self):
        assert AsymmetricQuantSimCompressor(8, 4).capabilities.compressor_type == "quantization"

    def test_supports_quantization_is_true(self):
        assert AsymmetricQuantSimCompressor(8, 4).capabilities.supports_quantization is True

    def test_supports_token_dropping_is_false(self):
        assert AsymmetricQuantSimCompressor(8, 4).capabilities.supports_token_dropping is False

    def test_asdict_has_no_forbidden_fields(self):
        d = asdict(AsymmetricQuantSimCompressor(8, 4).capabilities)
        for key in d:
            assert key not in _FORBIDDEN


# ===========================================================================
# 4.  Quantisation ranges (pure tensor tests — no model needed)
# ===========================================================================

class TestQuantisationRanges:
    """Verify the quantised values stay within the correct clamped range."""

    def _make_large_tensor(self):
        """A tensor whose values span a wide range so clamping is exercised."""
        return torch.linspace(-100.0, 100.0, steps=200)

    def test_int8_range_minus128_to_127(self):
        from exactkv.compressors.asymmetric_sim import _quantize
        q, _ = _quantize(self._make_large_tensor(), 8)
        assert int(q.min().item()) >= -128
        assert int(q.max().item()) <= 127
        assert q.dtype == torch.int8

    def test_int4_sim_range_minus8_to_7(self):
        from exactkv.compressors.asymmetric_sim import _quantize
        q, _ = _quantize(self._make_large_tensor(), 4)
        assert int(q.min().item()) >= -8
        assert int(q.max().item()) <= 7
        assert q.dtype == torch.int8

    def test_int2_sim_range_minus2_to_1(self):
        from exactkv.compressors.asymmetric_sim import _quantize
        q, _ = _quantize(self._make_large_tensor(), 2)
        assert int(q.min().item()) >= -2
        assert int(q.max().item()) <= 1
        assert q.dtype == torch.int8

    def test_all_zero_tensor_safe_int8(self):
        from exactkv.compressors.asymmetric_sim import _quantize
        q, scale = _quantize(torch.zeros(10), 8)
        assert scale == 1.0
        assert (q == 0).all()

    def test_all_zero_tensor_safe_int4(self):
        from exactkv.compressors.asymmetric_sim import _quantize
        q, scale = _quantize(torch.zeros(10), 4)
        assert scale == 1.0
        assert (q == 0).all()

    def test_all_zero_tensor_safe_int2(self):
        from exactkv.compressors.asymmetric_sim import _quantize
        q, scale = _quantize(torch.zeros(10), 2)
        assert scale == 1.0
        assert (q == 0).all()


# ===========================================================================
# 5.  Full-precision passthrough (pure tensor tests — no model needed)
# ===========================================================================

class TestFullPrecisionPassthrough:
    def test_full_k_matches_original_values(self):
        state = _make_synthetic_full_state()
        comp = AsymmetricQuantSimCompressor(None, 4)
        compressed = comp.compress(state)
        mat = comp.materialize_for_draft(compressed)
        # Extract K tensors from original and materialized
        from exactkv.cache.utils import extract_kv_tensors
        orig_k, _, _ = extract_kv_tensors(state.past_key_values)
        mat_k, _, _ = extract_kv_tensors(mat)
        for ok, mk in zip(orig_k, mat_k):
            assert torch.allclose(ok, mk), "Full-precision K side should be bit-identical"

    def test_full_v_matches_original_values(self):
        state = _make_synthetic_full_state()
        comp = AsymmetricQuantSimCompressor(4, None)
        compressed = comp.compress(state)
        mat = comp.materialize_for_draft(compressed)
        from exactkv.cache.utils import extract_kv_tensors
        _, orig_v, _ = extract_kv_tensors(state.past_key_values)
        _, mat_v, _ = extract_kv_tensors(mat)
        for ov, mv in zip(orig_v, mat_v):
            assert torch.allclose(ov, mv), "Full-precision V side should be bit-identical"

    def test_full_full_both_match_original(self):
        state = _make_synthetic_full_state()
        comp = AsymmetricQuantSimCompressor(None, None)
        compressed = comp.compress(state)
        mat = comp.materialize_for_draft(compressed)
        from exactkv.cache.utils import extract_kv_tensors
        ok, ov, _ = extract_kv_tensors(state.past_key_values)
        mk, mv, _ = extract_kv_tensors(mat)
        for a, b in zip(ok, mk):
            assert torch.allclose(a, b)
        for a, b in zip(ov, mv):
            assert torch.allclose(a, b)


# ===========================================================================
# 6.  K and V use different ranges in the same cache
# ===========================================================================

class TestDifferentRangesPerSide:
    def test_k8_v4_different_quantised_values(self):
        """K quantised to INT8 range and V to INT4 range should differ."""
        state = _make_synthetic_full_state()
        comp = AsymmetricQuantSimCompressor(8, 4)
        compressed = comp.compress(state)
        # K data clamps to ≤ [-128, 127]; V data clamps to ≤ [-8, 7]
        for layer in compressed.data["layers"]:
            k_q = layer["k_data"]["q"]
            v_q = layer["v_data"]["q"]
            assert int(k_q.min().item()) >= -128
            assert int(k_q.max().item()) <= 127
            assert int(v_q.min().item()) >= -8
            assert int(v_q.max().item()) <= 7

    def test_k4_v8_reversed(self):
        state = _make_synthetic_full_state()
        comp = AsymmetricQuantSimCompressor(4, 8)
        compressed = comp.compress(state)
        for layer in compressed.data["layers"]:
            k_q = layer["k_data"]["q"]
            v_q = layer["v_data"]["q"]
            assert int(k_q.max().item()) <= 7    # INT4 K
            assert int(v_q.max().item()) <= 127  # INT8 V


# ===========================================================================
# 7.  No mutation of FullKVState
# ===========================================================================

class TestNoMutation:
    def test_compress_does_not_mutate_state(self, full_state):
        from exactkv.cache.utils import extract_kv_tensors
        comp = AsymmetricQuantSimCompressor(8, 4)
        orig_k, orig_v, _ = extract_kv_tensors(full_state.past_key_values)
        orig_k_copy = [t.clone() for t in orig_k]
        orig_v_copy = [t.clone() for t in orig_v]
        _ = comp.compress(full_state)
        new_k, new_v, _ = extract_kv_tensors(full_state.past_key_values)
        for ok, nk in zip(orig_k_copy, new_k):
            assert torch.allclose(ok, nk), "compress() must not mutate K tensors"
        for ov, nv in zip(orig_v_copy, new_v):
            assert torch.allclose(ov, nv), "compress() must not mutate V tensors"

    def test_materialize_does_not_mutate_compressed(self, full_state):
        comp = AsymmetricQuantSimCompressor(8, 4)
        compressed = comp.compress(full_state)
        # snapshot the compressed data
        k_q_snapshots = [
            layer["k_data"]["q"].clone()
            for layer in compressed.data["layers"]
        ]
        _ = comp.materialize_for_draft(compressed)
        for snap, layer in zip(k_q_snapshots, compressed.data["layers"]):
            assert torch.equal(snap, layer["k_data"]["q"]), (
                "materialize_for_draft() must not mutate compressed data"
            )


# ===========================================================================
# 8.  Materialized cache is forward-usable
# ===========================================================================

class TestMaterializeForwardUsable:
    @pytest.mark.parametrize("k, v", [(8, 4), (4, 8), (None, 4), (8, None)])
    def test_forward_does_not_crash(self, runtime, full_state, k, v):
        """Model.forward with the materialized cache must not raise."""
        comp = AsymmetricQuantSimCompressor(k, v)
        compressed = comp.compress(full_state)
        mat_cache = comp.materialize_for_draft(compressed)
        next_token = torch.tensor([[full_state.next_token_id]], dtype=torch.long,
                                   device=full_state.device)
        with torch.no_grad():
            out = runtime.model(input_ids=next_token, past_key_values=mat_cache,
                                use_cache=True)
        assert out.logits is not None


# ===========================================================================
# 9.  update_after_commit
# ===========================================================================

class TestUpdateAfterCommit:
    def test_refreshes_from_new_state(self, runtime, full_state):
        comp = AsymmetricQuantSimCompressor(8, 4)
        compressed = comp.compress(full_state)
        # Simulate a second prefill that produces a slightly different full state
        from exactkv.runtime.prefill import prefill_to_full_state
        new_state = prefill_to_full_state(runtime, "def hello():")
        updated = comp.update_after_commit(compressed, new_state)
        # The updated state should reflect the new sequence length
        assert updated.logical_seq_len == new_state.seq_len


# ===========================================================================
# 10.  CompressionStats
# ===========================================================================

class TestCompressionStats:
    @pytest.mark.parametrize("k, v", [
        (8, 4), (8, 2), (4, 8), (None, 4), (4, None),
        (8, None), (None, 8), (None, None), (8, 8),
    ])
    def test_stats_non_negative(self, full_state, k, v):
        comp = AsymmetricQuantSimCompressor(k, v)
        compressed = comp.compress(full_state)
        stats = comp.stats(compressed)
        assert stats.full_bytes > 0
        assert stats.compressed_bytes > 0
        assert stats.compression_ratio > 0.0
        assert stats.memory_reduction_factor > 0.0
        assert stats.num_layers > 0

    def test_full_full_ratio_close_to_one(self, full_state):
        """full/full passthrough: compressed_bytes == full_bytes (fp32 original)."""
        comp = AsymmetricQuantSimCompressor(None, None)
        compressed = comp.compress(full_state)
        stats = comp.stats(compressed)
        assert abs(stats.compression_ratio - 1.0) < 0.01

    def test_8_4_compressed_smaller_than_full(self, full_state):
        comp = AsymmetricQuantSimCompressor(8, 4)
        compressed = comp.compress(full_state)
        stats = comp.stats(compressed)
        assert stats.compressed_bytes < stats.full_bytes

    def test_stats_no_forbidden_fields(self, full_state):
        from dataclasses import asdict
        comp = AsymmetricQuantSimCompressor(8, 4)
        compressed = comp.compress(full_state)
        stats = comp.stats(compressed)
        d = asdict(stats)
        for key in d:
            assert key not in _FORBIDDEN


# ===========================================================================
# 11.  supports_real_bytes_claim consistency
# ===========================================================================

class TestRealBytesClaim:
    @pytest.mark.parametrize("k, v", [
        (8, 4), (8, 2), (4, 8), (None, 4), (4, None),
    ])
    def test_false_when_any_side_simulated(self, k, v):
        assert AsymmetricQuantSimCompressor(k, v).capabilities.supports_real_bytes_claim is False

    @pytest.mark.parametrize("k, v", [
        (8, None), (None, 8), (None, None), (8, 8),
    ])
    def test_true_when_no_simulated_side(self, k, v):
        assert AsymmetricQuantSimCompressor(k, v).capabilities.supports_real_bytes_claim is True


# ===========================================================================
# 12.  Direct ExactKV smoke — without registry registration
# ===========================================================================

class TestDirectExactKVSmoke:
    """
    Gate: asymmetric ExactKV gate.

    Instantiate directly (not via registry) and run ExactKVGenerator.
    Checks:
      * ExactKV output_ids == generate_full_greedy output_ids (zero failures)
      * Acceptance bookkeeping reconciles (drafted == accepted + rejected)
      * Cache alignment holds after every round
    """

    _PROMPTS = [
        "The capital of France is",
        "def fibonacci(n):",
    ]
    _DRAFT_LENS = [4, 8]
    _MAX_NEW = 12

    @pytest.mark.parametrize("k, v", [
        (8, 4),
        (4, 8),
        (None, 4),
        (8, None),
    ])
    @pytest.mark.parametrize("prompt", _PROMPTS)
    @pytest.mark.parametrize("draft_len", _DRAFT_LENS)
    def test_exactkv_output_matches_full_greedy(self, runtime, k, v, prompt, draft_len):
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy

        comp = AsymmetricQuantSimCompressor(k, v)
        full_res = generate_full_greedy(runtime, prompt, self._MAX_NEW)
        ekv_res = ExactKVGenerator(
            runtime, comp, draft_len=draft_len
        ).generate(prompt, self._MAX_NEW)

        full_ids = full_res.generated_ids.squeeze(0).tolist()
        ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
        assert full_ids == ekv_ids, (
            f"ExactKV mismatch for k={k}, v={v}, prompt={prompt!r}, "
            f"draft_len={draft_len}.\n"
            f"  full:   {full_ids}\n"
            f"  exactkv:{ekv_ids}"
        )

    @pytest.mark.parametrize("k, v", [(8, 4), (4, 8), (None, 4), (8, None)])
    def test_acceptance_bookkeeping_reconciles(self, runtime, k, v):
        from exactkv.metrics.acceptance import summarize_acceptance
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        comp = AsymmetricQuantSimCompressor(k, v)
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=4).generate(
            self._PROMPTS[0], self._MAX_NEW
        )
        acc = summarize_acceptance(ekv_res.traces)
        assert acc.total_drafted == acc.total_accepted + acc.total_rejected, (
            f"Bookkeeping mismatch: drafted={acc.total_drafted}, "
            f"accepted={acc.total_accepted}, rejected={acc.total_rejected}"
        )

    @pytest.mark.parametrize("k, v", [(8, 4), (4, 8)])
    def test_cache_alignment_holds(self, runtime, k, v):
        """After every round the full seq len must equal the compressed logical len."""
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        comp = AsymmetricQuantSimCompressor(k, v)
        # ExactKVGenerator asserts cache alignment internally; if it runs
        # to completion without AssertionError the invariant held throughout.
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=4).generate(
            self._PROMPTS[0], self._MAX_NEW
        )
        # Output must be non-empty
        assert ekv_res.output_ids.shape[1] > 0
