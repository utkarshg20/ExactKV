"""V5 Phase A: workspace-aware memory summary tests.

Gate: workspace-memory stats gate.

Verifies:
  * MemorySummary carries all new V5 fields.
  * CompressionStats carries all new V5 fields.
  * Every registered compressor returns non-negative workspace fields.
  * total_kv_footprint_bytes reconciles as stored + materialized + metadata + temporary.
  * compressed_bytes remains backward compatible (== stored + metadata).
  * NoOp: stored == full; materialized == full; metadata == 0; scratch == 0.
  * Int8: stored < full; materialized == full; metadata > 0; real bytes claimed.
  * Int4Sim: stored reflects int8 container reality (NOT /4); sim warning present.
  * Asymmetric k_full_v8: full K side + int8 V side; real bytes claimed.
  * Asymmetric k8_v_full: int8 K side + full V side; real bytes claimed.
  * Asymmetric k8_v4_sim: sub-INT8 sim flag; supports_real_bytes_claim=False.
  * All materialized_working_kv_bytes == full_bytes for current compressors.
  * No forbidden performance data fields in any output.
"""
from __future__ import annotations

import dataclasses
import os

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import exactkv.compressors  # noqa: F401 — registers built-ins

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"

# ---------------------------------------------------------------------------
# Forbidden performance field names (data keys / dict entries — not prose)
# ---------------------------------------------------------------------------
_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})


def _assert_no_forbidden_fields(d: dict, context: str = "") -> None:
    hits = _FORBIDDEN_FIELDS & d.keys()
    assert not hits, f"Forbidden performance fields {hits} found in {context or 'dict'}"


# ---------------------------------------------------------------------------
# Synthetic helper: build a minimal fake past_key_values tuple for unit tests
# (avoids loading a full model for pure unit tests)
# ---------------------------------------------------------------------------

def _make_fake_full_state(
    num_layers: int = 2,
    seq_len: int = 8,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> "FullKVState":
    """Build a FullKVState with random fp32 tensors (no model required)."""
    from exactkv.cache.full_state import FullKVState

    pkv = tuple(
        (
            torch.randn(1, 1, seq_len, head_dim, dtype=dtype),
            torch.randn(1, 1, seq_len, head_dim, dtype=dtype),
        )
        for _ in range(num_layers)
    )
    prompt_ids = torch.zeros(1, seq_len, dtype=torch.long)
    generated_ids = torch.zeros(1, 0, dtype=torch.long)
    full_sequence_ids = torch.zeros(1, seq_len, dtype=torch.long)
    return FullKVState(
        past_key_values=pkv,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        full_sequence_ids=full_sequence_ids,
        device=torch.device("cpu"),
        dtype=dtype,
        metadata={"next_token_id": 42},
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures (model-backed, for integration gates)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def full_state_model(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state
    return prefill_to_full_state(runtime, PROMPT)


# ---------------------------------------------------------------------------
# 1. Dataclass field presence
# ---------------------------------------------------------------------------

class TestMemorySummaryFields:
    """MemorySummary carries all V5 fields with appropriate defaults."""

    def test_has_stored_kv_bytes(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "stored_kv_bytes" in field_names

    def test_has_materialized_working_kv_bytes(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "materialized_working_kv_bytes" in field_names

    def test_has_metadata_bytes(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "metadata_bytes" in field_names

    def test_has_temporary_workspace_bytes(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "temporary_workspace_bytes" in field_names

    def test_has_total_kv_footprint_bytes(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "total_kv_footprint_bytes" in field_names

    def test_has_supports_real_bytes_claim(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "supports_real_bytes_claim" in field_names

    def test_has_is_simulated(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "is_simulated" in field_names

    def test_has_memory_claim_note(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        assert "memory_claim_note" in field_names

    def test_existing_fields_present(self):
        from exactkv.metrics.memory import MemorySummary
        field_names = {f.name for f in dataclasses.fields(MemorySummary)}
        for name in ("full_bytes", "compressed_bytes", "compression_ratio",
                     "memory_reduction_factor"):
            assert name in field_names, f"backward-compat field missing: {name}"

    def test_v1_construction_still_works(self):
        """Constructing with only original 4 fields must not raise."""
        from exactkv.metrics.memory import MemorySummary
        ms = MemorySummary(
            full_bytes=1000,
            compressed_bytes=250,
            compression_ratio=0.25,
            memory_reduction_factor=4.0,
        )
        assert ms.stored_kv_bytes == 0
        assert ms.materialized_working_kv_bytes == 0
        assert ms.metadata_bytes == 0
        assert ms.temporary_workspace_bytes == 0
        assert ms.total_kv_footprint_bytes == 0

    def test_to_dict_contains_new_fields(self):
        from exactkv.metrics.memory import MemorySummary
        ms = MemorySummary(
            full_bytes=1000,
            compressed_bytes=250,
            compression_ratio=0.25,
            memory_reduction_factor=4.0,
            stored_kv_bytes=242,
            materialized_working_kv_bytes=1000,
            metadata_bytes=8,
            temporary_workspace_bytes=0,
            total_kv_footprint_bytes=1250,
        )
        d = ms.to_dict()
        for key in ("stored_kv_bytes", "materialized_working_kv_bytes",
                    "metadata_bytes", "temporary_workspace_bytes",
                    "total_kv_footprint_bytes"):
            assert key in d, f"to_dict missing: {key}"
        _assert_no_forbidden_fields(d, "MemorySummary.to_dict()")


class TestCompressionStatsFields:
    """CompressionStats carries all V5 workspace fields with zero defaults."""

    def test_has_workspace_fields(self):
        from exactkv.compressors.base import CompressionStats
        field_names = {f.name for f in dataclasses.fields(CompressionStats)}
        for name in ("stored_kv_bytes", "materialized_working_kv_bytes",
                     "metadata_bytes", "temporary_workspace_bytes",
                     "total_kv_footprint_bytes"):
            assert name in field_names, f"CompressionStats missing: {name}"

    def test_existing_fields_present(self):
        from exactkv.compressors.base import CompressionStats
        field_names = {f.name for f in dataclasses.fields(CompressionStats)}
        for name in ("compressor_name", "full_bytes", "compressed_bytes",
                     "compression_ratio", "memory_reduction_factor",
                     "seq_len", "num_layers"):
            assert name in field_names

    def test_v1_construction_still_works(self):
        from exactkv.compressors.base import CompressionStats
        cs = CompressionStats(
            compressor_name="test",
            full_bytes=1000,
            compressed_bytes=250,
            compression_ratio=0.25,
            memory_reduction_factor=4.0,
            seq_len=8,
            num_layers=2,
        )
        assert cs.stored_kv_bytes == 0
        assert cs.materialized_working_kv_bytes == 0
        assert cs.metadata_bytes == 0
        assert cs.temporary_workspace_bytes == 0
        assert cs.total_kv_footprint_bytes == 0


# ---------------------------------------------------------------------------
# 2. NoOp unit tests (synthetic state)
# ---------------------------------------------------------------------------

class TestNoOpWorkspaceStats:
    @pytest.fixture
    def noop_stats(self):
        from exactkv.compressors.noop import NoOpCompressor
        comp = NoOpCompressor()
        state = _make_fake_full_state()
        compressed = comp.compress(state)
        return comp.stats(compressed)

    def test_stored_equals_full(self, noop_stats):
        assert noop_stats.stored_kv_bytes == noop_stats.full_bytes

    def test_materialized_equals_full(self, noop_stats):
        assert noop_stats.materialized_working_kv_bytes == noop_stats.full_bytes

    def test_metadata_is_zero(self, noop_stats):
        assert noop_stats.metadata_bytes == 0

    def test_scratch_is_zero(self, noop_stats):
        assert noop_stats.temporary_workspace_bytes == 0

    def test_total_reconciles(self, noop_stats):
        expected = (
            noop_stats.stored_kv_bytes
            + noop_stats.materialized_working_kv_bytes
            + noop_stats.metadata_bytes
            + noop_stats.temporary_workspace_bytes
        )
        assert noop_stats.total_kv_footprint_bytes == expected

    def test_backward_compat_compressed_bytes(self, noop_stats):
        # For NoOp: stored == full, metadata == 0, so compressed == full.
        assert noop_stats.compressed_bytes == noop_stats.full_bytes

    def test_all_non_negative(self, noop_stats):
        for f in ("stored_kv_bytes", "materialized_working_kv_bytes",
                  "metadata_bytes", "temporary_workspace_bytes",
                  "total_kv_footprint_bytes"):
            val = getattr(noop_stats, f)
            assert val >= 0, f"noop {f} is negative: {val}"

    def test_no_forbidden_fields(self, noop_stats):
        _assert_no_forbidden_fields(dataclasses.asdict(noop_stats), "noop stats")


# ---------------------------------------------------------------------------
# 3. Int8 unit tests (synthetic state)
# ---------------------------------------------------------------------------

class TestInt8WorkspaceStats:
    @pytest.fixture
    def int8_stats(self):
        from exactkv.compressors.int8 import Int8Compressor
        comp = Int8Compressor()
        state = _make_fake_full_state()
        compressed = comp.compress(state)
        return comp.stats(compressed)

    def test_stored_less_than_full(self, int8_stats):
        assert int8_stats.stored_kv_bytes < int8_stats.full_bytes, (
            "INT8 stored_kv_bytes should be less than full_bytes"
        )

    def test_materialized_equals_full(self, int8_stats):
        assert int8_stats.materialized_working_kv_bytes == int8_stats.full_bytes

    def test_metadata_positive(self, int8_stats):
        assert int8_stats.metadata_bytes > 0

    def test_scratch_is_zero(self, int8_stats):
        assert int8_stats.temporary_workspace_bytes == 0

    def test_total_reconciles(self, int8_stats):
        expected = (
            int8_stats.stored_kv_bytes
            + int8_stats.materialized_working_kv_bytes
            + int8_stats.metadata_bytes
            + int8_stats.temporary_workspace_bytes
        )
        assert int8_stats.total_kv_footprint_bytes == expected

    def test_backward_compat_compressed_bytes(self, int8_stats):
        # compressed_bytes == stored_kv_bytes + metadata_bytes.
        assert int8_stats.compressed_bytes == (
            int8_stats.stored_kv_bytes + int8_stats.metadata_bytes
        )

    def test_total_ge_stored(self, int8_stats):
        assert int8_stats.total_kv_footprint_bytes >= int8_stats.stored_kv_bytes

    def test_total_ge_materialized(self, int8_stats):
        assert int8_stats.total_kv_footprint_bytes >= int8_stats.materialized_working_kv_bytes

    def test_all_non_negative(self, int8_stats):
        for f in ("stored_kv_bytes", "materialized_working_kv_bytes",
                  "metadata_bytes", "temporary_workspace_bytes",
                  "total_kv_footprint_bytes"):
            val = getattr(int8_stats, f)
            assert val >= 0, f"int8 {f} is negative: {val}"


# ---------------------------------------------------------------------------
# 4. Int4Sim unit tests (synthetic state)
# ---------------------------------------------------------------------------

class TestInt4SimWorkspaceStats:
    @pytest.fixture
    def int4_stats(self):
        from exactkv.compressors.int4_sim import Int4SimCompressor
        comp = Int4SimCompressor()
        state = _make_fake_full_state()
        compressed = comp.compress(state)
        return comp.stats(compressed)

    @pytest.fixture
    def int4_comp(self):
        from exactkv.compressors.int4_sim import Int4SimCompressor
        return Int4SimCompressor()

    def test_stored_is_int8_container_not_packed_4bit(self, int4_stats):
        # Int8 and Int4Sim have the same element storage (1 byte each, int8 container).
        # So stored_kv_bytes for int4_sim should equal the int8 case for same tensors.
        # More concretely: stored_kv_bytes should NOT be full_bytes / 4 (packed 4-bit).
        # It should equal (full_bytes / 4) * 1 = full_bytes/4 for tensor bytes...
        # Actually: fp32 is 4 bytes/element; int8 is 1 byte/element → stored = full/4.
        # Theoretical packed 4-bit would be full/8. Check we're NOT reporting full/8.
        theoretical_packed = int4_stats.full_bytes // 8
        assert int4_stats.stored_kv_bytes != theoretical_packed or theoretical_packed == 0, (
            "stored_kv_bytes must reflect int8 container bytes, not packed INT4"
        )
        # More direct: stored should be approximately full/4 (int8 vs fp32).
        expected_int8_stored = int4_stats.full_bytes // 4
        assert abs(int4_stats.stored_kv_bytes - expected_int8_stored) <= 4, (
            f"int4_sim stored_kv_bytes ({int4_stats.stored_kv_bytes}) should equal "
            f"int8-container bytes (~{expected_int8_stored}), not packed 4-bit ({theoretical_packed})"
        )

    def test_materialized_equals_full(self, int4_stats):
        assert int4_stats.materialized_working_kv_bytes == int4_stats.full_bytes

    def test_metadata_positive(self, int4_stats):
        assert int4_stats.metadata_bytes > 0

    def test_total_reconciles(self, int4_stats):
        expected = (
            int4_stats.stored_kv_bytes
            + int4_stats.materialized_working_kv_bytes
            + int4_stats.metadata_bytes
            + int4_stats.temporary_workspace_bytes
        )
        assert int4_stats.total_kv_footprint_bytes == expected

    def test_backward_compat_compressed_bytes(self, int4_stats):
        assert int4_stats.compressed_bytes == (
            int4_stats.stored_kv_bytes + int4_stats.metadata_bytes
        )

    def test_capabilities_still_not_real_bytes(self, int4_comp):
        assert int4_comp.capabilities.supports_real_bytes_claim is False

    def test_capabilities_note_mentions_simulation(self, int4_comp):
        note = int4_comp.capabilities.notes.lower()
        assert "simulat" in note or "int8" in note, (
            "capabilities.notes should mention simulation or int8 container"
        )

    def test_all_non_negative(self, int4_stats):
        for f in ("stored_kv_bytes", "materialized_working_kv_bytes",
                  "metadata_bytes", "temporary_workspace_bytes",
                  "total_kv_footprint_bytes"):
            val = getattr(int4_stats, f)
            assert val >= 0, f"int4_sim {f} is negative: {val}"


# ---------------------------------------------------------------------------
# 5. DebugNoise unit tests (synthetic state)
# ---------------------------------------------------------------------------

class TestDebugNoiseWorkspaceStats:
    @pytest.fixture
    def noise_stats(self):
        from exactkv.compressors.debug_noise import DebugNoiseCompressor
        comp = DebugNoiseCompressor()
        state = _make_fake_full_state()
        compressed = comp.compress(state)
        return comp.stats(compressed)

    def test_stored_non_negative(self, noise_stats):
        assert noise_stats.stored_kv_bytes >= 0

    def test_materialized_non_negative(self, noise_stats):
        assert noise_stats.materialized_working_kv_bytes >= 0

    def test_metadata_zero(self, noise_stats):
        assert noise_stats.metadata_bytes == 0

    def test_scratch_zero(self, noise_stats):
        assert noise_stats.temporary_workspace_bytes == 0

    def test_total_reconciles(self, noise_stats):
        expected = (
            noise_stats.stored_kv_bytes
            + noise_stats.materialized_working_kv_bytes
            + noise_stats.metadata_bytes
            + noise_stats.temporary_workspace_bytes
        )
        assert noise_stats.total_kv_footprint_bytes == expected

    def test_all_non_negative(self, noise_stats):
        for f in ("stored_kv_bytes", "materialized_working_kv_bytes",
                  "metadata_bytes", "temporary_workspace_bytes",
                  "total_kv_footprint_bytes"):
            val = getattr(noise_stats, f)
            assert val >= 0, f"debug_noise {f} is negative: {val}"


# ---------------------------------------------------------------------------
# 6. Asymmetric compressor unit tests (synthetic state)
# ---------------------------------------------------------------------------

class TestAsymmetricWorkspaceStats:

    def _get_stats(self, name: str):
        from exactkv.compressors import get_compressor
        comp = get_compressor(name)
        state = _make_fake_full_state()
        compressed = comp.compress(state)
        return comp, comp.stats(compressed)

    # k_full_v8: full K side + INT8 V side — real bytes claimed
    def test_k_full_v8_supports_real_bytes_claim(self):
        comp, _ = self._get_stats("k_full_v8")
        assert comp.capabilities.supports_real_bytes_claim is True

    def test_k_full_v8_materialized_equals_full(self):
        _, s = self._get_stats("k_full_v8")
        assert s.materialized_working_kv_bytes == s.full_bytes

    def test_k_full_v8_total_reconciles(self):
        _, s = self._get_stats("k_full_v8")
        expected = s.stored_kv_bytes + s.materialized_working_kv_bytes + s.metadata_bytes + s.temporary_workspace_bytes
        assert s.total_kv_footprint_bytes == expected

    # k8_v_full: INT8 K side + full V side — real bytes claimed
    def test_k8_v_full_supports_real_bytes_claim(self):
        comp, _ = self._get_stats("k8_v_full")
        assert comp.capabilities.supports_real_bytes_claim is True

    def test_k8_v_full_materialized_equals_full(self):
        _, s = self._get_stats("k8_v_full")
        assert s.materialized_working_kv_bytes == s.full_bytes

    def test_k8_v_full_total_reconciles(self):
        _, s = self._get_stats("k8_v_full")
        expected = s.stored_kv_bytes + s.materialized_working_kv_bytes + s.metadata_bytes + s.temporary_workspace_bytes
        assert s.total_kv_footprint_bytes == expected

    # k8_v4_sim: sub-INT8 sim flag; supports_real_bytes_claim=False
    def test_k8_v4_sim_not_real_bytes(self):
        comp, _ = self._get_stats("k8_v4_sim")
        assert comp.capabilities.supports_real_bytes_claim is False

    def test_k8_v4_sim_is_simulated(self):
        comp, _ = self._get_stats("k8_v4_sim")
        assert comp.capabilities.is_simulated is True

    def test_k8_v4_sim_materialized_equals_full(self):
        _, s = self._get_stats("k8_v4_sim")
        assert s.materialized_working_kv_bytes == s.full_bytes

    def test_k8_v4_sim_total_reconciles(self):
        _, s = self._get_stats("k8_v4_sim")
        expected = s.stored_kv_bytes + s.materialized_working_kv_bytes + s.metadata_bytes + s.temporary_workspace_bytes
        assert s.total_kv_footprint_bytes == expected

    def test_k8_v4_sim_all_non_negative(self):
        _, s = self._get_stats("k8_v4_sim")
        for f in ("stored_kv_bytes", "materialized_working_kv_bytes",
                  "metadata_bytes", "temporary_workspace_bytes",
                  "total_kv_footprint_bytes"):
            assert getattr(s, f) >= 0

    # k_full_v4_sim: V side simulated
    def test_k_full_v4_sim_not_real_bytes(self):
        comp, _ = self._get_stats("k_full_v4_sim")
        assert comp.capabilities.supports_real_bytes_claim is False

    def test_k_full_v4_sim_total_reconciles(self):
        _, s = self._get_stats("k_full_v4_sim")
        expected = s.stored_kv_bytes + s.materialized_working_kv_bytes + s.metadata_bytes + s.temporary_workspace_bytes
        assert s.total_kv_footprint_bytes == expected

    # k4_v_full_sim: K side simulated
    def test_k4_v_full_sim_not_real_bytes(self):
        comp, _ = self._get_stats("k4_v_full_sim")
        assert comp.capabilities.supports_real_bytes_claim is False

    def test_k4_v_full_sim_total_reconciles(self):
        _, s = self._get_stats("k4_v_full_sim")
        expected = s.stored_kv_bytes + s.materialized_working_kv_bytes + s.metadata_bytes + s.temporary_workspace_bytes
        assert s.total_kv_footprint_bytes == expected


# ---------------------------------------------------------------------------
# 7. All registered compressors: bulk non-negative and formula gate
# ---------------------------------------------------------------------------

class TestAllRegisteredCompressors:

    @pytest.fixture(params=[
        "noop", "int8", "int4_sim", "debug_noise",
        "k8_v4_sim", "k8_v2_sim", "k4_v8_sim",
        "k_full_v4_sim", "k4_v_full_sim",
        "k8_v_full", "k_full_v8",
    ])
    def compressor_stats(self, request):
        from exactkv.compressors import get_compressor
        comp = get_compressor(request.param)
        state = _make_fake_full_state()
        compressed = comp.compress(state)
        return comp, comp.stats(compressed)

    def test_all_workspace_fields_non_negative(self, compressor_stats):
        _, s = compressor_stats
        for f in ("stored_kv_bytes", "materialized_working_kv_bytes",
                  "metadata_bytes", "temporary_workspace_bytes",
                  "total_kv_footprint_bytes"):
            val = getattr(s, f)
            assert val >= 0, f"{s.compressor_name}: {f} = {val} (negative)"

    def test_total_reconciles(self, compressor_stats):
        _, s = compressor_stats
        expected = (
            s.stored_kv_bytes
            + s.materialized_working_kv_bytes
            + s.metadata_bytes
            + s.temporary_workspace_bytes
        )
        assert s.total_kv_footprint_bytes == expected, (
            f"{s.compressor_name}: total_kv_footprint_bytes {s.total_kv_footprint_bytes} "
            f"!= stored+working+meta+scratch {expected}"
        )

    def test_total_ge_stored(self, compressor_stats):
        _, s = compressor_stats
        assert s.total_kv_footprint_bytes >= s.stored_kv_bytes

    def test_total_ge_materialized(self, compressor_stats):
        _, s = compressor_stats
        assert s.total_kv_footprint_bytes >= s.materialized_working_kv_bytes

    def test_materialized_equals_full_for_all_current(self, compressor_stats):
        """All current compressors dequantise to full precision for attention."""
        _, s = compressor_stats
        assert s.materialized_working_kv_bytes == s.full_bytes, (
            f"{s.compressor_name}: materialized_working_kv_bytes "
            f"({s.materialized_working_kv_bytes}) != full_bytes ({s.full_bytes})"
        )

    def test_compressed_bytes_backward_compat(self, compressor_stats):
        """compressed_bytes must equal stored_kv_bytes + metadata_bytes."""
        _, s = compressor_stats
        assert s.compressed_bytes == s.stored_kv_bytes + s.metadata_bytes, (
            f"{s.compressor_name}: compressed_bytes ({s.compressed_bytes}) != "
            f"stored ({s.stored_kv_bytes}) + metadata ({s.metadata_bytes})"
        )

    def test_no_forbidden_fields_in_stats_dict(self, compressor_stats):
        _, s = compressor_stats
        _assert_no_forbidden_fields(dataclasses.asdict(s), f"{s.compressor_name} stats dict")


# ---------------------------------------------------------------------------
# 8. Integration gate: MemorySummary from estimate_kv_memory (model-backed)
# ---------------------------------------------------------------------------

class TestEstimateKVMemoryIntegration:
    """Model-backed integration: estimate_kv_memory returns complete MemorySummary."""

    @pytest.fixture(scope="class")
    def noop_summary(self, runtime):
        from exactkv.metrics.memory import estimate_kv_memory
        from exactkv.compressors.noop import NoOpCompressor
        return estimate_kv_memory(runtime, PROMPT, NoOpCompressor())

    @pytest.fixture(scope="class")
    def int8_summary(self, runtime):
        from exactkv.metrics.memory import estimate_kv_memory
        from exactkv.compressors.int8 import Int8Compressor
        return estimate_kv_memory(runtime, PROMPT, Int8Compressor())

    @pytest.fixture(scope="class")
    def int4_summary(self, runtime):
        from exactkv.metrics.memory import estimate_kv_memory
        from exactkv.compressors.int4_sim import Int4SimCompressor
        return estimate_kv_memory(runtime, PROMPT, Int4SimCompressor())

    def test_noop_stored_eq_full(self, noop_summary):
        assert noop_summary.stored_kv_bytes == noop_summary.full_bytes

    def test_noop_materialized_eq_full(self, noop_summary):
        assert noop_summary.materialized_working_kv_bytes == noop_summary.full_bytes

    def test_noop_metadata_zero(self, noop_summary):
        assert noop_summary.metadata_bytes == 0

    def test_noop_scratch_zero(self, noop_summary):
        assert noop_summary.temporary_workspace_bytes == 0

    def test_noop_total_reconciles(self, noop_summary):
        expected = (
            noop_summary.stored_kv_bytes
            + noop_summary.materialized_working_kv_bytes
            + noop_summary.metadata_bytes
            + noop_summary.temporary_workspace_bytes
        )
        assert noop_summary.total_kv_footprint_bytes == expected

    def test_int8_stored_lt_full(self, int8_summary):
        assert int8_summary.stored_kv_bytes < int8_summary.full_bytes

    def test_int8_materialized_eq_full(self, int8_summary):
        assert int8_summary.materialized_working_kv_bytes == int8_summary.full_bytes

    def test_int8_metadata_gt_zero(self, int8_summary):
        assert int8_summary.metadata_bytes > 0

    def test_int8_supports_real_bytes(self, int8_summary):
        assert int8_summary.supports_real_bytes_claim is True

    def test_int8_not_simulated(self, int8_summary):
        assert int8_summary.is_simulated is False

    def test_int8_total_reconciles(self, int8_summary):
        expected = (
            int8_summary.stored_kv_bytes
            + int8_summary.materialized_working_kv_bytes
            + int8_summary.metadata_bytes
            + int8_summary.temporary_workspace_bytes
        )
        assert int8_summary.total_kv_footprint_bytes == expected

    def test_int4_is_simulated(self, int4_summary):
        assert int4_summary.is_simulated is True

    def test_int4_not_real_bytes(self, int4_summary):
        assert int4_summary.supports_real_bytes_claim is False

    def test_int4_note_mentions_simulation(self, int4_summary):
        note = int4_summary.memory_claim_note.lower()
        assert "simulat" in note, "memory_claim_note must mention simulation"

    def test_int4_materialized_eq_full(self, int4_summary):
        assert int4_summary.materialized_working_kv_bytes == int4_summary.full_bytes

    def test_int4_total_reconciles(self, int4_summary):
        expected = (
            int4_summary.stored_kv_bytes
            + int4_summary.materialized_working_kv_bytes
            + int4_summary.metadata_bytes
            + int4_summary.temporary_workspace_bytes
        )
        assert int4_summary.total_kv_footprint_bytes == expected

    def test_summary_to_dict_no_forbidden_fields(self, int8_summary):
        _assert_no_forbidden_fields(int8_summary.to_dict(), "estimate_kv_memory int8 summary")
