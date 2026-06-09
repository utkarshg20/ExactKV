"""V6 Phase B: BackendAdapter proof-of-concept gate.

Gate: backend adapter boundary gate.

Verifies:
  * BackendAdapter base class satisfies KVCompressor protocol shape.
  * PassThroughBackendAdapter is registered as backend_passthrough.
  * get_compressor("backend_passthrough") returns a fresh instance.
  * Capabilities include V6 backend identity fields.
  * Existing compressors are unaffected (backward compatibility).
  * compress() does not mutate FullKVState.
  * materialize_for_draft() returns a forward-usable cache.
  * update_after_commit() refreshes from authoritative full state.
  * stats() includes all V5 workspace fields.
  * total_kv_footprint_bytes reconciles.
  * No forbidden performance fields appear.
  * ExactKV with backend_passthrough matches generate_full_greedy on:
      - 2 prompts × 2 draft lengths
  * acceptance_rate == 1.0, total_rejected == 0, total_corrections == 0.
  * Cache alignment holds every round.
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
DTYPE = "float32"

# ---------------------------------------------------------------------------
# Forbidden performance field names (data keys — not prose)
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
# Synthetic helper (no model required)
# ---------------------------------------------------------------------------

def _make_fake_full_state(
    num_layers: int = 2,
    seq_len: int = 8,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> "FullKVState":
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
# 1. Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_backend_passthrough_in_list_compressors(self):
        from exactkv.compressors import list_compressors

        names = list_compressors()
        assert "backend_passthrough" in names, (
            f"'backend_passthrough' missing from registry: {names}"
        )

    def test_get_backend_passthrough_returns_passthrough_adapter(self):
        from exactkv.compressors import get_compressor
        from exactkv.compressors.backend_adapter import PassThroughBackendAdapter

        comp = get_compressor("backend_passthrough")
        assert isinstance(comp, PassThroughBackendAdapter)

    def test_get_compressor_returns_fresh_instance(self):
        from exactkv.compressors import get_compressor

        a = get_compressor("backend_passthrough")
        b = get_compressor("backend_passthrough")
        assert a is not b, "get_compressor should return new instances each call"

    def test_backend_adapter_is_abstract(self):
        from exactkv.compressors.backend_adapter import BackendAdapter

        with pytest.raises(TypeError):
            BackendAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 2. Capabilities — backend identity fields
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_passthrough_has_backend_identity_fields(self):
        from exactkv.compressors import get_compressor

        caps = get_compressor("backend_passthrough").capabilities
        assert caps.backend_name == "passthrough"
        assert caps.backend_version == "0"
        assert caps.adapter_name == "PassThroughBackendAdapter"
        assert caps.adapter_version == "0.1.0"

    def test_passthrough_is_not_simulated(self):
        from exactkv.compressors import get_compressor

        caps = get_compressor("backend_passthrough").capabilities
        assert caps.is_simulated is False

    def test_passthrough_does_not_claim_real_bytes(self):
        from exactkv.compressors import get_compressor

        caps = get_compressor("backend_passthrough").capabilities
        assert caps.supports_real_bytes_claim is False

    def test_passthrough_notes_mention_poc(self):
        from exactkv.compressors import get_compressor

        caps = get_compressor("backend_passthrough").capabilities
        assert "proof-of-concept" in caps.notes.lower() or "poc" in caps.notes.lower()

    def test_existing_compressors_have_none_backend_fields(self):
        """Existing compressors must not need changes — backend fields default to None."""
        from exactkv.compressors import get_compressor

        for name in ("noop", "int8", "int4_sim", "debug_noise"):
            caps = get_compressor(name).capabilities
            assert caps.backend_name is None, (
                f"{name}: backend_name should be None, got {caps.backend_name!r}"
            )
            assert caps.backend_version is None
            assert caps.adapter_name is None
            assert caps.adapter_version is None

    def test_compressor_capabilities_dataclass_has_backend_fields(self):
        from exactkv.compressors.base import CompressorCapabilities

        field_names = {f.name for f in dataclasses.fields(CompressorCapabilities)}
        for field in ("backend_name", "backend_version", "adapter_name", "adapter_version"):
            assert field in field_names, f"CompressorCapabilities missing field {field!r}"


# ---------------------------------------------------------------------------
# 3. Unit tests — compress / materialize / update / stats (no model)
# ---------------------------------------------------------------------------

class TestPassThroughUnit:
    @pytest.fixture
    def adapter(self):
        from exactkv.compressors.backend_adapter import PassThroughBackendAdapter
        return PassThroughBackendAdapter()

    @pytest.fixture
    def full_state(self):
        return _make_fake_full_state()

    def test_compress_does_not_mutate_full_state(self, adapter, full_state):
        """compress() must not modify the authoritative FullKVState."""
        from exactkv.cache.utils import kv_total_bytes

        orig_seq_len = full_state.seq_len
        orig_next_token = full_state.next_token_id
        orig_bytes = kv_total_bytes(full_state.past_key_values)

        # Snapshot tensor values before compress
        orig_tensors = [
            (k.clone(), v.clone())
            for k, v in full_state.past_key_values
        ]

        compressed = adapter.compress(full_state)

        assert full_state.seq_len == orig_seq_len
        assert full_state.next_token_id == orig_next_token
        assert kv_total_bytes(full_state.past_key_values) == orig_bytes

        for (orig_k, orig_v), (new_k, new_v) in zip(
            orig_tensors, full_state.past_key_values
        ):
            assert torch.equal(orig_k, new_k), "FullKVState key tensor was mutated"
            assert torch.equal(orig_v, new_v), "FullKVState value tensor was mutated"

        assert compressed.logical_seq_len == orig_seq_len
        assert compressed.compressor_name == "backend_passthrough"

    def test_compress_returns_compressed_kv_state(self, adapter, full_state):
        from exactkv.cache.compressed_state import CompressedKVState

        result = adapter.compress(full_state)
        assert isinstance(result, CompressedKVState)
        assert result.metadata["next_token_id"] == full_state.next_token_id

    def test_materialize_returns_forward_usable_cache(self, adapter, full_state):
        from exactkv.cache.utils import kv_seq_len

        compressed = adapter.compress(full_state)
        cache = adapter.materialize_for_draft(compressed)

        assert cache is not None
        assert kv_seq_len(cache) == full_state.seq_len

    def test_materialize_does_not_mutate_compressed_data(self, adapter, full_state):
        compressed = adapter.compress(full_state)
        orig_k = [t.clone() for t in compressed.data["k"]]
        orig_v = [t.clone() for t in compressed.data["v"]]

        adapter.materialize_for_draft(compressed)

        for orig, cur in zip(orig_k, compressed.data["k"]):
            assert torch.equal(orig, cur), "compressed.data['k'] was mutated by materialize"
        for orig, cur in zip(orig_v, compressed.data["v"]):
            assert torch.equal(orig, cur), "compressed.data['v'] was mutated by materialize"

    def test_update_after_commit_refreshes_from_full_state(self, adapter, full_state):
        compressed = adapter.compress(full_state)

        # Build a new full state with a longer sequence
        new_state = _make_fake_full_state(num_layers=2, seq_len=12)
        updated = adapter.update_after_commit(compressed, new_state)

        assert updated.logical_seq_len == new_state.seq_len
        assert updated.compressor_name == "backend_passthrough"
        # Updated state must be independent of the old compressed state
        assert updated is not compressed

    def test_stats_has_all_v5_workspace_fields(self, adapter, full_state):
        compressed = adapter.compress(full_state)
        stats = adapter.stats(compressed)

        field_names = {f.name for f in dataclasses.fields(type(stats))}
        for field in (
            "stored_kv_bytes",
            "materialized_working_kv_bytes",
            "metadata_bytes",
            "temporary_workspace_bytes",
            "total_kv_footprint_bytes",
        ):
            assert field in field_names
            assert getattr(stats, field) >= 0, f"{field} must be non-negative"

    def test_stats_total_kv_footprint_reconciles(self, adapter, full_state):
        compressed = adapter.compress(full_state)
        stats = adapter.stats(compressed)

        expected = (
            stats.stored_kv_bytes
            + stats.materialized_working_kv_bytes
            + stats.metadata_bytes
            + stats.temporary_workspace_bytes
        )
        assert stats.total_kv_footprint_bytes == expected, (
            f"total_kv_footprint_bytes={stats.total_kv_footprint_bytes} "
            f"!= stored({stats.stored_kv_bytes}) + materialized({stats.materialized_working_kv_bytes}) "
            f"+ metadata({stats.metadata_bytes}) + temporary({stats.temporary_workspace_bytes}) "
            f"= {expected}"
        )

    def test_stats_no_compression_for_passthrough(self, adapter, full_state):
        """Pass-through stores full-precision bytes — no memory reduction."""
        compressed = adapter.compress(full_state)
        stats = adapter.stats(compressed)

        assert stats.compression_ratio == pytest.approx(1.0)
        assert stats.memory_reduction_factor == pytest.approx(1.0)
        assert stats.stored_kv_bytes == stats.full_bytes
        assert stats.metadata_bytes == 0
        assert stats.temporary_workspace_bytes == 0

    def test_stats_no_forbidden_performance_fields(self, adapter, full_state):
        compressed = adapter.compress(full_state)
        stats = adapter.stats(compressed)
        stats_dict = dataclasses.asdict(stats)
        _assert_no_forbidden_fields(stats_dict, context="CompressionStats")


# ---------------------------------------------------------------------------
# 4. Integration gate — ExactKV with backend_passthrough (requires model)
# ---------------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "Write a Python function that adds two numbers.",
]
LENGTHS = [8, 20]
DRAFT_LENGTHS = [2, 4]


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
@pytest.mark.parametrize("draft_len", DRAFT_LENGTHS)
def test_passthrough_output_equals_full_greedy(
    runtime,
    prompt: str,
    max_new_tokens: int,
    draft_len: int,
) -> None:
    """Gate 1: ExactKV(backend_passthrough) output_ids must match generate_full_greedy."""
    from exactkv.compressors import get_compressor
    from exactkv.runtime.exactkv_generator import ExactKVGenerator
    from exactkv.runtime.generation import generate_full_greedy

    compressor = get_compressor("backend_passthrough")
    generator = ExactKVGenerator(runtime=runtime, compressor=compressor, draft_len=draft_len)

    full_result = generate_full_greedy(runtime, prompt, max_new_tokens)
    exactkv_result = generator.generate(prompt, max_new_tokens)

    expected = full_result.generated_ids
    actual = exactkv_result.output_ids

    assert actual.shape == expected.shape, (
        f"Shape mismatch: expected {expected.shape}, got {actual.shape}\n"
        f"  expected text: {runtime.decode(expected)!r}\n"
        f"  actual text:   {exactkv_result.output_text!r}"
    )
    mismatch = (actual != expected).nonzero(as_tuple=True)
    assert len(mismatch[0]) == 0, (
        f"Token mismatch at positions {mismatch}\n"
        f"  expected ids: {expected.tolist()}\n"
        f"  actual ids:   {actual.tolist()}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
@pytest.mark.parametrize("draft_len", DRAFT_LENGTHS)
def test_passthrough_acceptance_rate_is_one(
    runtime,
    prompt: str,
    max_new_tokens: int,
    draft_len: int,
) -> None:
    """Gate 2: acceptance_rate must be exactly 1.0 for pass-through adapter."""
    from exactkv.compressors import get_compressor
    from exactkv.runtime.exactkv_generator import ExactKVGenerator

    compressor = get_compressor("backend_passthrough")
    generator = ExactKVGenerator(runtime=runtime, compressor=compressor, draft_len=draft_len)
    result = generator.generate(prompt, max_new_tokens)

    assert result.acceptance_rate == 1.0, (
        f"acceptance_rate={result.acceptance_rate} (expected 1.0)"
    )
    assert result.total_rejected == 0, (
        f"total_rejected={result.total_rejected} (expected 0)"
    )
    assert result.total_corrections == 0, (
        f"total_corrections={result.total_corrections} (expected 0)"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new_tokens", LENGTHS)
@pytest.mark.parametrize("draft_len", DRAFT_LENGTHS)
def test_passthrough_per_round_invariants(
    runtime,
    prompt: str,
    max_new_tokens: int,
    draft_len: int,
) -> None:
    """Gate 3–7: per-round trace invariants for pass-through adapter."""
    from exactkv.compressors import get_compressor
    from exactkv.runtime.exactkv_generator import ExactKVGenerator

    compressor = get_compressor("backend_passthrough")
    generator = ExactKVGenerator(runtime=runtime, compressor=compressor, draft_len=draft_len)
    result = generator.generate(prompt, max_new_tokens)

    assert result.num_rounds > 0, "Expected at least one round"

    for trace in result.traces:
        acc = trace.acceptance

        assert acc.all_matched is True, (
            f"Round {trace.round_idx}: all_matched=False\n"
            f"  draft:    {trace.draft_tokens}\n"
            f"  verifier: {acc.verifier_tokens}"
        )
        assert acc.correction_token is None, (
            f"Round {trace.round_idx}: correction_token={acc.correction_token}"
        )
        assert acc.bonus_token is None, (
            f"Round {trace.round_idx}: bonus_token={acc.bonus_token}"
        )
        assert acc.rejected_tokens == [], (
            f"Round {trace.round_idx}: rejected_tokens={acc.rejected_tokens}"
        )
        assert trace.full_seq_len_after == trace.compressed_seq_len_after, (
            f"Round {trace.round_idx}: cache misalignment: "
            f"full={trace.full_seq_len_after}, "
            f"compressed={trace.compressed_seq_len_after}"
        )
