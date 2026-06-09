"""V8 Phase B: ServingCacheLifecycleHarness gate tests."""
from __future__ import annotations

import copy
import dataclasses
import os

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import exactkv.compressors  # noqa: F401 — registers built-ins

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len, kv_total_bytes
from exactkv.serving.cache_lifecycle import (
    AUTHORITATIVE_FULL,
    COMPRESSED_DRAFT,
    ServingCacheLifecycleHarness,
    build_blocks,
    validate_retained_logical_positions,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden_fields(obj: object, context: str = "") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        assert not hits, f"Forbidden fields {hits} in {context or 'dict'}"
        for v in obj.values():
            _assert_no_forbidden_fields(v, context)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_forbidden_fields(item, context)


def _make_fake_full_state(
    num_layers: int = 2,
    seq_len: int = 8,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> FullKVState:
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
# Block mapping unit tests (no model)
# ---------------------------------------------------------------------------


class TestBlockMapping:
    def test_identity_logical_physical_builds_correct_blocks(self):
        blocks = build_blocks(logical_seq_len=10, physical_seq_len=10, block_size=4)
        assert len(blocks) == 3
        assert blocks[0].block_id == 0
        assert blocks[0].logical_start == 0 and blocks[0].logical_end == 4
        assert blocks[0].physical_start == 0 and blocks[0].physical_end == 4
        assert blocks[-1].physical_end == 10
        assert sum(b.physical_end - b.physical_start for b in blocks) == 10

    def test_physical_shorter_than_logical_requires_retained_positions(self):
        with pytest.raises(ValueError, match="retained_logical_positions"):
            build_blocks(logical_seq_len=10, physical_seq_len=6, block_size=4)

    def test_explicit_retained_positions_work(self):
        retained = (0, 1, 2, 5, 6, 7)
        blocks = build_blocks(
            logical_seq_len=10,
            physical_seq_len=6,
            block_size=4,
            retained_logical_positions=retained,
        )
        assert len(blocks) == 2
        assert blocks[0].logical_start == 0
        assert blocks[0].logical_end == 6  # retained[3] + 1 for physical slot 3
        assert blocks[1].logical_start == 6
        assert blocks[1].logical_end == 8

    def test_invalid_retained_positions_fail(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_retained_logical_positions(
                [0, 2, 2], logical_seq_len=10, physical_seq_len=3
            )
        with pytest.raises(ValueError, match="out of range"):
            validate_retained_logical_positions(
                [0, 1, 10], logical_seq_len=10, physical_seq_len=3
            )
        with pytest.raises(ValueError, match="length"):
            validate_retained_logical_positions(
                [0, 1], logical_seq_len=10, physical_seq_len=3
            )


# ---------------------------------------------------------------------------
# Harness ownership and lifecycle (no model)
# ---------------------------------------------------------------------------


class TestHarnessOwnership:
    def test_authoritative_and_compressed_are_separate_owners(self):
        full = _make_fake_full_state(seq_len=8)
        from exactkv.compressors import get_compressor

        compressor = get_compressor("noop")
        compressed = compressor.compress(full)

        harness = ServingCacheLifecycleHarness(block_size=4)
        auth_id = harness.register_authoritative_full(full)
        comp_id = harness.register_compressed_cache(compressed, compressor=compressor)

        assert auth_id != comp_id
        summary = harness.summarize()
        assert summary["authoritative_cache_id"] == auth_id
        assert summary["compressed_cache_id"] == comp_id
        assert summary["verification_uses"] == AUTHORITATIVE_FULL
        harness.validate_invariants()

    def test_compressed_cannot_replace_authoritative(self):
        full = _make_fake_full_state(seq_len=4)
        harness = ServingCacheLifecycleHarness()
        auth_id = harness.register_authoritative_full(full)

        from exactkv.compressors import get_compressor

        compressed = get_compressor("noop").compress(full)
        with pytest.raises(ValueError, match="cannot replace authoritative_full"):
            harness.register_compressed_cache(
                compressed, cache_id=auth_id, compressor=get_compressor("noop")
            )

    def test_append_committed_tokens_updates_logical_lengths(self):
        full = _make_fake_full_state(seq_len=6)
        from exactkv.compressors import get_compressor

        compressor = get_compressor("noop")
        compressed = compressor.compress(full)

        harness = ServingCacheLifecycleHarness(block_size=4)
        harness.register_authoritative_full(full)
        harness.register_compressed_cache(compressed, compressor=compressor)
        harness.append_committed_tokens(3)
        harness.validate_invariants()

        summary = harness.summarize()
        assert summary["authoritative_logical_seq_len"] == 9
        assert summary["compressed_logical_seq_len"] == 9

    def test_validate_invariants_catches_bad_block_ranges(self):
        harness = ServingCacheLifecycleHarness()
        full = _make_fake_full_state(seq_len=4)
        harness.register_authoritative_full(full)
        auth = harness._entries[harness._authoritative_id]
        auth.blocks[0] = dataclasses.replace(
            auth.blocks[0], physical_end=auth.blocks[0].physical_start
        )
        with pytest.raises(ValueError, match="physical range reversed"):
            harness.validate_invariants()

    def test_full_state_registration_does_not_mutate_state(self):
        full = _make_fake_full_state(seq_len=8)
        snap = copy.deepcopy(full)
        harness = ServingCacheLifecycleHarness()
        harness.register_authoritative_full(full)
        assert full.seq_len == snap.seq_len
        assert full.metadata == snap.metadata
        k0, v0 = full.past_key_values[0]
        sk0, sv0 = snap.past_key_values[0]
        assert torch.equal(k0, sk0)
        assert torch.equal(v0, sv0)

    def test_compressed_state_registration_does_not_mutate_state(self):
        full = _make_fake_full_state(seq_len=8)
        from exactkv.compressors import get_compressor

        compressor = get_compressor("int8")
        compressed = compressor.compress(full)
        snap = copy.deepcopy(compressed)

        harness = ServingCacheLifecycleHarness()
        harness.register_compressed_cache(compressed, compressor=compressor)
        assert compressed.logical_seq_len == snap.logical_seq_len
        assert compressed.metadata == snap.metadata
        assert compressed.compressor_name == snap.compressor_name

    def test_pruned_registration_requires_retained_positions(self):
        full = _make_fake_full_state(seq_len=8)
        from exactkv.compressors import get_compressor

        compressed = get_compressor("noop").compress(full)
        harness = ServingCacheLifecycleHarness()
        with pytest.raises(ValueError, match="retained_logical_positions"):
            harness.register_compressed_cache(
                compressed,
                physical_seq_len=5,
                compressor=get_compressor("noop"),
            )

    def test_summarize_has_no_forbidden_fields(self):
        full = _make_fake_full_state(seq_len=6)
        from exactkv.compressors import get_compressor

        compressor = get_compressor("noop")
        compressed = compressor.compress(full)
        harness = ServingCacheLifecycleHarness()
        harness.register_authoritative_full(full)
        harness.register_compressed_cache(compressed, compressor=compressor)
        summary = harness.summarize()
        _assert_no_forbidden_fields(summary, "summarize()")
        # Keys only — disclaimer prose may name forbidden metrics explicitly.
        assert not (_FORBIDDEN_FIELDS & summary.keys())


# ---------------------------------------------------------------------------
# Model-backed integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime

    return ModelRuntime(MODEL_NAME, device="cpu", dtype=DTYPE)


class TestHarnessWithCompressors:
    def test_harness_with_k8_v4_sim(self, runtime):
        from exactkv.compressors import get_compressor
        from exactkv.runtime.prefill import prefill_to_full_state

        compressor = get_compressor("k8_v4_sim")
        full_state = prefill_to_full_state(runtime, "Hello world")
        compressed = compressor.compress(full_state)

        harness = ServingCacheLifecycleHarness(block_size=8)
        harness.register_authoritative_full(full_state)
        harness.register_compressed_cache(compressed, compressor=compressor)
        harness.validate_invariants()

        summary = harness.summarize()
        comp_entry = next(e for e in summary["entries"] if e["owner"] == COMPRESSED_DRAFT)
        assert comp_entry["compressor_name"] == "k8_v4_sim"
        assert comp_entry["is_simulated"] is True
        assert comp_entry["stored_kv_bytes"] is not None
        assert comp_entry["total_kv_footprint_bytes"] is not None

    def test_harness_with_backend_passthrough(self, runtime):
        from exactkv.compressors import get_compressor
        from exactkv.runtime.prefill import prefill_to_full_state

        compressor = get_compressor("backend_passthrough")
        full_state = prefill_to_full_state(runtime, "ExactKV harness smoke")
        compressed = compressor.compress(full_state)

        harness = ServingCacheLifecycleHarness()
        harness.register_authoritative_full(full_state)
        harness.register_compressed_cache(compressed, compressor=compressor)
        harness.validate_invariants()

        summary = harness.summarize()
        auth = next(e for e in summary["entries"] if e["owner"] == AUTHORITATIVE_FULL)
        comp = next(e for e in summary["entries"] if e["owner"] == COMPRESSED_DRAFT)
        assert auth["supports_real_bytes_claim"] is True
        assert comp["supports_real_bytes_claim"] is False  # passthrough PoC capability
        assert comp["is_simulated"] is False

    def test_exactkv_smoke_with_harness_exactness_gate(self, runtime):
        from exactkv.compressors import get_compressor
        from exactkv.metrics.exactness import token_exact_match
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy
        from exactkv.runtime.prefill import prefill_to_full_state

        prompt = "The capital of France is"
        compressor = get_compressor("int8")
        harness = ServingCacheLifecycleHarness(block_size=16)

        full_state = prefill_to_full_state(runtime, prompt)
        harness.register_authoritative_full(full_state)
        compressed = compressor.compress(full_state)
        harness.register_compressed_cache(compressed, compressor=compressor)
        harness.validate_invariants()

        gen = ExactKVGenerator(runtime, compressor, draft_len=4)
        result = gen.generate(prompt, max_new_tokens=8)

        for trace in result.traces:
            harness.append_committed_tokens(
                trace.acceptance.num_accepted
                + (1 if trace.acceptance.correction_token is not None else 0)
            )
            harness.validate_invariants()

        full_out = generate_full_greedy(runtime, prompt, max_new_tokens=8)
        assert token_exact_match(result.output_ids, full_out.generated_ids)
        assert harness.summarize()["invariants_valid"] is True

        summary = harness.summarize()
        _assert_no_forbidden_fields(summary, "post-run summarize()")
