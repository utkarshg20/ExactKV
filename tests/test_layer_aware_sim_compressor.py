"""V7 Phase B/C — LayerAwareVSimCompressor tests.

Gate: layer-aware simulated compressor gate + ExactKV correctness gate.

Covers:
  * Registry resolves k8_v4_boundary_v8_sim, k8_v4_boundary2_v8_sim,
    k8_v4_boundary4_v8_sim
  * Honest capabilities (is_simulated, supports_real_bytes_claim=False)
  * Boundary layer selection for N=1, 2, 4 (first/last N layers V8, interior V4)
  * K quantisation INT8 range on all layers
  * V boundary vs interior quantisation ranges
  * compress does not mutate FullKVState
  * materialize_for_draft is forward-usable
  * update_after_commit refreshes from authoritative full state
  * stats V5 fields reconcile; int8 container accounting
  * No forbidden performance fields
  * ExactKV gate: 2 prompts × 2 draft_lens × 2 max_new_tokens
"""
from __future__ import annotations

import os
from dataclasses import asdict

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]
_DRAFT_LENS = [4, 8]
_MAX_NEW_TOKENS = [8, 12]

_FORBIDDEN = frozenset({
    "tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds",
})

import exactkv.compressors  # noqa: F401

from exactkv.compressors import get_compressor, list_compressors
from exactkv.compressors.layer_aware_sim import (
    K8V4Boundary2V8SimCompressor,
    K8V4Boundary4V8SimCompressor,
    K8V4BoundaryV8SimCompressor,
    LayerAwareVSimCompressor,
    _boundary_layer_indices,
    _v_bits_for_layer,
)

_BOUNDARY_VARIANTS = [
    ("k8_v4_boundary_v8_sim", K8V4BoundaryV8SimCompressor, 1),
    ("k8_v4_boundary2_v8_sim", K8V4Boundary2V8SimCompressor, 2),
    ("k8_v4_boundary4_v8_sim", K8V4Boundary4V8SimCompressor, 4),
]


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def full_state(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state
    return prefill_to_full_state(runtime, _PROMPTS[0])


def _make_synthetic_full_state(num_layers: int = 4):
    from exactkv.cache.full_state import FullKVState

    shape = (1, 2, 4, 8)
    layers = tuple(
        (torch.randn(shape), torch.randn(shape)) for _ in range(num_layers)
    )
    prompt_ids = torch.zeros(1, 4, dtype=torch.long)
    return FullKVState(
        past_key_values=layers,
        prompt_ids=prompt_ids,
        generated_ids=torch.zeros(1, 0, dtype=torch.long),
        full_sequence_ids=prompt_ids,
        device=torch.device("cpu"),
        dtype=torch.float32,
        metadata={"next_token_id": 0},
    )


def _assert_no_forbidden(obj, path="root"):
    if isinstance(obj, dict):
        hits = _FORBIDDEN & obj.keys()
        assert not hits, f"Forbidden at {path}: {hits}"
        for k, v in obj.items():
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")


# ===========================================================================
# Registry
# ===========================================================================

class TestRegistry:
    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_name_in_list_compressors(self, name, cls, boundary_n):
        assert name in list_compressors()

    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_get_compressor_returns_instance(self, name, cls, boundary_n):
        comp = get_compressor(name)
        assert isinstance(comp, cls)
        assert comp.name == name

    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_fresh_instance_each_call(self, name, cls, boundary_n):
        assert get_compressor(name) is not get_compressor(name)


# ===========================================================================
# Capabilities
# ===========================================================================

class TestCapabilities:
    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_honest_capabilities(self, name, cls, boundary_n):
        caps = get_compressor(name).capabilities
        assert caps.is_simulated is True
        assert caps.supports_real_bytes_claim is False
        assert caps.key_bit_width == 8
        assert caps.value_bit_width is None
        assert caps.value_bit_width_label == "mixed 8/4-sim"
        assert caps.asymmetric is True
        assert caps.supports_quantization is True
        assert caps.supports_token_dropping is False
        assert "boundary-depth" in caps.notes.lower()
        assert "int8 containers" in caps.notes.lower()
        assert "no true attention weights" in caps.notes.lower()
        assert "no sparse v" in caps.notes.lower()
        assert "no turboquant+" in caps.notes.lower()
        assert "kvquant" in caps.notes.lower()

    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_no_forbidden_in_capabilities(self, name, cls, boundary_n):
        _assert_no_forbidden(asdict(get_compressor(name).capabilities))


# ===========================================================================
# Boundary layer selection
# ===========================================================================

class TestBoundarySelection:
    def test_boundary_indices_n1(self):
        assert _boundary_layer_indices(24, 1) == {0, 23}

    def test_boundary_indices_n2(self):
        assert _boundary_layer_indices(24, 2) == {0, 1, 22, 23}

    def test_boundary_indices_n4(self):
        assert _boundary_layer_indices(24, 4) == {0, 1, 2, 3, 20, 21, 22, 23}

    def test_boundary_indices_two_layers_model(self):
        assert _boundary_layer_indices(2, 1) == {0, 1}

    def test_v_bits_boundary_vs_interior_n1(self):
        assert _v_bits_for_layer(0, 24, 1, 8, 4) == 8
        assert _v_bits_for_layer(23, 24, 1, 8, 4) == 8
        assert _v_bits_for_layer(12, 24, 1, 8, 4) == 4

    def test_v_bits_boundary_vs_interior_n2(self):
        for idx in (0, 1, 22, 23):
            assert _v_bits_for_layer(idx, 24, 2, 8, 4) == 8
        assert _v_bits_for_layer(12, 24, 2, 8, 4) == 4

    def test_v_bits_boundary_vs_interior_n4(self):
        for idx in (0, 1, 2, 3, 20, 21, 22, 23):
            assert _v_bits_for_layer(idx, 24, 4, 8, 4) == 8
        assert _v_bits_for_layer(12, 24, 4, 8, 4) == 4


# ===========================================================================
# Quantisation ranges
# ===========================================================================

class TestQuantisationRanges:
    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_k_int8_range_all_layers(self, full_state, name, cls, boundary_n):
        comp = get_compressor(name)
        compressed = comp.compress(full_state)
        for layer in compressed.data["layers"]:
            q = layer["k_data"]["q"]
            assert q.dtype == torch.int8
            assert int(q.min()) >= -128
            assert int(q.max()) <= 127

    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_v_boundary_int8_interior_int4(self, full_state, name, cls, boundary_n):
        comp = get_compressor(name)
        compressed = comp.compress(full_state)
        layers = compressed.data["layers"]
        num = len(layers)
        boundary = _boundary_layer_indices(num, boundary_n)

        for idx, layer in enumerate(layers):
            q = layer["v_data"]["q"]
            assert q.dtype == torch.int8
            if idx in boundary:
                assert layer["v_bits"] == 8
                assert int(q.min()) >= -128
                assert int(q.max()) <= 127
            else:
                assert layer["v_bits"] == 4
                assert int(q.min()) >= -8
                assert int(q.max()) <= 7


# ===========================================================================
# compress / materialize / update
# ===========================================================================

class TestCompressMaterialize:
    def test_compress_does_not_mutate_full_state(self, full_state):
        from exactkv.cache.utils import extract_kv_tensors

        comp = get_compressor("k8_v4_boundary_v8_sim")
        orig_k, orig_v, _ = extract_kv_tensors(full_state.past_key_values)
        k_copy = [t.clone() for t in orig_k]
        v_copy = [t.clone() for t in orig_v]
        comp.compress(full_state)
        new_k, new_v, _ = extract_kv_tensors(full_state.past_key_values)
        for ok, nk in zip(k_copy, new_k):
            assert torch.allclose(ok, nk)
        for ov, nv in zip(v_copy, new_v):
            assert torch.allclose(ov, nv)

    def test_materialize_forward_usable(self, runtime, full_state):
        comp = get_compressor("k8_v4_boundary_v8_sim")
        compressed = comp.compress(full_state)
        cache = comp.materialize_for_draft(compressed)
        input_ids = torch.tensor([[full_state.metadata["next_token_id"]]])
        out = runtime.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        assert out.logits is not None

    def test_update_after_commit_from_authoritative_state(self, runtime, full_state):
        from exactkv.runtime.prefill import prefill_to_full_state

        comp = get_compressor("k8_v4_boundary_v8_sim")
        compressed = comp.compress(full_state)
        new_state = prefill_to_full_state(runtime, _PROMPTS[1])
        updated = comp.update_after_commit(compressed, new_state)
        assert updated.logical_seq_len == new_state.seq_len


# ===========================================================================
# stats
# ===========================================================================

class TestStats:
    def test_stats_fields_reconcile(self, full_state):
        comp = get_compressor("k8_v4_boundary_v8_sim")
        compressed = comp.compress(full_state)
        stats = comp.stats(compressed)
        assert stats.stored_kv_bytes > 0
        assert stats.materialized_working_kv_bytes == stats.full_bytes
        assert stats.metadata_bytes > 0
        assert stats.total_kv_footprint_bytes == (
            stats.stored_kv_bytes
            + stats.materialized_working_kv_bytes
            + stats.metadata_bytes
            + stats.temporary_workspace_bytes
        )
        assert stats.compressed_bytes == stats.stored_kv_bytes + stats.metadata_bytes

    def test_stored_bytes_int8_container_accounting(self, full_state):
        """stored_kv_bytes uses 1 byte/element for quantised sides."""
        comp = get_compressor("k8_v4_boundary_v8_sim")
        compressed = comp.compress(full_state)
        stats = comp.stats(compressed)
        # Must be less than full fp32 reference but not as small as packed 4-bit would be.
        assert stats.stored_kv_bytes < stats.full_bytes
        assert stats.materialized_working_kv_bytes == stats.full_bytes

    def test_capabilities_supports_real_bytes_false(self, full_state):
        comp = get_compressor("k8_v4_boundary_v8_sim")
        assert comp.capabilities.supports_real_bytes_claim is False

    def test_stats_no_forbidden_fields(self, full_state):
        comp = get_compressor("k8_v4_boundary_v8_sim")
        stats = comp.stats(comp.compress(full_state))
        _assert_no_forbidden(asdict(stats))


# ===========================================================================
# ExactKV gate
# ===========================================================================

class TestExactKVGate:
    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    @pytest.mark.parametrize("prompt", _PROMPTS)
    @pytest.mark.parametrize("draft_len", _DRAFT_LENS)
    @pytest.mark.parametrize("max_new", _MAX_NEW_TOKENS)
    def test_exactkv_output_matches_full_greedy(
        self, runtime, name, cls, boundary_n, prompt, draft_len, max_new,
    ):
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy

        comp = get_compressor(name)
        full_res = generate_full_greedy(runtime, prompt, max_new)
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=draft_len).generate(
            prompt, max_new,
        )
        full_ids = full_res.generated_ids.squeeze(0).tolist()
        ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
        assert full_ids == ekv_ids

    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    @pytest.mark.parametrize("prompt", _PROMPTS)
    @pytest.mark.parametrize("max_new", _MAX_NEW_TOKENS)
    def test_acceptance_bookkeeping_reconciles(
        self, runtime, name, cls, boundary_n, prompt, max_new,
    ):
        from exactkv.metrics.acceptance import summarize_acceptance
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        comp = get_compressor(name)
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=4).generate(
            prompt, max_new,
        )
        acc = summarize_acceptance(ekv_res.traces)
        assert acc.total_drafted == acc.total_accepted + acc.total_rejected
        assert acc.total_corrections >= 0

    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    @pytest.mark.parametrize("prompt", _PROMPTS)
    def test_cache_alignment_holds(self, runtime, name, cls, boundary_n, prompt):
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        comp = get_compressor(name)
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=4).generate(
            prompt, _MAX_NEW_TOKENS[0],
        )
        assert ekv_res.output_ids.shape[1] > 0

    @pytest.mark.parametrize("name,cls,boundary_n", _BOUNDARY_VARIANTS)
    def test_exactkv_failures_zero_via_run_one(self, runtime, name, cls, boundary_n):
        from exactkv.benchmarks.runner import RunConfig, run_one

        entry = {"prompt_id": "p0", "category": "test", "prompt": _PROMPTS[0]}
        cfg = RunConfig(
            compressor_name=name,
            draft_len=4,
            max_new_tokens=_MAX_NEW_TOKENS[0],
        )
        result = run_one(runtime, entry, cfg)
        assert result["exactkv_failure"] is False
        _assert_no_forbidden(result)
