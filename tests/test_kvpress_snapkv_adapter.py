"""V13 Phase 5b: restricted SnapKV experimental adapter safety gates.

Verifies ``snapkv_experimental`` factory-only adapter (kvpress SnapKVPress):
  * NOT in default compressor registry.
  * Lazy kvpress import; hook isolation mirrors KVPressKnormAdapter.
  * ExactKV exactness on a short prompt panel when kvpress is installed.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_KVPRESS_INSTALLED = importlib.util.find_spec("kvpress") is not None

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
_KV_PROMPT = "The capital of France is Paris and the river Seine flows through it."
_MAX_NEW = 8
_DRAFT_LEN = 4


@pytest.fixture(scope="module")
def kvpress_runtime():
    from exactkv.runtime.model_runtime import ModelRuntime

    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


class TestSnapKVExperimentalRegistry:
    def test_not_in_default_registry(self):
        import exactkv.compressors  # noqa: F401

        from exactkv.compressors import get_compressor, list_compressors

        assert "snapkv_experimental" not in list_compressors()
        with pytest.raises(ValueError, match="snapkv_experimental"):
            get_compressor("snapkv_experimental")


@pytest.mark.skipif(not _KVPRESS_INSTALLED, reason="kvpress optional extra not installed")
class TestKVPressSnapKVExperimentalAdapter:
    def test_module_import_does_not_load_kvpress(self):
        sys.modules.pop("kvpress", None)
        import exactkv.compressors.kvpress_snapkv  # noqa: F401

        assert "kvpress" not in sys.modules

    def test_constructing_adapter_imports_kvpress_lazily(self, kvpress_runtime):
        sys.modules.pop("kvpress", None)
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

        create_snapkv_experimental_adapter(kvpress_runtime, compression_ratio=0.5)
        assert "kvpress" in sys.modules

    def test_adapter_name_and_capabilities(self, kvpress_runtime):
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

        adapter = create_snapkv_experimental_adapter(kvpress_runtime)
        assert adapter.name == "snapkv_experimental"
        assert adapter.capabilities.name == "snapkv_experimental"
        assert adapter.capabilities.backend_name == "kvpress"
        assert adapter.capabilities.supports_real_bytes_claim is True
        assert "restricted experimental" in adapter.capabilities.notes.lower()

    def test_verification_mode_passes_when_no_hooks(self, kvpress_runtime):
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

        adapter = create_snapkv_experimental_adapter(kvpress_runtime)
        with adapter.verification_mode():
            pass

    def test_verification_mode_fails_when_hooks_manually_active(self, kvpress_runtime):
        from exactkv.compressors.kvpress_knorm import _iter_attention_modules
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

        adapter = create_snapkv_experimental_adapter(kvpress_runtime)
        attn = next(_iter_attention_modules(kvpress_runtime.model))

        def _noop_hook(module, inp, out):  # noqa: ARG001
            return out

        handle = attn.register_forward_hook(_noop_hook)
        try:
            with pytest.raises(RuntimeError, match="hooks must not be active"):
                with adapter.verification_mode():
                    pass
        finally:
            handle.remove()

    def test_hook_count_returns_to_original_after_compression_replay(self, kvpress_runtime):
        from exactkv.compressors.kvpress_knorm import count_attention_forward_hooks
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_snapkv_experimental_adapter(kvpress_runtime)
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)
        hooks_before = count_attention_forward_hooks(kvpress_runtime.model)

        compressed = adapter.compress(full_state)

        hooks_after = count_attention_forward_hooks(kvpress_runtime.model)
        assert hooks_after == hooks_before
        assert compressed.data["__hook_count_before__"] == hooks_before
        assert compressed.data["__hook_count_after__"] == hooks_before
        assert compressed.data["__hook_count_during__"] > hooks_before

    def test_physical_kv_shorter_than_logical_seq_len(self, kvpress_runtime):
        from exactkv.cache.utils import kv_seq_len, kv_total_bytes
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_snapkv_experimental_adapter(kvpress_runtime, compression_ratio=0.5)
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)
        full_bytes = kv_total_bytes(full_state.past_key_values)
        logical_len = full_state.seq_len

        compressed = adapter.compress(full_state)
        cache = adapter.materialize_for_draft(compressed)
        physical_len = kv_seq_len(cache)

        assert physical_len < logical_len
        assert compressed.logical_seq_len == logical_len
        assert compressed.data["__physical_seq_len__"] == physical_len
        assert kv_total_bytes(cache) < full_bytes

    def test_stored_bytes_match_materialized_working_bytes(self, kvpress_runtime):
        from exactkv.cache.utils import kv_total_bytes
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_snapkv_experimental_adapter(kvpress_runtime, compression_ratio=0.5)
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)
        compressed = adapter.compress(full_state)
        stats = adapter.stats(compressed)
        pruned_bytes = kv_total_bytes(compressed.data["dynamic_cache"])

        assert stats.stored_kv_bytes == pruned_bytes
        assert stats.materialized_working_kv_bytes == pruned_bytes
        assert stats.metadata_bytes == 0
        assert adapter.capabilities.supports_real_bytes_claim is True

    def test_full_authoritative_state_unchanged_after_verify(self, kvpress_runtime):
        from exactkv.cache.utils import extract_kv_tensors, kv_seq_len, kv_total_bytes
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter
        from exactkv.runtime.prefill import prefill_to_full_state
        from exactkv.verification.engine import VerificationEngine

        adapter = create_snapkv_experimental_adapter(kvpress_runtime)
        engine = VerificationEngine(kvpress_runtime)
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)

        full_bytes_before = kv_total_bytes(full_state.past_key_values)
        seq_before = full_state.seq_len
        k_before, v_before, _ = extract_kv_tensors(full_state.past_key_values)
        k_snap = [t.clone() for t in k_before]
        v_snap = [t.clone() for t in v_before]

        adapter.compress(full_state)
        draft = [full_state.next_token_id]
        with adapter.verification_mode():
            engine.verify_sequential(full_state, draft)

        assert kv_total_bytes(full_state.past_key_values) == full_bytes_before
        assert full_state.seq_len == seq_before
        assert kv_seq_len(full_state.past_key_values) == seq_before
        k_after, v_after, _ = extract_kv_tensors(full_state.past_key_values)
        for orig_k, cur_k in zip(k_snap, k_after):
            assert torch.equal(orig_k, cur_k)
        for orig_v, cur_v in zip(v_snap, v_after):
            assert torch.equal(orig_v, cur_v)

    def test_exactkv_matches_full_greedy_sequential(self, kvpress_runtime):
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter
        from exactkv.metrics.acceptance import summarize_acceptance
        from exactkv.metrics.exactness import token_exact_match
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy

        adapter = create_snapkv_experimental_adapter(kvpress_runtime, compression_ratio=0.5)
        full_res = generate_full_greedy(kvpress_runtime, _KV_PROMPT, _MAX_NEW)
        ekv_res = ExactKVGenerator(
            kvpress_runtime, adapter, draft_len=_DRAFT_LEN, verification_method="sequential"
        ).generate(_KV_PROMPT, _MAX_NEW)

        assert token_exact_match(full_res.generated_ids, ekv_res.output_ids)
        acceptance = summarize_acceptance(ekv_res.traces)
        assert acceptance.total_drafted == acceptance.total_accepted + acceptance.total_rejected

    def test_exactkv_matches_full_greedy_span(self, kvpress_runtime):
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter
        from exactkv.metrics.exactness import token_exact_match
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy

        adapter = create_snapkv_experimental_adapter(kvpress_runtime, compression_ratio=0.5)
        full_res = generate_full_greedy(kvpress_runtime, _KV_PROMPT, _MAX_NEW)
        ekv_res = ExactKVGenerator(
            kvpress_runtime, adapter, draft_len=_DRAFT_LEN, verification_method="span"
        ).generate(_KV_PROMPT, _MAX_NEW)

        assert token_exact_match(full_res.generated_ids, ekv_res.output_ids)
