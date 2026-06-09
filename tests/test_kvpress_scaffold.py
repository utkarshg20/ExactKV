"""V6 Phase C: kvpress safety gates and restricted KVPressKnormAdapter.

Verifies:
  * Optional ``kvpress`` extra is defined but not loaded on default import paths.
  * ``BackendAdapter.verification_mode()`` default is a no-op.
  * ``ExactKVGenerator`` wraps verification inside ``verification_mode()`` when present.
  * Explicit ``import kvpress`` (when installed) does not break pass-through smoke.
  * ``KVPressKnormAdapter`` (KnormPress only) when ``[kvpress]`` extra is installed.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_KVPRESS_INSTALLED = importlib.util.find_spec("kvpress") is not None


# ---------------------------------------------------------------------------
# Helpers (shared with test_backend_adapter_poc patterns)
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
# 1. Default import paths do not load kvpress
# ---------------------------------------------------------------------------

class TestKvpressNotOnDefaultImportPath:
    def test_import_exactkv_compressors_does_not_load_kvpress(self):
        """Importing exactkv.compressors must not pull in kvpress."""
        # Clear kvpress from sys.modules if a prior optional test left it.
        sys.modules.pop("kvpress", None)
        import exactkv.compressors  # noqa: F401

        assert "kvpress" not in sys.modules

    def test_kvpress_not_in_sys_modules_after_poc_imports(self):
        """Typical Phase B test imports must not register kvpress."""
        sys.modules.pop("kvpress", None)
        import exactkv.compressors  # noqa: F401
        from exactkv.compressors.backend_adapter import PassThroughBackendAdapter
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        assert PassThroughBackendAdapter is not None
        assert ExactKVGenerator is not None
        assert "kvpress" not in sys.modules


# ---------------------------------------------------------------------------
# 2. verification_mode() default contract
# ---------------------------------------------------------------------------

class TestVerificationModeDefault:
    def test_backend_adapter_has_verification_mode(self):
        from exactkv.compressors.backend_adapter import BackendAdapter

        assert hasattr(BackendAdapter, "verification_mode")
        assert callable(BackendAdapter.verification_mode)

    def test_default_verification_mode_is_noop(self):
        from exactkv.compressors.backend_adapter import PassThroughBackendAdapter
        from exactkv.cache.utils import kv_total_bytes

        adapter = PassThroughBackendAdapter()
        full_state = _make_fake_full_state()
        compressed = adapter.compress(full_state)
        orig_full_bytes = kv_total_bytes(full_state.past_key_values)
        orig_k = [t.clone() for t in compressed.data["k"]]

        with adapter.verification_mode():
            pass

        assert kv_total_bytes(full_state.past_key_values) == orig_full_bytes
        for orig, cur in zip(orig_k, compressed.data["k"]):
            assert torch.equal(orig, cur)

    def test_passthrough_inherits_default_verification_mode(self):
        from exactkv.compressors.backend_adapter import PassThroughBackendAdapter

        adapter = PassThroughBackendAdapter()
        entered = False
        with adapter.verification_mode():
            entered = True
        assert entered


# ---------------------------------------------------------------------------
# 3. Generator wraps verification inside verification_mode()
# ---------------------------------------------------------------------------

class TestGeneratorVerificationGuard:
    def test_verify_called_inside_verification_mode_for_backend_adapter(self):
        from exactkv.compressors.backend_adapter import PassThroughBackendAdapter
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.model_runtime import ModelRuntime
        from exactkv.verification.acceptance import AcceptanceResult

        runtime = MagicMock(spec=ModelRuntime)
        adapter = PassThroughBackendAdapter()
        generator = ExactKVGenerator(runtime=runtime, compressor=adapter, draft_len=2)

        fake_acceptance = AcceptanceResult(
            draft_tokens=[1, 2],
            verifier_tokens=[1],
            accepted_tokens=[1],
            correction_token=None,
            rejected_tokens=[],
            bonus_token=None,
            all_matched=False,
            num_accepted=1,
            num_rejected=0,
        )

        mode_mock = MagicMock()
        mode_mock.__enter__ = MagicMock(return_value=None)
        mode_mock.__exit__ = MagicMock(return_value=False)

        full_state = _make_fake_full_state()
        with patch.object(
            PassThroughBackendAdapter,
            "verification_mode",
            return_value=mode_mock,
        ) as patched_mode:
            with patch.object(
                generator.engine,
                "verify_sequential",
                return_value=fake_acceptance,
            ) as patched_verify:
                result = generator._verify_draft_tokens(full_state, [1, 2])

        patched_mode.assert_called_once()
        mode_mock.__enter__.assert_called_once()
        patched_verify.assert_called_once_with(full_state, [1, 2])
        assert result is fake_acceptance

    def test_noop_compressor_skips_verification_mode_guard(self):
        from exactkv.compressors.noop import NoOpCompressor
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.model_runtime import ModelRuntime
        from exactkv.verification.acceptance import AcceptanceResult

        runtime = MagicMock(spec=ModelRuntime)
        compressor = NoOpCompressor()
        generator = ExactKVGenerator(runtime=runtime, compressor=compressor, draft_len=2)

        assert not hasattr(compressor, "verification_mode")

        fake_acceptance = AcceptanceResult(
            draft_tokens=[3],
            verifier_tokens=[3],
            accepted_tokens=[3],
            correction_token=None,
            rejected_tokens=[],
            bonus_token=None,
            all_matched=True,
            num_accepted=1,
            num_rejected=0,
        )

        full_state = _make_fake_full_state()
        with patch.object(
            generator.engine,
            "verify_sequential",
            return_value=fake_acceptance,
        ) as patched_verify:
            result = generator._verify_draft_tokens(full_state, [3])

        patched_verify.assert_called_once_with(full_state, [3])
        assert result is fake_acceptance


# ---------------------------------------------------------------------------
# 4. Optional lazy kvpress import regression (skipped when not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _KVPRESS_INSTALLED, reason="kvpress optional extra not installed")
class TestKvpressLazyImportRegression:
    def test_explicit_kvpress_import_then_passthrough_smoke(self):
        """Explicit import kvpress must not break pass-through adapter unit smoke."""
        import kvpress  # noqa: F401 — intentional lazy import inside test

        from exactkv.compressors.backend_adapter import PassThroughBackendAdapter
        from exactkv.cache.utils import kv_seq_len

        adapter = PassThroughBackendAdapter()
        full_state = _make_fake_full_state()
        compressed = adapter.compress(full_state)
        cache = adapter.materialize_for_draft(compressed)

        assert compressed.logical_seq_len == full_state.seq_len
        assert kv_seq_len(cache) == full_state.seq_len


# ---------------------------------------------------------------------------
# 5. Restricted KVPressKnormAdapter (skipped when kvpress not installed)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
_KV_PROMPT = "The capital of France is Paris and the river"


@pytest.fixture(scope="module")
def kvpress_runtime():
    from exactkv.runtime.model_runtime import ModelRuntime

    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.mark.skipif(not _KVPRESS_INSTALLED, reason="kvpress optional extra not installed")
class TestKVPressKnormAdapter:
    def test_kvpress_knorm_module_import_does_not_load_kvpress(self):
        """Importing the adapter module must not eagerly import kvpress."""
        sys.modules.pop("kvpress", None)
        import exactkv.compressors.kvpress_knorm  # noqa: F401

        assert "kvpress" not in sys.modules

    def test_constructing_adapter_imports_kvpress_lazily(self, kvpress_runtime):
        sys.modules.pop("kvpress", None)
        from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter

        create_kvpress_knorm_adapter(kvpress_runtime, compression_ratio=0.5)
        assert "kvpress" in sys.modules

    def test_verification_mode_passes_when_no_hooks(self, kvpress_runtime):
        from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter

        adapter = create_kvpress_knorm_adapter(kvpress_runtime)
        with adapter.verification_mode():
            pass

    def test_verification_mode_fails_when_hooks_manually_active(self, kvpress_runtime):
        from exactkv.compressors.kvpress_knorm import (
            create_kvpress_knorm_adapter,
            _iter_attention_modules,
        )

        adapter = create_kvpress_knorm_adapter(kvpress_runtime)
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
        from exactkv.compressors.kvpress_knorm import (
            count_attention_forward_hooks,
            create_kvpress_knorm_adapter,
        )
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_kvpress_knorm_adapter(kvpress_runtime)
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)
        hooks_before = count_attention_forward_hooks(kvpress_runtime.model)

        compressed = adapter.compress(full_state)

        hooks_after = count_attention_forward_hooks(kvpress_runtime.model)
        assert hooks_after == hooks_before
        assert compressed.data["__hook_count_before__"] == hooks_before
        assert compressed.data["__hook_count_after__"] == hooks_before
        assert compressed.data["__hook_count_during__"] > hooks_before

    def test_full_authoritative_state_unchanged_after_verify(self, kvpress_runtime):
        import copy

        from exactkv.cache.utils import extract_kv_tensors, kv_seq_len, kv_total_bytes
        from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter
        from exactkv.runtime.prefill import prefill_to_full_state
        from exactkv.verification.engine import VerificationEngine

        adapter = create_kvpress_knorm_adapter(kvpress_runtime)
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

    def test_physical_kv_shorter_than_logical_seq_len(self, kvpress_runtime):
        from exactkv.cache.utils import kv_seq_len, kv_total_bytes
        from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_kvpress_knorm_adapter(kvpress_runtime, compression_ratio=0.5)
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
        from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_kvpress_knorm_adapter(kvpress_runtime, compression_ratio=0.5)
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)
        compressed = adapter.compress(full_state)
        stats = adapter.stats(compressed)
        pruned_bytes = kv_total_bytes(compressed.data["dynamic_cache"])

        assert stats.stored_kv_bytes == pruned_bytes
        assert stats.materialized_working_kv_bytes == pruned_bytes
        assert stats.materialized_working_kv_bytes == stats.stored_kv_bytes
        assert stats.metadata_bytes == 0
        assert adapter.capabilities.supports_real_bytes_claim is True

    def test_draft_next_token_may_differ_from_full_kv_is_allowed(self, kvpress_runtime):
        """Lossy KnormPress draft prediction may diverge; verification uses full KV."""
        from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_kvpress_knorm_adapter(kvpress_runtime, compression_ratio=0.5)
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)
        compressed = adapter.compress(full_state)

        full_next = full_state.next_token_id
        draft_next = compressed.metadata["next_token_id"]
        assert isinstance(full_next, int)
        assert isinstance(draft_next, int)
        # Divergence is expected and acceptable for this lossy backend.

    def test_model_mutation_safety_main_model_unchanged_with_isolation(self, kvpress_runtime):
        from exactkv.compressors.kvpress_knorm import (
            create_kvpress_knorm_adapter,
            snapshot_attention_model_state,
        )
        from exactkv.runtime.prefill import prefill_to_full_state

        adapter = create_kvpress_knorm_adapter(
            kvpress_runtime, compression_ratio=0.5, isolate_compression_model=True
        )
        full_state = prefill_to_full_state(kvpress_runtime, _KV_PROMPT)
        before = snapshot_attention_model_state(kvpress_runtime.model)

        adapter.compress(full_state)

        after = snapshot_attention_model_state(kvpress_runtime.model)
        assert before["hook_counts"] == after["hook_counts"]
        assert before["attn_module_ids"] == after["attn_module_ids"]
        assert before["rotary_emb_ids"] == after["rotary_emb_ids"]

    def test_rotary_emb_mutates_without_isolation_documents_known_blocker(self):
        """Documents that kvpress permanently mutates rotary_emb without isolation.

        Default ``isolate_compression_model=True`` avoids this on the verification
        model.  Do not run compression on the main model with hooks unless
        isolation is disabled intentionally.
        """
        from exactkv.compressors.kvpress_knorm import (
            create_kvpress_knorm_adapter,
            snapshot_attention_model_state,
        )
        from exactkv.runtime.model_runtime import ModelRuntime
        from exactkv.runtime.prefill import prefill_to_full_state

        # Fresh runtime — this test permanently mutates the model under compression.
        runtime = ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)
        adapter = create_kvpress_knorm_adapter(
            runtime, compression_ratio=0.5, isolate_compression_model=False
        )
        full_state = prefill_to_full_state(runtime, _KV_PROMPT)
        before = snapshot_attention_model_state(runtime.model)
        adapter.compress(full_state)
        after = snapshot_attention_model_state(runtime.model)

        # Known kvpress behaviour: rotary_emb assignment is not restored on exit.
        assert before["rotary_emb_ids"] != after["rotary_emb_ids"], (
            "Expected rotary_emb mutation without isolation; if kvpress fixes this, "
            "revisit whether isolate_compression_model=True remains necessary."
        )
        assert before["hook_counts"] == after["hook_counts"]
        assert before["attn_module_ids"] == after["attn_module_ids"]
