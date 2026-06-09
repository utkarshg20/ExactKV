"""V6 Phase C scaffold: kvpress safety gates (no KVPressKnormAdapter yet).

Verifies:
  * Optional ``kvpress`` extra is defined but not loaded on default import paths.
  * ``BackendAdapter.verification_mode()`` default is a no-op.
  * ``ExactKVGenerator`` wraps verification inside ``verification_mode()`` when present.
  * Explicit ``import kvpress`` (when installed) does not break pass-through smoke.
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
