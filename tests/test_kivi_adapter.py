"""V9 Phase D2: KIVI offline adapter gate.

Verifies:
  * Optional KIVI path is not loaded on default import paths.
  * ``KIVIOfflineAdapter`` when upstream models.utils_quant is available.
  * ExactKV smoke exactness gate on Qwen2.5-0.5B (2 prompts × 2 draft lengths).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

def _kivi_utils_available() -> bool:
    try:
        return importlib.util.find_spec("models.utils_quant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_KIVI_AVAILABLE = _kivi_utils_available()

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden_fields(d: dict, context: str = "") -> None:
    hits = _FORBIDDEN_FIELDS & d.keys()
    assert not hits, f"Forbidden performance fields {hits} found in {context or 'dict'}"


def _make_fake_full_state(
    num_layers: int = 2,
    seq_len: int = 8,
    head_dim: int = 64,
    num_heads: int = 2,
    dtype: torch.dtype = torch.float32,
) -> "FullKVState":
    from exactkv.cache.full_state import FullKVState

    pkv = tuple(
        (
            torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype),
            torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype),
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
# 1. Default import isolation
# ---------------------------------------------------------------------------

class TestKiviNotOnDefaultImportPath:
    def test_import_exactkv_compressors_does_not_load_kivi(self):
        sys.modules.pop("models", None)
        sys.modules.pop("models.utils_quant", None)
        import exactkv.compressors  # noqa: F401

        assert "models.utils_quant" not in sys.modules

    def test_import_backend_adapter_does_not_load_kivi(self):
        sys.modules.pop("models", None)
        sys.modules.pop("models.utils_quant", None)
        from exactkv.compressors.backend_adapter import BackendAdapter  # noqa: F401

        assert BackendAdapter is not None
        assert "models.utils_quant" not in sys.modules

    def test_import_kivi_adapter_module_does_not_load_kivi(self):
        sys.modules.pop("models", None)
        sys.modules.pop("models.utils_quant", None)
        import exactkv.compressors.kivi_adapter  # noqa: F401

        assert "models.utils_quant" not in sys.modules

    def test_kivi_not_in_default_registry(self):
        from exactkv.compressors import list_compressors

        names = list_compressors()
        assert "kivi_offline_k2_v2" not in names
        assert not any(n.startswith("kivi_offline") for n in names)


# ---------------------------------------------------------------------------
# 2. Adapter unit + smoke (skipped when KIVI unavailable)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
PROMPTS = [
    "The capital of France is",
    "Write a Python function that adds two numbers.",
]
DRAFT_LENGTHS = [2, 4]
MAX_NEW_TOKENS = 8


@pytest.fixture(scope="module")
def kivi_runtime():
    if not _KIVI_AVAILABLE:
        pytest.skip("KIVI models.utils_quant not importable (set PYTHONPATH to KIVI repo)")
    from exactkv.runtime.model_runtime import ModelRuntime

    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture
def kivi_adapter_unit():
    if not _KIVI_AVAILABLE:
        pytest.skip("KIVI models.utils_quant not importable")
    from unittest.mock import MagicMock

    from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter

    runtime = MagicMock()
    runtime.device = torch.device("cpu")
    runtime.model = MagicMock()
    runtime.model.config = MagicMock(hidden_size=896, num_attention_heads=14)
    logits = torch.zeros(1, 1, 151936)
    logits[0, 0, 100] = 10.0
    runtime.forward.return_value = MagicMock(
        logits=logits,
        past_key_values=None,
    )
    return create_kivi_offline_adapter(runtime, head_dim=64, k_bits=2, v_bits=2)


@pytest.fixture
def kivi_adapter(kivi_runtime):
    from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter

    return create_kivi_offline_adapter(kivi_runtime, head_dim=64, k_bits=2, v_bits=2)


@pytest.mark.skipif(not _KIVI_AVAILABLE, reason="KIVI models.utils_quant not importable")
class TestKiviAdapterImport:
    def test_constructing_adapter_imports_kivi_lazily(self, kivi_runtime):
        sys.modules.pop("models.utils_quant", None)
        from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter

        create_kivi_offline_adapter(kivi_runtime, head_dim=64)
        assert importlib.util.find_spec("models.utils_quant") is not None


@pytest.mark.skipif(not _KIVI_AVAILABLE, reason="KIVI models.utils_quant not importable")
class TestKiviAdapterUnit:
    def test_factory_creates_adapter(self, kivi_adapter_unit):
        from exactkv.compressors.kivi_adapter import KIVIOfflineAdapter

        assert isinstance(kivi_adapter_unit, KIVIOfflineAdapter)
        assert kivi_adapter_unit.name == "kivi_offline_k2_v2"

    def test_capabilities_include_backend_identity(self, kivi_adapter_unit):
        caps = kivi_adapter_unit.capabilities
        assert caps.backend_name == "kivi"
        assert caps.adapter_name == "KIVIOfflineAdapter"
        assert caps.adapter_version == "0.1.0"
        assert caps.is_simulated is False
        assert caps.supports_quantization is True
        assert caps.supports_token_dropping is False
        assert caps.supports_real_bytes_claim is False
        assert caps.key_bit_width_label == "kivi_k2_offline"
        assert caps.value_bit_width_label == "kivi_v2_offline"
        assert "not kivi production" in caps.notes.lower()
        assert "not kvquant" in caps.notes.lower()
        assert "not packed-bit" in caps.notes.lower()

    def test_compress_does_not_mutate_full_state(self, kivi_adapter_unit):
        from exactkv.cache.utils import kv_total_bytes

        full_state = _make_fake_full_state(head_dim=64)
        orig_bytes = kv_total_bytes(full_state.past_key_values)
        orig_tensors = [(k.clone(), v.clone()) for k, v in full_state.past_key_values]

        compressed = kivi_adapter_unit.compress(full_state)

        assert kv_total_bytes(full_state.past_key_values) == orig_bytes
        for (orig_k, orig_v), (cur_k, cur_v) in zip(orig_tensors, full_state.past_key_values):
            assert torch.equal(orig_k, cur_k)
            assert torch.equal(orig_v, cur_v)
        assert compressed.logical_seq_len == full_state.seq_len

    def test_materialize_returns_forward_usable_cache(self, kivi_adapter_unit):
        from exactkv.cache.utils import kv_seq_len

        full_state = _make_fake_full_state(head_dim=64)
        compressed = kivi_adapter_unit.compress(full_state)
        cache = kivi_adapter_unit.materialize_for_draft(compressed)

        assert cache is not None
        assert kv_seq_len(cache) == full_state.seq_len

    def test_materialize_does_not_mutate_backend_data(self, kivi_adapter_unit):
        full_state = _make_fake_full_state(head_dim=64)
        compressed = kivi_adapter_unit.compress(full_state)
        stored_before = compressed.data["__stored_kv_bytes__"]

        kivi_adapter_unit.materialize_for_draft(compressed)

        assert compressed.data["__stored_kv_bytes__"] == stored_before

    def test_stats_fields_reconcile(self, kivi_adapter_unit):
        full_state = _make_fake_full_state(head_dim=64)
        compressed = kivi_adapter_unit.compress(full_state)
        stats = kivi_adapter_unit.stats(compressed)

        expected_total = (
            stats.stored_kv_bytes
            + stats.materialized_working_kv_bytes
            + stats.metadata_bytes
            + stats.temporary_workspace_bytes
        )
        assert stats.total_kv_footprint_bytes == expected_total
        _assert_no_forbidden_fields(dataclasses.asdict(stats), context="CompressionStats")

    def test_stored_bytes_honest_accounting(self, kivi_adapter_unit):
        full_state = _make_fake_full_state(head_dim=64, seq_len=16, num_layers=4)
        compressed = kivi_adapter_unit.compress(full_state)
        stats = kivi_adapter_unit.stats(compressed)

        assert stats.stored_kv_bytes > 0
        assert stats.compressed_bytes == max(stats.stored_kv_bytes + stats.metadata_bytes, 1)
        assert stats.materialized_working_kv_bytes == stats.full_bytes
        assert kivi_adapter_unit.capabilities.supports_real_bytes_claim is False

    def test_unsupported_head_dim_raises(self, kivi_adapter_unit):
        full_state = _make_fake_full_state(head_dim=32)
        with pytest.raises(ValueError, match="head_dim"):
            kivi_adapter_unit.compress(full_state)


@pytest.mark.skipif(not _KIVI_AVAILABLE, reason="KIVI models.utils_quant not importable")
class TestKiviExactnessSmoke:
    @pytest.mark.parametrize("prompt", PROMPTS)
    @pytest.mark.parametrize("draft_len", DRAFT_LENGTHS)
    def test_output_equals_full_greedy(
        self,
        kivi_runtime,
        prompt: str,
        draft_len: int,
    ) -> None:
        from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy

        adapter = create_kivi_offline_adapter(kivi_runtime, head_dim=64, k_bits=2, v_bits=2)
        generator = ExactKVGenerator(runtime=kivi_runtime, compressor=adapter, draft_len=draft_len)

        full_result = generate_full_greedy(kivi_runtime, prompt, MAX_NEW_TOKENS)
        exactkv_result = generator.generate(prompt, MAX_NEW_TOKENS)

        expected = full_result.generated_ids
        actual = exactkv_result.output_ids

        assert actual.shape == expected.shape
        mismatch = (actual != expected).nonzero(as_tuple=True)
        assert len(mismatch[0]) == 0, (
            f"exactkv_output_ids != full_output_ids\n"
            f"  expected: {expected.tolist()}\n"
            f"  actual:   {actual.tolist()}"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    @pytest.mark.parametrize("draft_len", DRAFT_LENGTHS)
    def test_acceptance_counts_reconcile(
        self,
        kivi_runtime,
        prompt: str,
        draft_len: int,
    ) -> None:
        from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        adapter = create_kivi_offline_adapter(kivi_runtime, head_dim=64, k_bits=2, v_bits=2)
        generator = ExactKVGenerator(runtime=kivi_runtime, compressor=adapter, draft_len=draft_len)
        result = generator.generate(prompt, MAX_NEW_TOKENS)

        assert result.num_rounds > 0
        trace_accepted = sum(t.acceptance.num_accepted for t in result.traces)
        trace_rejected = sum(t.acceptance.num_rejected for t in result.traces)
        trace_corrections = sum(
            1 for t in result.traces if t.acceptance.correction_token is not None
        )

        assert result.total_accepted == trace_accepted
        assert result.total_rejected == trace_rejected
        assert result.total_corrections == trace_corrections

        for trace in result.traces:
            assert trace.full_seq_len_after == trace.compressed_seq_len_after

        denom = result.total_accepted + result.total_rejected
        if denom > 0:
            assert result.acceptance_rate == pytest.approx(result.total_accepted / denom)
