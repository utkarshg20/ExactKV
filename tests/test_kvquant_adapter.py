"""V9 Phase D5: KVQuant simquant adapter gate.

Verifies:
  * Optional KVQuant path is not loaded on default import paths.
  * ``KVQuantSimAdapter`` when kvquant + quantizers pickle are available.
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

_QUANTIZERS_PATH = os.environ.get("EXACTKV_KVQUANT_QUANTIZERS", "")


def _kvquant_available() -> bool:
    try:
        return importlib.util.find_spec("kvquant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _quantizers_available() -> bool:
    return bool(_QUANTIZERS_PATH) and os.path.isfile(_QUANTIZERS_PATH)


_KVQUANT_ENV = _kvquant_available() and _quantizers_available() and torch.cuda.is_available()

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


# ---------------------------------------------------------------------------
# 1. Default import isolation
# ---------------------------------------------------------------------------

class TestKvquantNotOnDefaultImportPath:
    def test_import_exactkv_compressors_does_not_load_kvquant(self):
        sys.modules.pop("kvquant", None)
        import exactkv.compressors  # noqa: F401

        assert "kvquant" not in sys.modules

    def test_import_backend_adapter_does_not_load_kvquant(self):
        sys.modules.pop("kvquant", None)
        from exactkv.compressors.backend_adapter import BackendAdapter  # noqa: F401

        assert BackendAdapter is not None
        assert "kvquant" not in sys.modules

    def test_import_kvquant_adapter_module_does_not_load_kvquant(self):
        sys.modules.pop("kvquant", None)
        import exactkv.compressors.kvquant_adapter  # noqa: F401

        assert "kvquant" not in sys.modules

    def test_kvquant_not_in_default_registry(self):
        from exactkv.compressors import list_compressors

        names = list_compressors()
        assert "kvquant_sim_qwen05b" not in names
        assert not any(n.startswith("kvquant_sim") for n in names)


# ---------------------------------------------------------------------------
# 2. Adapter unit + smoke (skipped when KVQuant env unavailable)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float16"
PROMPTS = [
    "The capital of France is",
    "Write a Python function that adds two numbers.",
]
DRAFT_LENGTHS = [2, 4]
MAX_NEW_TOKENS = 8


@pytest.fixture(scope="module")
def kvquant_runtime():
    if not _KVQUANT_ENV:
        pytest.skip(
            "KVQuant env unavailable (kvquant import, EXACTKV_KVQUANT_QUANTIZERS, CUDA)"
        )
    from exactkv.runtime.model_runtime import ModelRuntime

    return ModelRuntime(model_name=MODEL_NAME, device="cuda", dtype=DTYPE)


@pytest.fixture(scope="module")
def kvquant_adapter(kvquant_runtime):
    from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter

    return create_kvquant_sim_adapter(kvquant_runtime, quantizers_path=_QUANTIZERS_PATH)


@pytest.mark.skipif(not _KVQUANT_ENV, reason="KVQuant env unavailable")
class TestKvquantAdapterImport:
    def test_constructing_adapter_imports_kvquant_lazily(self, kvquant_runtime):
        sys.modules.pop("kvquant", None)
        from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter

        create_kvquant_sim_adapter(kvquant_runtime, quantizers_path=_QUANTIZERS_PATH)
        assert importlib.util.find_spec("kvquant") is not None


@pytest.mark.skipif(not _KVQUANT_ENV, reason="KVQuant env unavailable")
class TestKvquantAdapterUnit:
    def test_factory_creates_adapter(self, kvquant_adapter):
        from exactkv.compressors.kvquant_adapter import KVQuantSimAdapter

        assert isinstance(kvquant_adapter, KVQuantSimAdapter)
        assert kvquant_adapter.name == "kvquant_sim_qwen05b"

    def test_capabilities_include_backend_identity(self, kvquant_adapter):
        caps = kvquant_adapter.capabilities
        assert caps.backend_name == "kvquant"
        assert caps.adapter_name == "KVQuantSimAdapter"
        assert caps.adapter_version == "0.1.0"
        assert caps.is_simulated is False
        assert caps.supports_quantization is True
        assert caps.supports_token_dropping is False
        assert caps.supports_real_bytes_claim is False
        assert caps.key_bit_width_label == "kvquant_sim_k"
        assert caps.value_bit_width_label == "kvquant_sim_v"
        assert "not kvquant deployment cuda" in caps.notes.lower()
        assert "not post-rope" in caps.notes.lower()
        assert "not in the default" in caps.notes.lower()

    def test_verifier_model_unmodified(self, kvquant_runtime, kvquant_adapter):
        from exactkv.compressors.kvquant_adapter import _has_quant_linear_sim

        assert not _has_quant_linear_sim(kvquant_runtime.model)
        assert _has_quant_linear_sim(kvquant_adapter._draft_model)

    def test_draft_is_deepcopy(self, kvquant_runtime, kvquant_adapter):
        assert kvquant_adapter._draft_model is not kvquant_runtime.model

    def test_compress_from_full_state(self, kvquant_runtime, kvquant_adapter):
        from exactkv.runtime.prefill import prefill_to_full_state

        full_state = prefill_to_full_state(kvquant_runtime, "Hello world")
        compressed = kvquant_adapter.compress(full_state)

        assert compressed.logical_seq_len == full_state.seq_len
        assert compressed.data["past_key_values"] is not None
        assert "__compressed_next_token_id__" in compressed.data

    def test_materialize_returns_forward_usable_cache(self, kvquant_runtime, kvquant_adapter):
        from exactkv.cache.utils import kv_seq_len
        from exactkv.runtime.prefill import prefill_to_full_state

        full_state = prefill_to_full_state(kvquant_runtime, "Hello world")
        compressed = kvquant_adapter.compress(full_state)
        cache = kvquant_adapter.materialize_for_draft(compressed)

        assert cache is not None
        assert kv_seq_len(cache) == full_state.seq_len

    def test_materialize_does_not_mutate_backend_data(self, kvquant_runtime, kvquant_adapter):
        from exactkv.runtime.prefill import prefill_to_full_state

        full_state = prefill_to_full_state(kvquant_runtime, "Hello world")
        compressed = kvquant_adapter.compress(full_state)
        stored_before = compressed.data["__stored_kv_bytes__"]

        kvquant_adapter.materialize_for_draft(compressed)

        assert compressed.data["__stored_kv_bytes__"] == stored_before

    def test_stats_fields_reconcile(self, kvquant_runtime, kvquant_adapter):
        from exactkv.runtime.prefill import prefill_to_full_state

        full_state = prefill_to_full_state(kvquant_runtime, "Hello world")
        compressed = kvquant_adapter.compress(full_state)
        stats = kvquant_adapter.stats(compressed)

        expected_total = (
            stats.stored_kv_bytes
            + stats.materialized_working_kv_bytes
            + stats.metadata_bytes
            + stats.temporary_workspace_bytes
        )
        assert stats.total_kv_footprint_bytes == expected_total
        _assert_no_forbidden_fields(dataclasses.asdict(stats), context="CompressionStats")

    def test_stored_bytes_honest_accounting(self, kvquant_runtime, kvquant_adapter):
        from exactkv.runtime.prefill import prefill_to_full_state

        full_state = prefill_to_full_state(kvquant_runtime, "Hello world")
        compressed = kvquant_adapter.compress(full_state)
        stats = kvquant_adapter.stats(compressed)

        assert stats.stored_kv_bytes == os.path.getsize(_QUANTIZERS_PATH)
        assert stats.compressed_bytes == max(stats.stored_kv_bytes + stats.metadata_bytes, 1)
        assert stats.materialized_working_kv_bytes == stats.full_bytes
        assert kvquant_adapter.capabilities.supports_real_bytes_claim is False


@pytest.mark.skipif(not _KVQUANT_ENV, reason="KVQuant env unavailable")
class TestKvquantExactnessSmoke:
    @pytest.mark.parametrize("prompt", PROMPTS)
    @pytest.mark.parametrize("draft_len", DRAFT_LENGTHS)
    def test_output_equals_full_greedy(
        self,
        kvquant_runtime,
        prompt: str,
        draft_len: int,
    ) -> None:
        from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy

        adapter = create_kvquant_sim_adapter(kvquant_runtime, quantizers_path=_QUANTIZERS_PATH)
        generator = ExactKVGenerator(runtime=kvquant_runtime, compressor=adapter, draft_len=draft_len)

        full_result = generate_full_greedy(kvquant_runtime, prompt, MAX_NEW_TOKENS)
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
        kvquant_runtime,
        prompt: str,
        draft_len: int,
    ) -> None:
        from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        adapter = create_kvquant_sim_adapter(kvquant_runtime, quantizers_path=_QUANTIZERS_PATH)
        generator = ExactKVGenerator(runtime=kvquant_runtime, compressor=adapter, draft_len=draft_len)
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
