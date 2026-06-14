"""Tests for Phase 12A HF full-KV restore smoke (no model download by default)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from exactkv.cache.dual_cache import CacheResidency
from exactkv.cache.hf_kv_restore import (
    CLAIM_NOTE,
    EXPERIMENT_046_ID,
    FORBIDDEN_CLAIMS,
    HfKvRestoreError,
    PrefillKVCapture,
    build_storage_payload_from_cache,
    detect_hf_cache_format,
    restore_cache_from_storage_payload,
    store_prefill_payload,
    summarize_hf_cache,
    validate_exp046_report,
)
from exactkv.cache.storage import (
    FileKVStorageBackend,
    InMemoryKVStorageBackend,
    KVStorageHandle,
)
from exactkv.cache.utils import extract_kv_tensors, kv_seq_len

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_046_FULL_KV_RESTORE_SMOKE.md"


def _legacy_tuple_cache(*, layers: int = 2, seq: int = 4) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (torch.randn(1, 2, seq, 8), torch.randn(1, 2, seq, 8)) for _ in range(layers)
    )


def _capture_from_cache(cache: object) -> PrefillKVCapture:
    summary = summarize_hf_cache(cache)
    return PrefillKVCapture(
        model_name="test-model",
        device="cpu",
        dtype="torch.float32",
        prompt_ids=torch.tensor([[1, 2, 3]]),
        next_token_id=42,
        past_key_values=cache,
        cache_summary=summary,
    )


def test_detect_legacy_tuple_format() -> None:
    cache = _legacy_tuple_cache()
    assert detect_hf_cache_format(cache) == "tuple"


def test_unsupported_cache_format_raises() -> None:
    class WeirdCache:
        pass

    with pytest.raises(HfKvRestoreError, match="Unsupported"):
        detect_hf_cache_format(WeirdCache())


def test_summarize_shape_dtype_accounting() -> None:
    cache = _legacy_tuple_cache(layers=3, seq=5)
    summary = summarize_hf_cache(cache)
    assert summary.cache_format == "tuple"
    assert summary.layer_count == 3
    assert summary.seq_len == 5
    assert "float" in summary.dtype_summary


def test_storage_payload_roundtrip_in_memory() -> None:
    cache = _legacy_tuple_cache()
    capture = _capture_from_cache(cache)
    payload = build_storage_payload_from_cache(capture)
    backend = InMemoryKVStorageBackend()
    handle = KVStorageHandle(namespace="test", key="roundtrip", version="1")
    store_prefill_payload(backend, handle, payload, residency=CacheResidency.CPU)
    loaded = backend.get(handle).payload
    restored, next_id = restore_cache_from_storage_payload(loaded)
    assert next_id == 42
    assert detect_hf_cache_format(restored) == "tuple"
    assert kv_seq_len(restored) == kv_seq_len(cache)


def test_storage_payload_roundtrip_file(tmp_path: Path) -> None:
    cache = _legacy_tuple_cache()
    capture = _capture_from_cache(cache)
    payload = build_storage_payload_from_cache(capture)
    backend = FileKVStorageBackend(tmp_path)
    handle = KVStorageHandle(namespace="test", key="file", version="1")
    store_prefill_payload(backend, handle, payload, residency=CacheResidency.DISK)
    loaded = backend.get(handle).payload
    restored, _ = restore_cache_from_storage_payload(loaded)
    k0, v0, _ = extract_kv_tensors(cache)
    k1, v1, _ = extract_kv_tensors(restored)
    assert torch.allclose(k0[0], k1[0])
    assert torch.allclose(v0[0], v1[0])


def test_rebuild_from_payload_matches_legacy() -> None:
    cache = _legacy_tuple_cache()
    capture = _capture_from_cache(cache)
    payload = build_storage_payload_from_cache(capture)
    restored, _ = restore_cache_from_storage_payload(payload)
    assert isinstance(restored, tuple)
    assert len(restored) == capture.cache_summary.layer_count


def test_validate_exp046_report_schema() -> None:
    report = {
        "experiment_id": EXPERIMENT_046_ID,
        "model": "Qwen/Qwen2.5-0.5B",
        "device": "cpu",
        "dtype": "float32",
        "prompt_count": 4,
        "storage_backends_tested": ["in_memory_kv_storage"],
        "cache_format_detected": "tuple",
        "token_exact_match_count": 4,
        "failures_count": 0,
        "per_prompt": [],
        "claim_note": CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    assert validate_exp046_report(report) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "full-kv restore smoke",
        "not a serving runtime",
        "vllm",
        "lmcache",
        "remote prefix",
        "generation and verification behavior is unchanged",
        "throughput",
        "vericache",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
        assert phrase not in text
