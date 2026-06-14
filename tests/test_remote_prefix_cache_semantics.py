"""Tests for Phase 11H remote prefix cache semantics and loopback mock."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from exactkv.cache.dual_cache import CacheResidency
from exactkv.cache.remote_prefix import (
    LoopbackPrefixCache,
    PrefixCacheEntry,
    PrefixCacheMode,
    PrefixCacheStatus,
    PrefixIdentity,
    PrefixRestorePlan,
    build_prefix_identity,
    build_prefix_restore_plan,
    build_remote_placeholder_entry,
    check_prefix_compatibility,
    smoke_loopback_prefix,
    validate_prefix_cache_entry,
    validate_prefix_restore_plan,
)
from exactkv.cache.storage import FileKVStorageBackend, InMemoryKVStorageBackend

_DOC = Path(__file__).resolve().parents[1] / "docs" / "REMOTE_PREFIX_CACHE_SEMANTICS.md"


def _identity(**overrides: object) -> PrefixIdentity:
    base = build_prefix_identity(
        model_id="model-a",
        tokenizer_id="tok-a",
        prompt="test prompt",
        token_ids=[10, 20, 30],
        payload={"k": torch.randn(1, 2, 4)},
    )
    data = base.to_dict()
    data.update(overrides)
    return PrefixIdentity.from_dict(data)


def _payload() -> dict[str, torch.Tensor]:
    return {"k": torch.randn(1, 2, 4), "v": torch.randn(1, 2, 4, 8)}


def test_prefix_identity_serializes() -> None:
    identity = _identity()
    restored = PrefixIdentity.from_dict(identity.to_dict())
    assert restored == identity
    json.dumps(identity.to_dict(), sort_keys=True)


def test_prefix_cache_entry_serializes() -> None:
    identity = _identity()
    entry = build_remote_placeholder_entry(identity)
    restored = PrefixCacheEntry.from_dict(entry.to_dict())
    assert restored.identity == identity
    assert restored.cache_mode is PrefixCacheMode.REMOTE_PLACEHOLDER


def test_prefix_restore_plan_serializes() -> None:
    identity = _identity()
    entry = build_remote_placeholder_entry(identity)
    plan = build_prefix_restore_plan(entry, identity)
    restored = PrefixRestorePlan.from_dict(plan.to_dict())
    assert restored.compatible == plan.compatible
    assert restored.entry.identity == identity


def test_compatible_identity_allows_restore() -> None:
    identity = _identity()
    entry = build_remote_placeholder_entry(identity)
    plan = build_prefix_restore_plan(entry, identity)
    assert plan.compatible
    assert plan.restore_allowed
    assert not plan.fallback_required
    assert validate_prefix_restore_plan(plan) == []


def test_model_mismatch_blocks_restore() -> None:
    stored = _identity()
    expected = _identity(model_id="model-b")
    entry = build_remote_placeholder_entry(stored)
    plan = build_prefix_restore_plan(entry, expected)
    assert not plan.compatible
    assert plan.fallback_required
    assert not plan.restore_allowed
    assert any("model_id" in r for r in plan.compatibility_reasons)


def test_tokenizer_mismatch_blocks_restore() -> None:
    stored = _identity()
    expected = _identity(tokenizer_id="tok-b")
    entry = build_remote_placeholder_entry(stored)
    plan = build_prefix_restore_plan(entry, expected)
    assert not plan.restore_allowed
    assert any("tokenizer_id" in r for r in plan.compatibility_reasons)


def test_prefix_length_mismatch_blocks_restore() -> None:
    stored = _identity()
    expected = _identity(prefix_token_count=99)
    entry = build_remote_placeholder_entry(stored)
    plan = build_prefix_restore_plan(entry, expected)
    assert not plan.restore_allowed
    assert any("prefix_token_count" in r for r in plan.compatibility_reasons)


def test_cache_version_mismatch_blocks_restore() -> None:
    stored = _identity()
    expected = _identity(cache_version="2")
    entry = build_remote_placeholder_entry(stored)
    plan = build_prefix_restore_plan(entry, expected)
    assert not plan.restore_allowed
    assert any("cache_version" in r for r in plan.compatibility_reasons)


def test_loopback_in_memory_store_retrieve() -> None:
    backend = InMemoryKVStorageBackend()
    cache = LoopbackPrefixCache(backend, residency=CacheResidency.CPU)
    identity = build_prefix_identity(
        model_id="m",
        tokenizer_id="t",
        prompt="p",
        token_ids=[1, 2],
        payload=_payload(),
    )
    entry = cache.store(identity, _payload())
    assert entry.status is PrefixCacheStatus.LOOPBACK_MOCK
    assert cache.exists(identity)
    restored = cache.retrieve(identity)
    assert restored.storage_handle == entry.storage_handle
    plan = cache.build_restore_plan(identity)
    assert plan.restore_allowed
    assert validate_prefix_restore_plan(plan) == []


def test_loopback_file_store_retrieve(tmp_path: Path) -> None:
    backend = FileKVStorageBackend(tmp_path)
    plan = smoke_loopback_prefix(backend, residency=CacheResidency.DISK)
    assert plan.restore_allowed
    assert plan.entry.storage_metadata.residency is CacheResidency.DISK


def test_remote_placeholder_cannot_be_active() -> None:
    identity = _identity()
    entry = build_remote_placeholder_entry(identity)
    assert entry.remote_placeholder_active is False
    entry.remote_placeholder_active = True
    errors = validate_prefix_cache_entry(entry)
    assert any("remote_placeholder_active" in e for e in errors)


def test_missing_remote_claim_note_fails() -> None:
    identity = _identity()
    entry = build_remote_placeholder_entry(identity)
    entry.claim_note = "missing keywords"
    errors = validate_prefix_cache_entry(entry)
    assert any("placeholder" in e or "remote" in e for e in errors)


def test_experimental_active_fails() -> None:
    identity = _identity()
    entry = build_remote_placeholder_entry(identity)
    entry.status = PrefixCacheStatus.EXPERIMENTAL_ACTIVE
    errors = validate_prefix_cache_entry(entry)
    assert any("EXPERIMENTAL_ACTIVE" in e for e in errors)


def test_check_prefix_compatibility_empty_when_match() -> None:
    identity = _identity()
    entry = build_remote_placeholder_entry(identity)
    assert check_prefix_compatibility(identity, entry) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "remote-prefix-cache semantics spike",
        "not a remote prefix cache runtime",
        "does not perform network i/o",
        "lmcache is not imported",
        "vllm is not imported",
        "production serving",
        "generation and verification behavior is unchanged",
        "throughput",
        "loopback",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "production-ready remote prefix", "lmcache integrated today"):
        assert phrase not in text


def test_no_lmcache_or_vllm_dependency() -> None:
    import exactkv.cache.remote_prefix as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import lmcache" not in source
    assert "from lmcache" not in source
    assert "import vllm" not in source
    assert "from vllm" not in source


def test_package_exports() -> None:
    from exactkv.cache import LoopbackPrefixCache, PrefixIdentity, smoke_loopback_prefix

    backend = InMemoryKVStorageBackend()
    plan = smoke_loopback_prefix(backend)
    assert plan.restore_allowed
