"""Remote prefix cache semantics and loopback mock (Phase 11H).

Prefix identity, compatibility checks, and a loopback store/retrieve helper built on
``KVStorageBackend``. **No network I/O** and **not** wired into ``ExactKVGenerator``.

This is a remote-prefix-cache semantics spike, not a remote prefix cache runtime.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch

from exactkv.cache.dual_cache import CacheResidency
from exactkv.cache.storage import (
    KVStorageBackend,
    KVStorageHandle,
    KVStorageMetadata,
    KVStorageNotFoundError,
    StoredKVEntry,
    build_verifier_storage_metadata,
    iter_tensors,
    summarize_tensor_payload,
)

_CLAIM_NOTE = (
    "Remote prefix cache semantics spike (Phase 11H). Loopback mock only — "
    "no network I/O. No performance, deployment, or remote-runtime claims."
)

_REMOTE_PLACEHOLDER_CLAIM = (
    "Remote prefix placeholder semantics only. Not active remote prefix caching. "
    "No network I/O. Loopback mock is the only executable path in Phase 11H."
)

_FORBIDDEN_CLAIM_TERMS = (
    "speedup",
    "throughput improvement",
    "latency improvement",
    "memory savings",
    "production serving",
    "remote prefix cache runtime",
    "remote prefix caching exists",
)

_NEGATION_PREFIXES = ("no ", "not ", "without ", "never ", "non-", "does not ")


class PrefixCacheMode(str, Enum):
    """How a prefix cache entry is backed (metadata only for non-loopback modes)."""

    LOCAL_LOOPBACK = "LOCAL_LOOPBACK"
    REMOTE_PLACEHOLDER = "REMOTE_PLACEHOLDER"
    LMCACHE_FUTURE = "LMCACHE_FUTURE"


class PrefixCacheStatus(str, Enum):
    """Lifecycle status for a prefix cache entry or restore path."""

    CONTRACT_ONLY = "CONTRACT_ONLY"
    LOOPBACK_MOCK = "LOOPBACK_MOCK"
    REMOTE_BLOCKED = "REMOTE_BLOCKED"
    EXPERIMENTAL_ACTIVE = "EXPERIMENTAL_ACTIVE"


@dataclass
class PrefixIdentity:
    """Stable metadata identifying a prefix cache entry."""

    model_id: str
    tokenizer_id: str
    prompt_hash: str
    token_ids_hash: str
    prefix_token_count: int
    dtype_summary: str = ""
    shape_summary: str = ""
    cache_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrefixIdentity:
        return cls(
            model_id=str(data["model_id"]),
            tokenizer_id=str(data["tokenizer_id"]),
            prompt_hash=str(data["prompt_hash"]),
            token_ids_hash=str(data["token_ids_hash"]),
            prefix_token_count=int(data.get("prefix_token_count", 0)),
            dtype_summary=str(data.get("dtype_summary", "")),
            shape_summary=str(data.get("shape_summary", "")),
            cache_version=str(data.get("cache_version", "1")),
        )

    def storage_key(self) -> str:
        return f"{self.prompt_hash}__{self.token_ids_hash}"


@dataclass
class PrefixCacheEntry:
    """Prefix cache record linking identity to a stored KV handle."""

    identity: PrefixIdentity
    storage_handle: KVStorageHandle
    storage_metadata: KVStorageMetadata
    cache_mode: PrefixCacheMode = PrefixCacheMode.LOCAL_LOOPBACK
    status: PrefixCacheStatus = PrefixCacheStatus.LOOPBACK_MOCK
    claim_note: str = _CLAIM_NOTE
    remote_placeholder_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "storage_handle": asdict(self.storage_handle),
            "storage_metadata": self.storage_metadata.to_dict(),
            "cache_mode": self.cache_mode.value,
            "status": self.status.value,
            "claim_note": self.claim_note,
            "remote_placeholder_active": self.remote_placeholder_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrefixCacheEntry:
        return cls(
            identity=PrefixIdentity.from_dict(data["identity"]),
            storage_handle=KVStorageHandle(**data["storage_handle"]),
            storage_metadata=KVStorageMetadata.from_dict(data["storage_metadata"]),
            cache_mode=PrefixCacheMode(data.get("cache_mode", PrefixCacheMode.LOCAL_LOOPBACK.value)),
            status=PrefixCacheStatus(data.get("status", PrefixCacheStatus.LOOPBACK_MOCK.value)),
            claim_note=str(data.get("claim_note", _CLAIM_NOTE)),
            remote_placeholder_active=bool(data.get("remote_placeholder_active", False)),
        )


@dataclass
class PrefixRestorePlan:
    """Safe restore decision for a prefix cache entry (metadata only)."""

    entry: PrefixCacheEntry
    compatible: bool
    compatibility_reasons: list[str] = field(default_factory=list)
    fallback_required: bool = True
    restore_allowed: bool = False
    claim_note: str = _CLAIM_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "compatible": self.compatible,
            "compatibility_reasons": list(self.compatibility_reasons),
            "fallback_required": self.fallback_required,
            "restore_allowed": self.restore_allowed,
            "claim_note": self.claim_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrefixRestorePlan:
        return cls(
            entry=PrefixCacheEntry.from_dict(data["entry"]),
            compatible=bool(data.get("compatible", False)),
            compatibility_reasons=list(data.get("compatibility_reasons", [])),
            fallback_required=bool(data.get("fallback_required", True)),
            restore_allowed=bool(data.get("restore_allowed", False)),
            claim_note=str(data.get("claim_note", _CLAIM_NOTE)),
        )


def hash_prompt(prompt: str) -> str:
    """Stable short hash for a prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def hash_token_ids(token_ids: list[int]) -> str:
    """Stable short hash for a token-id prefix."""
    payload = ",".join(str(t) for t in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_prefix_identity(
    *,
    model_id: str,
    tokenizer_id: str,
    prompt: str,
    token_ids: list[int],
    payload: Any | None = None,
    cache_version: str = "1",
) -> PrefixIdentity:
    """Construct ``PrefixIdentity`` with optional dtype/shape from a tensor payload."""
    dtype_summary = ""
    shape_summary = ""
    if payload is not None:
        _, _, dtype_summary, shape_summary = summarize_tensor_payload(payload)
    return PrefixIdentity(
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        prompt_hash=hash_prompt(prompt),
        token_ids_hash=hash_token_ids(token_ids),
        prefix_token_count=len(token_ids),
        dtype_summary=dtype_summary,
        shape_summary=shape_summary,
        cache_version=cache_version,
    )


def check_prefix_compatibility(
    expected: PrefixIdentity,
    entry: PrefixCacheEntry,
) -> list[str]:
    """Return human-readable compatibility violations (empty if compatible)."""
    errors: list[str] = []
    stored = entry.identity

    if expected.model_id != stored.model_id:
        errors.append(f"model_id mismatch: expected={expected.model_id}, stored={stored.model_id}")
    if expected.tokenizer_id != stored.tokenizer_id:
        errors.append(
            f"tokenizer_id mismatch: expected={expected.tokenizer_id}, "
            f"stored={stored.tokenizer_id}"
        )
    if expected.prefix_token_count != stored.prefix_token_count:
        errors.append(
            f"prefix_token_count mismatch: expected={expected.prefix_token_count}, "
            f"stored={stored.prefix_token_count}"
        )
    if expected.cache_version != stored.cache_version:
        errors.append(
            f"cache_version mismatch: expected={expected.cache_version}, "
            f"stored={stored.cache_version}"
        )
    if expected.prompt_hash != stored.prompt_hash:
        errors.append("prompt_hash mismatch")
    if expected.token_ids_hash != stored.token_ids_hash:
        errors.append("token_ids_hash mismatch")

    if expected.dtype_summary and stored.dtype_summary:
        if expected.dtype_summary != stored.dtype_summary:
            errors.append(
                f"dtype_summary mismatch: expected={expected.dtype_summary}, "
                f"stored={stored.dtype_summary}"
            )
    if expected.shape_summary and stored.shape_summary:
        if expected.shape_summary != stored.shape_summary:
            errors.append(
                f"shape_summary mismatch: expected={expected.shape_summary}, "
                f"stored={stored.shape_summary}"
            )

    meta = entry.storage_metadata
    if stored.dtype_summary and meta.dtype_summary and stored.dtype_summary != meta.dtype_summary:
        errors.append("storage_metadata dtype_summary incompatible with identity")
    if stored.shape_summary and meta.shape_summary and stored.shape_summary != meta.shape_summary:
        errors.append("storage_metadata shape_summary incompatible with identity")

    return errors


def build_prefix_restore_plan(
    entry: PrefixCacheEntry,
    expected: PrefixIdentity,
    *,
    claim_note: str | None = None,
) -> PrefixRestorePlan:
    """Build a restore plan from stored entry vs expected identity."""
    reasons = check_prefix_compatibility(expected, entry)
    compatible = len(reasons) == 0
    return PrefixRestorePlan(
        entry=entry,
        compatible=compatible,
        compatibility_reasons=reasons,
        fallback_required=not compatible,
        restore_allowed=compatible,
        claim_note=claim_note or entry.claim_note,
    )


def _encodes_positive_forbidden_claim(text_lower: str, term: str) -> bool:
    start = 0
    while True:
        pos = text_lower.find(term, start)
        if pos == -1:
            return False
        window = text_lower[max(0, pos - 40):pos]
        if not any(neg in window for neg in _NEGATION_PREFIXES):
            return True
        start = pos + len(term)


def validate_prefix_identity(identity: PrefixIdentity) -> list[str]:
    errors: list[str] = []
    if not identity.model_id.strip():
        errors.append("model_id required")
    if not identity.tokenizer_id.strip():
        errors.append("tokenizer_id required")
    if identity.prefix_token_count <= 0:
        errors.append("prefix_token_count must be positive")
    if not identity.prompt_hash.strip():
        errors.append("prompt_hash required")
    if not identity.token_ids_hash.strip():
        errors.append("token_ids_hash required")
    return errors


def validate_prefix_cache_entry(entry: PrefixCacheEntry) -> list[str]:
    """Validate prefix cache entry invariants and claim guards."""
    errors = validate_prefix_identity(entry.identity)

    if entry.status is PrefixCacheStatus.EXPERIMENTAL_ACTIVE:
        errors.append("EXPERIMENTAL_ACTIVE is forbidden in Phase 11H")

    if entry.remote_placeholder_active:
        errors.append("remote_placeholder_active must remain False in Phase 11H")

    if entry.cache_mode is PrefixCacheMode.REMOTE_PLACEHOLDER:
        note = entry.claim_note.lower()
        if not note.strip():
            errors.append("REMOTE_PLACEHOLDER requires claim_note")
        elif "placeholder" not in note and "remote" not in note:
            errors.append("REMOTE_PLACEHOLDER requires remote/placeholder caveat in claim_note")
        if entry.status is PrefixCacheStatus.LOOPBACK_MOCK:
            errors.append("REMOTE_PLACEHOLDER cannot use LOOPBACK_MOCK status")

    if entry.cache_mode is PrefixCacheMode.LMCACHE_FUTURE:
        if "lmcache" not in entry.claim_note.lower() and "future" not in entry.claim_note.lower():
            errors.append("LMCACHE_FUTURE requires future/lmcache caveat in claim_note")

    if not entry.claim_note.strip():
        errors.append("claim_note required on prefix cache entry")

    note_lower = entry.claim_note.lower()
    for term in _FORBIDDEN_CLAIM_TERMS:
        if _encodes_positive_forbidden_claim(note_lower, term):
            errors.append(f"claim_note must not encode positive forbidden claim: {term}")

    return errors


def validate_prefix_restore_plan(plan: PrefixRestorePlan) -> list[str]:
    """Validate restore plan consistency."""
    errors = validate_prefix_cache_entry(plan.entry)

    if plan.compatible and plan.compatibility_reasons:
        errors.append("compatible=True cannot have compatibility_reasons")
    if not plan.compatible and not plan.compatibility_reasons:
        errors.append("compatible=False requires compatibility_reasons")
    if plan.compatible and plan.fallback_required:
        errors.append("compatible entry must not require fallback")
    if not plan.compatible and not plan.fallback_required:
        errors.append("incompatible entry must require fallback")
    if plan.compatible and not plan.restore_allowed:
        errors.append("compatible entry must allow restore")
    if not plan.compatible and plan.restore_allowed:
        errors.append("incompatible entry must not allow restore")

    if not plan.claim_note.strip():
        errors.append("claim_note required on prefix restore plan")

    return errors


class LoopbackPrefixCache:
    """Loopback prefix cache using an existing ``KVStorageBackend`` (no network I/O)."""

    def __init__(
        self,
        backend: KVStorageBackend,
        *,
        residency: CacheResidency = CacheResidency.CPU,
        backend_label: str = "loopback_prefix_cache",
        claim_note: str = _CLAIM_NOTE,
    ) -> None:
        self._backend = backend
        self._residency = residency
        self._backend_label = backend_label
        self._claim_note = claim_note

    def _handle_for(self, identity: PrefixIdentity) -> KVStorageHandle:
        return KVStorageHandle(
            namespace=f"prefix/{identity.model_id}",
            key=identity.storage_key(),
            version=identity.cache_version,
        )

    def store(self, identity: PrefixIdentity, payload: Any) -> PrefixCacheEntry:
        """Store a prefix payload and return a cache entry record."""
        id_errors = validate_prefix_identity(identity)
        if id_errors:
            raise ValueError("; ".join(id_errors))

        handle = self._handle_for(identity)
        metadata = build_verifier_storage_metadata(
            payload,
            residency=self._residency,
            backend_name=self._backend_label,
            claim_note=self._claim_note,
        )
        self._backend.put(handle, payload, metadata)
        entry = PrefixCacheEntry(
            identity=identity,
            storage_handle=handle,
            storage_metadata=metadata,
            cache_mode=PrefixCacheMode.LOCAL_LOOPBACK,
            status=PrefixCacheStatus.LOOPBACK_MOCK,
            claim_note=self._claim_note,
            remote_placeholder_active=False,
        )
        entry_errors = validate_prefix_cache_entry(entry)
        if entry_errors:
            raise ValueError("; ".join(entry_errors))
        return entry

    def retrieve(self, identity: PrefixIdentity) -> PrefixCacheEntry:
        """Retrieve a stored prefix entry by identity."""
        handle = self._handle_for(identity)
        stored: StoredKVEntry = self._backend.get(handle)
        entry = PrefixCacheEntry(
            identity=identity,
            storage_handle=handle,
            storage_metadata=stored.metadata,
            cache_mode=PrefixCacheMode.LOCAL_LOOPBACK,
            status=PrefixCacheStatus.LOOPBACK_MOCK,
            claim_note=self._claim_note,
            remote_placeholder_active=False,
        )
        return entry

    def exists(self, identity: PrefixIdentity) -> bool:
        return self._backend.exists(self._handle_for(identity))

    def build_restore_plan(
        self,
        identity: PrefixIdentity,
        *,
        expected: PrefixIdentity | None = None,
    ) -> PrefixRestorePlan:
        """Retrieve entry (if present) and build a compatibility restore plan."""
        expected_identity = expected or identity
        if not self.exists(identity):
            placeholder = PrefixCacheEntry(
                identity=identity,
                storage_handle=self._handle_for(identity),
                storage_metadata=build_verifier_storage_metadata(
                    {},
                    residency=self._residency,
                    backend_name=self._backend_label,
                    claim_note=self._claim_note,
                ),
                cache_mode=PrefixCacheMode.LOCAL_LOOPBACK,
                status=PrefixCacheStatus.LOOPBACK_MOCK,
                claim_note=self._claim_note,
            )
            return PrefixRestorePlan(
                entry=placeholder,
                compatible=False,
                compatibility_reasons=["prefix cache entry not found"],
                fallback_required=True,
                restore_allowed=False,
                claim_note=self._claim_note,
            )
        entry = self.retrieve(identity)
        return build_prefix_restore_plan(entry, expected_identity, claim_note=self._claim_note)


def smoke_loopback_prefix(
    backend: KVStorageBackend,
    *,
    residency: CacheResidency = CacheResidency.CPU,
) -> PrefixRestorePlan:
    """Tiny synthetic tensor smoke: store, retrieve, restore plan."""
    payload = {"k": torch.randn(1, 2, 4), "v": torch.randn(1, 2, 4, 8)}
    identity = build_prefix_identity(
        model_id="smoke-model",
        tokenizer_id="smoke-tokenizer",
        prompt="hello prefix",
        token_ids=[1, 2, 3, 4],
        payload=payload,
    )
    cache = LoopbackPrefixCache(backend, residency=residency)
    cache.store(identity, payload)
    plan = cache.build_restore_plan(identity)
    plan_errors = validate_prefix_restore_plan(plan)
    if plan_errors:
        raise ValueError("; ".join(plan_errors))
    return plan


def build_remote_placeholder_entry(identity: PrefixIdentity) -> PrefixCacheEntry:
    """Metadata-only remote placeholder entry (not storable via loopback)."""
    handle = KVStorageHandle(
        namespace=f"remote_placeholder/{identity.model_id}",
        key=identity.storage_key(),
        version=identity.cache_version,
    )
    metadata = build_verifier_storage_metadata(
        {},
        residency=CacheResidency.REMOTE,
        backend_name="remote_prefix_placeholder",
        claim_note=_REMOTE_PLACEHOLDER_CLAIM,
    )
    if identity.dtype_summary:
        metadata.dtype_summary = identity.dtype_summary
    if identity.shape_summary:
        metadata.shape_summary = identity.shape_summary
    return PrefixCacheEntry(
        identity=identity,
        storage_handle=handle,
        storage_metadata=metadata,
        cache_mode=PrefixCacheMode.REMOTE_PLACEHOLDER,
        status=PrefixCacheStatus.REMOTE_BLOCKED,
        claim_note=_REMOTE_PLACEHOLDER_CLAIM,
        remote_placeholder_active=False,
    )
