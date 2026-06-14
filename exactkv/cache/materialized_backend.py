"""Materialized compressed-draft backend — design spike (Phase 11D).

Describes draft-side cache metadata when a compressor stores compressed KV and
materializes dequant tensors for attention. **Isolated contract only** — not wired
into ``ExactKVGenerator`` or verification.

This is an isolated contract/design spike, not a serving runtime.
The backend materializes decompressed K/V for use and therefore does not prove
active GPU memory savings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch

from exactkv.cache.dual_cache import (
    CacheMaterialization,
    CacheResidency,
    CacheRole,
    CacheView,
    DualCacheState,
    validate_cache_view,
    validate_dual_cache_state,
)
from exactkv.cache.storage import (
    KVStorageBackend,
    KVStorageHandle,
    KVStorageMetadata,
    StoredKVEntry,
    build_verifier_storage_metadata,
    cache_view_from_storage_metadata,
    iter_tensors,
    smoke_store_verifier_payload,
    summarize_tensor_payload,
)

_IDENTITY_CLAIM = (
    "Identity draft backend: no compression. Design spike only; "
    "no active GPU memory savings claim."
)
_SIMULATED_CLAIM = (
    "Simulated compressed draft (e.g. k8_v4_sim): sub-INT8 values in int8 containers. "
    "Materialized working KV equals full precision for attention. "
    "supports_real_bytes_claim=False. No active GPU memory savings."
)
_EXTERNAL_ADAPTER_CLAIM = (
    "External adapter materialized draft (SpectralQuant-style): compresses K/V then "
    "materialises dequant tensors for draft. Factory-only restricted probe — not hot "
    "compressed attention. No active GPU memory savings."
)


class DraftBackendKind(str, Enum):
    """Classification of materialized draft backend paths."""

    IDENTITY = "IDENTITY"
    SIMULATED_COMPRESSED = "SIMULATED_COMPRESSED"
    REAL_COMPRESSED_MATERIALIZED = "REAL_COMPRESSED_MATERIALIZED"
    EXTERNAL_ADAPTER_MATERIALIZED = "EXTERNAL_ADAPTER_MATERIALIZED"


@dataclass
class MaterializedDraftMetadata:
    """Metadata for a materialized compressed-draft cache description."""

    backend_name: str
    backend_kind: DraftBackendKind
    source_dtype_summary: str
    stored_bytes: int
    materialized_bytes: int
    metadata_bytes: int
    temporary_workspace_bytes: int
    supports_real_bytes_claim: bool
    claim_note: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backend_kind"] = self.backend_kind.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterializedDraftMetadata:
        return cls(
            backend_name=str(data["backend_name"]),
            backend_kind=DraftBackendKind(data["backend_kind"]),
            source_dtype_summary=str(data.get("source_dtype_summary", "")),
            stored_bytes=int(data.get("stored_bytes", 0)),
            materialized_bytes=int(data.get("materialized_bytes", 0)),
            metadata_bytes=int(data.get("metadata_bytes", 0)),
            temporary_workspace_bytes=int(data.get("temporary_workspace_bytes", 0)),
            supports_real_bytes_claim=bool(data.get("supports_real_bytes_claim", False)),
            claim_note=str(data.get("claim_note", "")),
        )

    def total_accounted_bytes(self) -> int:
        return (
            self.stored_bytes
            + self.materialized_bytes
            + self.metadata_bytes
            + self.temporary_workspace_bytes
        )


@dataclass
class DualCacheFootprint:
    """Conservative combined draft + verifier byte accounting."""

    draft_stored_bytes: int
    draft_materialized_bytes: int
    draft_metadata_bytes: int
    draft_workspace_bytes: int
    verifier_payload_bytes: int
    verifier_metadata_bytes: int

    @property
    def draft_total(self) -> int:
        return (
            self.draft_stored_bytes
            + self.draft_materialized_bytes
            + self.draft_metadata_bytes
            + self.draft_workspace_bytes
        )

    @property
    def verifier_total(self) -> int:
        return self.verifier_payload_bytes + self.verifier_metadata_bytes

    @property
    def combined_total(self) -> int:
        return self.draft_total + self.verifier_total

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name, val in (
            ("draft_stored_bytes", self.draft_stored_bytes),
            ("draft_materialized_bytes", self.draft_materialized_bytes),
            ("draft_metadata_bytes", self.draft_metadata_bytes),
            ("draft_workspace_bytes", self.draft_workspace_bytes),
            ("verifier_payload_bytes", self.verifier_payload_bytes),
            ("verifier_metadata_bytes", self.verifier_metadata_bytes),
        ):
            if val < 0:
                errors.append(f"{name} must be non-negative")
        return errors


class MaterializedDraftBackend(ABC):
    """Describe a materialized draft path from a tensor payload."""

    @abstractmethod
    def describe(self, payload: Any) -> MaterializedDraftMetadata:
        """Build draft metadata from a synthetic payload."""

    def build_cache_view(self, metadata: MaterializedDraftMetadata) -> CacheView:
        return build_draft_cache_view(metadata)


def _materialization_for_kind(kind: DraftBackendKind) -> CacheMaterialization:
    if kind is DraftBackendKind.IDENTITY:
        return CacheMaterialization.FULL
    if kind is DraftBackendKind.SIMULATED_COMPRESSED:
        return CacheMaterialization.SIMULATED
    return CacheMaterialization.MATERIALIZED


def build_draft_metadata_from_payload(
    payload: Any,
    *,
    backend_name: str,
    backend_kind: DraftBackendKind,
    stored_ratio: float = 0.5,
    metadata_bytes: int = 64,
    temporary_workspace_bytes: int = 0,
    claim_note: str | None = None,
) -> MaterializedDraftMetadata:
    """Derive conservative draft metadata from tiny synthetic tensors."""
    _, full_bytes, dtype_summary, _ = summarize_tensor_payload(payload)
    if backend_kind is DraftBackendKind.IDENTITY:
        stored = full_bytes
        materialized = full_bytes
        supports_real = False
        note = claim_note or _IDENTITY_CLAIM
    elif backend_kind is DraftBackendKind.SIMULATED_COMPRESSED:
        stored = max(int(full_bytes * stored_ratio), 0)
        materialized = full_bytes
        supports_real = False
        note = claim_note or _SIMULATED_CLAIM
    elif backend_kind is DraftBackendKind.REAL_COMPRESSED_MATERIALIZED:
        stored = max(int(full_bytes * stored_ratio), 1)
        materialized = full_bytes
        supports_real = False
        note = claim_note or (
            "Real compressed storage with materialized working KV for attention. "
            "Diagnostic byte counts only; no active GPU memory savings."
        )
    else:  # EXTERNAL_ADAPTER_MATERIALIZED
        stored = max(int(full_bytes * 0.4), 1)
        materialized = full_bytes
        supports_real = False
        note = claim_note or _EXTERNAL_ADAPTER_CLAIM

    return MaterializedDraftMetadata(
        backend_name=backend_name,
        backend_kind=backend_kind,
        source_dtype_summary=dtype_summary,
        stored_bytes=stored,
        materialized_bytes=materialized,
        metadata_bytes=metadata_bytes,
        temporary_workspace_bytes=temporary_workspace_bytes,
        supports_real_bytes_claim=supports_real,
        claim_note=note,
    )


class SyntheticMaterializedDraftBackend(MaterializedDraftBackend):
    """Concrete spike backend parameterized by ``DraftBackendKind``."""

    def __init__(
        self,
        *,
        backend_name: str,
        backend_kind: DraftBackendKind,
        stored_ratio: float = 0.5,
    ) -> None:
        self.backend_name = backend_name
        self.backend_kind = backend_kind
        self.stored_ratio = stored_ratio

    def describe(self, payload: Any) -> MaterializedDraftMetadata:
        return build_draft_metadata_from_payload(
            payload,
            backend_name=self.backend_name,
            backend_kind=self.backend_kind,
            stored_ratio=self.stored_ratio,
        )


def validate_materialized_draft_metadata(metadata: MaterializedDraftMetadata) -> list[str]:
    """Claim guards for materialized draft metadata."""
    errors: list[str] = []

    for name, val in (
        ("stored_bytes", metadata.stored_bytes),
        ("materialized_bytes", metadata.materialized_bytes),
        ("metadata_bytes", metadata.metadata_bytes),
        ("temporary_workspace_bytes", metadata.temporary_workspace_bytes),
    ):
        if val < 0:
            errors.append(f"{name} must be non-negative")

    if not metadata.claim_note.strip():
        errors.append("claim_note required on materialized draft metadata")

    if metadata.backend_kind in (
        DraftBackendKind.SIMULATED_COMPRESSED,
        DraftBackendKind.EXTERNAL_ADAPTER_MATERIALIZED,
        DraftBackendKind.REAL_COMPRESSED_MATERIALIZED,
    ):
        if metadata.supports_real_bytes_claim:
            errors.append(
                f"{metadata.backend_kind.value}: simulated/materialized draft cannot "
                "claim supports_real_bytes_claim=True"
            )

    if metadata.backend_kind is DraftBackendKind.SIMULATED_COMPRESSED:
        if "simulated" not in metadata.claim_note.lower() and "sim" not in metadata.claim_note.lower():
            errors.append("simulated compressed draft requires simulated caveat in claim_note")

    if metadata.backend_kind is DraftBackendKind.EXTERNAL_ADAPTER_MATERIALIZED:
        if "materializ" not in metadata.claim_note.lower():
            errors.append("external adapter materialized draft requires materializing caveat")

    # Materialized working copy dominates — stored bytes alone are not GPU savings.
    if metadata.materialized_bytes > 0 and metadata.stored_bytes < metadata.materialized_bytes:
        if metadata.supports_real_bytes_claim:
            errors.append(
                "stored_bytes < materialized_bytes cannot imply active GPU memory savings"
            )

    return errors


def build_draft_cache_view(metadata: MaterializedDraftMetadata) -> CacheView:
    """Build a draft ``CacheView`` from materialized draft metadata."""
    return CacheView(
        role=CacheRole.DRAFT,
        backend_name=metadata.backend_name,
        residency=CacheResidency.GPU,
        materialization=_materialization_for_kind(metadata.backend_kind),
        kv_bytes=metadata.stored_bytes,
        metadata_bytes=metadata.metadata_bytes,
        temporary_workspace_bytes=metadata.temporary_workspace_bytes + metadata.materialized_bytes,
        supports_real_bytes_claim=False,
        claim_note=metadata.claim_note,
    )


def compute_dual_cache_footprint(
    draft: MaterializedDraftMetadata,
    verifier: KVStorageMetadata,
) -> DualCacheFootprint:
    """Conservative combined footprint from draft + stored verifier metadata."""
    return DualCacheFootprint(
        draft_stored_bytes=draft.stored_bytes,
        draft_materialized_bytes=draft.materialized_bytes,
        draft_metadata_bytes=draft.metadata_bytes,
        draft_workspace_bytes=draft.temporary_workspace_bytes,
        verifier_payload_bytes=verifier.total_payload_bytes,
        verifier_metadata_bytes=verifier.metadata_bytes,
    )


def dual_cache_with_materialized_draft_and_verifier(
    draft_metadata: MaterializedDraftMetadata,
    verifier_metadata: KVStorageMetadata,
) -> DualCacheState:
    """Combine materialized draft view with stored verifier view."""
    draft_view = build_draft_cache_view(draft_metadata)
    verifier_view = cache_view_from_storage_metadata(verifier_metadata)
    return DualCacheState(
        draft=draft_view,
        verifier=verifier_view,
        notes=(
            "Materialized draft + stored verifier dual-cache spike (Phase 11D). "
            "Not a serving runtime."
        ),
    )


def validate_materialized_dual_cache(
    draft_metadata: MaterializedDraftMetadata,
    verifier_metadata: KVStorageMetadata,
    payload: Any | None = None,
) -> list[str]:
    """Validate draft metadata, optional payload tensors, and dual-cache state."""
    errors = validate_materialized_draft_metadata(draft_metadata)
    if payload is not None:
        _, full_bytes, _, _ = summarize_tensor_payload(payload)
        if full_bytes > 0 and draft_metadata.materialized_bytes != full_bytes:
            errors.append(
                f"materialized_bytes ({draft_metadata.materialized_bytes}) should "
                f"match payload full bytes ({full_bytes}) for spike"
            )
    footprint = compute_dual_cache_footprint(draft_metadata, verifier_metadata)
    errors.extend(footprint.validate())
    state = dual_cache_with_materialized_draft_and_verifier(
        draft_metadata, verifier_metadata
    )
    errors.extend(validate_dual_cache_state(state))
    return errors


def smoke_materialized_dual_cache(
    verifier_backend: KVStorageBackend,
    handle: KVStorageHandle,
    full_payload: Any,
    draft_kind: DraftBackendKind,
    *,
    verifier_residency: CacheResidency,
    backend_name: str | None = None,
) -> tuple[DualCacheState, DualCacheFootprint, StoredKVEntry]:
    """Tiny smoke: store verifier, build materialized draft, validate dual cache."""
    stored = smoke_store_verifier_payload(
        verifier_backend,
        handle,
        full_payload,
        residency=verifier_residency,
    )
    name = backend_name or draft_kind.value.lower()
    draft_backend = SyntheticMaterializedDraftBackend(
        backend_name=name,
        backend_kind=draft_kind,
    )
    draft_meta = draft_backend.describe(full_payload)
    errors = validate_materialized_dual_cache(
        draft_meta, stored.metadata, full_payload
    )
    if errors:
        raise ValueError("; ".join(errors))
    state = dual_cache_with_materialized_draft_and_verifier(
        draft_meta, stored.metadata
    )
    footprint = compute_dual_cache_footprint(draft_meta, stored.metadata)
    return state, footprint, stored
