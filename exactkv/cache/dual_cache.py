"""Dual-cache contract layer for VeriCache systems parity (Phase 11B).

This module defines **metadata contracts** for separating draft (compressed/lossy)
and verifier (authoritative full) KV cache views. It does **not** implement storage,
offload, serving integration, or generation/verification behavior changes.

The dual-cache abstraction is a contract layer, not a serving runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CacheRole(str, Enum):
    """Logical role of a KV cache view in the draft-verify loop."""

    DRAFT = "DRAFT"
    VERIFIER = "VERIFIER"


class CacheResidency(str, Enum):
    """Where KV tensors logically reside (metadata only in Phase 11B)."""

    GPU = "GPU"
    CPU = "CPU"
    DISK = "DISK"
    REMOTE = "REMOTE"
    UNKNOWN = "UNKNOWN"


class CacheMaterialization(str, Enum):
    """How KV values are represented for attention / verification."""

    COMPRESSED = "COMPRESSED"
    MATERIALIZED = "MATERIALIZED"
    FULL = "FULL"
    SIMULATED = "SIMULATED"
    UNKNOWN = "UNKNOWN"


@dataclass
class CacheView:
    """Serializable description of one KV cache side (draft or verifier).

    Byte fields are **accounting estimates** — not peak-device profiling and not
    evidence of active GPU memory savings unless explicitly validated elsewhere.
    """

    role: CacheRole
    backend_name: str
    residency: CacheResidency = CacheResidency.UNKNOWN
    materialization: CacheMaterialization = CacheMaterialization.UNKNOWN
    kv_bytes: int = 0
    metadata_bytes: int = 0
    temporary_workspace_bytes: int = 0
    supports_real_bytes_claim: bool = False
    claim_note: str = ""

    def total_accounted_bytes(self) -> int:
        return self.kv_bytes + self.metadata_bytes + self.temporary_workspace_bytes

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        d["residency"] = self.residency.value
        d["materialization"] = self.materialization.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheView:
        return cls(
            role=CacheRole(data["role"]),
            backend_name=str(data["backend_name"]),
            residency=CacheResidency(data.get("residency", CacheResidency.UNKNOWN.value)),
            materialization=CacheMaterialization(
                data.get("materialization", CacheMaterialization.UNKNOWN.value)
            ),
            kv_bytes=int(data.get("kv_bytes", 0)),
            metadata_bytes=int(data.get("metadata_bytes", 0)),
            temporary_workspace_bytes=int(data.get("temporary_workspace_bytes", 0)),
            supports_real_bytes_claim=bool(data.get("supports_real_bytes_claim", False)),
            claim_note=str(data.get("claim_note", "")),
        )


@dataclass
class DualCacheState:
    """Paired draft + verifier cache views with contract validation."""

    draft: CacheView
    verifier: CacheView
    schema_version: str = "1.0"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "notes": self.notes,
            "draft": self.draft.to_dict(),
            "verifier": self.verifier.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DualCacheState:
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            notes=str(data.get("notes", "")),
            draft=CacheView.from_dict(data["draft"]),
            verifier=CacheView.from_dict(data["verifier"]),
        )

    def validate(self) -> list[str]:
        return validate_dual_cache_state(self)


def validate_cache_view(view: CacheView) -> list[str]:
    """Return human-readable invariant violations for a single cache view."""
    errors: list[str] = []

    if view.kv_bytes < 0:
        errors.append(f"{view.role.value}: kv_bytes must be non-negative")
    if view.metadata_bytes < 0:
        errors.append(f"{view.role.value}: metadata_bytes must be non-negative")
    if view.temporary_workspace_bytes < 0:
        errors.append(f"{view.role.value}: temporary_workspace_bytes must be non-negative")

    if not view.supports_real_bytes_claim and not view.claim_note.strip():
        errors.append(
            f"{view.role.value}: claim_note required when supports_real_bytes_claim is False"
        )

    if view.materialization in (
        CacheMaterialization.MATERIALIZED,
        CacheMaterialization.SIMULATED,
    ):
        if view.supports_real_bytes_claim:
            errors.append(
                f"{view.role.value}: materialized/simulated cache cannot claim "
                "supports_real_bytes_claim=True (no active GPU memory savings)"
            )

    if view.role is CacheRole.VERIFIER:
        if view.materialization not in (
            CacheMaterialization.FULL,
            CacheMaterialization.UNKNOWN,
        ):
            errors.append(
                f"verifier cache must be FULL or UNKNOWN materialization, "
                f"got {view.materialization.value}"
            )

    if view.role is CacheRole.DRAFT and view.materialization is CacheMaterialization.FULL:
        if view.backend_name not in ("noop", "backend_passthrough", "passthrough"):
            # Full materialization on draft side is allowed for identity compressors only.
            pass  # identity compressors may draft from full KV — not an error

    # Residency beyond GPU without explicit note is not a savings claim by itself,
    # but claiming real bytes on non-GPU unvalidated tiers is blocked.
    if view.residency in (CacheResidency.CPU, CacheResidency.DISK, CacheResidency.REMOTE):
        if view.supports_real_bytes_claim and not view.claim_note.strip():
            errors.append(
                f"{view.role.value}: off-GPU residency ({view.residency.value}) "
                "requires explicit claim_note when supports_real_bytes_claim is True"
            )

    return errors


def validate_dual_cache_state(state: DualCacheState) -> list[str]:
    """Return human-readable invariant violations for a dual-cache pair."""
    errors: list[str] = []

    if state.draft.role is not CacheRole.DRAFT:
        errors.append("draft view must have role DRAFT")
    if state.verifier.role is not CacheRole.VERIFIER:
        errors.append("verifier view must have role VERIFIER")
    if state.draft.role is state.verifier.role:
        errors.append("draft and verifier roles must be distinct")

    errors.extend(validate_cache_view(state.draft))
    errors.extend(validate_cache_view(state.verifier))

    draft_total = state.draft.total_accounted_bytes()
    verifier_total = state.verifier.total_accounted_bytes()
    if draft_total < 0 or verifier_total < 0:
        errors.append("total accounted bytes must be non-negative")

    # Draft must not imply active GPU savings vs verifier without evidence.
    if (
        state.draft.supports_real_bytes_claim
        and state.draft.kv_bytes < state.verifier.kv_bytes
        and state.draft.materialization
        in (CacheMaterialization.MATERIALIZED, CacheMaterialization.SIMULATED)
    ):
        errors.append(
            "materializing/simulated draft cannot claim real-byte savings "
            "smaller than verifier KV"
        )

    return errors


def build_identity_dual_cache(
    *,
    backend_name: str = "noop",
    kv_bytes: int = 0,
    residency: CacheResidency = CacheResidency.GPU,
) -> DualCacheState:
    """Construct a valid identity (no-compression) dual-cache description.

    Useful for tests and future reporting adapters. Does not touch runtime state.
    """
    note = (
        "Identity dual-cache: draft and verifier share full KV semantics. "
        "No compression; no active GPU memory savings claimed."
    )
    full_view = CacheView(
        role=CacheRole.VERIFIER,
        backend_name=backend_name,
        residency=residency,
        materialization=CacheMaterialization.FULL,
        kv_bytes=kv_bytes,
        metadata_bytes=0,
        temporary_workspace_bytes=0,
        supports_real_bytes_claim=True,
        claim_note=note,
    )
    draft_view = CacheView(
        role=CacheRole.DRAFT,
        backend_name=backend_name,
        residency=residency,
        materialization=CacheMaterialization.FULL,
        kv_bytes=kv_bytes,
        metadata_bytes=0,
        temporary_workspace_bytes=0,
        supports_real_bytes_claim=True,
        claim_note=note,
    )
    return DualCacheState(draft=draft_view, verifier=full_view, notes=note)
