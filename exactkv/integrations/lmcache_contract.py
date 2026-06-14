"""LMCache prototype path contracts (Phase 11G).

Metadata and integration gates for a **future** LMCache / prefix-cache prototype.
**Does not** import LMCache or add runtime integration.

This is an LMCache prototype contract, not an LMCache integration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from exactkv.integrations.vllm_contract import (
    VLLMIntegrationStatus,
    build_default_vllm_prototype_plan,
)

_CLAIM_NOTE = (
    "LMCache prototype contract spike (Phase 11G). Metadata and gates only — "
    "not runtime integration. No performance, deployment, remote-prefix, or "
    "resource-usage claims."
)

_ALLOWED_CLAIMS = (
    "LMCache prototype contract metadata exists",
    "Integration gates documented for future prefix-cache prototype work",
    "Storage manager to LMCache KV tier mapping described in design docs only",
    "Exactness gate required before any performance claim",
    "Remote-prefix gate required before any remote-prefix-cache claim",
)

_FORBIDDEN_CLAIMS = (
    "speedup",
    "latency improvement",
    "throughput improvement",
    "memory savings",
    "remote prefix caching",
    "production serving",
    "lmcache integrated",
    "lmcache integration exists",
)

_NEGATION_PREFIXES = ("no ", "not ", "without ", "never ", "non-", "does not ")


class LMCacheIntegrationStatus(str, Enum):
    """Lifecycle status for LMCache prototype work (metadata only)."""

    NOT_STARTED = "NOT_STARTED"
    CONTRACT_ONLY = "CONTRACT_ONLY"
    PROTOTYPE_BLOCKED = "PROTOTYPE_BLOCKED"
    PROTOTYPE_READY = "PROTOTYPE_READY"
    EXPERIMENTAL_ACTIVE = "EXPERIMENTAL_ACTIVE"


class LMCacheCapability(str, Enum):
    """LMCache / prefix-cache capabilities a prototype would need to address."""

    LOCAL_PREFIX_CACHE = "LOCAL_PREFIX_CACHE"
    REMOTE_PREFIX_CACHE = "REMOTE_PREFIX_CACHE"
    KV_SERIALIZATION = "KV_SERIALIZATION"
    KV_RESTORE = "KV_RESTORE"
    ASYNC_LOAD = "ASYNC_LOAD"
    CACHE_EVICTION = "CACHE_EVICTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class LMCachePrototypeGate:
    """Single LMCache integration gate with evidence and blocker fields."""

    gate_name: str
    required: bool
    satisfied: bool
    evidence: str = ""
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LMCachePrototypeGate:
        return cls(
            gate_name=str(data["gate_name"]),
            required=bool(data.get("required", True)),
            satisfied=bool(data.get("satisfied", False)),
            evidence=str(data.get("evidence", "")),
            blocker=str(data.get("blocker", "")),
        )


@dataclass
class LMCachePrototypePlan:
    """Serializable LMCache prototype plan — gates, claims, and status metadata."""

    status: LMCacheIntegrationStatus
    capabilities_required: list[LMCacheCapability] = field(default_factory=list)
    gates: list[LMCachePrototypeGate] = field(default_factory=list)
    allowed_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    claim_note: str = _CLAIM_NOTE
    dependency_import_attempted: bool = False
    remote_prefix_cache_active: bool = False
    vllm_contract_status: str = VLLMIntegrationStatus.CONTRACT_ONLY.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "capabilities_required": [c.value for c in self.capabilities_required],
            "gates": [g.to_dict() for g in self.gates],
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "claim_note": self.claim_note,
            "dependency_import_attempted": self.dependency_import_attempted,
            "remote_prefix_cache_active": self.remote_prefix_cache_active,
            "vllm_contract_status": self.vllm_contract_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LMCachePrototypePlan:
        return cls(
            status=LMCacheIntegrationStatus(data["status"]),
            capabilities_required=[
                LMCacheCapability(v) for v in data.get("capabilities_required", [])
            ],
            gates=[LMCachePrototypeGate.from_dict(g) for g in data.get("gates", [])],
            allowed_claims=list(data.get("allowed_claims", [])),
            forbidden_claims=list(data.get("forbidden_claims", [])),
            claim_note=str(data.get("claim_note", _CLAIM_NOTE)),
            dependency_import_attempted=bool(data.get("dependency_import_attempted", False)),
            remote_prefix_cache_active=bool(data.get("remote_prefix_cache_active", False)),
            vllm_contract_status=str(
                data.get("vllm_contract_status", VLLMIntegrationStatus.CONTRACT_ONLY.value)
            ),
        )

    def unsatisfied_required_gates(self) -> list[LMCachePrototypeGate]:
        return [g for g in self.gates if g.required and not g.satisfied]


def _default_capabilities() -> list[LMCacheCapability]:
    return [
        LMCacheCapability.LOCAL_PREFIX_CACHE,
        LMCacheCapability.KV_SERIALIZATION,
        LMCacheCapability.KV_RESTORE,
        LMCacheCapability.ASYNC_LOAD,
        LMCacheCapability.CACHE_EVICTION,
    ]


def _default_gates() -> list[LMCachePrototypeGate]:
    """Standard integration gates for Stage 6 prototype readiness."""
    return [
        LMCachePrototypeGate(
            gate_name="optional_dependency_isolation",
            required=True,
            satisfied=True,
            evidence="LMCache not listed in pyproject.toml core or dev dependencies",
        ),
        LMCachePrototypeGate(
            gate_name="no_required_lmcache_import",
            required=True,
            satisfied=True,
            evidence="exactkv.integrations.lmcache_contract does not import LMCache",
        ),
        LMCachePrototypeGate(
            gate_name="local_prefix_cache_semantics_identified",
            required=True,
            satisfied=True,
            evidence=(
                "Local prefix reuse maps to verifier-tier KVStorageBackend round-trip "
                "(Phase 11C) and CacheRole.VERIFIER residency metadata (Phase 11B)"
            ),
        ),
        LMCachePrototypeGate(
            gate_name="remote_prefix_cache_semantics_identified",
            required=True,
            satisfied=True,
            evidence="Remote prefix documented as Stage 7 target — not active in Phase 11G",
            blocker="Remote prefix runtime not implemented; remote_prefix_cache_active must stay False",
        ),
        LMCachePrototypeGate(
            gate_name="full_kv_serialization_mapping",
            required=True,
            satisfied=True,
            evidence="KVStorageBackend serialize/store path maps to LMCache KV_SERIALIZATION capability",
        ),
        LMCachePrototypeGate(
            gate_name="full_kv_restore_mapping",
            required=True,
            satisfied=True,
            evidence="StoredKVEntry reload maps to LMCache KV_RESTORE capability",
            blocker="Exp 017: LMCache async restore vs synchronous verify not solved",
        ),
        LMCachePrototypeGate(
            gate_name="verifier_cache_correctness_gate",
            required=True,
            satisfied=True,
            evidence="Authoritative verifier role must round-trip without draft contamination",
        ),
        LMCachePrototypeGate(
            gate_name="async_load_blocking_semantics_documented",
            required=True,
            satisfied=True,
            evidence="ASYNC_LOAD capability documented; blocking verify path remains HF default",
        ),
        LMCachePrototypeGate(
            gate_name="eviction_invalidation_semantics_documented",
            required=True,
            satisfied=True,
            evidence="CACHE_EVICTION capability documented; no eviction runtime in Phase 11G",
        ),
        LMCachePrototypeGate(
            gate_name="vllm_contract_interaction_identified",
            required=True,
            satisfied=True,
            evidence=(
                "Phase 11F vLLMPrototypePlan is CONTRACT_ONLY; LMCache tiers verifier KV "
                "behind future vLLM worker — no active vLLM integration"
            ),
            blocker="vLLM prototype runtime not started; LMCache must not assume active vLLM",
        ),
        LMCachePrototypeGate(
            gate_name="exactness_test_plan",
            required=True,
            satisfied=True,
            evidence="Bounded panel with exactkv_failures == 0 gate before any performance claim",
        ),
        LMCachePrototypeGate(
            gate_name="rollback_fallback_path",
            required=True,
            satisfied=False,
            evidence="",
            blocker="No prototype runtime — HF ExactKVGenerator + in-memory storage remain sole path",
        ),
        LMCachePrototypeGate(
            gate_name="no_speed_claim_before_benchmark",
            required=True,
            satisfied=True,
            evidence="Stage 8 throughput harness required before speed/latency claims",
        ),
        LMCachePrototypeGate(
            gate_name="no_memory_claim_before_measurement",
            required=True,
            satisfied=True,
            evidence="Active memory measurement gate (Exp 031 methodology) before savings claims",
        ),
        LMCachePrototypeGate(
            gate_name="no_production_claim_before_serving_tests",
            required=True,
            satisfied=True,
            evidence="Multi-request serving tests required before production-serving claims",
        ),
        LMCachePrototypeGate(
            gate_name="remote_prefix_gate_before_remote_claim",
            required=True,
            satisfied=True,
            evidence="Remote-prefix prototype (Stage 7) required before remote-prefix-cache claims",
        ),
    ]


def build_default_lmcache_prototype_plan() -> LMCachePrototypePlan:
    """Factory for the Phase 11G default contract-only plan."""
    vllm_plan = build_default_vllm_prototype_plan()
    return LMCachePrototypePlan(
        status=LMCacheIntegrationStatus.CONTRACT_ONLY,
        capabilities_required=_default_capabilities(),
        gates=_default_gates(),
        allowed_claims=list(_ALLOWED_CLAIMS),
        forbidden_claims=list(_FORBIDDEN_CLAIMS),
        claim_note=_CLAIM_NOTE,
        dependency_import_attempted=False,
        remote_prefix_cache_active=False,
        vllm_contract_status=vllm_plan.status.value,
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


def _vllm_contract_is_active() -> bool:
    """True when vLLM contract indicates integration beyond contract-only metadata."""
    vllm_plan = build_default_vllm_prototype_plan()
    if vllm_plan.dependency_import_attempted:
        return True
    return vllm_plan.status in (
        VLLMIntegrationStatus.EXPERIMENTAL_ACTIVE,
        VLLMIntegrationStatus.PROTOTYPE_READY,
    )


def validate_lmcache_prototype_plan(plan: LMCachePrototypePlan) -> list[str]:
    """Return human-readable plan invariant violations."""
    errors: list[str] = []

    if plan.status is LMCacheIntegrationStatus.EXPERIMENTAL_ACTIVE:
        errors.append("EXPERIMENTAL_ACTIVE is forbidden in Phase 11G")

    if plan.dependency_import_attempted:
        errors.append("dependency_import_attempted must remain False — LMCache is not imported")

    if plan.remote_prefix_cache_active:
        errors.append("remote_prefix_cache_active must remain False in Phase 11G")

    if not plan.claim_note.strip():
        errors.append("claim_note required on LMCache prototype plan")

    if plan.status is LMCacheIntegrationStatus.PROTOTYPE_READY:
        unsatisfied = plan.unsatisfied_required_gates()
        if unsatisfied:
            names = ", ".join(g.gate_name for g in unsatisfied)
            errors.append(f"PROTOTYPE_READY requires all required gates satisfied; blocked: {names}")

    if _vllm_contract_is_active():
        errors.append(
            "vLLM contract must remain contract-only — active vLLM integration blocks LMCache plan"
        )

    if plan.vllm_contract_status not in (
        VLLMIntegrationStatus.CONTRACT_ONLY.value,
        VLLMIntegrationStatus.NOT_STARTED.value,
        VLLMIntegrationStatus.PROTOTYPE_BLOCKED.value,
    ):
        errors.append(
            f"vllm_contract_status must be contract-only or blocked; got {plan.vllm_contract_status}"
        )

    for term in _FORBIDDEN_CLAIMS:
        if term not in plan.forbidden_claims:
            errors.append(f"forbidden_claims must include: {term}")

    for claim in plan.allowed_claims:
        lower = claim.lower()
        for term in _FORBIDDEN_CLAIMS:
            if _encodes_positive_forbidden_claim(lower, term):
                errors.append(f"allowed_claims must not encode positive forbidden claim: {term}")

    note_lower = plan.claim_note.lower()
    for term in _FORBIDDEN_CLAIMS:
        if _encodes_positive_forbidden_claim(note_lower, term):
            errors.append(f"claim_note must not encode positive forbidden claim: {term}")

    return errors


def assert_lmcache_not_required() -> None:
    """Runtime check that this contract module does not import LMCache."""
    import sys

    for name in ("lmcache", "lm_cache"):
        if name in sys.modules:
            raise RuntimeError(
                "LMCache must not be imported when loading LMCache prototype contracts"
            )
