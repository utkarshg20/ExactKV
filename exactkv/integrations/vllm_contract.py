"""vLLM prototype path contracts (Phase 11F).

Metadata and integration gates for a **future** vLLM prototype. **Does not**
import vLLM or add runtime integration.

This is a vLLM prototype contract, not a vLLM integration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

_CLAIM_NOTE = (
    "vLLM prototype contract spike (Phase 11F). Metadata and gates only — "
    "not runtime integration. No performance, deployment, or resource-usage claims."
)

_ALLOWED_CLAIMS = (
    "vLLM prototype contract metadata exists",
    "Integration gates documented for future prototype work",
    "Dual-cache to paged-KV mapping described in design docs only",
    "Exactness gate required before any performance claim",
)

_FORBIDDEN_CLAIMS = (
    "speedup",
    "latency improvement",
    "throughput improvement",
    "memory savings",
    "production serving",
    "vLLM integrated",
    "vLLM integration exists",
)

_NEGATION_PREFIXES = ("no ", "not ", "without ", "never ", "non-", "does not ")


class VLLMIntegrationStatus(str, Enum):
    """Lifecycle status for vLLM prototype work (metadata only)."""

    NOT_STARTED = "NOT_STARTED"
    CONTRACT_ONLY = "CONTRACT_ONLY"
    PROTOTYPE_BLOCKED = "PROTOTYPE_BLOCKED"
    PROTOTYPE_READY = "PROTOTYPE_READY"
    EXPERIMENTAL_ACTIVE = "EXPERIMENTAL_ACTIVE"


class VLLMCacheCapability(str, Enum):
    """vLLM cache/scheduler capabilities a prototype would need to address."""

    PAGED_KV_CACHE = "PAGED_KV_CACHE"
    CUSTOM_CACHE_MANAGER = "CUSTOM_CACHE_MANAGER"
    PREFILL_DECODE_SPLIT = "PREFILL_DECODE_SPLIT"
    BATCH_SCHEDULER = "BATCH_SCHEDULER"
    ATTENTION_BACKEND_HOOK = "ATTENTION_BACKEND_HOOK"
    UNKNOWN = "UNKNOWN"


@dataclass
class VLLMPrototypeGate:
    """Single integration gate with evidence and blocker fields."""

    gate_name: str
    required: bool
    satisfied: bool
    evidence: str = ""
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VLLMPrototypeGate:
        return cls(
            gate_name=str(data["gate_name"]),
            required=bool(data.get("required", True)),
            satisfied=bool(data.get("satisfied", False)),
            evidence=str(data.get("evidence", "")),
            blocker=str(data.get("blocker", "")),
        )


@dataclass
class VLLMPrototypePlan:
    """Serializable vLLM prototype plan — gates, claims, and status metadata."""

    status: VLLMIntegrationStatus
    capabilities_required: list[VLLMCacheCapability] = field(default_factory=list)
    gates: list[VLLMPrototypeGate] = field(default_factory=list)
    allowed_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    claim_note: str = _CLAIM_NOTE
    dependency_import_attempted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "capabilities_required": [c.value for c in self.capabilities_required],
            "gates": [g.to_dict() for g in self.gates],
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "claim_note": self.claim_note,
            "dependency_import_attempted": self.dependency_import_attempted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VLLMPrototypePlan:
        return cls(
            status=VLLMIntegrationStatus(data["status"]),
            capabilities_required=[
                VLLMCacheCapability(v) for v in data.get("capabilities_required", [])
            ],
            gates=[VLLMPrototypeGate.from_dict(g) for g in data.get("gates", [])],
            allowed_claims=list(data.get("allowed_claims", [])),
            forbidden_claims=list(data.get("forbidden_claims", [])),
            claim_note=str(data.get("claim_note", _CLAIM_NOTE)),
            dependency_import_attempted=bool(data.get("dependency_import_attempted", False)),
        )

    def unsatisfied_required_gates(self) -> list[VLLMPrototypeGate]:
        return [g for g in self.gates if g.required and not g.satisfied]


def _default_capabilities() -> list[VLLMCacheCapability]:
    return [
        VLLMCacheCapability.PAGED_KV_CACHE,
        VLLMCacheCapability.CUSTOM_CACHE_MANAGER,
        VLLMCacheCapability.PREFILL_DECODE_SPLIT,
        VLLMCacheCapability.ATTENTION_BACKEND_HOOK,
    ]


def _default_gates() -> list[VLLMPrototypeGate]:
    """Standard integration gates for Stage 5 prototype readiness."""
    return [
        VLLMPrototypeGate(
            gate_name="optional_dependency_isolation",
            required=True,
            satisfied=True,
            evidence="vLLM not listed in pyproject.toml core or dev dependencies",
        ),
        VLLMPrototypeGate(
            gate_name="no_required_vllm_import",
            required=True,
            satisfied=True,
            evidence="exactkv.integrations.vllm_contract does not import vLLM",
        ),
        VLLMPrototypeGate(
            gate_name="cache_api_mapping_identified",
            required=True,
            satisfied=True,
            evidence=(
                "DualCacheState draft/verifier roles map to compressed draft vs "
                "authoritative full KV; vLLM paged blocks documented as future adapter target"
            ),
        ),
        VLLMPrototypeGate(
            gate_name="draft_cache_role_mapping",
            required=True,
            satisfied=True,
            evidence="CacheRole.DRAFT ↔ CompressedKVState / materialized draft backend (Phase 11B–11D)",
        ),
        VLLMPrototypeGate(
            gate_name="verifier_cache_role_mapping",
            required=True,
            satisfied=True,
            evidence="CacheRole.VERIFIER ↔ FullKVState / storage manager (Phase 11B–11C)",
            blocker="Exp 017: vLLM paged KV does not safely export HF FullKVState for verify",
        ),
        VLLMPrototypeGate(
            gate_name="scheduler_mapping",
            required=True,
            satisfied=True,
            evidence="VerificationExecutionMode.FUTURE_VLLM placeholder in Phase 11E scheduler",
        ),
        VLLMPrototypeGate(
            gate_name="exactness_test_plan",
            required=True,
            satisfied=True,
            evidence="Bounded panel with exactkv_failures == 0 gate before any performance claim",
        ),
        VLLMPrototypeGate(
            gate_name="rollback_fallback_path",
            required=True,
            satisfied=False,
            evidence="",
            blocker="No prototype runtime — HF ExactKVGenerator remains sole active path",
        ),
        VLLMPrototypeGate(
            gate_name="no_speed_claim_before_benchmark",
            required=True,
            satisfied=True,
            evidence="Stage 8 throughput harness required before speed/latency claims",
        ),
        VLLMPrototypeGate(
            gate_name="no_memory_claim_before_measurement",
            required=True,
            satisfied=True,
            evidence="Active memory measurement gate (Exp 031 methodology) before savings claims",
        ),
        VLLMPrototypeGate(
            gate_name="no_production_claim_before_serving_tests",
            required=True,
            satisfied=True,
            evidence="Multi-request serving tests required before production-serving claims",
        ),
    ]


def build_default_vllm_prototype_plan() -> VLLMPrototypePlan:
    """Factory for the Phase 11F default contract-only plan."""
    return VLLMPrototypePlan(
        status=VLLMIntegrationStatus.CONTRACT_ONLY,
        capabilities_required=_default_capabilities(),
        gates=_default_gates(),
        allowed_claims=list(_ALLOWED_CLAIMS),
        forbidden_claims=list(_FORBIDDEN_CLAIMS),
        claim_note=_CLAIM_NOTE,
        dependency_import_attempted=False,
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


def validate_vllm_prototype_plan(plan: VLLMPrototypePlan) -> list[str]:
    """Return human-readable plan invariant violations."""
    errors: list[str] = []

    if plan.status is VLLMIntegrationStatus.EXPERIMENTAL_ACTIVE:
        errors.append("EXPERIMENTAL_ACTIVE is forbidden in Phase 11F")

    if plan.dependency_import_attempted:
        errors.append("dependency_import_attempted must remain False — vLLM is not imported")

    if not plan.claim_note.strip():
        errors.append("claim_note required on vLLM prototype plan")

    if plan.status is VLLMIntegrationStatus.PROTOTYPE_READY:
        unsatisfied = plan.unsatisfied_required_gates()
        if unsatisfied:
            names = ", ".join(g.gate_name for g in unsatisfied)
            errors.append(f"PROTOTYPE_READY requires all required gates satisfied; blocked: {names}")

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


def assert_vllm_not_required() -> None:
    """Runtime check that this contract module does not import vLLM."""
    import sys

    if "vllm" in sys.modules:
        raise RuntimeError("vLLM must not be imported when loading vLLM prototype contracts")
