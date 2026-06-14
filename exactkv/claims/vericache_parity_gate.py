"""VeriCache parity RC claim gate (Phase 11K).

Conservative classification of what VeriCache-related claims are allowed,
forbidden, or blocked pending evidence. **Does not** unlock parity claims.

ExactKV currently reproduces VeriCache-style algorithmic semantics, not the
full VeriCache serving system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from exactkv.benchmarks.paper_panel_contract import (
    PaperPanelStatus,
    build_default_paper_like_panel,
)
from exactkv.benchmarks.throughput_contract import (
    ThroughputClaimStatus,
    build_default_diagnostic_plan,
)
from exactkv.integrations.lmcache_contract import (
    LMCacheIntegrationStatus,
    build_default_lmcache_prototype_plan,
)
from exactkv.integrations.vllm_contract import (
    VLLMIntegrationStatus,
    build_default_vllm_prototype_plan,
)

_GATE_CLAIM_NOTE = (
    "VeriCache parity RC claim gate (Phase 11K). Classification metadata only — "
    "not a parity certification. Full VeriCache reproduction remains forbidden "
    "until all gates pass and are reviewed."
)

_NEGATION_PREFIXES = ("no ", "not ", "without ", "never ", "non-", "does not ")

_FORBIDDEN_POSITIVE_TERMS = (
    "speedup",
    "throughput improvement",
    "latency improvement",
    "memory savings",
    "production serving",
    "full vericache reproduction",
    "vericache parity complete",
)


class VeriCacheClaimCategory(str, Enum):
    """VeriCache-related claim categories tracked by the gate."""

    ALGORITHMIC_SEMANTICS = "ALGORITHMIC_SEMANTICS"
    CORRECTNESS_ON_TESTED_PANELS = "CORRECTNESS_ON_TESTED_PANELS"
    SYSTEMS_PARITY = "SYSTEMS_PARITY"
    VLLM_INTEGRATION = "VLLM_INTEGRATION"
    LMCACHE_INTEGRATION = "LMCACHE_INTEGRATION"
    REMOTE_PREFIX_CACHE = "REMOTE_PREFIX_CACHE"
    THROUGHPUT_BENEFIT = "THROUGHPUT_BENEFIT"
    MEMORY_BENEFIT = "MEMORY_BENEFIT"
    PRODUCTION_SERVING = "PRODUCTION_SERVING"
    PAPER_LIKE_REPRODUCTION = "PAPER_LIKE_REPRODUCTION"
    FULL_VERICACHE_REPRODUCTION = "FULL_VERICACHE_REPRODUCTION"


class VeriCacheClaimStatus(str, Enum):
    """Permitted language level for a claim category."""

    ALLOWED = "ALLOWED"
    ALLOWED_WITH_SCOPE = "ALLOWED_WITH_SCOPE"
    FORBIDDEN = "FORBIDDEN"
    CONTRACT_ONLY = "CONTRACT_ONLY"
    BLOCKED_PENDING_EVIDENCE = "BLOCKED_PENDING_EVIDENCE"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


@dataclass
class ClaimEvidenceRequirement:
    """Single evidence requirement for unlocking a claim category."""

    requirement_name: str
    required: bool
    satisfied: bool
    evidence: str = ""
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimEvidenceRequirement:
        return cls(
            requirement_name=str(data["requirement_name"]),
            required=bool(data.get("required", True)),
            satisfied=bool(data.get("satisfied", False)),
            evidence=str(data.get("evidence", "")),
            blocker=str(data.get("blocker", "")),
        )


@dataclass
class VeriCacheParityClaim:
    """Classification of one VeriCache-related claim category."""

    category: VeriCacheClaimCategory
    status: VeriCacheClaimStatus
    allowed_wording: list[str] = field(default_factory=list)
    forbidden_wording: list[str] = field(default_factory=list)
    evidence_requirements: list[ClaimEvidenceRequirement] = field(default_factory=list)
    human_review_required: bool = False
    claim_note: str = _GATE_CLAIM_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "status": self.status.value,
            "allowed_wording": list(self.allowed_wording),
            "forbidden_wording": list(self.forbidden_wording),
            "evidence_requirements": [r.to_dict() for r in self.evidence_requirements],
            "human_review_required": self.human_review_required,
            "claim_note": self.claim_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VeriCacheParityClaim:
        return cls(
            category=VeriCacheClaimCategory(data["category"]),
            status=VeriCacheClaimStatus(data["status"]),
            allowed_wording=list(data.get("allowed_wording", [])),
            forbidden_wording=list(data.get("forbidden_wording", [])),
            evidence_requirements=[
                ClaimEvidenceRequirement.from_dict(r)
                for r in data.get("evidence_requirements", [])
            ],
            human_review_required=bool(data.get("human_review_required", False)),
            claim_note=str(data.get("claim_note", _GATE_CLAIM_NOTE)),
        )


@dataclass
class VeriCacheParityClaimGate:
    """Aggregate claim gate for VeriCache parity RC review."""

    claims: list[VeriCacheParityClaim] = field(default_factory=list)
    audit_passed: bool = True
    paper_panel_claim_eligible: bool = False
    throughput_claim_allowed: bool = False
    memory_claim_allowed: bool = False
    serving_claim_allowed: bool = False
    full_parity_claim_allowed: bool = False
    claim_note: str = _GATE_CLAIM_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "audit_passed": self.audit_passed,
            "paper_panel_claim_eligible": self.paper_panel_claim_eligible,
            "throughput_claim_allowed": self.throughput_claim_allowed,
            "memory_claim_allowed": self.memory_claim_allowed,
            "serving_claim_allowed": self.serving_claim_allowed,
            "full_parity_claim_allowed": self.full_parity_claim_allowed,
            "claim_note": self.claim_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VeriCacheParityClaimGate:
        return cls(
            claims=[VeriCacheParityClaim.from_dict(c) for c in data.get("claims", [])],
            audit_passed=bool(data.get("audit_passed", True)),
            paper_panel_claim_eligible=bool(data.get("paper_panel_claim_eligible", False)),
            throughput_claim_allowed=bool(data.get("throughput_claim_allowed", False)),
            memory_claim_allowed=bool(data.get("memory_claim_allowed", False)),
            serving_claim_allowed=bool(data.get("serving_claim_allowed", False)),
            full_parity_claim_allowed=bool(data.get("full_parity_claim_allowed", False)),
            claim_note=str(data.get("claim_note", _GATE_CLAIM_NOTE)),
        )

    def claim_for(self, category: VeriCacheClaimCategory) -> VeriCacheParityClaim | None:
        for claim in self.claims:
            if claim.category is category:
                return claim
        return None


def _contract_snapshot() -> dict[str, Any]:
    """Read conservative status from Phase 11F–11J contracts."""
    paper = build_default_paper_like_panel()
    throughput = build_default_diagnostic_plan()
    vllm = build_default_vllm_prototype_plan()
    lmcache = build_default_lmcache_prototype_plan()
    return {
        "paper_panel_claim_eligible": paper.claim_eligible
        and paper.status is PaperPanelStatus.CLAIM_ELIGIBLE,
        "throughput_claim_allowed": throughput.claim_status
        is ThroughputClaimStatus.CLAIM_ALLOWED,
        "memory_claim_allowed": False,
        "serving_claim_allowed": False,
        "vllm_contract_only": vllm.status is VLLMIntegrationStatus.CONTRACT_ONLY,
        "lmcache_contract_only": lmcache.status is LMCacheIntegrationStatus.CONTRACT_ONLY,
        "remote_prefix_loopback_only": True,
    }


def _default_claims() -> list[VeriCacheParityClaim]:
    return [
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.ALGORITHMIC_SEMANTICS,
            status=VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
            allowed_wording=[
                "ExactKV implements VeriCache-style draft-then-verify algorithmic semantics on the HF harness",
                "Lossy compressed KV drafts; full KV verifies; exact prefix + correction commit",
            ],
            forbidden_wording=[
                "ExactKV invented the VeriCache algorithm",
                "Full VeriCache system reproduced",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="hf_generator_semantics",
                    required=True,
                    satisfied=True,
                    evidence="ExactKVGenerator + VerificationEngine on tested panels",
                ),
            ],
            claim_note="Algorithm layer only — not serving system parity",
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.CORRECTNESS_ON_TESTED_PANELS,
            status=VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
            allowed_wording=[
                "On [named panel], exactkv_failures == 0 and output matches full greedy",
                "Crash-test harness framing with cited experiment",
            ],
            forbidden_wording=[
                "Exactness on all models, compressors, and prompts without testing",
                "Universal correctness guarantee",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="cited_panel_exactness",
                    required=True,
                    satisfied=True,
                    evidence="Exp 012, 029, 033 and cited V10 panels",
                ),
            ],
            claim_note="Must cite specific panel — tested panels only",
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.SYSTEMS_PARITY,
            status=VeriCacheClaimStatus.FORBIDDEN,
            allowed_wording=[],
            forbidden_wording=[
                "ExactKV reproduces the VeriCache serving system",
                "Systems parity with VeriCache",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="vllm_lmcache_serving_runtime",
                    required=True,
                    satisfied=False,
                    blocker="Serving runtime not implemented",
                ),
            ],
            human_review_required=True,
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.VLLM_INTEGRATION,
            status=VeriCacheClaimStatus.CONTRACT_ONLY,
            allowed_wording=[
                "vLLM prototype contract metadata exists (Phase 11F)",
            ],
            forbidden_wording=[
                "vLLM integrated",
                "vLLM integration exists",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="vllm_prototype_runtime",
                    required=True,
                    satisfied=False,
                    blocker="Phase 11F contract only; Exp 017 no-go unchanged",
                ),
            ],
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.LMCACHE_INTEGRATION,
            status=VeriCacheClaimStatus.CONTRACT_ONLY,
            allowed_wording=[
                "LMCache prototype contract metadata exists (Phase 11G)",
            ],
            forbidden_wording=[
                "LMCache integrated",
                "LMCache integration exists",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="lmcache_prototype_runtime",
                    required=True,
                    satisfied=False,
                    blocker="Phase 11G contract only; Exp 017 no-go unchanged",
                ),
            ],
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.REMOTE_PREFIX_CACHE,
            status=VeriCacheClaimStatus.FORBIDDEN,
            allowed_wording=[
                "Prefix identity loopback mock on tiny tensors (Phase 11H)",
            ],
            forbidden_wording=[
                "Remote prefix cache runtime exists",
                "Remote prefix caching implemented",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="remote_prefix_runtime",
                    required=True,
                    satisfied=False,
                    blocker="Loopback mock only — no network I/O",
                ),
            ],
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.THROUGHPUT_BENEFIT,
            status=VeriCacheClaimStatus.FORBIDDEN,
            allowed_wording=[
                "Diagnostic timing on cited panel (Exp 030) — not a benefit claim",
            ],
            forbidden_wording=[
                "Throughput improvement",
                "Speedup over full greedy",
                "VeriCache throughput reproduced",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="throughput_claim_allowed",
                    required=True,
                    satisfied=False,
                    evidence="Phase 11I methodology only; Exp 030 shows ExactKV slower",
                    blocker="throughput_claim_allowed is False",
                ),
            ],
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.MEMORY_BENEFIT,
            status=VeriCacheClaimStatus.FORBIDDEN,
            allowed_wording=[
                "Diagnostic workspace accounting on tested path — not active VRAM savings",
            ],
            forbidden_wording=[
                "Active memory savings",
                "VeriCache memory benefits reproduced",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="active_memory_measurement",
                    required=True,
                    satisfied=False,
                    evidence="Exp 031 — no active VRAM savings at tested scale",
                    blocker="memory_claim_allowed is False",
                ),
            ],
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.PRODUCTION_SERVING,
            status=VeriCacheClaimStatus.FORBIDDEN,
            allowed_wording=[
                "Serving sidecar probe feasibility only (Exp 017)",
            ],
            forbidden_wording=[
                "Production serving ready",
                "Production serving implemented",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="multi_request_serving_tests",
                    required=True,
                    satisfied=False,
                    blocker="No serving runtime or batching",
                ),
            ],
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.PAPER_LIKE_REPRODUCTION,
            status=VeriCacheClaimStatus.BLOCKED_PENDING_EVIDENCE,
            allowed_wording=[
                "Paper-like reproduction panel contract exists (Phase 11J)",
            ],
            forbidden_wording=[
                "Paper-like VeriCache reproduction complete",
                "Paper numbers as ExactKV results",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="paper_panel_claim_eligible",
                    required=True,
                    satisfied=False,
                    blocker="Paper panel status CONTRACT_ONLY",
                ),
            ],
            human_review_required=True,
        ),
        VeriCacheParityClaim(
            category=VeriCacheClaimCategory.FULL_VERICACHE_REPRODUCTION,
            status=VeriCacheClaimStatus.FORBIDDEN,
            allowed_wording=[],
            forbidden_wording=[
                "ExactKV reproduces VeriCache",
                "Full VeriCache parity achieved",
                "VeriCache-equivalent functionality on all dimensions",
            ],
            evidence_requirements=[
                ClaimEvidenceRequirement(
                    requirement_name="all_parity_gates",
                    required=True,
                    satisfied=False,
                    blocker="Systems, throughput, memory, serving, paper panel gates not passed",
                ),
            ],
            human_review_required=True,
        ),
    ]


def build_default_vericache_parity_claim_gate() -> VeriCacheParityClaimGate:
    """Factory for Phase 11K conservative default claim gate."""
    snapshot = _contract_snapshot()
    return VeriCacheParityClaimGate(
        claims=_default_claims(),
        audit_passed=True,
        paper_panel_claim_eligible=snapshot["paper_panel_claim_eligible"],
        throughput_claim_allowed=snapshot["throughput_claim_allowed"],
        memory_claim_allowed=snapshot["memory_claim_allowed"],
        serving_claim_allowed=snapshot["serving_claim_allowed"],
        full_parity_claim_allowed=False,
        claim_note=_GATE_CLAIM_NOTE,
    )


def _encodes_positive_forbidden(text_lower: str, term: str) -> bool:
    start = 0
    while True:
        pos = text_lower.find(term, start)
        if pos == -1:
            return False
        window = text_lower[max(0, pos - 40):pos]
        if not any(neg in window for neg in _NEGATION_PREFIXES):
            return True
        start = pos + len(term)


def validate_vericache_parity_claim(claim: VeriCacheParityClaim) -> list[str]:
    """Validate a single claim classification."""
    errors: list[str] = []

    if not claim.forbidden_wording:
        errors.append(f"{claim.category.value}: forbidden_wording must be non-empty")

    if claim.status is VeriCacheClaimStatus.ALLOWED_WITH_SCOPE:
        scoped = any(
            token in " ".join(claim.allowed_wording).lower()
            for token in ("panel", "tested", "cited", "on [", "harness", "crash-test")
        )
        if not scoped and claim.category is not VeriCacheClaimCategory.ALGORITHMIC_SEMANTICS:
            errors.append(
                f"{claim.category.value}: ALLOWED_WITH_SCOPE requires scoped/tested-panel wording"
            )

    if claim.status in (VeriCacheClaimStatus.ALLOWED, VeriCacheClaimStatus.ALLOWED_WITH_SCOPE):
        if not claim.allowed_wording:
            errors.append(f"{claim.category.value}: allowed_wording required when status is allowed")

    if not claim.claim_note.strip():
        errors.append(f"{claim.category.value}: claim_note required")

    note_lower = claim.claim_note.lower()
    for term in _FORBIDDEN_POSITIVE_TERMS:
        if _encodes_positive_forbidden(note_lower, term):
            errors.append(f"{claim.category.value}: claim_note encodes forbidden positive claim: {term}")

    return errors


def validate_vericache_parity_claim_gate(gate: VeriCacheParityClaimGate) -> list[str]:
    """Return human-readable gate invariant violations."""
    errors: list[str] = []

    for claim in gate.claims:
        errors.extend(validate_vericache_parity_claim(claim))

    full = gate.claim_for(VeriCacheClaimCategory.FULL_VERICACHE_REPRODUCTION)
    if full and full.status in (
        VeriCacheClaimStatus.ALLOWED,
        VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
    ):
        errors.append("FULL_VERICACHE_REPRODUCTION cannot be allowed in Phase 11K")

    if gate.full_parity_claim_allowed:
        if not (
            gate.paper_panel_claim_eligible
            and gate.throughput_claim_allowed
            and gate.memory_claim_allowed
            and gate.serving_claim_allowed
        ):
            errors.append(
                "full_parity_claim_allowed requires paper, throughput, memory, and serving gates"
            )
        if full and not full.human_review_required:
            errors.append("FULL_VERICACHE_REPRODUCTION requires human_review_required")

    throughput = gate.claim_for(VeriCacheClaimCategory.THROUGHPUT_BENEFIT)
    if throughput and throughput.status in (
        VeriCacheClaimStatus.ALLOWED,
        VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
    ):
        if not gate.throughput_claim_allowed:
            errors.append("THROUGHPUT_BENEFIT cannot be allowed unless throughput_claim_allowed")

    memory = gate.claim_for(VeriCacheClaimCategory.MEMORY_BENEFIT)
    if memory and memory.status in (
        VeriCacheClaimStatus.ALLOWED,
        VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
    ):
        if not gate.memory_claim_allowed:
            errors.append("MEMORY_BENEFIT cannot be allowed unless memory_claim_allowed")

    vllm = gate.claim_for(VeriCacheClaimCategory.VLLM_INTEGRATION)
    if vllm and vllm.status is VeriCacheClaimStatus.ALLOWED:
        errors.append("VLLM_INTEGRATION cannot be ALLOWED while contract-only")

    lmcache = gate.claim_for(VeriCacheClaimCategory.LMCACHE_INTEGRATION)
    if lmcache and lmcache.status is VeriCacheClaimStatus.ALLOWED:
        errors.append("LMCACHE_INTEGRATION cannot be ALLOWED while contract-only")

    remote = gate.claim_for(VeriCacheClaimCategory.REMOTE_PREFIX_CACHE)
    if remote and remote.status in (
        VeriCacheClaimStatus.ALLOWED,
        VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
    ):
        errors.append("REMOTE_PREFIX_CACHE runtime cannot be allowed — loopback mock only")

    paper = gate.claim_for(VeriCacheClaimCategory.PAPER_LIKE_REPRODUCTION)
    if paper and paper.status in (
        VeriCacheClaimStatus.ALLOWED,
        VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
    ):
        if not gate.paper_panel_claim_eligible:
            errors.append(
                "PAPER_LIKE_REPRODUCTION cannot be allowed while paper panel is contract-only"
            )

    serving = gate.claim_for(VeriCacheClaimCategory.PRODUCTION_SERVING)
    if serving and serving.status in (
        VeriCacheClaimStatus.ALLOWED,
        VeriCacheClaimStatus.ALLOWED_WITH_SCOPE,
    ):
        if not gate.serving_claim_allowed:
            errors.append("PRODUCTION_SERVING cannot be allowed without serving tests")

    if not gate.claim_note.strip():
        errors.append("claim_note required on VeriCacheParityClaimGate")

    return errors
