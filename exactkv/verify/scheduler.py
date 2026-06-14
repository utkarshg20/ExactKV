"""Extended verification scheduler contracts (Phase 11E).

Policy metadata for how verification *could* be scheduled later. **Not** wired
into ``ExactKVGenerator`` or ``VerificationEngine``.

This is a scheduler contract layer, not a verification runtime rewrite.
Bonus-token acceptance remains disabled in all active policies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from exactkv.cache.dual_cache import DualCacheState

_FORBIDDEN_CLAIM_TERMS = (
    "speedup",
    "throughput improvement",
    "latency improvement",
    "tokens/sec",
    "memory savings",
    "production serving",
    "faster than",
)

_SCHEDULER_CLAIM_NOTE = (
    "Scheduler contract spike (Phase 11E). Policy metadata only — not a runtime "
    "rewrite. No performance, deployment, or resource-usage claims."
)

_SERVING_PLACEHOLDER_CLAIM = (
    "Serving-aware verification scheduler placeholder only. Not integrated with "
    "vLLM, LMCache, or batching. Placeholder metadata — not live deployment."
)

_NEGATION_PREFIXES = ("no ", "not ", "without ", "never ", "non-", "does not ")


class VerificationPolicyKind(str, Enum):
    """Verification scheduling policy classification."""

    SEQUENTIAL = "SEQUENTIAL"
    SPAN = "SPAN"
    BONUS_TOKEN_DISABLED = "BONUS_TOKEN_DISABLED"
    SERVING_AWARE_PLACEHOLDER = "SERVING_AWARE_PLACEHOLDER"


class VerificationCommitSemantics(str, Enum):
    """Token commit semantics allowed under scheduler policy."""

    EXACT_PREFIX_ONLY = "EXACT_PREFIX_ONLY"
    EXPERIMENTAL_DISABLED = "EXPERIMENTAL_DISABLED"


class VerificationExecutionMode(str, Enum):
    """Where verification executes (metadata only)."""

    LOCAL_HF = "LOCAL_HF"
    FUTURE_VLLM = "FUTURE_VLLM"
    FUTURE_LMCACHE = "FUTURE_LMCACHE"
    UNKNOWN = "UNKNOWN"


@dataclass
class VerificationPolicy:
    """Serializable verification policy metadata — no runtime behavior."""

    policy_name: str
    kind: VerificationPolicyKind
    max_draft_tokens: int
    span_size: int = 0
    bonus_token_acceptance_enabled: bool = False
    commit_semantics: VerificationCommitSemantics = VerificationCommitSemantics.EXACT_PREFIX_ONLY
    execution_mode: VerificationExecutionMode = VerificationExecutionMode.LOCAL_HF
    requires_dual_cache: bool = True
    runtime_integration_active: bool = False
    claim_note: str = _SCHEDULER_CLAIM_NOTE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["commit_semantics"] = self.commit_semantics.value
        d["execution_mode"] = self.execution_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationPolicy:
        return cls(
            policy_name=str(data["policy_name"]),
            kind=VerificationPolicyKind(data["kind"]),
            max_draft_tokens=int(data.get("max_draft_tokens", 0)),
            span_size=int(data.get("span_size", 0)),
            bonus_token_acceptance_enabled=bool(
                data.get("bonus_token_acceptance_enabled", False)
            ),
            commit_semantics=VerificationCommitSemantics(
                data.get("commit_semantics", VerificationCommitSemantics.EXACT_PREFIX_ONLY.value)
            ),
            execution_mode=VerificationExecutionMode(
                data.get("execution_mode", VerificationExecutionMode.LOCAL_HF.value)
            ),
            requires_dual_cache=bool(data.get("requires_dual_cache", True)),
            runtime_integration_active=bool(data.get("runtime_integration_active", False)),
            claim_note=str(data.get("claim_note", _SCHEDULER_CLAIM_NOTE)),
        )


@dataclass
class VerificationSchedulePlan:
    """Planned verification steps derived from a policy (metadata only)."""

    policy: VerificationPolicy
    planned_verify_steps: list[str] = field(default_factory=list)
    dual_cache_state_summary: dict[str, Any] | None = None
    allowed_runtime_integration: bool = False
    claim_note: str = _SCHEDULER_CLAIM_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "planned_verify_steps": list(self.planned_verify_steps),
            "dual_cache_state_summary": self.dual_cache_state_summary,
            "allowed_runtime_integration": self.allowed_runtime_integration,
            "claim_note": self.claim_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationSchedulePlan:
        return cls(
            policy=VerificationPolicy.from_dict(data["policy"]),
            planned_verify_steps=list(data.get("planned_verify_steps", [])),
            dual_cache_state_summary=data.get("dual_cache_state_summary"),
            allowed_runtime_integration=bool(data.get("allowed_runtime_integration", False)),
            claim_note=str(data.get("claim_note", _SCHEDULER_CLAIM_NOTE)),
        )


def validate_verification_policy(policy: VerificationPolicy) -> list[str]:
    """Return human-readable policy invariant violations."""
    errors: list[str] = []

    if policy.max_draft_tokens <= 0:
        errors.append("max_draft_tokens must be positive")

    if policy.bonus_token_acceptance_enabled:
        errors.append("bonus_token_acceptance_enabled must remain False (Phase 11E)")

    if policy.commit_semantics is not VerificationCommitSemantics.EXACT_PREFIX_ONLY:
        errors.append("commit_semantics must be EXACT_PREFIX_ONLY")

    if policy.kind is VerificationPolicyKind.SPAN:
        if policy.span_size <= 0:
            errors.append("span policy requires positive span_size")
    elif policy.kind is VerificationPolicyKind.SEQUENTIAL:
        if policy.span_size > 0:
            errors.append("sequential policy must not set span_size > 0")

    if policy.kind is VerificationPolicyKind.BONUS_TOKEN_DISABLED:
        if policy.bonus_token_acceptance_enabled:
            errors.append("BONUS_TOKEN_DISABLED policy cannot enable bonus tokens")

    if policy.kind is VerificationPolicyKind.SERVING_AWARE_PLACEHOLDER:
        note = policy.claim_note.lower()
        if not note.strip():
            errors.append("serving-aware placeholder requires claim_note")
        elif "placeholder" not in note and "serving" not in note:
            errors.append("serving-aware placeholder requires serving/placeholder caveat")

    if policy.execution_mode in (
        VerificationExecutionMode.FUTURE_VLLM,
        VerificationExecutionMode.FUTURE_LMCACHE,
    ):
        if policy.runtime_integration_active:
            errors.append(
                f"{policy.execution_mode.value} cannot be marked runtime_integration_active"
            )
        if not policy.claim_note.strip():
            errors.append(f"{policy.execution_mode.value} requires claim_note")

    if policy.runtime_integration_active and policy.execution_mode is not VerificationExecutionMode.LOCAL_HF:
        errors.append("only LOCAL_HF may be runtime_integration_active in Phase 11E")

    if not policy.claim_note.strip():
        errors.append("claim_note required on verification policy")

    note_lower = policy.claim_note.lower()
    for term in _FORBIDDEN_CLAIM_TERMS:
        if _encodes_positive_forbidden_claim(note_lower, term):
            errors.append(f"policy claim_note must not encode positive claim: {term}")

    return errors


def _encodes_positive_forbidden_claim(note_lower: str, term: str) -> bool:
    """True when *term* appears without a nearby negation prefix."""
    start = 0
    while True:
        pos = note_lower.find(term, start)
        if pos == -1:
            return False
        window = note_lower[max(0, pos - 40):pos]
        if not any(neg in window for neg in _NEGATION_PREFIXES):
            return True
        start = pos + len(term)


def validate_verification_schedule_plan(plan: VerificationSchedulePlan) -> list[str]:
    """Validate schedule plan and nested policy."""
    errors = validate_verification_policy(plan.policy)

    if plan.allowed_runtime_integration:
        if plan.policy.execution_mode is not VerificationExecutionMode.LOCAL_HF:
            errors.append("allowed_runtime_integration only valid for LOCAL_HF in Phase 11E")
        if plan.policy.kind is VerificationPolicyKind.SERVING_AWARE_PLACEHOLDER:
            errors.append("serving-aware placeholder cannot allow runtime integration")

    if not plan.planned_verify_steps:
        errors.append("planned_verify_steps must be non-empty")

    if not plan.claim_note.strip():
        errors.append("claim_note required on schedule plan")

    return errors


def _planned_steps_for_policy(policy: VerificationPolicy) -> list[str]:
    if policy.kind is VerificationPolicyKind.SEQUENTIAL:
        return [
            "draft up to max_draft_tokens from compressed KV",
            "verify tokens sequentially against authoritative full KV",
            "commit exact prefix + correction token only",
            "realign compressed KV from committed full state",
        ]
    if policy.kind is VerificationPolicyKind.SPAN:
        return [
            f"draft up to {policy.max_draft_tokens} tokens",
            f"verify span of size {policy.span_size} in one teacher-forced forward",
            "fall back to sequential verify on parity mismatch (future runtime)",
            "commit exact prefix + correction token only",
        ]
    if policy.kind is VerificationPolicyKind.BONUS_TOKEN_DISABLED:
        return [
            "verify draft tokens",
            "reject bonus-token acceptance path (disabled)",
            "commit exact prefix + correction token only",
        ]
    return [
        "placeholder serving-aware schedule — not executed",
        "metadata-only policy for future Stage 5/6 integration",
    ]


def build_schedule_plan(
    policy: VerificationPolicy,
    *,
    dual_cache: DualCacheState | None = None,
) -> VerificationSchedulePlan:
    """Build a schedule plan from policy metadata."""
    steps = _planned_steps_for_policy(policy)
    summary = dual_cache.to_dict() if dual_cache is not None else None
    allowed = (
        policy.execution_mode is VerificationExecutionMode.LOCAL_HF
        and policy.runtime_integration_active
        and policy.kind is not VerificationPolicyKind.SERVING_AWARE_PLACEHOLDER
    )
    return VerificationSchedulePlan(
        policy=policy,
        planned_verify_steps=steps,
        dual_cache_state_summary=summary,
        allowed_runtime_integration=allowed,
        claim_note=policy.claim_note,
    )


def sequential_policy(max_draft_tokens: int, *, policy_name: str = "sequential_default") -> VerificationPolicy:
    return VerificationPolicy(
        policy_name=policy_name,
        kind=VerificationPolicyKind.SEQUENTIAL,
        max_draft_tokens=max_draft_tokens,
        span_size=0,
        bonus_token_acceptance_enabled=False,
        commit_semantics=VerificationCommitSemantics.EXACT_PREFIX_ONLY,
        execution_mode=VerificationExecutionMode.LOCAL_HF,
        requires_dual_cache=True,
        runtime_integration_active=False,
        claim_note=_SCHEDULER_CLAIM_NOTE,
    )


def span_policy(
    max_draft_tokens: int,
    span_size: int,
    *,
    policy_name: str = "span_default",
) -> VerificationPolicy:
    return VerificationPolicy(
        policy_name=policy_name,
        kind=VerificationPolicyKind.SPAN,
        max_draft_tokens=max_draft_tokens,
        span_size=span_size,
        bonus_token_acceptance_enabled=False,
        commit_semantics=VerificationCommitSemantics.EXACT_PREFIX_ONLY,
        execution_mode=VerificationExecutionMode.LOCAL_HF,
        requires_dual_cache=True,
        runtime_integration_active=False,
        claim_note=_SCHEDULER_CLAIM_NOTE,
    )


def disabled_bonus_token_policy(
    *,
    max_draft_tokens: int = 8,
    policy_name: str = "bonus_token_disabled",
) -> VerificationPolicy:
    return VerificationPolicy(
        policy_name=policy_name,
        kind=VerificationPolicyKind.BONUS_TOKEN_DISABLED,
        max_draft_tokens=max_draft_tokens,
        span_size=0,
        bonus_token_acceptance_enabled=False,
        commit_semantics=VerificationCommitSemantics.EXACT_PREFIX_ONLY,
        execution_mode=VerificationExecutionMode.LOCAL_HF,
        requires_dual_cache=True,
        runtime_integration_active=False,
        claim_note=(
            "Bonus-token acceptance explicitly disabled. Exact prefix + correction only. "
            + _SCHEDULER_CLAIM_NOTE
        ),
    )


def serving_aware_placeholder_policy(
    *,
    max_draft_tokens: int = 8,
    policy_name: str = "serving_aware_placeholder",
) -> VerificationPolicy:
    return VerificationPolicy(
        policy_name=policy_name,
        kind=VerificationPolicyKind.SERVING_AWARE_PLACEHOLDER,
        max_draft_tokens=max_draft_tokens,
        span_size=0,
        bonus_token_acceptance_enabled=False,
        commit_semantics=VerificationCommitSemantics.EXACT_PREFIX_ONLY,
        execution_mode=VerificationExecutionMode.FUTURE_VLLM,
        requires_dual_cache=True,
        runtime_integration_active=False,
        claim_note=_SERVING_PLACEHOLDER_CLAIM,
    )
