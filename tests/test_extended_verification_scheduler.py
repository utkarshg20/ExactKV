"""Tests for Phase 11E extended verification scheduler contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.cache.dual_cache import build_identity_dual_cache
from exactkv.verify.scheduler import (
    VerificationCommitSemantics,
    VerificationExecutionMode,
    VerificationPolicy,
    VerificationPolicyKind,
    VerificationSchedulePlan,
    build_schedule_plan,
    disabled_bonus_token_policy,
    sequential_policy,
    serving_aware_placeholder_policy,
    span_policy,
    validate_verification_policy,
    validate_verification_schedule_plan,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXTENDED_VERIFICATION_SCHEDULER.md"


def test_sequential_policy_validates() -> None:
    policy = sequential_policy(4)
    assert validate_verification_policy(policy) == []
    plan = build_schedule_plan(policy)
    assert validate_verification_schedule_plan(plan) == []
    assert plan.policy.kind is VerificationPolicyKind.SEQUENTIAL


def test_span_policy_validates_positive_span() -> None:
    policy = span_policy(max_draft_tokens=8, span_size=4)
    assert validate_verification_policy(policy) == []
    plan = build_schedule_plan(policy)
    assert validate_verification_schedule_plan(plan) == []
    assert any("span" in step.lower() for step in plan.planned_verify_steps)


def test_span_policy_rejects_zero_span() -> None:
    policy = span_policy(max_draft_tokens=8, span_size=0)
    errors = validate_verification_policy(policy)
    assert any("span_size" in e for e in errors)


def test_sequential_policy_rejects_span_size() -> None:
    policy = sequential_policy(4)
    policy.span_size = 4
    errors = validate_verification_policy(policy)
    assert any("sequential" in e.lower() and "span" in e.lower() for e in errors)


def test_bonus_token_enabled_fails() -> None:
    policy = sequential_policy(4)
    policy.bonus_token_acceptance_enabled = True
    errors = validate_verification_policy(policy)
    assert any("bonus_token" in e for e in errors)


def test_disabled_bonus_token_policy_validates() -> None:
    policy = disabled_bonus_token_policy()
    assert validate_verification_policy(policy) == []
    assert policy.kind is VerificationPolicyKind.BONUS_TOKEN_DISABLED


def test_serving_placeholder_requires_caveat() -> None:
    policy = serving_aware_placeholder_policy()
    assert validate_verification_policy(policy) == []
    policy.claim_note = "missing keywords"
    errors = validate_verification_policy(policy)
    assert any("placeholder" in e or "serving" in e for e in errors)


def test_future_vllm_cannot_be_active() -> None:
    policy = serving_aware_placeholder_policy()
    policy.runtime_integration_active = True
    errors = validate_verification_policy(policy)
    assert any("runtime_integration_active" in e for e in errors)


def test_future_lmcache_cannot_be_active() -> None:
    policy = span_policy(8, 4)
    policy.execution_mode = VerificationExecutionMode.FUTURE_LMCACHE
    policy.runtime_integration_active = True
    policy.claim_note = "future lmcache placeholder only"
    errors = validate_verification_policy(policy)
    assert any("FUTURE_LMCACHE" in e or "runtime_integration_active" in e for e in errors)


def test_commit_semantics_must_be_exact_prefix() -> None:
    policy = sequential_policy(4)
    policy.commit_semantics = VerificationCommitSemantics.EXPERIMENTAL_DISABLED
    errors = validate_verification_policy(policy)
    assert any("EXACT_PREFIX_ONLY" in e for e in errors)


def test_policy_serializes() -> None:
    policy = span_policy(8, 4)
    raw = policy.to_dict()
    restored = VerificationPolicy.from_dict(raw)
    assert restored == policy
    json.dumps(raw, sort_keys=True)


def test_schedule_plan_serializes() -> None:
    policy = sequential_policy(4)
    plan = build_schedule_plan(policy, dual_cache=build_identity_dual_cache(kv_bytes=100))
    raw = plan.to_dict()
    restored = VerificationSchedulePlan.from_dict(raw)
    assert restored.policy == plan.policy
    assert restored.dual_cache_state_summary is not None


def test_backward_compatible_missing_fields() -> None:
    minimal = {
        "policy_name": "legacy",
        "kind": "SEQUENTIAL",
        "max_draft_tokens": 4,
    }
    policy = VerificationPolicy.from_dict(minimal)
    assert policy.span_size == 0
    assert policy.bonus_token_acceptance_enabled is False
    assert policy.execution_mode is VerificationExecutionMode.LOCAL_HF


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "scheduler contract layer",
        "generation and verification behavior is unchanged",
        "bonus-token acceptance remains disabled",
        "vllm",
        "lmcache",
        "placeholder",
        "throughput",
        "no speedup",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    assert "forbidden" in text
    for phrase in ("achieves speedup", "production serving ready", "latency improvement claim"):
        assert phrase not in text


def test_package_exports() -> None:
    from exactkv.verify import sequential_policy as sp

    assert sp(4).kind is VerificationPolicyKind.SEQUENTIAL
