"""Tests for offline attention tolerance policy (Phase 16I)."""
from __future__ import annotations

import math

import pytest
import torch

from exactkv.attention.streaming_quant_attention import FORBIDDEN_ATTENTION_CLAIMS
from exactkv.attention.tolerance_policy import (
    AttentionTolerancePolicy,
    MetricType,
    OfflineAttentionStatus,
    TopKAgreementSummary,
    evaluate_offline_attention_cell,
    layer_depth_aware_streaming_tolerance,
    strict_streaming_tolerance,
)


def test_strict_numeric_pass() -> None:
    policy = AttentionTolerancePolicy()
    decision = evaluate_offline_attention_cell(
        policy=policy,
        metric_type=MetricType.HIDDEN,
        prefix_layers=4,
        max_abs_error=1e-5,
    )
    assert decision.strict_numeric_pass is True
    assert decision.overall_offline_status == OfflineAttentionStatus.STRICT_NUMERIC_PASS


def test_strict_fail_depth_aware_pass() -> None:
    policy = AttentionTolerancePolicy()
    prefix = 24
    err = 0.002
    depth_tol = layer_depth_aware_streaming_tolerance(torch.float32, prefix)
    assert err <= depth_tol
    decision = evaluate_offline_attention_cell(
        policy=policy,
        metric_type=MetricType.LOGITS,
        prefix_layers=prefix,
        max_abs_error=err,
    )
    assert decision.strict_numeric_pass is False
    assert decision.depth_aware_numeric_pass is True
    assert decision.overall_offline_status == OfflineAttentionStatus.STRICT_FAIL_DEPTH_AWARE_PASS


def test_topk_does_not_imply_exactness() -> None:
    policy = AttentionTolerancePolicy()
    decision = evaluate_offline_attention_cell(
        policy=policy,
        metric_type=MetricType.LOGITS,
        prefix_layers=24,
        max_abs_error=0.1,
        topk=TopKAgreementSummary(top1_agreement=True, top5_overlap=5, top10_overlap=10),
    )
    assert decision.topk_supplementary_pass is True
    assert decision.strict_numeric_pass is False
    assert decision.overall_offline_status == OfflineAttentionStatus.TOPK_AGREES_NUMERIC_DRIFT_PRESENT
    assert "not imply exactness" in decision.interpretation_note.lower() or "supplementary" in decision.interpretation_note.lower()


def test_teacher_forced_local_alignment_free_running_accumulation() -> None:
    policy = AttentionTolerancePolicy()
    decision = evaluate_offline_attention_cell(
        policy=policy,
        metric_type=MetricType.HIDDEN,
        prefix_layers=24,
        max_abs_error=0.5,
        teacher_forced_max_attn_error=1e-6,
        teacher_forced_max_post_mlp_error=1e-6,
        free_running_max_post_mlp_error=0.5,
        root_cause_classification="free_running_accumulation",
    )
    assert decision.teacher_forced_local_alignment_pass is True
    assert decision.overall_offline_status == (
        OfflineAttentionStatus.LOCAL_ALIGNMENT_PASS_FREE_RUNNING_ACCUMULATION
    )


def test_local_attention_mismatch() -> None:
    policy = AttentionTolerancePolicy()
    decision = evaluate_offline_attention_cell(
        policy=policy,
        metric_type=MetricType.ATTENTION_CONTEXT,
        prefix_layers=8,
        max_abs_error=0.02,
        teacher_forced_max_attn_error=0.02,
        root_cause_classification="local_attention_mismatch",
    )
    assert decision.overall_offline_status == OfflineAttentionStatus.LOCAL_ATTENTION_MISMATCH


def test_blocked_classification() -> None:
    policy = AttentionTolerancePolicy()
    decision = evaluate_offline_attention_cell(
        policy=policy,
        metric_type=MetricType.HIDDEN,
        prefix_layers=0,
        max_abs_error=0.0,
        blockers=["load failed"],
    )
    assert decision.overall_offline_status == OfflineAttentionStatus.BLOCKED


def test_attention_context_uses_strict_only() -> None:
    policy = AttentionTolerancePolicy()
    strict, depth = policy.tolerance_for_metric(MetricType.ATTENTION_CONTEXT, 24)
    assert strict == depth == strict_streaming_tolerance(torch.float32)


def test_policy_formula_depth_aware() -> None:
    policy = AttentionTolerancePolicy()
    assert policy.depth_aware_tolerance(24) == pytest.approx(5e-4 * math.sqrt(24))
