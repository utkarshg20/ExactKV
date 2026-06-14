"""Tests for Phase 11I throughput benchmark harness contracts."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.benchmarks.throughput_contract import (
    ThroughputBenchmarkMode,
    ThroughputBenchmarkPlan,
    ThroughputClaimStatus,
    ThroughputDiagnosticResult,
    TimingMetricKind,
    build_default_diagnostic_plan,
    build_diagnostic_result_stub,
    validate_throughput_benchmark_plan,
    validate_throughput_diagnostic_result,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "THROUGHPUT_BENCHMARK_HARNESS.md"


def test_default_plan_serializes() -> None:
    plan = build_default_diagnostic_plan()
    raw = plan.to_dict()
    restored = ThroughputBenchmarkPlan.from_dict(raw)
    assert restored == plan
    json.dumps(raw, sort_keys=True)


def test_diagnostic_only_plan_validates() -> None:
    plan = build_default_diagnostic_plan()
    assert validate_throughput_benchmark_plan(plan) == []


def test_speed_claim_rejected_unless_claim_allowed() -> None:
    plan = build_default_diagnostic_plan()
    plan.allowed_claims = ["ExactKV achieves speedup over full greedy"]
    errors = validate_throughput_benchmark_plan(plan)
    assert any("CLAIM_ALLOWED" in e or "speed" in e for e in errors)


def test_claim_allowed_fails_without_exactness_gate() -> None:
    plan = build_default_diagnostic_plan()
    plan.claim_status = ThroughputClaimStatus.CLAIM_ALLOWED
    plan.exactness_gate_required = False
    plan.model = "Qwen/Qwen2.5-0.5B"
    plan.hardware = {"gpu_device_name": "A5000"}
    errors = validate_throughput_benchmark_plan(plan)
    assert any("exactness_gate" in e for e in errors)


def test_claim_allowed_fails_without_hardware() -> None:
    plan = build_default_diagnostic_plan()
    plan.claim_status = ThroughputClaimStatus.CLAIM_ALLOWED
    plan.model = "Qwen/Qwen2.5-0.5B"
    plan.baseline_arm = "full_greedy"
    errors = validate_throughput_benchmark_plan(plan)
    assert any("hardware" in e for e in errors)


def test_claim_allowed_fails_without_baseline() -> None:
    plan = build_default_diagnostic_plan()
    plan.claim_status = ThroughputClaimStatus.CLAIM_ALLOWED
    plan.model = "Qwen/Qwen2.5-0.5B"
    plan.hardware = {"gpu_device_name": "A5000"}
    plan.baseline_arm = ""
    errors = validate_throughput_benchmark_plan(plan)
    assert any("baseline_arm" in e for e in errors)


def test_placeholder_serving_cannot_claim_runtime() -> None:
    plan = build_default_diagnostic_plan()
    plan.mode = ThroughputBenchmarkMode.SERVING_PLACEHOLDER
    plan.runtime_placeholder_active = True
    errors = validate_throughput_benchmark_plan(plan)
    assert any("runtime_placeholder_active" in e for e in errors)


def test_placeholder_batched_requires_caveat() -> None:
    plan = build_default_diagnostic_plan()
    plan.mode = ThroughputBenchmarkMode.BATCHED_PLACEHOLDER
    plan.claim_note = "missing keywords"
    errors = validate_throughput_benchmark_plan(plan)
    assert any("placeholder" in e or "BATCHED" in e for e in errors)


def test_diagnostic_result_serializes() -> None:
    stub = build_diagnostic_result_stub()
    raw = stub.to_dict()
    restored = ThroughputDiagnosticResult.from_dict(raw)
    assert restored.tokens_per_second == stub.tokens_per_second
    assert validate_throughput_diagnostic_result(restored) == []


def test_hide_negative_results_fails() -> None:
    stub = build_diagnostic_result_stub()
    stub.hide_negative_results = True
    errors = validate_throughput_diagnostic_result(stub)
    assert any("hide_negative_results" in e for e in errors)


def test_negative_status_preserved() -> None:
    stub = build_diagnostic_result_stub(negative_vs_baseline=True)
    assert stub.claim_status is ThroughputClaimStatus.NEGATIVE_OR_NEUTRAL
    assert stub.tokens_per_second < (stub.baseline_tokens_per_second() or 0)
    assert validate_throughput_diagnostic_result(stub) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "methodology and contract layer",
        "not a throughput result",
        "do not support a speedup claim",
        "exactness",
        "baseline",
        "warmup",
        "placeholder",
        "vllm",
        "lmcache",
        "vericache throughput",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "throughput improvement claim", "production serving ready"):
        assert phrase not in text


def test_sample_count_must_be_positive() -> None:
    plan = build_default_diagnostic_plan()
    plan.sample_count = 0
    errors = validate_throughput_benchmark_plan(plan)
    assert any("sample_count" in e for e in errors)


def test_metrics_must_be_named() -> None:
    plan = build_default_diagnostic_plan()
    plan.metrics_required = []
    errors = validate_throughput_benchmark_plan(plan)
    assert any("metrics_required" in e for e in errors)
