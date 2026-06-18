"""Tests for Experiment 089 integration design review (Phase 17D)."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.attention.phase16_closeout import ALLOWED_CLAIMS, FORBIDDEN_CLAIMS
from exactkv.demo.integration_design_review import (
    EXPERIMENT_089_ID,
    GATE_POLICY_BEFORE_TOKEN_COMMIT,
    RECOMMENDED_NEXT_PHASE,
    RISK_REGISTER,
    SOURCE_DOCS,
    run_exp089_integration_design_review,
    validate_exp089_report,
)

LEVEL_IDS = (
    "L0_demo_only",
    "L1_external_shadow_observer",
    "L2_live_diagnostic_observer",
    "L3_guarded_draft_shadow_no_commit",
    "L4_verifier_mediated_compressed_draft",
    "L5_real_backend_integration",
)

IMPLEMENTED = {"L0_demo_only", "L1_external_shadow_observer", "L2_live_diagnostic_observer"}
FUTURE = {
    "L3_guarded_draft_shadow_no_commit",
    "L4_verifier_mediated_compressed_draft",
    "L5_real_backend_integration",
}


def test_schema_validates(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    assert report["experiment_id"] == EXPERIMENT_089_ID
    assert validate_exp089_report(report) == []


def test_integration_levels_l0_through_l5(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    levels = report["integration_levels"]
    for level_id in LEVEL_IDS:
        assert level_id in levels


def test_implemented_and_future_levels_labeled(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    levels = report["integration_levels"]
    for lid in IMPLEMENTED:
        assert levels[lid]["status"] == "implemented"
    assert levels["L3_guarded_draft_shadow_no_commit"]["status"] == "not_implemented"
    assert levels["L4_verifier_mediated_compressed_draft"]["status"] == "not_implemented"
    assert levels["L5_real_backend_integration"]["status"] == "deferred"


def test_token_commit_gate_policy_exists(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    gates = report["gate_policy_before_token_commit_changes"]
    assert len(gates) == len(GATE_POLICY_BEFORE_TOKEN_COMMIT)
    gate_ids = {g["gate_id"] for g in gates}
    assert "baseline_vs_integrated_token_parity" in gate_ids
    assert "shadow_cannot_bypass_verifier" in gate_ids
    assert "full_verifier_source_of_truth" in gate_ids


def test_risk_register_includes_required_risks(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    found = {r["risk_id"] for r in report["risk_register"]}
    expected = {r["risk_id"] for r in RISK_REGISTER}
    assert found == expected
    for risk in report["risk_register"]:
        assert risk.get("severity")
        assert risk.get("mitigation")
        assert risk.get("current_status")


def test_recommended_next_phase(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE
    assert report["recommended_next_phase"] == "phase18a_integration_safety_spec"
    assert report["recommended_next_phase_reason"]


def test_missing_evidence_without_invention(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    assert len(report["source_docs_missing"]) == len(SOURCE_DOCS)
    assert report["source_docs_found"] == []
    l2 = report["integration_levels"]["L2_live_diagnostic_observer"]
    assert "missing" in " ".join(l2["evidence"]).lower() or "not_loaded" in json.dumps(l2)


def test_allowed_and_forbidden_claims_match_freeze(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    assert report["allowed_claims"] == list(ALLOWED_CLAIMS)
    assert report["forbidden_claims"] == list(FORBIDDEN_CLAIMS)


def test_found_docs_inventory(tmp_path: Path) -> None:
    doc = tmp_path / SOURCE_DOCS[0]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Phase 16 closeout\n")
    report = run_exp089_integration_design_review(root=tmp_path)
    assert SOURCE_DOCS[0] in report["source_docs_found"]


def test_no_forbidden_positive_claim_phrases(tmp_path: Path) -> None:
    report = run_exp089_integration_design_review(root=tmp_path)
    for level in report["integration_levels"].values():
        for field in ("title", "implementation_risk", "claim_risk"):
            dumped = json.dumps(level.get(field, "")).lower()
            for forbidden in (
                "speedup achieved",
                "throughput improved",
                "latency reduced",
                "tokens_per_second",
                "runtime_seconds",
                "active_gpu_memory_savings",
                "production_memory_savings",
                "production serving supported",
                "vericache throughput reproduced",
                "vericache serving reproduced",
                "streaming attention integrated into token commit",
                "long-context support proven",
                "broad model-family support proven",
            ):
                assert forbidden not in dumped


def test_real_repo_review_validates() -> None:
    report = run_exp089_integration_design_review(root=Path("."))
    assert validate_exp089_report(report) == []
    assert report["status"] in ("review_complete", "review_partial")
