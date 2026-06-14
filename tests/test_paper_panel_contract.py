"""Tests for Phase 11J paper-like reproduction panel contracts."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.benchmarks.paper_panel_contract import (
    PaperLikeReproductionPanel,
    PaperPanelCompressorSpec,
    PaperPanelStatus,
    PaperPanelWorkloadSpec,
    build_default_paper_like_panel,
    validate_paper_like_reproduction_panel,
    validate_paper_panel_compressor,
    validate_paper_panel_workload,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "PAPER_LIKE_REPRODUCTION_PANEL.md"


def test_default_panel_serializes() -> None:
    panel = build_default_paper_like_panel()
    raw = panel.to_dict()
    restored = PaperLikeReproductionPanel.from_dict(raw)
    assert restored.status is PaperPanelStatus.CONTRACT_ONLY
    assert len(restored.models) == len(panel.models)
    json.dumps(raw, sort_keys=True)


def test_default_panel_status_contract_only() -> None:
    panel = build_default_paper_like_panel()
    assert panel.status is PaperPanelStatus.CONTRACT_ONLY


def test_default_panel_not_claim_eligible() -> None:
    panel = build_default_paper_like_panel()
    assert panel.claim_eligible is False
    assert validate_paper_like_reproduction_panel(panel) == []


def test_claim_eligible_fails_without_exactness_gate() -> None:
    panel = build_default_paper_like_panel()
    panel.status = PaperPanelStatus.CLAIM_ELIGIBLE
    panel.claim_eligible = True
    panel.exactness_gate_passed = False
    errors = validate_paper_like_reproduction_panel(panel)
    assert any("exactness_gate" in e for e in errors)


def test_claim_eligible_fails_without_throughput_gate() -> None:
    panel = build_default_paper_like_panel()
    panel.status = PaperPanelStatus.CLAIM_ELIGIBLE
    panel.claim_eligible = True
    panel.exactness_gate_passed = True
    panel.throughput_gate_passed = False
    errors = validate_paper_like_reproduction_panel(panel)
    assert any("throughput_gate" in e for e in errors)


def test_claim_eligible_fails_without_memory_gate() -> None:
    panel = build_default_paper_like_panel()
    panel.status = PaperPanelStatus.CLAIM_ELIGIBLE
    panel.claim_eligible = True
    panel.exactness_gate_passed = True
    panel.throughput_gate_passed = True
    panel.memory_gate_passed = False
    errors = validate_paper_like_reproduction_panel(panel)
    assert any("memory_gate" in e for e in errors)


def test_simulated_compressor_cannot_be_paper_equivalent_real() -> None:
    spec = PaperPanelCompressorSpec(
        compressor_name="int4_sim",
        implementation_status="built_in",
        real_or_simulated="simulated",
        claim_note="simulated row",
        paper_equivalent_real_backend=True,
    )
    errors = validate_paper_panel_compressor(spec)
    assert any("simulated" in e and "paper-equivalent" in e for e in errors)


def test_shard_cannot_be_paper_equivalent_integrated() -> None:
    spec = PaperPanelCompressorSpec(
        compressor_name="shard_external_drafter",
        implementation_status="restricted_probe",
        real_or_simulated="restricted_external",
        claim_note="shard probe",
        paper_equivalent_integrated_compressor=True,
    )
    errors = validate_paper_panel_compressor(spec)
    assert any("Shard" in e or "shard" in e for e in errors)


def test_spectralquant_cannot_claim_active_memory_savings() -> None:
    spec = PaperPanelCompressorSpec(
        compressor_name="spectralquant_experimental",
        implementation_status="restricted_probe",
        real_or_simulated="materializing_adapter",
        claim_note="adapter row",
        claims_active_memory_savings=True,
    )
    errors = validate_paper_panel_compressor(spec)
    assert any("memory savings" in e for e in errors)


def test_non_paper_workload_requires_caveat() -> None:
    spec = PaperPanelWorkloadSpec(
        workload_name="custom_suite",
        exact_match_to_vericache_paper=False,
        implemented=True,
        claim_note="missing keywords",
    )
    errors = validate_paper_panel_workload(spec)
    assert any("caveat" in e or "paper" in e for e in errors)


def test_paper_numbers_cannot_be_marked_exactkv_results() -> None:
    panel = build_default_paper_like_panel()
    panel.paper_numbers_as_exactkv_results = True
    errors = validate_paper_like_reproduction_panel(panel)
    assert any("paper_numbers_as_exactkv_results" in e for e in errors)


def test_forbidden_claims_present() -> None:
    panel = build_default_paper_like_panel()
    forbidden = {c.lower() for c in panel.forbidden_claims}
    for term in (
        "speedup",
        "throughput improvement",
        "vericache throughput reproduced",
        "paper numbers as exactkv results",
    ):
        assert term in forbidden


def test_default_panel_covers_required_dimensions() -> None:
    panel = build_default_paper_like_panel()
    model_ids = {m.model_id for m in panel.models}
    assert any("0.5" in m.parameter_scale or "0.5B" in m.parameter_scale for m in panel.models)
    assert any("llama" in m.model_id.lower() for m in panel.models)
    compressor_names = {c.compressor_name for c in panel.compressors}
    assert "int8" in compressor_names
    assert any("shard" in n for n in compressor_names)
    assert any("spectralquant" in n for n in compressor_names)
    metric_names = {m.metric_name for m in panel.metrics}
    assert "exactness_exactkv_failures" in metric_names
    assert "throughput_tokens_per_second" in metric_names
    workload_names = {w.workload_name for w in panel.workloads}
    assert any("long_context" in n for n in workload_names)
    assert any("remote_prefix" in n for n in workload_names)


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "panel contract",
        "not a reproduction result",
        "has not reproduced vericache throughput",
        "has not reproduced vericache memory",
        "has not reproduced vericache production serving",
        "external paper numbers are not exactkv results",
        "exactness",
        "throughput",
        "memory",
        "shard",
        "spectralquant",
        "stage 10",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "vericache reproduction complete today", "paper numbers are exactkv results"):
        assert phrase not in text


def test_performance_claim_rejected_on_contract_only() -> None:
    panel = build_default_paper_like_panel()
    panel.allowed_claims = ["ExactKV achieves throughput improvement on panel"]
    errors = validate_paper_like_reproduction_panel(panel)
    assert any("throughput" in e or "CLAIM_ELIGIBLE" in e for e in errors)
