"""Tests for Experiment 116 instability regime extraction (Phase 21O)."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.analysis.l4_instability_regime_extractor import (
    EXPERIMENT_116_ID,
    REGIME_NAMES,
    classify_regime,
    compute_cell_instability_score,
    load_exp115_report,
    normalize_exp115_metrics,
    run_exp116_instability_regime_extraction,
    validate_exp116_report,
)
from exactkv.safety.l4_runtime_coupling_stress_panel import (
    EXPECTED_STRESS_CELL_COUNT,
    run_exp115_l4_runtime_coupling_stress_panel,
)


def _minimal_exp115_report() -> dict:
    return run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)


def test_load_exp115_report_from_generated() -> None:
    exp115 = _minimal_exp115_report()
    path = Path("reports/test_exp116_exp115_fixture.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(exp115) + "\n")
    loaded = load_exp115_report(path)
    assert loaded["total_cells"] == EXPECTED_STRESS_CELL_COUNT


def test_deterministic_output() -> None:
    exp115 = _minimal_exp115_report()
    r1 = run_exp116_instability_regime_extraction(exp115)
    r2 = run_exp116_instability_regime_extraction(exp115)
    r1.pop("generated_at", None)
    r2.pop("generated_at", None)
    assert r1 == r2


def test_full_pipeline_on_exp115() -> None:
    report = run_exp116_instability_regime_extraction(_minimal_exp115_report())
    assert report["experiment_id"] == EXPERIMENT_116_ID
    assert report["source_total_cells"] == 144
    assert report["analysis_only"] is True
    assert report["exactkv_generator_modified"] is False
    assert validate_exp116_report(report).valid is True


def test_regime_classification_coverage() -> None:
    report = run_exp116_instability_regime_extraction(_minimal_exp115_report())
    regimes = report["instability_regimes"]
    for name in REGIME_NAMES:
        assert name in regimes
    assert sum(len(v) for v in regimes.values()) == 144


def test_interaction_completeness() -> None:
    report = run_exp116_instability_regime_extraction(_minimal_exp115_report())
    interactions = report["interaction_effects"]
    assert len(interactions["compressor_length"]) == 12
    assert len(interactions["model_compressor"]) == 8
    assert len(interactions["model_length"]) == 6


def test_phase_boundaries_present() -> None:
    exp115 = _minimal_exp115_report()
    report = run_exp116_instability_regime_extraction(exp115)
    boundaries = report["phase_boundaries"]
    assert "compressor_thresholds" in boundaries
    assert "length_thresholds" in boundaries
    assert len(boundaries["model_sensitivity_order"]) == 2


def test_failure_taxonomy_fields() -> None:
    report = run_exp116_instability_regime_extraction(_minimal_exp115_report())
    taxonomy = report["failure_taxonomy"]
    for key in (
        "verifier_mismatch",
        "proposal_instability",
        "compressor_drift",
        "length_collapse",
        "cross_model_divergence",
    ):
        assert key in taxonomy
        assert isinstance(taxonomy[key], int)


def test_stability_surface_grid() -> None:
    report = run_exp116_instability_regime_extraction(_minimal_exp115_report())
    surface = report["stability_surface"]
    assert len(surface["grid"]) == 144
    assert surface["peak_stability_regions"]
    assert surface["valley_instability_regions"]


def test_normalize_metrics() -> None:
    exp115 = _minimal_exp115_report()
    metrics = normalize_exp115_metrics(exp115)
    assert "cross_model_agreement_rate" in metrics
    assert "verifier_stability_score" in metrics


def test_classify_regime_thresholds() -> None:
    assert classify_regime(0.1) == "stable"
    assert classify_regime(0.3) == "moderate_drift"
    assert classify_regime(0.6) == "high_divergence"
    assert classify_regime(0.9) == "failure_prone"


def test_cell_instability_score_blocked_generation() -> None:
    cell = {"generation_completed": False, "exactkv_failures": 0}
    score = compute_cell_instability_score(
        cell,
        group_context={},
        global_metrics=normalize_exp115_metrics(_minimal_exp115_report()),
    )
    assert score == 1.0


def test_missing_regime_fails_validation() -> None:
    report = run_exp116_instability_regime_extraction(_minimal_exp115_report())
    del report["instability_regimes"]["stable"]
    assert validate_exp116_report(report).valid is False
