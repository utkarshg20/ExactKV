"""Tests for Experiment 115 L4 runtime coupling stress panel (Phase 21N)."""
from __future__ import annotations

from exactkv.safety.l4_runtime_coupling_stress_panel import (
    DEFAULT_STRESS_COMPRESSORS,
    DEFAULT_STRESS_MAX_NEW_TOKENS,
    DEFAULT_STRESS_MODELS,
    EXPECTED_STRESS_CELL_COUNT,
    EXPERIMENT_115_ID,
    build_deterministic_stress_cache,
    build_deterministic_stress_generation_fn,
    compute_cross_model_consistency_metrics,
    compute_divergence_heatmap,
    compute_proposal_instability_rate,
    compute_verifier_stability_score,
    default_stress_panel_prompts,
    run_exp115_l4_runtime_coupling_stress_panel,
    validate_exp115_report,
    validate_exp115_stress_panel_report,
)


def test_stress_panel_144_cells_deterministic() -> None:
    report = run_exp115_l4_runtime_coupling_stress_panel(
        deterministic_mode=True,
    )
    assert report["experiment_id"] == EXPERIMENT_115_ID
    assert report["total_cells"] == EXPECTED_STRESS_CELL_COUNT
    assert report["expected_cells"] == 144
    assert report["successful_generation_cells"] == 144
    assert report["panel_outcome"] == "stress_panel_complete"
    assert report["trace_only"] is True
    assert report["runtime_commit_authorized"] is False
    assert validate_exp115_stress_panel_report(report).valid is True


def test_deterministic_replay_consistency() -> None:
    r1 = run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)
    r2 = run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)
    assert r1["decision_status_counts"] == r2["decision_status_counts"]
    assert r1["cross_model_metrics"] == r2["cross_model_metrics"]
    assert r1["verifier_stability_score"] == r2["verifier_stability_score"]


def test_cross_model_metrics_computed() -> None:
    report = run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)
    metrics = report["cross_model_metrics"]
    assert "cross_model_agreement_rate" in metrics
    assert "cross_model_prefix_stability" in metrics
    assert "cross_model_failure_delta" in metrics
    assert metrics["comparable_groups"] > 0


def test_divergence_heatmap_shape() -> None:
    prompts = default_stress_panel_prompts(6)
    cache = build_deterministic_stress_cache(
        models=DEFAULT_STRESS_MODELS,
        prompts=prompts,
        compressors=DEFAULT_STRESS_COMPRESSORS,
        max_new_tokens_values=DEFAULT_STRESS_MAX_NEW_TOKENS,
    )
    gen = build_deterministic_stress_generation_fn()
    cells_report = run_exp115_l4_runtime_coupling_stress_panel(
        prompts=prompts,
        cached_outputs=cache,
        generation_fn=gen,
    )
    heatmap = cells_report["divergence_heatmap"]["divergence"]
    for model in DEFAULT_STRESS_MODELS:
        assert model in heatmap
        for comp in DEFAULT_STRESS_COMPRESSORS:
            assert comp in heatmap[model]
            for mnt in DEFAULT_STRESS_MAX_NEW_TOKENS:
                assert str(mnt) in heatmap[model][comp]


def test_metric_correctness_on_controlled_cells() -> None:
    cells = [
        {
            "model_name": "m1",
            "prompt_id": "p0",
            "compressor": "noop",
            "max_new_tokens": 4,
            "generation_completed": True,
            "exactkv_failures": 0,
            "decisions": ["ACCEPT_PREFIX"],
            "trace_records": [
                {
                    "proposal_tokens": [1, 2, 3],
                    "decision": "ACCEPT_PREFIX",
                    "prefix_length": 3,
                },
            ],
        },
        {
            "model_name": "m2",
            "prompt_id": "p0",
            "compressor": "noop",
            "max_new_tokens": 4,
            "generation_completed": True,
            "exactkv_failures": 0,
            "decisions": ["ACCEPT_PREFIX"],
            "trace_records": [
                {
                    "proposal_tokens": [1, 2, 3],
                    "decision": "ACCEPT_PREFIX",
                    "prefix_length": 3,
                },
            ],
        },
        {
            "model_name": "m1",
            "prompt_id": "p0",
            "compressor": "int8",
            "max_new_tokens": 4,
            "generation_completed": True,
            "exactkv_failures": 0,
            "decisions": ["REJECT"],
            "trace_records": [
                {
                    "proposal_tokens": [9, 9],
                    "decision": "REJECT",
                    "prefix_length": 0,
                },
            ],
        },
        {
            "model_name": "m2",
            "prompt_id": "p0",
            "compressor": "int8",
            "max_new_tokens": 4,
            "generation_completed": True,
            "exactkv_failures": 0,
            "decisions": ["ACCEPT_PREFIX"],
            "trace_records": [
                {
                    "proposal_tokens": [1, 2],
                    "decision": "ACCEPT_PREFIX",
                    "prefix_length": 2,
                },
            ],
        },
    ]
    metrics = compute_cross_model_consistency_metrics(cells)
    assert metrics.cross_model_agreement_rate == 0.5
    assert metrics.cross_model_prefix_stability == 0.5
    assert compute_verifier_stability_score(cells) == 0.5
    assert compute_proposal_instability_rate(cells) == 0.5
    heat = compute_divergence_heatmap(
        cells,
        models=["m1", "m2"],
        compressors=["noop", "int8"],
        max_new_tokens_values=[4],
    )
    assert heat["divergence"]["m1"]["int8"]["4"] == 1.0
    assert heat["divergence"]["m2"]["noop"]["4"] == 0.0


def test_no_runtime_mutation_flags() -> None:
    report = run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)
    assert report["exactkv_generator_modified"] is False
    assert report["default_runtime_changed"] is False
    assert report["l4_activation"] is False
    gates = report["safety_gate_summary"]
    assert gates["proposal_used_for_token_commit"] is False
    assert gates["verifier_is_source_of_truth"] is True


def test_no_commit_leakage() -> None:
    report = run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)
    for cell in report["cells"]:
        for rec in cell.get("trace_records") or []:
            assert rec.get("dry_run_decision_used_for_token_commit") is False
            assert rec.get("exposed_to_generator") is False


def test_commit_leakage_fails_validation() -> None:
    report = run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)
    report["cells"][0]["trace_records"][0]["dry_run_decision_used_for_token_commit"] = True
    assert validate_exp115_stress_panel_report(report).valid is False
    assert validate_exp115_report(report)


def test_failure_conditions_tracked() -> None:
    report = run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)
    fc = report["failure_conditions_detected"]
    assert "missing_verifier_evidence" in fc
    assert "proposal_verifier_mismatch" in fc
    assert "compressor_induced_divergence" in fc
    assert "length_dependent_instability" in fc
