"""Tests for Experiment 087 broader model validation panel (Phase 17B)."""
from __future__ import annotations

import json

from exactkv.demo.broader_model_validation import (
    CLAIM_SCOPE_NOTE,
    DEFAULT_MODEL_IDS,
    EXPERIMENT_087_ID,
    OPTIONAL_MODEL_IDS,
    resolve_model_panel,
    run_exp087_broader_model_validation_panel,
    validate_exp087_report,
)

TOKENS = [100, 101, 102, 103]


def _baseline_fn(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": TOKENS,
        "generated_text": "out",
        "exactkv_failures": 0,
        "token_exact_match": True,
        "blockers": [],
    }


def _guarded_fn(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": TOKENS,
        "generated_text": "out",
        "exactkv_failures": 0,
        "token_exact_match": True,
        "live_snapshots": [],
        "decode_time_shadow_cells": [],
        "decode_time_shadow_callback_count": 0,
        "decode_time_shadow_successful_callbacks": 0,
        "decode_time_shadow_exception_callbacks": 0,
        "blockers": [],
    }


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "model_ids": ["model_a", "model_b"],
        "prompts": [("p0", "hello"), ("p1", "world")],
        "compressors_requested": ["noop", "int8"],
        "baseline_generation_fn": _baseline_fn,
        "guarded_generation_fn": _guarded_fn,
    }
    defaults.update(overrides)
    return run_exp087_broader_model_validation_panel(**defaults)


def test_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_087_ID
    assert validate_exp087_report(report) == []


def test_default_models_requested() -> None:
    default, optional, panel = resolve_model_panel()
    assert default == list(DEFAULT_MODEL_IDS)
    assert optional == []
    assert panel == list(DEFAULT_MODEL_IDS)


def test_optional_models_only_behind_flag() -> None:
    _, optional, panel = resolve_model_panel(include_optional_models=True)
    assert optional == list(OPTIONAL_MODEL_IDS)
    assert len(panel) == len(DEFAULT_MODEL_IDS) + len(OPTIONAL_MODEL_IDS)


def test_blocked_model_recorded_without_fake_success() -> None:
    def _loader(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("model blocked")

    report = run_exp087_broader_model_validation_panel(
        model_ids=["blocked_model"],
        prompts=[("p0", "hi")],
        compressors_requested=["noop"],
        runtime_loader=_loader,
        allow_model_blocked=True,
    )
    assert report["models_blocked"]
    assert report["models_blocked"][0]["model_id"] == "blocked_model"
    assert report["total_cells"] == 0
    assert report["status"] == "blocked"


def test_all_default_models_blocked() -> None:
    def _loader(**kwargs: object) -> None:
        del kwargs
        raise OSError("no model")

    report = run_exp087_broader_model_validation_panel(
        prompts=[("p0", "hi")],
        compressors_requested=["noop"],
        runtime_loader=_loader,
        allow_model_blocked=True,
    )
    assert len(report["models_blocked"]) == len(DEFAULT_MODEL_IDS)
    assert report["status"] == "blocked"
    assert "all_default_models_blocked" in report["blockers"]


def test_cell_parity_aggregation() -> None:
    report = _run_panel()
    assert report["total_cells"] == 8
    assert report["baseline_vs_guarded_token_match_cells"] == 8
    assert report["baseline_vs_guarded_text_match_cells"] == 8


def test_safety_gate_aggregation() -> None:
    report = _run_panel()
    assert report["safety_gate_summary"]["cells_all_gates_ok"] == 8


def test_claim_scope_note_present() -> None:
    report = _run_panel()
    assert CLAIM_SCOPE_NOTE in report["claim_scope_note"]
    assert "model-scoped" in report["claim_scope_note"].lower()


def test_no_forbidden_positive_claim_phrases() -> None:
    report = _run_panel()
    for mr in report["model_results"]:
        for cell in mr["cells"]:
            for field in ("baseline_generated_token_ids", "blockers"):
                dumped = json.dumps(cell.get(field, "")).lower()
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
                    "broad model-family support proven",
                ):
                    assert forbidden not in dumped


def test_parity_failure_marks_report_failed() -> None:
    def _bad_guarded(**kwargs: object) -> dict:
        out = _guarded_fn(**kwargs)
        out["generated_token_ids"] = [999]
        return out

    report = _run_panel(guarded_generation_fn=_bad_guarded)
    assert report["status"] == "failed"
    assert report["baseline_vs_guarded_token_match_cells"] < report["total_cells"]
