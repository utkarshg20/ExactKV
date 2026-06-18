"""Tests for Experiment 088 longer-context validation panel (Phase 17C)."""
from __future__ import annotations

import json

from exactkv.demo.long_context_validation import (
    CLAIM_SCOPE_NOTE,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_FAMILIES,
    DEFAULT_TARGET_CONTEXT_TOKENS,
    EXPERIMENT_088_ID,
    FAMILY_FILLERS,
    build_panel_cells,
    generate_family_long_prompt,
    resolve_model_panel,
    run_exp088_long_context_validation_panel,
    validate_exp088_report,
)

TOKENS = [200, 201, 202, 203]


def _word_tokenize(text: str) -> list[int]:
    return list(range(len(text.split())))


def _word_decode(ids: list[int]) -> str:
    return " ".join(f"w{i}" for i in ids)


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
        "model_id": "test_model",
        "target_context_tokens": (32, 64),
        "prompt_families": ("factual", "structured"),
        "compressors_requested": ["noop", "int8"],
        "tokenize_fn": _word_tokenize,
        "baseline_generation_fn": _baseline_fn,
        "guarded_generation_fn": _guarded_fn,
    }
    defaults.update(overrides)
    return run_exp088_long_context_validation_panel(**defaults)


def test_prompt_families_deterministic() -> None:
    for family in DEFAULT_PROMPT_FAMILIES:
        assert family in FAMILY_FILLERS
        t1, a1 = generate_family_long_prompt(
            family, 40, tokenize_fn=_word_tokenize, decode_fn=_word_decode,
        )
        t2, a2 = generate_family_long_prompt(
            family, 40, tokenize_fn=_word_tokenize, decode_fn=_word_decode,
        )
        assert t1 == t2
        assert a1 == a2
        assert a1 > 0


def test_target_vs_actual_token_count_fields() -> None:
    specs = build_panel_cells(
        target_context_tokens=[48],
        prompt_families=["code"],
        compressors=["noop"],
        tokenize_fn=_word_tokenize,
        decode_fn=_word_decode,
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec["target_context_tokens"] == 48
    assert spec["actual_prompt_token_count"] > 0
    assert spec["prompt_family"] == "code"


def test_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_088_ID
    assert validate_exp088_report(report) == []


def test_default_model_panel() -> None:
    default, optional, panel = resolve_model_panel()
    assert default == [DEFAULT_MODEL_ID]
    assert optional == []
    assert panel == [DEFAULT_MODEL_ID]


def test_instruct_only_behind_flag() -> None:
    _, optional, panel = resolve_model_panel(include_instruct=True)
    assert len(optional) == 1
    assert len(panel) == 2


def test_blocked_model_recorded_without_fake_success() -> None:
    def _loader(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("model blocked")

    report = run_exp088_long_context_validation_panel(
        model_id="blocked_model",
        target_context_tokens=[32],
        prompt_families=["factual"],
        compressors_requested=["noop"],
        runtime_loader=_loader,
        allow_model_blocked=True,
    )
    assert report["models_blocked"]
    assert report["models_blocked"][0]["model_id"] == "blocked_model"
    assert report["total_cells"] == 0
    assert report["status"] == "blocked"


def test_default_model_blocked() -> None:
    def _loader(**kwargs: object) -> None:
        del kwargs
        raise OSError("no model")

    report = run_exp088_long_context_validation_panel(
        target_context_tokens=[32],
        prompt_families=["factual"],
        compressors_requested=["noop"],
        runtime_loader=_loader,
        allow_model_blocked=True,
    )
    assert report["models_blocked"]
    assert report["status"] == "blocked"
    assert "default_model_blocked" in report["blockers"]


def test_cell_parity_aggregation() -> None:
    report = _run_panel()
    # 1 model × 2 lengths × 2 families × 2 compressors = 8
    assert report["total_cells"] == 8
    assert report["baseline_vs_guarded_token_match_cells"] == 8
    assert report["baseline_vs_guarded_text_match_cells"] == 8


def test_context_length_summary() -> None:
    report = _run_panel()
    summary = report["context_length_summary"]
    assert "32" in summary
    assert "64" in summary
    assert summary["32"]["cells"] == 4
    assert summary["64"]["cells"] == 4


def test_safety_gate_aggregation() -> None:
    report = _run_panel()
    assert report["safety_gate_summary"]["cells_all_gates_ok"] == 8


def test_claim_scope_note_present() -> None:
    report = _run_panel()
    assert CLAIM_SCOPE_NOTE in report["claim_scope_note"]
    assert "context-length-scoped" in report["claim_scope_note"].lower()


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
                    "long-context support proven",
                ):
                    assert forbidden not in dumped


def test_default_panel_cell_count() -> None:
    specs = build_panel_cells(
        target_context_tokens=DEFAULT_TARGET_CONTEXT_TOKENS,
        prompt_families=DEFAULT_PROMPT_FAMILIES,
        compressors=["noop", "int8"],
        tokenize_fn=_word_tokenize,
        decode_fn=_word_decode,
    )
    assert len(specs) == 18
