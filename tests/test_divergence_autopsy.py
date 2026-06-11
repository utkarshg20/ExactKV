"""Tests for divergence autopsy helpers (Experiment 019)."""
from __future__ import annotations

import json

import pytest

from exactkv.analysis.divergence_autopsy import (
    FORBIDDEN_AUTOPSY_FIELDS,
    assert_autopsy_artifact_safe,
    build_repair_hypotheses,
    classify_token_text,
    load_autopsy_prompt_subset,
    logit_margin,
    structured_output_state,
    top_k_token_ids,
)
from exactkv.runtime.model_runtime import ModelRuntime


def test_load_autopsy_prompt_subset_count_and_suites():
    prompts = load_autopsy_prompt_subset(per_suite=5)
    assert len(prompts) == 25
    suites = {p["v10_suite"] for p in prompts}
    assert suites == {
        "long_context",
        "retrieval_copy",
        "tool_json",
        "code_structured",
        "core_v2",
    }
    ids_by_suite: dict[str, list[str]] = {}
    for p in prompts:
        ids_by_suite.setdefault(p["v10_suite"], []).append(p["prompt_id"])
    for suite, ids in ids_by_suite.items():
        assert len(ids) == 5
        assert ids == sorted(ids)


def test_classify_token_types():
    assert classify_token_text(" ") == "whitespace"
    assert classify_token_text("{") == "bracket"
    assert classify_token_text('"') == "quote"
    assert classify_token_text(",") == "punctuation"
    assert classify_token_text("42") == "numeric"
    assert classify_token_text("hello") == "wordpiece/other"


def test_structured_output_state():
    state = structured_output_state('{"a": [1, 2')
    assert state["jsonish_prefix"] is True
    assert state["unmatched_brackets"] is True
    assert state["bracket_depth"] >= 1


def test_top_k_and_margin():
    import torch

    logits = torch.tensor([0.1, 2.0, 1.0, 0.5])
    assert top_k_token_ids(logits, 3) == [1, 2, 3]
    assert logit_margin(logits, 1, 0) == pytest.approx(1.9)


def test_assert_autopsy_artifact_safe_rejects_forbidden():
    with pytest.raises(ValueError, match="throughput"):
        assert_autopsy_artifact_safe({"throughput": 1.0})
    assert_autopsy_artifact_safe({"gpu_peak_allocated_during_run_bytes": 100})


def test_forbidden_fields_not_in_standard_report_schema():
    from exactkv.benchmarks.reports import validate_report

    sample = {
        "manifest": {"model_name": "test"},
        "results": [
            {
                "prompt_id": "p1",
                "compressor_name": "int8",
                "draft_len": 4,
                "exactkv_failure": False,
                "lossy": {"token_exact_match": True},
                "exactkv": {"acceptance": {"acceptance_rate": 1.0}},
            }
        ],
        "aggregate": {"total_runs": 1, "exactkv_failures": 0},
    }
    validate_report(sample)
    for field in FORBIDDEN_AUTOPSY_FIELDS:
        assert field not in json.dumps(sample)


def test_build_repair_hypotheses_returns_hypothesis_only():
    hyps = build_repair_hypotheses([], [], [], {}, [])
    assert hyps
    assert all(h["status"] == "hypothesis_only" for h in hyps)


def test_run_autopsy_cell_noop_cpu():
    from exactkv.analysis.divergence_autopsy import run_autopsy_cell
    from exactkv.compressors import get_compressor

    try:
        runtime = ModelRuntime("Qwen/Qwen2.5-0.5B", device="cpu", dtype="float32")
    except Exception as exc:
        pytest.skip(f"model unavailable offline: {exc}")
    prompts = load_autopsy_prompt_subset(per_suite=1)
    cell = run_autopsy_cell(
        runtime,
        prompts[0],
        get_compressor("noop"),
        draft_len=4,
        max_new_tokens=8,
        collect_kv_errors=False,
        collect_attention=False,
    )
    assert cell["exactkv_failure"] is False
    assert cell["lossy"]["token_exact_match"] is True
    assert_autopsy_artifact_safe(cell)
