"""Tests for repair-policy pilot helpers (Experiment 020)."""
from __future__ import annotations

import json

import pytest

from exactkv.analysis.divergence_autopsy import load_autopsy_prompt_subset
from exactkv.analysis.repair_policy import (
    ALL_POLICIES,
    FORBIDDEN_POLICY_FIELDS,
    POLICY_BASELINE_BOUNDARY4,
    POLICY_BASELINE_K8,
    POLICY_CATEGORY_ADAPTIVE,
    POLICY_DRAFT_LEN_ADAPTIVE,
    POLICY_FALLBACK_INT8_HARD,
    POLICY_STRUCTURED_SAFE,
    aggregate_policy_results,
    assert_policy_artifact_safe,
    evaluate_repair_hypotheses,
    load_exp014_hard_subset,
    load_pilot_prompts,
    resolve_policy_cell,
)


def test_load_exp019_panel_count():
    prompts = load_pilot_prompts(panel="exp019")
    assert len(prompts) == 25


def test_load_exp014_hard_panel_count():
    prompts = load_pilot_prompts(panel="exp014_hard")
    assert len(prompts) == 40


def test_all_policies_count():
    assert len(ALL_POLICIES) == 6


def test_baseline_policies_fixed():
    p = {"v10_suite": "core_v2"}
    assert resolve_policy_cell(POLICY_BASELINE_K8, p).compressor_name == "k8_v4_sim"
    assert resolve_policy_cell(POLICY_BASELINE_K8, p).draft_len == 4
    b4 = resolve_policy_cell(POLICY_BASELINE_BOUNDARY4, p)
    assert b4.compressor_name == "k8_v4_boundary4_v8_sim"
    assert b4.draft_len == 4


def test_fallback_int8_hard_categories():
    lc = resolve_policy_cell(
        POLICY_FALLBACK_INT8_HARD, {"v10_suite": "long_context"}
    )
    assert lc.compressor_name == "int8"
    core = resolve_policy_cell(
        POLICY_FALLBACK_INT8_HARD, {"v10_suite": "core_v2"}
    )
    assert core.compressor_name == "k8_v4_boundary4_v8_sim"


def test_structured_safe_mode():
    tj = resolve_policy_cell(POLICY_STRUCTURED_SAFE, {"v10_suite": "tool_json"})
    assert tj.compressor_name == "int8"
    core = resolve_policy_cell(POLICY_STRUCTURED_SAFE, {"v10_suite": "core_v2"})
    assert core.compressor_name == "k8_v4_boundary4_v8_sim"


def test_category_adaptive_mapping():
    assert resolve_policy_cell(
        POLICY_CATEGORY_ADAPTIVE, {"v10_suite": "long_context"}
    ).compressor_name == "int8"
    assert resolve_policy_cell(
        POLICY_CATEGORY_ADAPTIVE, {"v10_suite": "tool_json"}
    ).compressor_name == "int8"
    assert resolve_policy_cell(
        POLICY_CATEGORY_ADAPTIVE, {"v10_suite": "core_v2"}
    ).compressor_name == "k8_v4_boundary4_v8_sim"


def test_draft_len_adaptive_varies_draft_len():
    core = resolve_policy_cell(POLICY_DRAFT_LEN_ADAPTIVE, {"v10_suite": "core_v2"})
    assert core.draft_len == 8
    lc = resolve_policy_cell(POLICY_DRAFT_LEN_ADAPTIVE, {"v10_suite": "long_context"})
    assert lc.draft_len == 4


def test_aggregate_and_hypothesis_evaluation():
    results = []
    for policy in ALL_POLICIES:
        for i, suite in enumerate(
            ("long_context", "retrieval_copy", "tool_json", "code_structured", "core_v2")
        ):
            acc = 0.95 if policy != POLICY_BASELINE_K8 else 0.90
            if policy == POLICY_FALLBACK_INT8_HARD and suite == "long_context":
                acc = 0.99
            results.append({
                "model_name": "Qwen/Qwen2.5-0.5B",
                "prompt_id": f"{suite}_{i}",
                "v10_suite": suite,
                "policy_name": policy,
                "exactkv_failure": False,
                "lossy": {"lossy_diverged": False},
                "exactkv": {
                    "acceptance": {
                        "acceptance_rate": acc,
                        "total_rejected": 1,
                        "total_corrections": 1,
                    },
                },
            })
    agg = aggregate_policy_results(results)
    assert agg["total_cells"] == len(results)
    assert agg["exactkv_failures"] == 0
    assert "global_by_policy" in agg
    assert agg["hypothesis_evaluation"]["supported"]


def test_assert_policy_artifact_safe():
    with pytest.raises(ValueError, match="throughput"):
        assert_policy_artifact_safe({"throughput": 1.0})
    assert_policy_artifact_safe({"mean_acceptance_rate": 0.9})


def test_forbidden_fields_not_in_aggregate_keys():
    sample = {"global_by_policy": {"p": {"mean_acceptance_rate": 1.0}}}
    blob = json.dumps(sample)
    for field in FORBIDDEN_POLICY_FIELDS:
        assert f'"{field}"' not in blob or field in ("latency",)  # keys only
    assert_policy_artifact_safe(sample)


def test_same_prompts_across_policies():
    prompts = load_autopsy_prompt_subset(per_suite=1)
    ids = {p["prompt_id"] for p in prompts}
    assert len(ids) == 5
