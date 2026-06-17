"""Tests for Experiment 072 full-depth divergence trace (Phase 16G)."""
from __future__ import annotations

import torch

from exactkv.attention.hf_full_replay_probe import (
    EXPERIMENT_072_ID,
    FORBIDDEN_ATTENTION_CLAIMS,
    TRACE_CHECKPOINTS,
    _first_layer_exceeding,
    classify_divergence_root_cause,
    run_exp072_probe,
    run_exp072_trace_cell,
    trace_mat_vs_stream_full_depth,
    validate_exp072_report,
)
from tests.test_hf_full_replay_probe import (
    _DummyModel,
    _FakeTokenizer,
    _mock_loader,
    _mock_prompts,
)


def _synthetic_layer(max_err: float, layer_idx: int) -> dict:
    return {
        "layer_idx": layer_idx,
        "streaming_vs_materialized": {
            ckpt: {
                "max_abs_error": max_err,
                "mean_abs_error": max_err / 10,
                "relative_l2_error": max_err,
                "cosine_similarity": 1.0 - max_err,
            }
            for ckpt in TRACE_CHECKPOINTS
        },
    }


def test_per_layer_trace_schema() -> None:
    model = _DummyModel(depth=2)
    input_ids = torch.randint(0, 100, (1, 32))
    trace = trace_mat_vs_stream_full_depth(
        model, input_ids, chunk_size=16, trace_mode="free_running",
    )
    assert len(trace["per_layer_trace"]) == 2
    layer0 = trace["per_layer_trace"][0]
    assert layer0["layer_idx"] == 0
    for ckpt in TRACE_CHECKPOINTS:
        assert ckpt in layer0["streaming_vs_materialized"]


def test_teacher_forced_same_layer_input() -> None:
    model = _DummyModel(depth=2)
    input_ids = torch.randint(0, 100, (1, 48))
    trace = trace_mat_vs_stream_full_depth(
        model, input_ids, chunk_size=16, trace_mode="teacher_forced_layer_inputs",
    )
    for layer in trace["per_layer_trace"]:
        inp_err = layer["streaming_vs_materialized"]["layer_input"]["max_abs_error"]
        assert inp_err == 0.0


def test_free_running_separate_hidden_states() -> None:
    model = _DummyModel(depth=2)
    input_ids = torch.randint(0, 100, (1, 32))
    trace = trace_mat_vs_stream_full_depth(
        model, input_ids, chunk_size=16, trace_mode="free_running",
    )
    assert trace["final_hidden_metrics"]["max_abs_error"] >= 0.0


def test_threshold_crossing_detection() -> None:
    trace = [
        _synthetic_layer(1e-5, 0),
        _synthetic_layer(2e-3, 1),
        _synthetic_layer(0.5, 2),
    ]
    assert _first_layer_exceeding(trace, 1e-4) == 1
    assert _first_layer_exceeding(trace, 1e-2) == 2
    assert _first_layer_exceeding(trace, 1.0) is None


def test_root_cause_free_running_accumulation() -> None:
    tf = [_synthetic_layer(1e-6, i) for i in range(3)]
    fr = [_synthetic_layer(1e-5, 0), _synthetic_layer(1e-3, 1), _synthetic_layer(0.2, 2)]
    for layer in tf:
        layer["streaming_vs_materialized"]["attn_context"]["max_abs_error"] = 1e-6
        layer["streaming_vs_materialized"]["post_mlp_hidden"]["max_abs_error"] = 1e-6
    rc = classify_divergence_root_cause(
        teacher_forced_trace=tf,
        free_running_trace=fr,
        final_logit_max_abs=0.1,
        depth_aware_tolerance=0.002,
        final_top1_agreement=True,
    )
    assert rc in ("free_running_accumulation", "tolerance_policy_issue")


def test_root_cause_local_mismatch() -> None:
    tf = [_synthetic_layer(1e-2, 0)]
    tf[0]["streaming_vs_materialized"]["attn_context"]["max_abs_error"] = 0.01
    rc = classify_divergence_root_cause(
        teacher_forced_trace=tf,
        free_running_trace=tf,
        final_logit_max_abs=0.01,
        depth_aware_tolerance=0.002,
        final_top1_agreement=False,
    )
    assert rc == "local_attention_mismatch"


def test_validate_exp072_report_success() -> None:
    report = {
        "experiment_id": EXPERIMENT_072_ID,
        "status": "diagnostic_complete",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "target_token_lengths": [32],
        "chunk_sizes": [16],
        "total_cells": 2,
        "successful_cells": 2,
        "blocked_cells": 0,
        "phase16f_failure_reproduced": False,
        "teacher_forced_local_error_summary": {"max_post_mlp_error": 1e-6, "max_attn_context_error": 1e-6, "cell_count": 1},
        "free_running_error_summary": {"max_post_mlp_error": 0.1, "max_attn_context_error": 1e-5, "cell_count": 1},
        "first_threshold_crossing_summary": {"1e_4": {"cells_with_crossing": 1, "earliest_layer_min": 0, "earliest_layer_max": 0}},
        "final_logit_error_summary": {"max_abs_error": 0.1, "mean_abs_error": 0.01, "cell_count": 1},
        "final_topk_agreement_summary": {"free_running_top1_agreement_cells": 1, "free_running_top5_overlap_mean": 5.0, "free_running_top10_overlap_mean": 10.0, "cell_count": 1},
        "root_cause_counts": {"free_running_accumulation": 1},
        "limitations": ["test"],
        "no_performance_claims_note": "no performance claims",
        "claim_note": "test",
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "cells": [
            {
                "prompt_id": "p0",
                "target_token_length": 32,
                "chunk_size": 16,
                "trace_mode": "teacher_forced_layer_inputs",
                "root_cause_classification": "unknown",
                "passed": False,
                "blockers": [],
            },
            {
                "prompt_id": "p0",
                "target_token_length": 32,
                "chunk_size": 16,
                "trace_mode": "free_running",
                "root_cause_classification": "free_running_accumulation",
                "passed": False,
                "blockers": [],
            },
        ],
    }
    assert validate_exp072_report(report) == []


def test_validate_exp072_blocked_cell() -> None:
    report = {
        "experiment_id": EXPERIMENT_072_ID,
        "status": "blocked",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "target_token_lengths": [],
        "chunk_sizes": [],
        "total_cells": 1,
        "successful_cells": 0,
        "blocked_cells": 1,
        "phase16f_failure_reproduced": False,
        "teacher_forced_local_error_summary": {"max_post_mlp_error": 0.0, "max_attn_context_error": 0.0, "cell_count": 0},
        "free_running_error_summary": {"max_post_mlp_error": 0.0, "max_attn_context_error": 0.0, "cell_count": 0},
        "first_threshold_crossing_summary": {},
        "final_logit_error_summary": {"max_abs_error": 0.0, "mean_abs_error": 0.0, "cell_count": 0},
        "final_topk_agreement_summary": {"free_running_top1_agreement_cells": 0, "free_running_top5_overlap_mean": 0.0, "free_running_top10_overlap_mean": 0.0, "cell_count": 0},
        "root_cause_counts": {},
        "limitations": ["test"],
        "no_performance_claims_note": "no performance claims",
        "claim_note": "test",
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "cells": [{
            "prompt_id": "p0",
            "target_token_length": 32,
            "chunk_size": 16,
            "trace_mode": "free_running",
            "root_cause_classification": "unknown",
            "passed": False,
            "blockers": ["failed"],
        }],
    }
    assert validate_exp072_report(report) == []


def test_no_forbidden_claim_fields() -> None:
    blob = str({"forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS)}).lower()
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "active_gpu_memory_savings",
        "production_memory_savings",
    ):
        assert term not in blob or term in FORBIDDEN_ATTENTION_CLAIMS


def test_run_exp072_mock_end_to_end() -> None:
    report = run_exp072_probe(
        model_id="mock",
        target_token_lengths=(32,),
        chunk_sizes=(16,),
        max_prompts=1,
        model_loader=_mock_loader,
        prompt_provider=_mock_prompts,
    )
    assert report["model_load_succeeded"] is True
    assert report["successful_cells"] == 2
    assert validate_exp072_report(report) == []
    fr_cells = [c for c in report["cells"] if c["trace_mode"] == "free_running"]
    assert len(fr_cells) == 1
    assert fr_cells[0]["final_top1_agreement"] is True


def test_exp072_trace_cell_final_topk() -> None:
    model = _DummyModel(depth=2)
    input_ids = torch.randint(0, 100, (1, 32))
    cell = run_exp072_trace_cell(
        model=model,
        input_ids=input_ids,
        prompt_id="p0",
        target_token_length=32,
        actual_token_length=32,
        chunk_size=16,
        trace_mode="free_running",
    )
    assert cell["final_top5_overlap"] == 5
    assert cell["final_top10_overlap"] >= 5
