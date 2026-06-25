"""Tests for Experiment 114 L4 minimal runtime coupling layer (Phase 21M)."""
from __future__ import annotations

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.l4_runtime_coupling import (
    EXPERIMENT_114_ID,
    build_runtime_trace_records,
    build_synthetic_model_output,
    build_verifier_evidence_from_round_trace,
    extract_proposals_from_model_outputs,
    load_model_outputs,
    run_exp114_l4_minimal_runtime_coupling_panel,
    run_verifier_comparison,
    validate_exp114_panel_report,
    validate_exp114_report,
)
from exactkv.safety.l4_runtime_coupling import ModelOutputRecord

MATCH_TOKENS = (10, 11, 12, 13)
MISMATCH_TOKENS = (10, 11, 99, 13)


def _full_match_gen(**kwargs: object) -> dict:
    del kwargs
    return build_synthetic_model_output(
        prompt_id="p0",
        compressor="noop",
        draft_tokens=MATCH_TOKENS,
        verifier_tokens=MATCH_TOKENS,
    )


def _mismatch_gen(**kwargs: object) -> dict:
    del kwargs
    return build_synthetic_model_output(
        prompt_id="p0",
        compressor="noop",
        draft_tokens=MISMATCH_TOKENS,
        verifier_tokens=MATCH_TOKENS,
    )


def _missing_verifier_gen(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": [10, 11],
        "exactkv_failures": 0,
        "result_traces": [
            {
                "round_idx": 0,
                "draft_tokens": list(MATCH_TOKENS),
                "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
            },
        ],
    }


def test_model_output_ingestion_cached() -> None:
    cached = {
        "p0|noop|4": _full_match_gen(),
        "p0|int8|4": _full_match_gen(),
        "p1|noop|4": _full_match_gen(),
        "p1|int8|4": _full_match_gen(),
    }
    records = load_model_outputs(
        "Qwen/Qwen2.5-0.5B",
        [("p0", "hello"), ("p1", "world")],
        4,
        compressors=["noop", "int8"],
        cached_outputs=cached,
    )
    assert len(records) == 4
    assert all(r.ingestion_source == "cached_outputs" for r in records)
    assert all(r.generation_output["generation_completed"] for r in records)


def test_model_output_ingestion_generation_fn() -> None:
    records = load_model_outputs(
        "test-model",
        [("p0", "hello")],
        4,
        compressors=["noop"],
        generation_fn=_full_match_gen,
    )
    assert len(records) == 1
    assert records[0].ingestion_source == "generation_fn"


def test_proposal_extraction_correctness() -> None:
    gen = _full_match_gen()
    proposals = extract_proposals_from_model_outputs(
        gen,
        prompt_id="p0",
        compressor="noop",
    )
    assert len(proposals) == 1
    assert proposals[0].proposal_token_ids == MATCH_TOKENS
    assert proposals[0].proposal_source == PROPOSAL_SOURCE_ROUND_LOG
    assert proposals[0].trace_id == "p0|noop|round_0"


def test_verifier_comparison_full_match() -> None:
    evidence = build_verifier_evidence_from_round_trace(
        {
            "draft_tokens": list(MATCH_TOKENS),
            "acceptance": {"verifier_tokens": list(MATCH_TOKENS)},
        },
        proposal_token_ids=MATCH_TOKENS,
    )
    result = run_verifier_comparison(MATCH_TOKENS, evidence)
    assert result.decision == "ACCEPT_PREFIX"
    assert result.prefix_match_length == len(MATCH_TOKENS)
    assert result.mismatch_index is None
    assert result.dry_run_decision_used_for_token_commit is False


def test_verifier_comparison_reject() -> None:
    evidence = build_verifier_evidence_from_round_trace(
        {
            "draft_tokens": list(MISMATCH_TOKENS),
            "acceptance": {"verifier_tokens": list(MATCH_TOKENS)},
        },
        proposal_token_ids=MISMATCH_TOKENS,
    )
    result = run_verifier_comparison(MISMATCH_TOKENS, evidence)
    assert result.decision == "REJECT"
    assert result.mismatch_index == 2
    assert result.prefix_match_length == 2


def test_missing_evidence_blocking() -> None:
    result = run_verifier_comparison(
        MATCH_TOKENS,
        {
            "verifier_evidence_available": False,
            "verifier_block_reason": "no explicit verifier evidence in trace",
        },
    )
    assert result.decision == "BLOCK_MISSING_EVIDENCE"
    assert result.prefix_match_length == 0


def test_full_match_acceptance_pipeline() -> None:
    record = ModelOutputRecord(
        model_name="test",
        prompt_id="p0",
        prompt_text="hello",
        compressor="noop",
        max_new_tokens=4,
        generation_output=_full_match_gen(),
        ingestion_source="generation_fn",
    )
    traces = build_runtime_trace_records(record)
    assert len(traces) == 1
    assert traces[0].decision == "ACCEPT_PREFIX"
    assert traces[0].prefix_length == len(MATCH_TOKENS)
    assert traces[0].dry_run_decision_used_for_token_commit is False


def test_missing_evidence_pipeline() -> None:
    record = ModelOutputRecord(
        model_name="test",
        prompt_id="p0",
        prompt_text="hello",
        compressor="noop",
        max_new_tokens=4,
        generation_output=_missing_verifier_gen(),
        ingestion_source="generation_fn",
    )
    traces = build_runtime_trace_records(record)
    assert len(traces) == 1
    assert traces[0].decision == "BLOCK_MISSING_EVIDENCE"


def test_panel_report_with_generation_fn() -> None:
    report = run_exp114_l4_minimal_runtime_coupling_panel(
        model_name="test-model",
        prompts=[("p0", "hello"), ("p1", "world")],
        compressors=["noop", "int8"],
        max_new_tokens=4,
        generation_fn=_full_match_gen,
    )
    assert report["experiment_id"] == EXPERIMENT_114_ID
    assert report["status"] == "panel_complete"
    assert report["trace_records_total"] >= 2
    assert report["l4_activation"] is False
    assert report["runtime_commit_authorized"] is False
    assert validate_exp114_panel_report(report).valid is True
    assert validate_exp114_report(report) == []


def test_mismatch_panel_decision() -> None:
    report = run_exp114_l4_minimal_runtime_coupling_panel(
        model_name="test-model",
        prompts=[("p0", "hello")],
        compressors=["noop"],
        max_new_tokens=4,
        generation_fn=_mismatch_gen,
    )
    cell = report["cells"][0]
    assert cell["decisions"][0] == "REJECT"


def test_commit_flag_fails_validation() -> None:
    report = run_exp114_l4_minimal_runtime_coupling_panel(
        model_name="test-model",
        prompts=[("p0", "hello")],
        compressors=["noop"],
        generation_fn=_full_match_gen,
    )
    report["cells"][0]["trace_records"][0]["dry_run_decision_used_for_token_commit"] = True
    assert validate_exp114_panel_report(report).valid is False
