"""Tests for Experiment 080 ExactKV round-log shadow observer (Phase 16O)."""
from __future__ import annotations

import json

import torch

from exactkv.attention.generation_shadow_observer import (
    BLOCKED_MISSING_ROUND_LOG,
    DEFAULT_EXP080_COMPRESSORS,
    EXPERIMENT_080_ID,
    SHADOW_FORBIDDEN_CLAIMS,
    GenerationOutput,
    build_round_boundary_input_ids,
    default_exp080_prompts,
    extract_round_log_entries,
    run_exp080_round_log_shadow_panel,
    validate_exp080_report,
)
from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace


def _acceptance(
    *,
    accepted: int,
    rejected: int = 0,
    correction: int | None = None,
) -> AcceptanceResult:
    draft = list(range(100, 104))
    verifier = draft[:accepted] + ([correction] if correction is not None else [])
    return AcceptanceResult(
        draft_tokens=draft,
        verifier_tokens=verifier,
        accepted_tokens=draft[:accepted],
        correction_token=correction,
        rejected_tokens=draft[accepted:],
        bonus_token=None,
        all_matched=correction is None and rejected == 0,
        num_accepted=accepted,
        num_rejected=rejected,
    )


def _fake_traces(prompt_len: int = 5) -> list[VerificationTrace]:
    return [
        VerificationTrace(
            round_idx=0,
            draft_tokens=[100, 101, 102, 103],
            acceptance=_acceptance(accepted=4),
            full_seq_len_before=prompt_len,
            full_seq_len_after=prompt_len + 4,
            compressed_seq_len_after=prompt_len + 4,
        ),
        VerificationTrace(
            round_idx=1,
            draft_tokens=[104, 105, 106, 107],
            acceptance=_acceptance(accepted=2, rejected=2, correction=199),
            full_seq_len_before=prompt_len + 4,
            full_seq_len_after=prompt_len + 7,
            compressed_seq_len_after=prompt_len + 7,
        ),
    ]


def _fake_generation(**kwargs: object) -> GenerationOutput:
    max_new = int(kwargs.get("max_new_tokens", 8))
    token_ids = list(range(100, 100 + max_new))
    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
    return GenerationOutput(
        generation_completed=True,
        generation_output_text="generated output",
        generation_output_token_ids=token_ids,
        prompt_ids=prompt_ids,
        full_sequence_ids=torch.cat(
            [prompt_ids, torch.tensor([token_ids], dtype=torch.long)], dim=1,
        ),
        exactkv_traces=_fake_traces(),
    )


def _fake_shadow_replay(**kwargs: object) -> dict:
    input_ids = kwargs["input_ids"]
    rnd = int(input_ids.shape[-1]) - 5
    top1 = rnd < 10
    return {
        "num_layers_replayed": 2,
        "streaming_vs_materialized_logit_metrics": {
            "max_abs_error": 0.001 * rnd,
            "top1_agreement": top1,
            "top5_overlap": 5,
            "top10_overlap": 10,
        },
        "full_vs_streaming_logit_metrics": {"max_abs_error": 0.002 * rnd},
        "blockers": [],
    }


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "prompts": default_exp080_prompts()[:2],
        "max_new_tokens": 8,
        "compressors_requested": ["noop"],
        "generation_fn": _fake_generation,
        "shadow_replay_fn": _fake_shadow_replay,
    }
    defaults.update(overrides)
    return run_exp080_round_log_shadow_panel(**defaults)


def test_round_log_extraction_from_fake_traces() -> None:
    gen = _fake_generation()
    entries, blockers = extract_round_log_entries(gen)
    assert blockers == []
    assert len(entries) == 2
    assert entries[0]["round_index"] == 0
    assert entries[0]["draft_length"] == 4
    assert entries[0]["accepted_token_count"] == 4
    assert entries[1]["rejected_or_corrected_token_count"] == 3


def test_missing_optional_fields_become_null() -> None:
    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="x",
        generation_output_token_ids=[10],
        prompt_ids=torch.tensor([[1, 2, 3]]),
        exactkv_traces=[{
            "round_idx": 0,
            "draft_tokens": None,
            "acceptance": None,
            "full_seq_len_before": 3,
            "full_seq_len_after": 4,
        }],
    )
    entries, _ = extract_round_log_entries(gen)
    assert entries[0]["draft_length"] is None
    assert entries[0]["accepted_token_count"] is None


def test_missing_round_log_blocks_by_default() -> None:
    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="x",
        generation_output_token_ids=[10, 11],
        prompt_ids=torch.tensor([[1, 2, 3]]),
    )
    report = _run_panel(generation_fn=lambda **k: gen)
    cell = report["generation_cells"][0]
    assert cell["round_log_available"] is False
    assert cell["round_shadow_cells"] == []
    assert BLOCKED_MISSING_ROUND_LOG in " ".join(cell["blockers"])


def test_fallback_prefix_ladder_only_when_enabled() -> None:
    gen = GenerationOutput(
        generation_completed=True,
        generation_output_text="x",
        generation_output_token_ids=[10, 11, 12],
        prompt_ids=torch.tensor([[1, 2, 3]]),
    )
    blocked = _run_panel(generation_fn=lambda **k: gen, fallback_prefix_ladder=False)
    assert blocked["generation_cells"][0]["round_shadow_cells"] == []

    fallback = _run_panel(generation_fn=lambda **k: gen, fallback_prefix_ladder=True)
    cell = fallback["generation_cells"][0]
    assert cell["fallback_prefix_ladder_used"] is True
    assert len(cell["round_shadow_cells"]) == 4  # k=0..3


def test_round_boundary_sequence_construction() -> None:
    gen = _fake_generation()
    entries, _ = extract_round_log_entries(gen)
    input_ids, blockers = build_round_boundary_input_ids(gen, entries[0])
    assert blockers == []
    assert input_ids is not None
    assert int(input_ids.shape[-1]) == entries[0]["prefix_length_after_round"]


def test_first_status_change_by_round() -> None:
    def _shadow(**kwargs: object) -> dict:
        input_ids = kwargs["input_ids"]
        status_val = int(input_ids.shape[-1])
        return {
            "num_layers_replayed": 2,
            "streaming_vs_materialized_logit_metrics": {
                "max_abs_error": 0.001,
                "top1_agreement": True,
            },
            "full_vs_streaming_logit_metrics": {"max_abs_error": 0.001},
            "blockers": [],
            "_seq_len": status_val,
        }

    report = _run_panel(shadow_replay_fn=_shadow)
    assert "first_status_change_summary" in report


def test_first_top1_mismatch_by_round() -> None:
    report = _run_panel(max_new_tokens=8)
    summary = report["first_top1_mismatch_summary"]
    assert "cells_with_top1_mismatch" in summary
    assert "mismatches" in summary


def test_accepted_prefix_correlation_summary() -> None:
    report = _run_panel()
    corr = report["accepted_prefix_correlation_summary"]
    assert "partial_acceptance_round_count" in corr
    assert corr["partial_acceptance_round_count"] >= 1
    assert "not a causality claim" in corr["description"]


def test_safety_gates_remain_false() -> None:
    report = _run_panel()
    assert report["generation_modified_by_shadow"] is False
    assert report["shadow_used_for_token_commit"] is False
    assert report["default_runtime_changed"] is False
    for cell in report["generation_cells"]:
        gates = cell["safety_gates"]
        assert gates["shadow_used_for_token_commit"] is False
        assert gates["generation_modified_by_shadow"] is False
        assert gates["default_runtime_changed"] is False
        assert gates["generated_output_unchanged"] is True


def test_report_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_080_ID
    assert validate_exp080_report(report) == []
    assert report["round_log_available_cells"] >= 1
    assert report["total_round_shadow_cells"] == 2 * 2  # 2 prompts, 2 rounds each


def test_no_forbidden_claim_fields() -> None:
    report = _run_panel()
    assert set(SHADOW_FORBIDDEN_CLAIMS).issubset(set(report.get("forbidden_claims", [])))
    for cell in report["generation_cells"]:
        dumped = json.dumps(cell).lower()
        for forbidden in (
            "throughput",
            "latency",
            "speedup",
            "tokens_per_second",
            "active_gpu_memory_savings",
            "production_memory_savings",
        ):
            assert forbidden not in dumped
