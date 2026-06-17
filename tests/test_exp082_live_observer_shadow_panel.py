"""Tests for Experiment 082 live observer + post-hoc shadow panel (Phase 16Q)."""
from __future__ import annotations

import json

import torch

from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS
from exactkv.attention.live_round_observer import (
    EXPERIMENT_082_ID,
    LiveRoundSnapshot,
    build_live_round_snapshot,
    compare_snapshots_to_traces,
    run_exp082_live_observer_shadow_panel,
    validate_exp082_report,
)
from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace

TOKENS = [100, 101, 102, 103]


def _acceptance() -> AcceptanceResult:
    return AcceptanceResult(
        draft_tokens=TOKENS[:2],
        verifier_tokens=TOKENS[:2],
        accepted_tokens=TOKENS[:2],
        correction_token=None,
        rejected_tokens=[],
        bonus_token=None,
        all_matched=True,
        num_accepted=2,
        num_rejected=0,
    )


def _fake_trace(round_idx: int, before: int, after: int) -> VerificationTrace:
    return VerificationTrace(
        round_idx=round_idx,
        draft_tokens=TOKENS[round_idx : round_idx + 2],
        acceptance=_acceptance(),
        full_seq_len_before=before,
        full_seq_len_after=after,
        compressed_seq_len_after=after,
    )


def _fake_snapshots() -> list[LiveRoundSnapshot]:
    prompt = (1, 2, 3)
    snaps = []
    for rnd, (before, after) in enumerate([(3, 5), (5, 7)]):
        before_gen = tuple(TOKENS[: rnd * 2])
        after_gen = tuple(TOKENS[: rnd * 2 + 2])
        snaps.append(
            build_live_round_snapshot(
                round_index=rnd,
                prompt_token_ids=prompt,
                generated_token_ids_before=before_gen,
                generated_token_ids_after=after_gen,
                draft_token_ids=TOKENS[rnd : rnd + 2],
                acceptance=_acceptance(),
                compressor_name="noop",
                max_new_tokens=8,
                full_seq_len_before=before,
                full_seq_len_after=after,
            )
        )
    return snaps


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


def _observer_fn(**kwargs: object) -> dict:
    del kwargs
    traces = [_fake_trace(0, 3, 5), _fake_trace(1, 5, 7)]
    snaps = _fake_snapshots()
    return {
        "generation_completed": True,
        "generated_token_ids": TOKENS,
        "generated_text": "out",
        "exactkv_failures": 0,
        "token_exact_match": True,
        "live_snapshots": snaps,
        "result_traces": traces,
        "snapshot_comparison": compare_snapshots_to_traces(snaps, traces),
        "observer_exceptions": [],
        "blockers": [],
    }


def _observer_fn_no_snapshots(**kwargs: object) -> dict:
    out = _observer_fn(**kwargs)
    out["live_snapshots"] = []
    out["snapshot_comparison"] = compare_snapshots_to_traces([], out["result_traces"])
    return out


def _fake_shadow(**kwargs: object) -> dict:
    return {
        "num_layers_replayed": 2,
        "streaming_vs_materialized_logit_metrics": {
            "max_abs_error": 0.001,
            "top1_agreement": True,
        },
        "full_vs_streaming_logit_metrics": {"max_abs_error": 0.001},
        "blockers": [],
    }


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "model_id": "mock",
        "prompts": [("p0", "hello")],
        "compressors_requested": ["noop"],
        "baseline_generation_fn": _baseline_fn,
        "observer_generation_fn": _observer_fn,
        "shadow_replay_fn": _fake_shadow,
    }
    defaults.update(overrides)
    return run_exp082_live_observer_shadow_panel(**defaults)


def test_baseline_vs_observer_parity_pass() -> None:
    report = _run_panel()
    assert report["status"] == "diagnostic_complete"
    assert report["baseline_vs_observer_token_match_cells"] == 1
    assert validate_exp082_report(report) == []


def test_baseline_vs_observer_parity_failure() -> None:
    def _bad_observer(**kwargs: object) -> dict:
        out = _observer_fn(**kwargs)
        out["generated_token_ids"] = [999]
        return out

    report = _run_panel(observer_generation_fn=_bad_observer)
    assert report["status"] == "failed"
    assert report["baseline_vs_observer_token_match_cells"] == 0


def test_live_snapshots_required_for_shadow() -> None:
    report = _run_panel(observer_generation_fn=_observer_fn_no_snapshots)
    cell = report["cells"][0]
    assert cell["posthoc_shadow_cells"] == []
    assert "blocked_missing_live_snapshots" in " ".join(cell["blockers"])
    assert report["status"] == "failed"


def test_posthoc_shadow_runs_after_generation() -> None:
    report = _run_panel()
    cell = report["cells"][0]
    assert len(cell["posthoc_shadow_cells"]) == 2
    assert report["posthoc_shadow_successful_cells"] == 2


def test_observer_and_shadow_cannot_affect_token_commit() -> None:
    report = _run_panel()
    assert report["observer_used_for_token_commit"] is False
    assert report["shadow_used_for_token_commit"] is False
    gates = report["cells"][0]["safety_gates"]
    assert gates["observer_used_for_token_commit"] is False
    assert gates["shadow_used_for_token_commit"] is False
    assert gates["generation_modified_by_shadow"] is False


def test_safety_gate_failure_marks_cell_failed() -> None:
    def _incomplete_observer(**kwargs: object) -> dict:
        del kwargs
        return {
            "generation_completed": False,
            "generated_token_ids": [],
            "generated_text": "",
            "blockers": [],
        }

    report = _run_panel(observer_generation_fn=_incomplete_observer)
    assert report["status"] == "failed"
    assert "safety_gate_failed" in " ".join(report["cells"][0]["blockers"])


def test_snapshot_vs_result_round_log_match_aggregation() -> None:
    report = _run_panel()
    assert report["snapshot_vs_result_round_log_match_cells"] == 1


def test_tolerance_policy_summary_by_round() -> None:
    report = _run_panel()
    assert "tolerance_policy_summary_by_round" in report
    assert isinstance(report["tolerance_policy_summary_by_round"], dict)


def test_topk_summary_by_round() -> None:
    report = _run_panel()
    assert "topk_agreement_summary_by_round" in report


def test_report_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_082_ID
    assert validate_exp082_report(report) == []


def test_no_forbidden_claim_fields() -> None:
    report = _run_panel()
    assert set(SHADOW_FORBIDDEN_CLAIMS).issubset(set(report.get("forbidden_claims", [])))
    for cell in report["cells"]:
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
