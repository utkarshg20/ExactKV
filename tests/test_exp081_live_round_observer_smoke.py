"""Tests for Experiment 081 live round observer smoke (Phase 16P)."""
from __future__ import annotations

import json

import torch

from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS
from exactkv.attention.live_round_observer import (
    EXPERIMENT_081_ID,
    LiveRoundObserver,
    LiveRoundSnapshot,
    build_live_round_snapshot,
    compare_snapshots_to_traces,
    run_exp081_live_round_observer_panel,
    validate_exp081_report,
)
from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace

TOKENS = [100, 101, 102, 103]


def _fake_trace(round_idx: int, before: int, after: int) -> VerificationTrace:
    return VerificationTrace(
        round_idx=round_idx,
        draft_tokens=TOKENS[round_idx : round_idx + 2],
        acceptance=AcceptanceResult(
            draft_tokens=TOKENS[round_idx : round_idx + 2],
            verifier_tokens=TOKENS[round_idx : round_idx + 2],
            accepted_tokens=TOKENS[round_idx : round_idx + 2],
            correction_token=None,
            rejected_tokens=[],
            bonus_token=None,
            all_matched=True,
            num_accepted=2,
            num_rejected=0,
        ),
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
                acceptance=_fake_trace(rnd, before, after).acceptance,
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
        "generated_text": "baseline text",
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
        "generated_text": "baseline text",
        "prompt_ids": torch.tensor([[1, 2, 3]]),
        "full_sequence_ids": torch.tensor([[1, 2, 3] + TOKENS]),
        "exactkv_failures": 0,
        "token_exact_match": True,
        "live_snapshots": snaps,
        "result_traces": traces,
        "snapshot_comparison": compare_snapshots_to_traces(snaps, traces),
        "observer_exceptions": [],
        "blockers": [],
    }


def _observer_fn_mismatch(**kwargs: object) -> dict:
    out = _observer_fn(**kwargs)
    out["generated_token_ids"] = [999]
    return out


def _observer_fn_with_exception(**kwargs: object) -> dict:
    out = _observer_fn(**kwargs)
    out["observer_exceptions"] = ["RuntimeError: boom"]
    return out


def test_report_schema_validates() -> None:
    report = run_exp081_live_round_observer_panel(
        model_id="mock",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        observer_generation_fn=_observer_fn,
    )
    assert report["experiment_id"] == EXPERIMENT_081_ID
    assert validate_exp081_report(report) == []
    assert report["baseline_vs_observer_token_match_cells"] == 1


def test_baseline_vs_observer_token_match_logic() -> None:
    ok = run_exp081_live_round_observer_panel(
        model_id="mock",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        observer_generation_fn=_observer_fn,
    )
    assert ok["status"] == "diagnostic_complete"

    bad = run_exp081_live_round_observer_panel(
        model_id="mock",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        observer_generation_fn=_observer_fn_mismatch,
    )
    assert bad["status"] == "failed"
    assert bad["baseline_vs_observer_token_match_cells"] == 0


def test_live_snapshot_vs_result_round_log_match() -> None:
    report = run_exp081_live_round_observer_panel(
        model_id="mock",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        observer_generation_fn=_observer_fn,
    )
    cell = report["cells"][0]
    assert cell["snapshot_vs_result_round_log_match"] is True
    assert cell["live_snapshot_count"] == 2


def test_missing_round_log_blocks_in_observer_fn() -> None:
    def _no_traces(**kwargs: object) -> dict:
        del kwargs
        return {
            "generation_completed": True,
            "generated_token_ids": TOKENS,
            "generated_text": "baseline text",
            "live_snapshots": [],
            "result_traces": [],
            "snapshot_comparison": compare_snapshots_to_traces([], []),
            "observer_exceptions": [],
            "blockers": [],
        }

    report = run_exp081_live_round_observer_panel(
        model_id="mock",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        observer_generation_fn=_no_traces,
    )
    cell = report["cells"][0]
    assert cell["live_snapshot_count"] == 0
    assert cell["snapshot_vs_result_round_log_match"] is True


def test_safety_gates_enforced() -> None:
    report = run_exp081_live_round_observer_panel(
        model_id="mock",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        observer_generation_fn=_observer_fn_with_exception,
    )
    assert report["observer_used_for_token_commit"] is False
    assert report["generation_modified_by_observer"] is False
    assert report["default_runtime_changed"] is False
    assert report["observer_exception_cells"] == 1
    for cell in report["cells"]:
        gates = cell["safety_gates"]
        assert gates["observer_return_value_ignored"] is True
        assert gates["observer_used_for_token_commit"] is False


def test_no_forbidden_claim_fields() -> None:
    report = run_exp081_live_round_observer_panel(
        model_id="mock",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        observer_generation_fn=_observer_fn,
    )
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


def test_live_round_observer_records_snapshots() -> None:
    obs = LiveRoundObserver()
    snap = build_live_round_snapshot(
        round_index=0,
        prompt_token_ids=(1,),
        generated_token_ids_before=(),
        generated_token_ids_after=(9,),
        draft_token_ids=[9],
        acceptance=None,
        compressor_name="noop",
        max_new_tokens=4,
    )
    obs.observe(snap)
    assert len(obs.snapshots) == 1
