"""Tests for Experiment 071 script and report integration (Phase 16F)."""
from __future__ import annotations

from exactkv.attention.hf_full_replay_probe import (
    EXPERIMENT_071_ID,
    DEFAULT_EXP071_REPORT,
    validate_exp071_report,
)
from tests.test_hf_full_replay_probe import _mock_loader, _mock_prompts
from exactkv.attention.hf_full_replay_probe import run_exp071_probe


def test_default_report_path() -> None:
    assert DEFAULT_EXP071_REPORT.name == "experiment_071_full_prefix_logit_drift_smoke.json"


def test_exp071_id() -> None:
    assert EXPERIMENT_071_ID == "exp071_full_prefix_logit_drift_smoke"


def test_run_exp071_integration() -> None:
    report = run_exp071_probe(
        model_id="mock",
        target_token_lengths=(32,),
        chunk_sizes=(16,),
        max_prompts=1,
        model_loader=_mock_loader,
        prompt_provider=_mock_prompts,
    )
    assert report["experiment_id"] == EXPERIMENT_071_ID
    assert validate_exp071_report(report) == []
