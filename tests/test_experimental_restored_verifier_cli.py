"""Tests for experimental restored-verifier CLI helper (no model by default)."""
from __future__ import annotations

import argparse
import os
from unittest.mock import patch

import pytest

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import DEFAULT_MODEL, VERIFIER_SOURCE
from exactkv.runtime.experimental import ExperimentalRestoredVerifierConfig
from exactkv.runtime.experimental_cli import (
    CLI_FLAG_DEST,
    EXP055_CLAIM_NOTE,
    add_experimental_restored_verifier_cli_args,
    cli_module_has_no_env_activation,
    report_to_exp055_json,
    resolve_experimental_cli_args,
    run_experimental_restored_verifier_from_cli,
    validate_exp055_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_experimental_restored_verifier_cli_args(parser)
    return parser


def test_parser_disabled_when_flag_absent() -> None:
    args = _parser().parse_args([])
    resolution = resolve_experimental_cli_args(args)
    assert not resolution.cli_flag_present
    assert not resolution.config.enabled
    assert resolution.parse_errors == []


def test_flag_present_creates_enabled_config() -> None:
    args = _parser().parse_args(["--experimental-restored-verifier"])
    resolution = resolve_experimental_cli_args(args)
    assert resolution.cli_flag_present
    assert resolution.config.enabled
    assert resolution.config.mode.value == "restored_verifier_offline"
    assert resolution.config.verifier_source == VERIFIER_SOURCE
    assert resolution.parse_errors == []


def test_env_var_cannot_activate_mode() -> None:
    assert cli_module_has_no_env_activation()
    with patch.dict(os.environ, {"EXACTKV_EXPERIMENTAL_RESTORED_VERIFIER": "1"}):
        args = _parser().parse_args([])
        resolution = resolve_experimental_cli_args(args)
    assert not resolution.cli_flag_present
    assert not resolution.config.enabled


def test_missing_required_enabled_args_fail() -> None:
    args = _parser().parse_args(
        [
            "--experimental-restored-verifier",
            "--prompt-ids",
            "",
            "--compressors",
            "",
        ]
    )
    resolution = resolve_experimental_cli_args(args)
    assert resolution.parse_errors


def test_safe_claim_note_present_when_enabled() -> None:
    args = _parser().parse_args(["--experimental-restored-verifier"])
    resolution = resolve_experimental_cli_args(args)
    assert "experimental" in resolution.config.claim_note.lower()
    assert "non-default" in resolution.config.claim_note.lower() or "not default" in resolution.config.claim_note.lower()
    assert EXP055_CLAIM_NOTE == resolution.config.claim_note


def test_disabled_cli_path_does_not_call_runner() -> None:
    args = _parser().parse_args([])
    with patch("exactkv.runtime.experimental_cli.run_experimental_restored_verifier") as mock_run:
        from exactkv.runtime.experimental import ExperimentalRuntimeResult

        mock_run.return_value = ExperimentalRuntimeResult(
            enabled=False,
            mode="default",
            status="disabled",
            runner_called=False,
        )
        resolution, result = run_experimental_restored_verifier_from_cli(args)
    mock_run.assert_called_once()
    call_cfg = mock_run.call_args[0][0]
    assert not call_cfg.enabled
    assert not resolution.cli_flag_present
    assert result.status == "disabled"


def test_enabled_cli_path_calls_runtime_wrapper() -> None:
    args = _parser().parse_args(["--experimental-restored-verifier"])
    with patch("exactkv.runtime.experimental_cli.run_experimental_restored_verifier") as mock_run:
        from exactkv.runtime.experimental import ExperimentalRuntimeResult

        mock_run.return_value = ExperimentalRuntimeResult(
            enabled=True,
            mode="restored_verifier_offline",
            status="pass",
            runner_called=True,
        )
        resolution, result = run_experimental_restored_verifier_from_cli(args)
    mock_run.assert_called_once()
    assert resolution.cli_flag_present
    assert resolution.config.enabled
    assert result.runner_called


def test_exp055_disabled_report_schema() -> None:
    args = _parser().parse_args([])
    resolution, result = run_experimental_restored_verifier_from_cli(args)
    payload = report_to_exp055_json(resolution, result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp055_report(payload) == []
    assert payload["cli_flag_present"] is False
    assert payload["runner_called"] is False


def test_exp055_enabled_report_schema() -> None:
    from exactkv.cache.restored_verifier_runner import (
        RestoredVerifierRunReport,
        aggregate_blockers,
        default_smoke_config,
    )
    from exactkv.runtime.experimental import (
        ExperimentalRuntimeResult,
        default_experimental_smoke_config,
    )
    from exactkv.runtime.experimental_cli import ExperimentalCliResolution

    cfg = default_smoke_config()
    runner_report = RestoredVerifierRunReport(
        experiment_id="exp055_experimental_restored_verifier_cli",
        config=cfg,
        cells=[],
        total_cells=2,
        token_exact_match_count=2,
        exactkv_failures=0,
        mean_acceptance=0.8,
        draft_divergence_count=3,
        blockers=aggregate_blockers([]),
        claim_note=EXP055_CLAIM_NOTE,
        status="pass",
    )
    resolution = ExperimentalCliResolution(
        cli_flag_present=True,
        config=default_experimental_smoke_config(),
        parse_errors=[],
    )
    result = ExperimentalRuntimeResult(
        enabled=True,
        mode="restored_verifier_offline",
        status="pass",
        runner_called=True,
        runner_report=runner_report,
    )
    payload = report_to_exp055_json(resolution, result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp055_report(payload) == []
    assert payload["cli_flag_present"] is True
    assert payload["model"] == DEFAULT_MODEL
