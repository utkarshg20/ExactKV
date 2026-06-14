"""Tests for experimental restored-verifier runtime (no model by default)."""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import VERIFIER_SOURCE
from exactkv.compressors import get_compressor
from exactkv.runtime.experimental import (
    EXPERIMENT_054_ID,
    EXP054_CLAIM_NOTE,
    ExperimentalRestoredVerifierConfig,
    ExperimentalRuntimeMode,
    default_experimental_smoke_config,
    report_to_exp054_json,
    run_experimental_restored_verifier,
    validate_exp054_report,
    validate_experimental_config,
)


def _enabled_config(**overrides: object) -> ExperimentalRestoredVerifierConfig:
    cfg = default_experimental_smoke_config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_disabled_config_does_not_call_runner() -> None:
    with patch("exactkv.runtime.experimental.run_restored_verifier") as mock_run:
        result = run_experimental_restored_verifier(
            ExperimentalRestoredVerifierConfig.disabled()
        )
    mock_run.assert_not_called()
    assert result.status == "disabled"
    assert not result.runner_called
    assert result.runner_report is None


def test_enabled_config_validates_required_fields() -> None:
    cfg = _enabled_config()
    assert validate_experimental_config(cfg) == []


def test_invalid_mode_fails() -> None:
    cfg = _enabled_config(mode=ExperimentalRuntimeMode.DEFAULT)
    errors = validate_experimental_config(cfg)
    assert any("RESTORED_VERIFIER_OFFLINE" in e for e in errors)


def test_missing_claim_caveat_fails() -> None:
    cfg = _enabled_config(claim_note="production serving ready")
    errors = validate_experimental_config(cfg)
    assert any("claim_note" in e for e in errors)


def test_verifier_source_must_be_reloaded_full_kv() -> None:
    cfg = _enabled_config(verifier_source="live_full_kv")
    errors = validate_experimental_config(cfg)
    assert any("verifier_source" in e for e in errors)


def test_no_implicit_env_activation_in_module() -> None:
    import exactkv.runtime.experimental as mod

    source = inspect.getsource(mod)
    assert "os.environ" not in source
    assert "getenv" not in source


def test_no_global_registry_mutation_on_disabled_run() -> None:
    before = {name: get_compressor(name).name for name in ("int8", "int4_sim", "k8_v4_sim")}
    run_experimental_restored_verifier(ExperimentalRestoredVerifierConfig.disabled())
    after = {name: get_compressor(name).name for name in ("int8", "int4_sim", "k8_v4_sim")}
    assert before == after


def test_invalid_enabled_config_does_not_call_runner() -> None:
    cfg = _enabled_config(verifier_source="wrong")
    with patch("exactkv.runtime.experimental.run_restored_verifier") as mock_run:
        result = run_experimental_restored_verifier(cfg)
    mock_run.assert_not_called()
    assert result.status == "invalid"
    assert result.validation_errors


def test_experimental_report_schema_disabled() -> None:
    result = run_experimental_restored_verifier(ExperimentalRestoredVerifierConfig.disabled())
    payload = report_to_exp054_json(result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp054_report(payload) == []
    assert payload["enabled"] is False
    assert payload["runner_called"] is False


def test_experimental_report_schema_enabled_pass() -> None:
    from exactkv.cache.restored_verifier_runner import (
        RestoredVerifierRunReport,
        aggregate_blockers,
        default_smoke_config,
    )

    cfg = default_smoke_config()
    cells_report = RestoredVerifierRunReport(
        experiment_id=EXPERIMENT_054_ID,
        config=cfg,
        cells=[],
        total_cells=2,
        token_exact_match_count=2,
        exactkv_failures=0,
        mean_acceptance=0.8,
        draft_divergence_count=3,
        semantic_divergence_count=1,
        no_real_drift_observed=False,
        blockers=aggregate_blockers([]),
        claim_note=EXP054_CLAIM_NOTE,
        status="pass",
    )
    from exactkv.runtime.experimental import ExperimentalRuntimeResult

    result = ExperimentalRuntimeResult(
        enabled=True,
        mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE.value,
        status="pass",
        runner_called=True,
        runner_report=cells_report,
    )
    payload = report_to_exp054_json(result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp054_report(payload) == []


def test_exactness_failure_preserved_in_report() -> None:
    from exactkv.cache.restored_verifier_runner import (
        RestoredVerifierCellResult,
        RestoredVerifierRunReport,
        aggregate_blockers,
        default_smoke_config,
    )
    from exactkv.runtime.experimental import ExperimentalRuntimeResult

    failed_cell = RestoredVerifierCellResult(
        prompt_id="offline_001",
        compressor_name="int4_sim",
        storage_backend="in_memory_kv_storage",
        draft_len=4,
        token_exact_match=False,
        exactkv_failure=1,
        accepted_prefix_lengths=[],
        first_divergence=2,
        draft_divergence_count=1,
    )
    cfg = default_smoke_config()
    report = RestoredVerifierRunReport(
        experiment_id=EXPERIMENT_054_ID,
        config=cfg,
        cells=[failed_cell],
        total_cells=1,
        token_exact_match_count=0,
        exactkv_failures=1,
        mean_acceptance=0.0,
        draft_divergence_count=1,
        blockers=aggregate_blockers([failed_cell]),
        claim_note=EXP054_CLAIM_NOTE,
        status="failed",
        first_divergences=[
            {
                "prompt_id": "offline_001",
                "compressor_name": "int4_sim",
                "storage_backend": "in_memory_kv_storage",
                "draft_len": 4,
                "first_divergence_idx": 2,
            }
        ],
    )
    result = ExperimentalRuntimeResult(
        enabled=True,
        mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE.value,
        status="failed",
        runner_called=True,
        runner_report=report,
    )
    payload = report_to_exp054_json(result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert payload["exactkv_failures"] == 1
    assert payload["token_exact_match_count"] == 0
    assert validate_exp054_report(payload) == []


def test_blockers_preserved_in_report() -> None:
    from exactkv.cache.restored_verifier_runner import (
        RestoredVerifierCellResult,
        RestoredVerifierRunReport,
        aggregate_blockers,
        default_smoke_config,
    )
    from exactkv.runtime.experimental import ExperimentalRuntimeResult

    cell = RestoredVerifierCellResult(
        prompt_id="offline_001",
        compressor_name="int8",
        storage_backend="in_memory_kv_storage",
        draft_len=4,
        token_exact_match=False,
        exactkv_failure=1,
        accepted_prefix_lengths=[],
        restore_blocker="HfKvRestoreError",
    )
    blockers = aggregate_blockers([cell])
    report = RestoredVerifierRunReport(
        experiment_id=EXPERIMENT_054_ID,
        config=default_smoke_config(),
        cells=[cell],
        total_cells=1,
        token_exact_match_count=0,
        exactkv_failures=1,
        mean_acceptance=0.0,
        draft_divergence_count=0,
        blockers=blockers,
        claim_note=EXP054_CLAIM_NOTE,
        status="failed",
    )
    result = ExperimentalRuntimeResult(
        enabled=True,
        mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE.value,
        status="failed",
        runner_called=True,
        runner_report=report,
    )
    payload = report_to_exp054_json(result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert payload["restore_blockers"]
    assert payload["blockers"]["restore_blockers"]
