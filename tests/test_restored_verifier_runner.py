"""Tests for restored-verifier runner API (no model download by default)."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import VERIFIER_SOURCE
from exactkv.cache.restored_verifier_runner import (
    EXPERIMENT_052_ID,
    EXP052_CLAIM_NOTE,
    RestoredVerifierCellResult,
    RestoredVerifierRunConfig,
    RestoredVerifierRunReport,
    aggregate_blockers,
    check_phase12f_exactness_gate,
    default_smoke_config,
    drift_result_to_cell,
    report_to_exp052_json,
    resolve_prompt_entries,
    validate_exactness_gate,
    validate_exp052_report,
)
from exactkv.cache.offline_verifier import OfflineDriftStressCellResult


def _cell(**overrides: object) -> RestoredVerifierCellResult:
    base = RestoredVerifierCellResult(
        prompt_id="offline_001",
        compressor_name="int4_sim",
        storage_backend="in_memory_kv_storage",
        draft_len=4,
        token_exact_match=True,
        exactkv_failure=0,
        accepted_prefix_lengths=[2, 2],
        mean_acceptance=0.75,
        draft_divergence_count=1,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _report(**overrides: object) -> RestoredVerifierRunReport:
    cfg = default_smoke_config()
    cells = [
        _cell(),
        _cell(prompt_id="offline_002", compressor_name="k8_v4_sim", draft_divergence_count=0),
    ]
    base = RestoredVerifierRunReport(
        experiment_id=EXPERIMENT_052_ID,
        config=cfg,
        cells=cells,
        total_cells=2,
        token_exact_match_count=2,
        exactkv_failures=0,
        mean_acceptance=0.625,
        draft_divergence_count=1,
        blockers=aggregate_blockers(cells),
        claim_note=EXP052_CLAIM_NOTE,
        status="pass",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_run_config_serialization_roundtrip() -> None:
    cfg = default_smoke_config(draft_len=8, max_new_tokens=16)
    restored = RestoredVerifierRunConfig.from_dict(cfg.to_dict())
    assert restored.model_id == cfg.model_id
    assert restored.draft_len == 8
    assert restored.max_new_tokens == 16
    assert restored.verifier_source == VERIFIER_SOURCE


def test_cell_result_serialization_roundtrip() -> None:
    cell = _cell(first_divergence=3, exactkv_failure=1, token_exact_match=False)
    data = cell.to_dict()
    restored = RestoredVerifierCellResult.from_dict(data)
    assert restored.first_divergence == 3
    assert restored.exactkv_failure == 1


def test_run_report_serialization() -> None:
    report = _report()
    data = report.to_dict()
    assert data["experiment_id"] == EXPERIMENT_052_ID
    assert len(data["cells"]) == 2
    assert data["total_cells"] == 2
    assert "config" in data


def test_exactness_gate_pass() -> None:
    assert validate_exactness_gate(_report()) == []


def test_exactness_gate_failure() -> None:
    failed_cell = _cell(
        prompt_id="offline_003",
        token_exact_match=False,
        exactkv_failure=1,
        first_divergence=2,
    )
    report = _report(
        cells=[_cell(), failed_cell],
        total_cells=2,
        token_exact_match_count=1,
        exactkv_failures=1,
        status="failed",
    )
    errors = validate_exactness_gate(report)
    assert errors


def test_blocker_aggregation() -> None:
    cells = [
        _cell(restore_blocker="HfKvRestoreError"),
        _cell(
            prompt_id="offline_002",
            draft_blocker="compress failed",
            verification_blocker="align failed",
        ),
    ]
    blockers = aggregate_blockers(cells)
    assert len(blockers["restore_blockers"]) == 1
    assert len(blockers["draft_blockers"]) == 1
    assert len(blockers["verification_blockers"]) == 1


def test_mean_acceptance_aggregation_in_report() -> None:
    report = _report()
    assert report.mean_acceptance == 0.625


def test_drift_result_to_cell_mapping() -> None:
    drift = OfflineDriftStressCellResult(
        prompt_id="offline_001",
        prompt="test",
        category="smoke",
        backend_name="in_memory_kv_storage",
        compressor_name="int8",
        draft_len=4,
        cache_format="dynamic_v5",
        draft_source="int8",
        verifier_source=VERIFIER_SOURCE,
        live_reference_token_ids=[1, 2],
        offline_output_token_ids=[1, 2],
        token_exact_match=True,
        exactkv_failures=0,
        accepted_prefix_lengths=[2],
        mean_acceptance=1.0,
        draft_divergence_count=0,
        semantic_divergence_count=0,
    )
    cell = drift_result_to_cell(drift, storage_backend="in_memory_kv_storage")
    assert cell.prompt_id == "offline_001"
    assert cell.exactkv_failure == 0
    assert cell.storage_backend == "in_memory_kv_storage"


def test_resolve_prompt_entries() -> None:
    entries = resolve_prompt_entries(["offline_001", "offline_002"])
    assert len(entries) == 2
    assert entries[0]["prompt_id"] == "offline_001"


def test_phase12f_gate_with_blocked_report() -> None:
    report_path = Path("reports/experiment_051_offline_verifier_cuda_drift_panel.json")
    if not report_path.is_file():
        allowed, reason = check_phase12f_exactness_gate(report_path)
        assert allowed
        assert "Phase 12E" in reason or "not found" in reason
    else:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        allowed, reason = check_phase12f_exactness_gate(report_path)
        if data.get("exactkv_failures", 0) > 0 and data.get("cells"):
            assert not allowed
        else:
            assert allowed


def test_report_to_exp052_json_schema() -> None:
    payload = report_to_exp052_json(_report())
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp052_report(payload) == []


def test_exp052_synthetic_report_schema() -> None:
    payload = report_to_exp052_json(_report())
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert payload["experiment_id"] == EXPERIMENT_052_ID
    assert validate_exp052_report(payload) == []


def test_exp052_failure_report_schema() -> None:
    failed = _cell(
        prompt_id="offline_003",
        token_exact_match=False,
        exactkv_failure=1,
        first_divergence=1,
    )
    report = _report(
        cells=[failed],
        total_cells=1,
        token_exact_match_count=0,
        exactkv_failures=1,
        status="failed",
        first_divergences=[
            {
                "prompt_id": "offline_003",
                "compressor_name": "int4_sim",
                "storage_backend": "in_memory_kv_storage",
                "draft_len": 4,
                "first_divergence_idx": 1,
            }
        ],
    )
    payload = report_to_exp052_json(report)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp052_report(payload) == []
