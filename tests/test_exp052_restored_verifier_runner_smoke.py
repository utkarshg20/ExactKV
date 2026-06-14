"""Tests for Experiment 052 restored-verifier runner smoke (no model by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import VERIFIER_SOURCE
from exactkv.cache.restored_verifier_runner import (
    EXPERIMENT_052_ID,
    RestoredVerifierCellResult,
    aggregate_blockers,
    default_smoke_config,
    report_to_exp052_json,
    validate_exp052_report,
    RestoredVerifierRunReport,
    EXP052_CLAIM_NOTE,
)

_DOC_EXP052 = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md"
)
_DOC_RUNNER = Path(__file__).resolve().parents[1] / "docs" / "RESTORED_VERIFIER_RUNNER.md"
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_052_restored_verifier_runner_smoke.json"
)


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


def _report() -> RestoredVerifierRunReport:
    cfg = default_smoke_config()
    cells = [
        _cell(),
        _cell(prompt_id="offline_002", compressor_name="k8_v4_sim", draft_divergence_count=0),
    ]
    return RestoredVerifierRunReport(
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

_DOC_EXP052 = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md"
)
_DOC_RUNNER = Path(__file__).resolve().parents[1] / "docs" / "RESTORED_VERIFIER_RUNNER.md"
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_052_restored_verifier_runner_smoke.json"
)


def test_exp052_report_schema_from_runner() -> None:
    payload = report_to_exp052_json(_report())
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp052_report(payload) == []


def test_default_smoke_config_fields() -> None:
    cfg = default_smoke_config()
    assert cfg.device == "cpu"
    assert cfg.dtype == "float32"
    assert cfg.draft_len == 4
    assert cfg.verifier_source == VERIFIER_SOURCE
    assert len(cfg.prompt_ids) >= 4
    assert "int4_sim" in cfg.compressor_names
    assert "int8" in cfg.compressor_names


def test_doc_exp052_caveats() -> None:
    text = _DOC_EXP052.read_text(encoding="utf-8").lower()
    for phrase in (
        "isolated restored-verifier runner",
        "not default runtime integration",
        "isolated experiment path",
        "vllm",
        "lmcache",
        "remote prefix",
        "generation and verification behavior is unchanged",
        "throughput",
        "vericache",
        "active memory savings",
    ):
        assert phrase in text, phrase


def test_doc_runner_caveats() -> None:
    text = _DOC_RUNNER.read_text(encoding="utf-8").lower()
    for phrase in (
        "isolated",
        "not default",
        "reloaded full kv",
        "vllm",
        "lmcache",
        "does not prove",
        "speed",
        "memory",
        "serving",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    for doc in (_DOC_EXP052, _DOC_RUNNER):
        text = doc.read_text(encoding="utf-8").lower()
        for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
            assert phrase not in text


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model smoke",
)
def test_model_smoke_one_cell() -> None:
    from exactkv.cache.restored_verifier_runner import (
        build_storage_backend,
        resolve_prompt_entries,
        run_restored_verifier_cell,
    )
    from exactkv.runtime.model_runtime import ModelRuntime

    cfg = default_smoke_config(max_new_tokens=12)
    runtime = ModelRuntime(cfg.model_id, device=cfg.device, dtype=cfg.dtype)
    backend = build_storage_backend(cfg.storage_backend_name)
    entry = resolve_prompt_entries(cfg.prompt_ids)[0]
    cell = run_restored_verifier_cell(
        runtime,
        config=cfg,
        prompt_id=entry["prompt_id"],
        prompt=entry["prompt"],
        category=entry["category"],
        backend=backend,
        compressor_name="int8",
    )
    assert not cell.restore_blocker, cell.restore_blocker
    assert not cell.draft_blocker, cell.draft_blocker
    assert not cell.verification_blocker, cell.verification_blocker
    assert cell.token_exact_match
    assert cell.exactkv_failure == 0


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp052_restored_verifier_runner_smoke.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp052_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_052_ID
    assert report["exactkv_failures"] == 0
