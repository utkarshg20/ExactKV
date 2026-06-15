"""Tests for Experiment 057 GPU memory accounting (no CUDA by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.metrics.gpu_memory_accounting import (
    EXPERIMENT_057_ID,
    EXP057_CLAIM_NOTE,
    check_exp056_exactness_gate,
    run_gpu_memory_accounting,
    validate_exp057_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_057_GPU_MEMORY_ACCOUNTING.md"
_EXP056 = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_056_cuda_restored_verifier_runtime_gate.json"
)
_EXP057 = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_057_gpu_memory_accounting.json"
)


def _pass_exp056_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": "exp056_cuda_restored_verifier_runtime_gate",
        "status": "pass",
        "cuda_available": True,
        "total_cells": 24,
        "exactkv_failures": 0,
        "token_exact_match_count": 24,
        "cuda_blockers": [],
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
    }
    base.update(overrides)
    return base


def test_exp056_gate_passes_on_local_report_if_present() -> None:
    if not _EXP056.is_file():
        pytest.skip("Exp 056 report not present")
    ok, blockers = check_exp056_exactness_gate(_EXP056)
    data = json.loads(_EXP056.read_text(encoding="utf-8"))
    if data.get("status") == "pass" and data.get("cuda_available"):
        assert ok, blockers
    else:
        assert not ok


def test_exactness_gate_failure_blocks_memory_diagnostic(tmp_path: Path) -> None:
    bad = tmp_path / "exp056_bad.json"
    bad.write_text(
        json.dumps(
            _pass_exp056_report(
                status="failed",
                exactkv_failures=1,
                token_exact_match_count=23,
            )
        ),
        encoding="utf-8",
    )
    ok, _ = check_exp056_exactness_gate(bad)
    assert not ok
    result = run_gpu_memory_accounting(exp056_report_path=bad)
    assert result.status == "blocked"
    assert not result.exp056_gate_passed


def test_run_blocked_without_cuda_when_gate_ok(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available — use EXACTKV_RUN_CUDA_SMOKE")
    good = tmp_path / "exp056_pass.json"
    good.write_text(json.dumps(_pass_exp056_report()), encoding="utf-8")
    result = run_gpu_memory_accounting(exp056_report_path=good)
    assert result.status == "blocked"
    assert result.exp056_gate_passed
    payload = result.to_dict()
    assert validate_exp057_report(payload) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "gpu memory accounting diagnostic",
        "not a memory savings claim",
        "active gpu memory savings are not claimed",
        "explicit experimental",
        "vllm",
        "lmcache",
        "remote prefix",
        "throughput",
        "vericache",
        "default exactkv generation behavior is unchanged",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves memory savings", "production serving ready", "faster than"):
        assert phrase not in text


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_CUDA_SMOKE") != "1",
    reason="Set EXACTKV_RUN_CUDA_SMOKE=1 for CUDA smoke",
)
def test_cuda_smoke_memory_accounting() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not _EXP056.is_file():
        pytest.skip("Exp 056 report required")
    ok, blockers = check_exp056_exactness_gate(_EXP056)
    if not ok:
        pytest.skip(f"Exp 056 gate not passed: {blockers}")
    result = run_gpu_memory_accounting(exp056_report_path=_EXP056)
    payload = result.to_dict()
    assert validate_exp057_report(payload) == []
    assert payload["active_gpu_memory_savings_claimed"] is False


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_CUDA_SMOKE") != "1",
    reason="Set EXACTKV_RUN_CUDA_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _EXP057.is_file():
        pytest.skip("Run scripts/research/run_exp057_gpu_memory_accounting.py first")
    report = json.loads(_EXP057.read_text(encoding="utf-8"))
    assert validate_exp057_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_057_ID
    assert report["active_gpu_memory_savings_claimed"] is False
