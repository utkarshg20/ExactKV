"""Tests for Experiment 055 experimental restored-verifier CLI (no model by default)."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL, FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import VERIFIER_SOURCE
from exactkv.runtime.experimental_cli import (
    EXPERIMENT_055_ID,
    EXP055_CLAIM_NOTE,
    validate_exp055_report,
)

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_055_EXPERIMENTAL_RESTORED_VERIFIER_CLI.md"
)
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_055_experimental_restored_verifier_cli.json"
)
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "run_exp055_experimental_restored_verifier_cli.py"
)


def _load_exp055_module():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("run_exp055_cli_module", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "explicit cli opt-in",
        "non-default experimental restored-verifier runtime",
        "--experimental-restored-verifier",
        "environment variables do not activate",
        "default exactkv generation behavior is unchanged",
        "vllm",
        "lmcache",
        "remote prefix",
        "throughput",
        "vericache",
        "active memory savings",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
        assert phrase not in text


def test_disabled_script_path_no_runner() -> None:
    mod = _load_exp055_module()

    with patch("exactkv.runtime.experimental_cli.run_experimental_restored_verifier") as mock_run:
        from exactkv.runtime.experimental import ExperimentalRuntimeResult

        mock_run.return_value = ExperimentalRuntimeResult(
            enabled=False,
            mode="default",
            status="disabled",
            runner_called=False,
        )
        code, report = mod.run_exp055_cli([])
    assert code == 0
    assert report["cli_flag_present"] is False
    assert report["runner_called"] is False
    mock_run.assert_called_once()


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model CLI smoke",
)
def test_model_smoke_enabled_cli() -> None:
    mod = _load_exp055_module()
    code, report = mod.run_exp055_cli(["--experimental-restored-verifier"])
    assert code == 0
    assert report["cli_flag_present"] is True
    assert report["exactkv_failures"] == 0
    assert report["token_exact_match_count"] == report["total_cells"]


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp055_experimental_restored_verifier_cli.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp055_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_055_ID
    assert report["exactkv_failures"] == 0
