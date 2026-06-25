"""Tests for project archaeology and lineage (Release Gate R2)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lineage_built() -> None:
    subprocess.run(
        [sys.executable, "scripts/build_project_lineage.py"],
        cwd=_ROOT,
        check=True,
    )


def test_lineage_artifacts_exist(lineage_built: None) -> None:
    for rel in (
        "docs/HISTORICAL_ARTIFACT_INVENTORY.md",
        "docs/PROJECT_LINEAGE.md",
        "docs/VERSION_LINEAGE.md",
        "reports/historical_artifact_inventory.json",
        "reports/historical_artifact_inventory.csv",
        "reports/project_lineage_graph.json",
        "reports/version_lineage.json",
        "reports/version_lineage.csv",
    ):
        assert (_ROOT / rel).is_file(), rel


def test_version_lineage_covers_v1_v21(lineage_built: None) -> None:
    data = json.loads((_ROOT / "reports/version_lineage.json").read_text(encoding="utf-8"))
    vids = {v["version_id"] for v in data.get("versions") or []}
    for n in range(1, 22):
        assert f"V{n}" in vids


def test_inventory_has_pre_a_artifacts(lineage_built: None) -> None:
    data = json.loads((_ROOT / "reports/historical_artifact_inventory.json").read_text())
    assert data["artifact_count"] >= 100
    assert data["pre_formal_pipeline_count"] >= 50


def test_project_lineage_states_pre_a_history(lineage_built: None) -> None:
    text = (_ROOT / "docs/PROJECT_LINEAGE.md").read_text(encoding="utf-8").lower()
    assert "did not start at phase a" in text
    assert "v1" in text and "v21" in text
    assert "verifier" in text
    assert "demo" in text


def test_lineage_validator_passes(lineage_built: None) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/check_project_lineage.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_scale_raw_has_mistral_cells() -> None:
    raw_path = _ROOT / "reports/scale_7b/raw.json"
    if not raw_path.is_file():
        pytest.skip("no scale raw")
    raw = json.loads(raw_path.read_text())
    mistral = [c for c in raw.get("cells") or [] if "mistral" in str(c.get("model_name", "")).lower()]
    assert len(mistral) == 750


def test_synthetic_validator_catches_bad_leaderboard() -> None:
    from exactkv.platform.leaderboard_aggregates import validate_leaderboard_against_raw

    errors = validate_leaderboard_against_raw(
        {"cells": [{"model_name": "mistralai/Mistral-7B-Instruct-v0.3"}]},
        {"entries": [{"model": "mistralai/Mistral-7B-Instruct-v0.3", "availability": "unavailable", "score": None}]},
    )
    assert errors
