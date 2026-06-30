"""Tests for external panel artifact validation."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.benchmarks.external_panel import run_external_panel, write_external_panel_outputs
from scripts.validate_external_panel_artifacts import (
    build_validation_report,
    validate_report,
)


def test_validate_deterministic_smoke_report(tmp_path: Path) -> None:
    report = run_external_panel("longbench", deterministic_mode=True, smoke=True)
    json_path = tmp_path / "longbench_raw.json"
    write_external_panel_outputs(report, json_path=json_path, markdown_path=tmp_path / "s.md")
    loaded = json.loads(json_path.read_text())
    result = validate_report(json_path, loaded)
    assert result["valid"] is True
    assert result["issues"] == []


def test_validate_catches_exactkv_failure_mismatch(tmp_path: Path) -> None:
    report = run_external_panel("mbpp", deterministic_mode=True, smoke=True)
    report["cells"][0]["exactkv_failure"] = True
    json_path = tmp_path / "mbpp_raw.json"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    result = validate_report(json_path, report)
    assert result["valid"] is False
    assert any("exactkv_failure mismatch" in issue for issue in result["issues"])


def test_build_validation_report_on_dir(tmp_path: Path) -> None:
    report = run_external_panel("mbpp", deterministic_mode=True, smoke=True)
    json_path = tmp_path / "mbpp_raw.json"
    write_external_panel_outputs(report, json_path=json_path, markdown_path=tmp_path / "s.md")
    summary = build_validation_report(tmp_path)
    assert summary["files_scanned"] == 1
    assert summary["overall_valid"] is True


def test_mbpp_loader_metadata() -> None:
    from exactkv.benchmarks.external_dataset_loaders import load_external_prompts

    rows = load_external_prompts("mbpp", source="pilot", max_prompts=4)
    assert len(rows) >= 4
    assert rows[0]["source"] == "bundled_pilot"
    assert rows[0]["task_id"]
    assert isinstance(rows[0].get("test_list"), list)


def test_deterministic_smoke_mbpp() -> None:
    report = run_external_panel("mbpp", deterministic_mode=True, smoke=True)
    assert report["dataset_family"] == "mbpp"
    assert report["cells_run"] > 0
    assert report["exactkv_failures"] == 0
    ok = [c for c in report["cells"] if c.get("status") == "ok"]
    assert ok[0].get("task_id")
    assert ok[0].get("source") == "bundled_pilot"
