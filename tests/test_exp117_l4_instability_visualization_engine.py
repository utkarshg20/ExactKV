"""Tests for Experiment 117 instability visualization engine (Phase 21P)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.analysis.l4_instability_regime_extractor import run_exp116_instability_regime_extraction
from exactkv.analysis.l4_instability_visualization_engine import (
    EXPERIMENT_117_ID,
    REGIME_NAMES,
    REQUIRED_VISUAL_FILES,
    build_phase_diagram_data,
    reconstruct_stability_surface_144,
    run_exp117_instability_visualization_engine,
    validate_exp117_manifest,
)
from exactkv.safety.l4_runtime_coupling_stress_panel import run_exp115_l4_runtime_coupling_stress_panel


@pytest.fixture
def exp115_report() -> dict:
    return run_exp115_l4_runtime_coupling_stress_panel(deterministic_mode=True)


@pytest.fixture
def exp116_report(exp115_report: dict) -> dict:
    return run_exp116_instability_regime_extraction(exp115_report)


@pytest.fixture
def tmp_visual_dir(tmp_path: Path) -> Path:
    return tmp_path / "visuals" / "exp117"


def test_all_six_visual_outputs_exist(exp115_report: dict, exp116_report: dict, tmp_visual_dir: Path) -> None:
    manifest = run_exp117_instability_visualization_engine(
        exp115_path=_write_json(tmp_visual_dir / "exp115.json", exp115_report),
        exp116_path=_write_json(tmp_visual_dir / "exp116.json", exp116_report),
        output_dir=tmp_visual_dir,
    )
    assert manifest["experiment_id"] == EXPERIMENT_117_ID
    for fname in REQUIRED_VISUAL_FILES:
        path = Path(manifest["visual_outputs"][fname])
        assert path.exists()
        assert path.stat().st_size > 0


def test_deterministic_rendering(exp115_report: dict, exp116_report: dict, tmp_visual_dir: Path) -> None:
    exp115_path = _write_json(tmp_visual_dir / "exp115.json", exp115_report)
    exp116_path = _write_json(tmp_visual_dir / "exp116.json", exp116_report)
    m1 = run_exp117_instability_visualization_engine(
        exp115_path=exp115_path,
        exp116_path=exp116_path,
        output_dir=tmp_visual_dir / "run1",
    )
    m2 = run_exp117_instability_visualization_engine(
        exp115_path=exp115_path,
        exp116_path=exp116_path,
        output_dir=tmp_visual_dir / "run2",
    )
    assert m1["stability_surface_144_cell"] == m2["stability_surface_144_cell"]
    assert m1["regime_coverage"] == m2["regime_coverage"]


def test_stability_surface_144_cells(exp116_report: dict) -> None:
    surface = reconstruct_stability_surface_144(exp116_report, expected_cells=144)
    assert len(surface["stability_surface_144_cell"]) == 144
    assert len(surface["cell_ids"]) == 144


def test_phase_diagram_grid_dimensions(exp115_report: dict, exp116_report: dict) -> None:
    compressors = list(exp115_report["compressors"])
    mnt = [int(x) for x in exp115_report["max_new_tokens_values"]]
    instability, _, regimes = build_phase_diagram_data(
        exp116_report,
        compressors=compressors,
        max_new_tokens_values=mnt,
    )
    assert instability.shape == (len(compressors), len(mnt))
    assert len(regimes) == len(compressors)
    assert len(regimes[0]) == len(mnt)


def test_regime_categories_present(exp115_report: dict, exp116_report: dict, tmp_visual_dir: Path) -> None:
    manifest = run_exp117_instability_visualization_engine(
        exp115_path=_write_json(tmp_visual_dir / "exp115.json", exp115_report),
        exp116_path=_write_json(tmp_visual_dir / "exp116.json", exp116_report),
        output_dir=tmp_visual_dir,
    )
    for name in REGIME_NAMES:
        assert name in manifest["regime_categories_present"]
    assert validate_exp117_manifest(manifest).valid is True


def test_no_runtime_flags(exp115_report: dict, exp116_report: dict, tmp_visual_dir: Path) -> None:
    manifest = run_exp117_instability_visualization_engine(
        exp115_path=_write_json(tmp_visual_dir / "exp115.json", exp115_report),
        exp116_path=_write_json(tmp_visual_dir / "exp116.json", exp116_report),
        output_dir=tmp_visual_dir,
    )
    assert manifest["analysis_only"] is True
    assert manifest["exactkv_generator_modified"] is False
    assert manifest["model_experiments_run"] is False


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")
    return path
