"""Schema / claim-boundary tests for the systems diagnostic panel."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.benchmarks.systems_diagnostic_panel import (
    CLAIM_BOUNDARY,
    FORBIDDEN_FIELDS,
    assert_no_forbidden_fields,
    run_systems_diagnostic_panel,
    write_systems_diagnostic_outputs,
)
from scripts.build_systems_diagnostic_pack import build_pack, render_md


def test_deterministic_panel_schema_and_forbidden_fields(tmp_path: Path) -> None:
    report = run_systems_diagnostic_panel(
        smoke=True,
        deterministic_mode=True,
    )
    assert report["schema"] == "exactkv.systems_diagnostic.v1"
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["n_cells"] == 1 * 3 * 1 * 1 * 4  # smoke: 1 model × 3 comp × 1×1×4
    assert report["exactkv_failures"] == 0
    assert_no_forbidden_fields(report)
    for cell in report["cells"]:
        for arm in ("full", "lossy", "exactkv"):
            assert "wall_clock_ms" in cell[arm]
            assert "gpu_peak_allocated_bytes" in cell[arm]
        assert "tokens_per_second" not in cell
        assert "speedup" not in cell

    out = tmp_path / "qwen_raw.json"
    write_systems_diagnostic_outputs(report, json_path=out, markdown_path=tmp_path / "s.md")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_cells"] == report["n_cells"]


def test_build_pack_from_deterministic_cells() -> None:
    report = run_systems_diagnostic_panel(smoke=True, deterministic_mode=True)
    pack = build_pack(report["cells"], sources=["synthetic"])
    assert pack["claim_boundary"] == CLAIM_BOUNDARY
    assert pack["n_cells"] == report["n_cells"]
    assert "peak_cuda_allocation_gib" in pack
    assert "path_wall_clock_ms" in pack
    assert not (FORBIDDEN_FIELDS & pack.keys())
    md = render_md(pack)
    assert "Peak CUDA" in md
    assert "NOT serving" in pack["claim_boundary"] or "not serving" in pack["claim_boundary"].lower()


def test_forbidden_field_guard() -> None:
    with pytest.raises(ValueError):
        assert_no_forbidden_fields({"speedup": 2.0})
