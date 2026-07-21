"""Schema / claim-boundary tests for the serving microbench panel."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.benchmarks.serving_microbench_panel import (
    CLAIM_BOUNDARY,
    FORBIDDEN_FIELDS,
    assert_no_forbidden_fields,
    run_serving_microbench_panel,
    write_serving_microbench_outputs,
)
from scripts.build_serving_microbench_pack import build_pack, render_md


def test_deterministic_serving_schema_and_forbidden_fields(tmp_path: Path) -> None:
    report = run_serving_microbench_panel(
        smoke=True,
        deterministic_mode=True,
    )
    assert report["schema"] == "exactkv.serving_microbench.v1"
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["n_cells"] == 1  # smoke det: 1 model × 1 comp × 1×1×1
    assert report["exactkv_failures"] == 0
    assert_no_forbidden_fields(report)
    cell = report["cells"][0]
    for arm in ("full", "lossy", "exactkv"):
        assert "completed_requests_per_sec" in cell[arm]
        assert "mean_ttft_like_ms" in cell[arm]
        assert "gpu_peak_allocated_bytes" in cell[arm]
        assert "peak_delta_vs_full_bytes" in cell[arm]
    assert "speedup" not in cell
    assert "throughput" not in cell

    out = tmp_path / "qwen_raw.json"
    write_serving_microbench_outputs(
        report, json_path=out, markdown_path=tmp_path / "s.md"
    )
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_cells"] == report["n_cells"]


def test_build_serving_pack() -> None:
    report = run_serving_microbench_panel(smoke=True, deterministic_mode=True)
    pack = build_pack(report["cells"], sources=["synthetic"])
    assert pack["n_cells"] == report["n_cells"]
    assert "completed_requests_per_sec" in pack
    assert "ttft_like_ms" in pack
    assert "peak_cuda_allocation_gib" in pack
    assert not (FORBIDDEN_FIELDS & pack.keys())
    md = render_md(pack)
    assert "requests/sec" in md.lower() or "Completed requests" in md
    assert "NOT vLLM" in pack["claim_boundary"] or "not vLLM" in pack["claim_boundary"]


def test_forbidden_field_guard() -> None:
    with pytest.raises(ValueError):
        assert_no_forbidden_fields({"speedup": 2.0})
