"""Tests for evidence-plus benchmark panel."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.benchmarks.evidence_plus_panel import (
    EVIDENCE_PLUS_ID,
    load_base_prompts,
    resolve_evidence_plus_compressor,
    run_evidence_plus_panel,
    write_evidence_plus_outputs,
)


def test_deterministic_smoke_panel() -> None:
    report = run_evidence_plus_panel(deterministic_mode=True, smoke=True)
    assert report["phase_id"] == EVIDENCE_PLUS_ID
    assert report["status"] == "benchmark_complete"
    assert report["cells_run"] > 0
    assert report["exactkv_failures"] == 0
    assert "512" in report["bucket_summary"]


def test_load_base_prompts_includes_long_context() -> None:
    prompts = load_base_prompts(max_prompts=4)
    assert len(prompts) >= 2
    assert any(p["prompt_id"].startswith("lc_") for p in prompts)


def test_external_compressor_resolution_graceful() -> None:
    res = resolve_evidence_plus_compressor("kivi_offline")
    assert res.compressor_name == "kivi_offline"
    assert res.backend_tier in ("RESTRICTED_ADAPTER", "UNAVAILABLE")

    res_r32 = resolve_evidence_plus_compressor("kivi_offline_r32")
    assert res_r32.compressor_name == "kivi_offline_r32"
    assert res_r32.backend_tier in ("RESTRICTED_ADAPTER", "UNAVAILABLE")

    for name in ("snapkv_experimental", "kvpress_knorm_experimental", "turboquant_experimental"):
        res = resolve_evidence_plus_compressor(name)
        assert res.compressor_name == name
        assert res.backend_tier in ("RESTRICTED_ADAPTER", "UNAVAILABLE")


def test_write_outputs(tmp_path: Path) -> None:
    report = run_evidence_plus_panel(deterministic_mode=True, smoke=True)
    json_path = tmp_path / "raw.json"
    md_path = tmp_path / "summary.md"
    write_evidence_plus_outputs(report, json_path=json_path, markdown_path=md_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["phase_id"] == EVIDENCE_PLUS_ID
    assert "# Evidence-Plus Panel Summary" in md_path.read_text()
