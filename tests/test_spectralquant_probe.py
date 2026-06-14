"""Tests for Experiment 042 SpectralQuant external probe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "probe_spectralquant.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_042_SPECTRALQUANT_PROBE.md"


def _run(*extra: str, json_out: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    import os

    merged = {k: v for k, v in os.environ.items() if k != "SPECTRALQUANT_REPO_PATH"}
    if env:
        merged.update(env)
    cmd = [sys.executable, str(_SCRIPT), *extra]
    if json_out is not None:
        cmd.extend(["--json-out", str(json_out)])
    return subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, env=merged, check=False)


def test_missing_repo_blocked(tmp_path: Path) -> None:
    out = tmp_path / "sq_probe.json"
    proc = _run(json_out=out)
    assert proc.returncode == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["probe_status"] == "blocked"
    assert report["repo_path_present"] is False


def test_report_schema_blocked() -> None:
    from exactkv.external.spectralquant_probe import blocked_report, validate_report

    report = blocked_report(reason="blocked: test", repo_path_present=False)
    validate_report(report)
    assert report["exactkv_failures"] is None


def test_report_schema_tensor_smoke() -> None:
    from exactkv.external.spectralquant_probe import build_report, validate_report

    report = build_report(
        probe_status="tensor_smoke_only",
        blocked_reason="",
        repo_path_present=True,
        import_success=True,
        dependency_blocker="",
        discovered_api_summary={"public_symbols": ["SpectralQuantEngine"]},
        classification={"categories": ["kv_cache_tensor_compression"]},
        tensor_smoke_result={
            "status": "pass",
            "output_shape_preserved_keys": True,
            "key_max_abs_error": 0.5,
        },
        model_probe_result={"attempted": False, "status": "restricted_no_go"},
        exactkv_failures=None,
        limitations=["tensor smoke only"],
        notes=[],
        recommendation="tensor_smoke_only",
    )
    validate_report(report)


def test_model_probe_assessment_no_generation_path() -> None:
    from exactkv.external.spectralquant_probe import (
        SpectralQuantImportResult,
        assess_model_probe_feasibility,
    )

    result = assess_model_probe_feasibility(
        classification={"generation_time_cache_path": False},
        import_result=SpectralQuantImportResult(
            success=True, reason="", repo_path=Path("/tmp/sq"), modules=("calibration",)
        ),
    )
    assert result["attempted"] is False
    assert result["status"] == "restricted_no_go"


def test_classify_api_from_layout() -> None:
    from exactkv.external.spectralquant_probe import (
        SpectralQuantImportResult,
        classify_api,
    )

    layout = {
        "src_modules": ["calibration.py", "spectralquant.py", "engine.py"],
        "has_experiments_dir": True,
        "has_turboquant_baseline": False,
    }
    cls = classify_api(
        layout=layout,
        import_result=SpectralQuantImportResult(
            success=True,
            reason="",
            repo_path=Path("/tmp/sq"),
            public_symbols=("SpectralQuantEngine", "EigenspectralCalibrator"),
        ),
    )
    assert "kv_cache_tensor_compression" in cls["categories"]
    assert cls["generation_time_cache_path"] is False


@pytest.mark.parametrize(
    "forbidden",
    [
        "spectralquant improves memory",
        "production serving ready",
        "spectralquant is an exactkv compressor",
        "tensor smoke proves exactness",
    ],
)
def test_docs_avoid_forbidden_claims(forbidden: str) -> None:
    assert forbidden not in _DOC.read_text(encoding="utf-8").lower()


def test_docs_include_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "not integrated as a default ExactKV compressor" in text
    assert "External SpectralQuant claims are not ExactKV results" in text
    assert "Tensor smoke results are not generation results" in text
    assert "No speedup" in text
    assert "scoped" in text.lower() or "bounded" in text.lower()
