"""Tests for Experiment 043 SpectralQuant real KV tensor smoke."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "research" / "run_exp043_spectralquant_real_kv_smoke.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_043_SPECTRALQUANT_REAL_KV_SMOKE.md"


def _run(*extra: str, json_out: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    import os

    merged = {k: v for k, v in os.environ.items() if k != "SPECTRALQUANT_REPO_PATH"}
    if env:
        merged.update(env)
    cmd = [sys.executable, str(_SCRIPT), "--json-out", str(json_out), *extra]
    return subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, env=merged, check=False)


def test_missing_repo_blocked(tmp_path: Path) -> None:
    out = tmp_path / "043.json"
    proc = _run(json_out=out)
    assert proc.returncode == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["label"] == "real_kv_tensor_smoke"
    assert report["repo_path_present"] is False


def test_report_schema_blocked() -> None:
    from exactkv.external.spectralquant_real_kv import validate_043_report

    report = {
        "experiment_id": "043_spectralquant_real_kv_smoke",
        "status": "blocked",
        "label": "real_kv_tensor_smoke",
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt": "",
        "repo_path_present": False,
        "import_success": False,
        "calibration": {"required": True},
        "kv_capture": {},
        "layer_results": [],
        "summary": {"per_layer_compression_works": False, "calibration_required": True},
        "limitations": [],
        "claims_forbidden": [],
        "recommendation": "blocked",
    }
    validate_043_report(report)


def test_report_schema_pass_shape() -> None:
    from exactkv.external.spectralquant_real_kv import validate_043_report

    report = {
        "experiment_id": "043_spectralquant_real_kv_smoke",
        "status": "pass",
        "label": "real_kv_tensor_smoke",
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt": "test",
        "repo_path_present": True,
        "import_success": True,
        "calibration": {"required": True, "n_calibration_prompts": 2},
        "kv_capture": {"num_layers": 24, "layer0_key_shape": [1, 2, 8, 64]},
        "layer_results": [{"layer_idx": 0, "key_shape_preserved": True, "value_shape_preserved": True}],
        "summary": {
            "per_layer_compression_works": True,
            "calibration_required": True,
            "key_max_abs_error": 1.0,
            "value_max_abs_error": 0.5,
        },
        "limitations": ["tensor smoke only"],
        "claims_forbidden": ["No speedup claim"],
        "recommendation": "real_kv_smoke_only",
    }
    validate_043_report(report)


def test_docs_exist_and_caveats() -> None:
    assert _DOC.is_file()
    text = _DOC.read_text(encoding="utf-8").lower()
    assert "real_kv_tensor_smoke" in text or "real kv" in text
    assert "not integrated" in text or "not a default" in text
    assert "not generation" in text or "not exactkv generation" in text
    assert "no speedup" in text
    forbidden = [
        "spectralquant is integrated",
        "spectralquant has exactkv acceptance",
        "exactkv_failures=0 for tensor smoke",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_not_default_registry() -> None:
    from exactkv.compressors import list_compressors

    assert "spectralquant_experimental" not in list_compressors()
