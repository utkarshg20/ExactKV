"""Tests for Experiment 044 SpectralQuant experimental adapter smoke."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "research" / "run_exp044_spectralquant_adapter_smoke.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_044_SPECTRALQUANT_ADAPTER_SMOKE.md"


def _run(json_out: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    import os

    merged = {k: v for k, v in os.environ.items() if k != "SPECTRALQUANT_REPO_PATH"}
    if env:
        merged.update(env)
    cmd = [sys.executable, str(_SCRIPT), "--json-out", str(json_out)]
    return subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, env=merged, check=False)


def test_missing_repo_blocked(tmp_path: Path) -> None:
    out = tmp_path / "044.json"
    proc = _run(json_out=out)
    assert proc.returncode == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["not_default_registry"] is True
    assert report["exactkv_failures"] is None


def test_report_schema_blocked() -> None:
    from exactkv.external.spectralquant_real_kv import validate_044_report

    report = {
        "experiment_id": "044_spectralquant_adapter_smoke",
        "status": "blocked",
        "adapter_name": "spectralquant_experimental",
        "not_default_registry": True,
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt_panel": [],
        "exactkv_failures": None,
        "acceptance_summary": None,
        "per_prompt": [],
        "memory_claim_note": "no active memory savings",
        "limitations": [],
        "claims_forbidden": [],
        "recommendation": "blocked",
    }
    validate_044_report(report)


def test_report_schema_pass() -> None:
    from exactkv.external.spectralquant_real_kv import validate_044_report

    report = {
        "experiment_id": "044_spectralquant_adapter_smoke",
        "status": "pass",
        "adapter_name": "spectralquant_experimental",
        "not_default_registry": True,
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt_panel": ["p0", "p1"],
        "exactkv_failures": 0,
        "acceptance_summary": {"mean_acceptance": 0.95, "n_prompts": 2},
        "per_prompt": [
            {"prompt_id": "p0", "exactkv_failures": 0, "acceptance_rate": 1.0},
        ],
        "memory_claim_note": "no active memory savings",
        "limitations": ["experimental"],
        "claims_forbidden": ["No speedup claim"],
        "recommendation": "restricted_adapter_go",
    }
    validate_044_report(report)


def test_factory_not_in_registry() -> None:
    from exactkv.compressors import list_compressors

    names = list_compressors()
    assert "spectralquant_experimental" not in names
    assert "spectralquant" not in names


def test_docs_exist_and_caveats() -> None:
    assert _DOC.is_file()
    text = _DOC.read_text(encoding="utf-8").lower()
    assert "factory-only" in text or "factory only" in text
    assert "not integrated" in text or "not default" in text
    assert (
        "no active memory" in text
        or "no memory savings" in text
        or "active gpu memory savings" in text
    )
    assert "experimental" in text
    forbidden = [
        "spectralquant is integrated as a default",
        "spectralquant improves memory",
        "spectralquant improves speed",
        "production-ready spectralquant",
    ]
    for phrase in forbidden:
        assert phrase not in text
