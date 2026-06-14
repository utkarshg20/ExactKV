"""Tests for Experiment 045 SpectralQuant restricted adapter panel."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "research" / "run_exp045_spectralquant_restricted_panel.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md"


def _run(json_out: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    import os

    merged = {k: v for k, v in os.environ.items() if k != "SPECTRALQUANT_REPO_PATH"}
    if env:
        merged.update(env)
    cmd = [sys.executable, str(_SCRIPT), "--json-out", str(json_out)]
    return subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, env=merged, check=False)


def test_missing_repo_blocked(tmp_path: Path) -> None:
    out = tmp_path / "045.json"
    proc = _run(out)
    assert proc.returncode == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["exactkv_failures"] is None
    assert report["leaderboard_decision"]["promote_to_restricted_backend"] is False


def test_report_schema_pass() -> None:
    from exactkv.external.spectralquant_real_kv import validate_045_report

    report = {
        "experiment_id": "045_spectralquant_restricted_panel",
        "status": "pass",
        "adapter_name": "spectralquant_experimental",
        "not_default_registry": True,
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt_count": 12,
        "calibration": {"calibration_prompt_count": 6, "required": True},
        "panel_composition": [{"prompt_id": "p0", "panel_category": "natural_language"}],
        "exactkv_failures": 0,
        "acceptance_summary": {
            "mean_acceptance": 0.62,
            "median_acceptance": 0.65,
            "min_acceptance": 0.37,
            "n_prompts": 12,
        },
        "divergence_summary": {"draft_divergence_prompt_count": 3, "examples": []},
        "reconstruction_error_summary": {"key_max_abs_error": 38.0, "value_max_abs_error": 0.8},
        "materializing_adapter": True,
        "memory_claim_note": "no active memory savings",
        "supports_real_bytes_claim": False,
        "leaderboard_decision": {
            "promote_to_restricted_backend": True,
            "min_prompts_required": 8,
            "exactkv_failures_required": 0,
            "tier_if_promoted": "RESTRICTED BACKEND",
            "tier_if_not": "SMOKE ONLY",
            "reason": "ok",
        },
        "limitations": ["small panel"],
        "claims_forbidden": ["No speedup claim"],
        "recommendation": "promote_restricted_backend",
        "per_prompt": [],
    }
    validate_045_report(report)


def test_leaderboard_promotion_rule() -> None:
    from exactkv.external.spectralquant_real_kv import (
        PANEL_PROMOTION_MIN_PROMPTS,
        leaderboard_promotion_decision,
    )

    yes = leaderboard_promotion_decision(exactkv_failures=0, prompt_count=12)
    assert yes["promote_to_restricted_backend"] is True
    no_fail = leaderboard_promotion_decision(exactkv_failures=1, prompt_count=12)
    assert no_fail["promote_to_restricted_backend"] is False
    few = leaderboard_promotion_decision(exactkv_failures=0, prompt_count=7)
    assert few["promote_to_restricted_backend"] is False
    assert PANEL_PROMOTION_MIN_PROMPTS == 8


def test_not_default_registry() -> None:
    from exactkv.compressors import list_compressors

    assert "spectralquant_experimental" not in list_compressors()


def test_docs_exist_and_caveats() -> None:
    assert _DOC.is_file()
    text = _DOC.read_text(encoding="utf-8").lower()
    assert "factory-only" in text or "factory only" in text
    assert "not default" in text or "not integrated" in text
    assert "materializ" in text
    assert "no speedup" in text
    assert "small" in text and "panel" in text
    forbidden = [
        "spectralquant is integrated as a default",
        "full benchmark coverage",
        "production serving readiness",
    ]
    for phrase in forbidden:
        assert phrase not in text
