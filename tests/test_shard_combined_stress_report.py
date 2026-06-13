"""Tests for Experiment 041 Shard combined stress."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "research" / "run_exp041_shard_combined_stress.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_041_SHARD_COMBINED_STRESS.md"


def _run(*extra: str, json_out: Path | None = None) -> subprocess.CompletedProcess[str]:
    import os

    env = {k: v for k, v in os.environ.items() if k != "SHARD_REPO_PATH"}
    cmd = [sys.executable, str(_SCRIPT), *extra]
    if json_out is not None:
        cmd.extend(["--json-out", str(json_out)])
    return subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, env=env, check=False)


def test_combined_settings() -> None:
    from exactkv.external.shard_combined_stress import COMBINED_SHARD_SETTINGS

    assert COMBINED_SHARD_SETTINGS["stream_bits"] == 4
    assert COMBINED_SHARD_SETTINGS["stream_qjl"] is False
    assert COMBINED_SHARD_SETTINGS["k_target_cr"] == 16.0


def test_combined_report_schema() -> None:
    from exactkv.external.shard_combined_stress import (
        build_combined_report,
        validate_combined_report,
    )

    result = {
        "prompt_count": 32,
        "blocked_prompt_count": 0,
        "tokenizer_alignment_pass": True,
        "exactkv_failures": 0,
        "accepted_prefix_mean": 100.0,
        "accepted_prefix_median": 128,
        "accepted_prefix_min": 14,
        "accepted_prefix_distribution": {"count": 32, "histogram": {128: 20}},
        "divergence_count": 12,
        "divergence_rate": 0.375,
        "semantic_divergence_count": 2,
        "formatting_divergence_count": 10,
        "divergence_examples": [],
        "prompt_results": [],
    }
    report = build_combined_report(
        combined_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=128,
        draft_len=4,
        shard_settings={"stream_bits": 4},
        result=result,
        notes=["test"],
        recommendation="stop_shard_bounded_probe_complete",
    )
    validate_combined_report(report)
    assert report["exp040_comparison"]
    assert "vs_length_128tok" in report["combined_vs_exp040"]


def test_exp040_comparison_fields() -> None:
    from exactkv.external.shard_combined_stress import build_exp040_comparison_summary

    vs = build_exp040_comparison_summary({
        "divergence_rate": 0.35,
        "divergence_count": 11,
        "accepted_prefix_mean": 105.0,
    })
    assert vs["vs_baseline_64tok"]["divergence_rate_delta"] > 0
    assert vs["vs_length_128tok"]["divergence_rate_delta"] > 0
    assert "increased_vs_length_128_or_stream_bits_4" in vs


def test_divergence_examples_validate() -> None:
    from exactkv.external.shard_combined_stress import summarize_combined_from_prompt_results, build_combined_report

    prompt_results = [
        {
            "token_alignment_pass": True,
            "exactkv_failure": False,
            "comparison": {
                "accepted_prefix_length": 60,
                "first_divergence_index": 60,
                "draft_token_id": 1,
                "verifier_token_id": 2,
                "draft_token_text": "sku",
                "verifier_token_text": "qty",
                "divergence_kind": "semantic_or_token_mismatch",
                "decoded_draft_prefix": "x",
                "decoded_verifier_prefix": "x",
            },
            "prompt_id": "struct_json_004",
            "panel_category": "structured_json",
        }
    ]
    result = summarize_combined_from_prompt_results(prompt_results)
    assert result["semantic_divergence_count"] == 1
    report = build_combined_report(
        combined_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=128,
        draft_len=4,
        shard_settings={},
        result=result,
        notes=[],
        recommendation="stop_shard_bounded_probe_complete",
    )
    assert report["exactkv_failures"] == 0
    assert len(report["divergence_examples"]) == 1


def test_script_blocked_without_shard_repo(tmp_path: Path) -> None:
    out = tmp_path / "combined.json"
    proc = _run(json_out=out)
    assert proc.returncode == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["combined_status"] == "blocked"


def test_docs_include_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "external-drafter probe" in text.lower()
    assert "not an integrated ExactKV compressor" in text
    assert "External Shard README claims are not ExactKV results" in text
    assert "No speedup" in text
    assert "scoped to this bounded panel only" in text.lower()
    assert "stream_bits=4" in text
    assert "128" in text


@pytest.mark.parametrize(
    "forbidden",
    [
        "shard is integrated",
        "shard improves memory",
        "production serving ready",
        "shard is an exactkv compressor",
    ],
)
def test_docs_avoid_forbidden_claims(forbidden: str) -> None:
    assert forbidden not in _DOC.read_text(encoding="utf-8").lower()
