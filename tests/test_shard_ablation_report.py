"""Tests for Experiment 040 Shard external-drafter ablation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "research" / "run_exp040_shard_ablation.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_040_SHARD_EXTERNAL_ABLATION.md"


def _run_ablation(
    *,
    env: dict[str, str] | None = None,
    extra: tuple[str, ...] = (),
    json_out: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    import os

    merged = {k: v for k, v in os.environ.items() if k != "SHARD_REPO_PATH"}
    if env:
        merged.update(env)
    cmd = [sys.executable, str(_SCRIPT), *extra]
    if json_out is not None:
        cmd.extend(["--json-out", str(json_out)])
    return subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=merged,
        check=False,
    )


def test_ablation_grid_has_five_settings() -> None:
    from exactkv.external.shard_ablation import build_ablation_grid

    tested, skipped = build_ablation_grid()
    assert len(tested) == 5
    names = {s["setting_name"] for s in tested}
    assert "baseline_64tok" in names
    assert "length_128tok" in names
    assert "stream_bits_4" in names
    assert skipped


def test_ablation_report_schema() -> None:
    from exactkv.external.shard_ablation import build_ablation_grid, build_ablation_report, validate_ablation_report

    settings, skipped = build_ablation_grid()
    row = {
        "setting_name": "baseline_64tok",
        "max_new_tokens": 64,
        "shard_settings": {"streaming": True, "stream_bits": 8},
        "status": "pass",
        "blocked_reason": "",
        "tokenizer_alignment_pass": True,
        "blocked_prompt_count": 0,
        "prompt_count": 32,
        "exactkv_failures": 0,
        "accepted_prefix_mean": 58.0,
        "accepted_prefix_median": 64,
        "accepted_prefix_min": 14,
        "accepted_prefix_distribution": {"count": 32, "histogram": {64: 26}},
        "divergence_count": 6,
        "divergence_rate": 0.1875,
        "semantic_divergence_count": 1,
        "formatting_divergence_count": 5,
        "divergence_examples": [],
    }
    report = build_ablation_report(
        ablation_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used="meta-llama/Llama-3.1-8B-Instruct",
        draft_len=4,
        prompt_count=32,
        settings_tested=[s["setting_name"] for s in settings],
        settings_skipped=skipped,
        setting_results=[row],
        notes=["test"],
        recommendation="expand_shard_lossy_ablation",
    )
    validate_ablation_report(report)


def test_blocked_setting_row_validates() -> None:
    from exactkv.external.shard_ablation import validate_setting_row

    validate_setting_row(
        {
            "setting_name": "stream_bits_2",
            "max_new_tokens": 64,
            "shard_settings": {},
            "status": "skipped",
            "blocked_reason": "unsupported",
            "tokenizer_alignment_pass": False,
            "blocked_prompt_count": 0,
            "prompt_count": 0,
            "exactkv_failures": None,
            "accepted_prefix_mean": None,
            "accepted_prefix_median": None,
            "accepted_prefix_min": None,
            "accepted_prefix_distribution": {"count": 0, "histogram": {}},
            "divergence_count": 0,
            "divergence_rate": None,
            "semantic_divergence_count": 0,
            "formatting_divergence_count": 0,
            "divergence_examples": [],
        }
    )


def test_divergence_examples_in_setting_row() -> None:
    from exactkv.external.shard_ablation import summarize_setting_result

    row = summarize_setting_result(
        setting_name="stream_bits_4",
        max_new_tokens=64,
        shard_settings={"stream_bits": 4},
        prompt_results=[
            {
                "prompt_id": "p1",
                "panel_category": "structured_json",
                "token_alignment_pass": True,
                "exactkv_failure": False,
                "comparison": {
                    "accepted_prefix_length": 10,
                    "first_divergence_index": 10,
                    "draft_token_id": 1,
                    "verifier_token_id": 2,
                    "draft_token_text": "a",
                    "verifier_token_text": "b",
                    "divergence_kind": "semantic_or_token_mismatch",
                    "decoded_draft_prefix": "x",
                    "decoded_verifier_prefix": "x",
                },
            }
        ],
    )
    assert row["semantic_divergence_count"] == 1
    assert len(row["divergence_examples"]) == 1


def test_exactkv_failures_field_exists() -> None:
    from exactkv.external.shard_ablation import summarize_setting_result

    row = summarize_setting_result(
        setting_name="test",
        max_new_tokens=64,
        shard_settings={},
        prompt_results=[
            {
                "token_alignment_pass": True,
                "exactkv_failure": True,
                "comparison": {"accepted_prefix_length": 0, "first_divergence_index": 0},
            }
        ],
    )
    assert row["exactkv_failures"] == 1


def test_script_blocked_without_shard_repo(tmp_path: Path) -> None:
    out_path = tmp_path / "ablation.json"
    result = _run_ablation(json_out=out_path)
    assert result.returncode == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ablation_status"] == "blocked"


def test_docs_include_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "external-drafter probe" in text.lower()
    assert "not an integrated ExactKV compressor" in text
    assert "External Shard README claims are not ExactKV results" in text
    assert "No speedup" in text
    assert "active memory savings" in text
    assert "production serving" in text
    assert "model accuracy improvement" in text
    assert "scoped to the tested panel only" in text.lower()
    assert "stream_bits=8" in text
    assert "lossless" in text.lower()


@pytest.mark.parametrize(
    "forbidden",
    [
        "2x speedup",
        "integrated as default",
        "production serving ready",
        "stream_bits=8 lossy",
    ],
)
def test_docs_avoid_forbidden_claims(forbidden: str) -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    assert forbidden.lower() not in text


def test_stream_bits_8_not_marked_lossy_in_grid() -> None:
    from exactkv.external.shard_ablation import build_ablation_grid

    tested, _ = build_ablation_grid()
    baseline = next(s for s in tested if s["setting_name"] == "baseline_64tok")
    assert baseline["shard_settings"]["stream_bits"] == 8
    assert "lossless" in baseline["shard_settings"]["label"].lower()
