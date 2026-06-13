"""Tests for Experiment 039 Shard external-drafter stress panel."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "research" / "run_exp039_shard_stress_panel.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_039_SHARD_EXTERNAL_STRESS_PANEL.md"


def _run_panel(
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


def test_stress_prompt_panel_has_eight_categories() -> None:
    from exactkv.external.shard_stress_panel import STRESS_CATEGORY_ORDER, build_stress_prompt_panel

    panel = build_stress_prompt_panel(per_category=4, max_prompts=48)
    assert 24 <= len(panel) <= 48
    cats = {p["panel_category"] for p in panel}
    for cat in STRESS_CATEGORY_ORDER:
        assert cat in cats


def test_blocked_report_schema(tmp_path: Path) -> None:
    from exactkv.external.shard_stress_panel import build_panel_report, validate_panel_report

    report = build_panel_report(
        panel_status="blocked",
        blocked_reason="blocked: test",
        shard_repo_path_present=False,
        shard_import_success=False,
        model_used=None,
        max_new_tokens=64,
        draft_len=4,
        shard_settings={"streaming": True, "stream_bits": 8},
        tokenizer_alignment_pass=False,
        prompt_results=[],
        notes=["blocked test"],
        recommendation="blocked",
    )
    validate_panel_report(report)
    assert report["exactkv_failures"] is None
    assert report["no_divergence_observed"] is False


def test_divergence_rows_validate() -> None:
    from exactkv.external.shard_stress_panel import build_panel_report, validate_panel_report

    prompt_results = [
        {
            "prompt_id": "p1",
            "panel_category": "structured_json",
            "blocked": False,
            "token_alignment_pass": True,
            "exactkv_failure": False,
            "comparison": {
                "accepted_prefix_length": 3,
                "first_divergence_index": 3,
                "draft_token_id": 99,
                "verifier_token_id": 12,
                "draft_token_text": "foo",
                "verifier_token_text": "bar",
                "divergence_kind": "semantic_or_token_mismatch",
                "decoded_draft_prefix": "abc",
                "decoded_verifier_prefix": "abc",
            },
        }
    ]
    report = build_panel_report(
        panel_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=64,
        draft_len=4,
        shard_settings={"streaming": True, "stream_bits": 8},
        tokenizer_alignment_pass=True,
        prompt_results=prompt_results,
        notes=["divergence present"],
        recommendation="restricted_go_with_divergence",
    )
    validate_panel_report(report)
    assert report["divergence_count"] == 1
    assert report["exactkv_failures"] == 0
    assert len(report["divergence_examples"]) == 1
    assert report["no_divergence_observed"] is False


def test_exactkv_failures_field_present() -> None:
    from exactkv.external.shard_stress_panel import build_panel_report

    report = build_panel_report(
        panel_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=64,
        draft_len=4,
        shard_settings={},
        tokenizer_alignment_pass=True,
        prompt_results=[
            {
                "token_alignment_pass": True,
                "exactkv_failure": True,
                "comparison": {
                    "accepted_prefix_length": 0,
                    "first_divergence_index": 0,
                },
            }
        ],
        notes=[],
        recommendation="restricted_go_verify_harness",
    )
    assert report["exactkv_failures"] == 1


def test_prefix_distribution() -> None:
    from exactkv.external.shard_stress_panel import prefix_distribution

    dist = prefix_distribution([16, 16, 8, 4, None])
    assert dist["count"] == 4
    assert dist["min"] == 4
    assert dist["max"] == 16
    assert dist["histogram"] == {4: 1, 8: 1, 16: 2}


def test_script_blocked_without_shard_repo(tmp_path: Path) -> None:
    out_path = tmp_path / "panel.json"
    result = _run_panel(json_out=out_path)
    assert result.returncode == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["panel_status"] == "blocked"
    assert report["shard_repo_path_present"] is False


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


@pytest.mark.parametrize(
    "forbidden",
    [
        "2x speedup",
        "10× memory",
        "production serving ready",
        "accuracy improvement over",
        "integrated as default",
    ],
)
def test_docs_avoid_forbidden_claims(forbidden: str) -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    assert forbidden.lower() not in text


def test_classify_divergence_kind() -> None:
    from exactkv.external.shard_stress_panel import classify_divergence_kind

    assert classify_divergence_kind(
        draft_text="{",
        verifier_text="[",
        draft_token_id=1,
        verifier_token_id=2,
    ) == "formatting_or_punctuation"
    assert classify_divergence_kind(
        draft_text="Foo",
        verifier_text="foo",
        draft_token_id=1,
        verifier_token_id=2,
    ) == "casing"
