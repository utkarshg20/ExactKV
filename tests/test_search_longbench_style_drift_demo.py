"""Tests for Experiment 037 LongBench-style drift search (Phase 10A)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SEARCH = _ROOT / "scripts" / "search_longbench_style_drift_demo.py"


def test_dry_run_lists_prompts() -> None:
    result = subprocess.run(
        [sys.executable, str(_SEARCH), "--dry-run"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "dry-run OK" in result.stdout
    assert "10 prompts" in result.stdout


def test_build_prompt_set_categories() -> None:
    from scripts.search_longbench_style_drift_demo import build_prompt_set

    specs = build_prompt_set()
    categories = {s.category for s in specs}
    assert "customer_success_summary" in categories
    assert "support_policy_qa" in categories
    assert "multi_doc_qa" in categories
    assert len(specs) >= 10


def test_qa_heuristic_passes_reference_answer() -> None:
    from scripts.search_longbench_style_drift_demo import build_prompt_set, task_heuristic

    spec = next(s for s in build_prompt_set() if s.prompt_id == "lb_pol_001")
    ok = task_heuristic(spec, "The refund window is 30 days after purchase.")
    assert ok["pass"] is True
    assert ok["mode"] == "qa_reference"


def test_summary_heuristic_requires_facts() -> None:
    from scripts.search_longbench_style_drift_demo import build_prompt_set, task_heuristic

    spec = next(s for s in build_prompt_set() if s.prompt_id == "lb_cs_001")
    assert task_heuristic(spec, "SSO blocker; medium renewal risk; Maya owns follow-up.")["pass"] is True
    assert task_heuristic(spec, "Nothing relevant here.")["pass"] is False


def test_score_rejects_punctuation_only_diff() -> None:
    from scripts.search_longbench_style_drift_demo import build_prompt_set, evaluate_cell, score_candidate

    spec = build_prompt_set()[0]
    cell = {
        "full_output_text": "Maya owns SSO follow-up; medium risk.",
        "lossy_output_text": "Maya owns SSO follow-up; medium risk!",
        "exactkv_output_text": "Maya owns SSO follow-up; medium risk.",
        "lossy_exact_match": False,
        "exactkv_exact_match": True,
        "exactkv_failure": False,
        "highlight_round": {
            "correction_token": 1,
            "num_rejected": 1,
            "first_rejected_text": ".",
            "correction_text": "!",
        },
    }
    ev = evaluate_cell(cell, spec)
    assert ev["is_score_preserving_candidate"] is True
    assert score_candidate(ev, spec) < 100.0


def test_score_prefers_semantic_diff() -> None:
    from scripts.search_longbench_style_drift_demo import build_prompt_set, evaluate_cell, score_candidate

    spec = next(s for s in build_prompt_set() if s.prompt_id == "lb_md_001")
    base = {
        "full_output_text": "The answer is: Maya",
        "lossy_output_text": "The billing migration checkpoint is assigned to Maya.",
        "exactkv_output_text": "The answer is: Maya",
        "lossy_exact_match": False,
        "exactkv_exact_match": True,
        "exactkv_failure": False,
        "lossy_first_divergence_idx": 1,
        "highlight_round": {
            "correction_token": 100,
            "num_rejected": 1,
            "first_rejected_text": "billing",
            "correction_text": "answer",
        },
    }
    ev = evaluate_cell(base, spec)
    assert ev["is_score_preserving_candidate"] is True
    semantic_score = score_candidate(ev, spec)
    punct_cell = dict(base)
    punct_cell["highlight_round"] = {
        "correction_token": 1,
        "num_rejected": 1,
        "first_rejected_text": ",",
        "correction_text": ";",
    }
    punct_ev = evaluate_cell(punct_cell, spec)
    assert semantic_score > score_candidate(punct_ev, spec)


def test_fixture_candidate_json_roundtrip() -> None:
    fixture = _ROOT / "tests" / "fixtures" / "experiment_037_fixture_candidate.json"
    if not fixture.is_file():
        pytest.skip("fixture not present")
    from scripts.search_longbench_style_drift_demo import build_prompt_set, evaluate_cell

    data = json.loads(fixture.read_text(encoding="utf-8"))
    spec = next(s for s in build_prompt_set() if s.prompt_id == data["prompt_id"])
    ev = evaluate_cell(data, spec)
    assert ev["is_score_preserving_candidate"] is True
