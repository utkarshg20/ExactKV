"""Tests for Phase C publication + demo layer."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.benchmarks.leaderboard_platform import load_phase_a_report
from exactkv.benchmarks.publication_layer import (
    PHASE_C_ID,
    extract_canonical_demos,
    run_phase_c_publication_layer,
    validate_phase_c_outputs,
    write_phase_c_outputs,
)


def test_extract_five_demos() -> None:
    phase_a = load_phase_a_report(Path("reports/phaseA_benchmark.json"))
    demos = extract_canonical_demos(phase_a)
    assert len(demos) == 5
    categories = {d["category"] for d in demos}
    assert "structured_output_drift" in categories
    assert "worst_case_compression" in categories
    assert "first_divergence_explosion" in categories


def test_demos_have_required_fields() -> None:
    phase_a = load_phase_a_report(Path("reports/phaseA_benchmark.json"))
    demos = extract_canonical_demos(phase_a)
    for d in demos:
        assert d.get("input_prompt")
        assert d.get("divergence_timeline")
        assert d.get("acceptance_decision_path")
        assert d.get("metrics") is not None
        assert "reports/phaseA_benchmark.json" in d["data_sources"]


def test_run_phase_c_pipeline(tmp_path: Path) -> None:
    result = run_phase_c_publication_layer(output_dir=tmp_path / "visuals")
    assert result["phase_id"] == PHASE_C_ID
    assert len(result["demos"]) == 5
    assert result["paper_draft_md"]
    assert result["blog_post_md"]
    assert result["x_thread_md"]
    assert result["linkedin_post_md"]
    assert result["visual_synthesis"]["first_divergence_map"]


def test_write_all_outputs(tmp_path: Path) -> None:
    result = run_phase_c_publication_layer(output_dir=tmp_path / "visuals")
    paths = write_phase_c_outputs(
        result,
        demo_pack_path=tmp_path / "demo_pack.json",
        paper_path=tmp_path / "paper_draft.md",
        blog_path=tmp_path / "blog_post.md",
        x_path=tmp_path / "x_thread.md",
        linkedin_path=tmp_path / "linkedin_post.md",
    )
    assert Path(paths["demo_pack"]).exists()
    pack = json.loads(Path(paths["demo_pack"]).read_text())
    assert validate_phase_c_outputs(pack).valid


def test_no_fabricated_output_text_flag() -> None:
    phase_a = load_phase_a_report(Path("reports/phaseA_benchmark.json"))
    demos = extract_canonical_demos(phase_a)
    for d in demos:
        assert d["outputs"]["full_reference"]["output_text_available"] is False or d["outputs"]["full_reference"].get("output_text")


def test_no_runtime_flags() -> None:
    result = run_phase_c_publication_layer()
    assert result["exactkv_generator_modified"] is False
    assert result["runtime_commit_authorized"] is False
    assert result["model_experiments_run"] is False
