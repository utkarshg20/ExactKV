"""Tests for external benchmark panels (LongBench, RULER, BFCL, HumanEval, MBPP)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.benchmarks.external_dataset_loaders import (
    load_external_prompts,
    load_jsonl_prompts,
    PILOT_PATHS,
)
from exactkv.benchmarks.external_panel import (
    EXTERNAL_PANEL_ID,
    build_external_context_prompt,
    load_external_panel_resume_index,
    run_external_panel,
    write_external_panel_outputs,
)


def test_bfcl_hf_loader_categories() -> None:
    from exactkv.benchmarks.external_dataset_loaders import BFCL_HF_CATEGORIES, _format_bfcl_prompt

    sample = {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Get weather in Austin."}]],
        "function": [{"name": "get_weather", "description": "weather tool"}],
    }
    text = _format_bfcl_prompt(sample)
    assert "get_weather" in text
    assert "Austin" in text
    assert len(BFCL_HF_CATEGORIES) == 4


@pytest.mark.parametrize("family", ["longbench", "ruler", "bfcl", "humaneval", "mbpp"])
def test_pilot_loaders(family: str) -> None:
    rows = load_external_prompts(family, source="pilot", max_prompts=4)
    assert len(rows) >= 2
    assert rows[0]["dataset_family"] == family
    assert rows[0]["prompt"].strip()


def test_longbench_pilot_categories() -> None:
    rows = load_external_prompts("longbench", source="pilot")
    categories = {r["category"] for r in rows}
    assert "narrativeqa" in categories
    assert "hotpotqa" in categories


def test_ruler_pilot_declared_lengths() -> None:
    rows = load_external_prompts("ruler", source="pilot")
    assert any(r.get("declared_context_tokens") == 4096 for r in rows)


def test_deterministic_smoke_longbench() -> None:
    report = run_external_panel("longbench", deterministic_mode=True, smoke=True)
    assert report["phase_id"] == EXTERNAL_PANEL_ID
    assert report["dataset_family"] == "longbench"
    assert report["status"] == "benchmark_complete"
    assert report["cells_run"] > 0
    assert report["exactkv_failures"] == 0
    assert report["category_summary"]


def test_deterministic_smoke_ruler() -> None:
    report = run_external_panel("ruler", deterministic_mode=True, smoke=True)
    assert report["dataset_family"] == "ruler"
    assert "512" in report["bucket_summary"]


def test_write_external_outputs(tmp_path: Path) -> None:
    report = run_external_panel("bfcl", deterministic_mode=True, smoke=True)
    json_path = tmp_path / "bfcl_raw.json"
    md_path = tmp_path / "bfcl_summary.md"
    write_external_panel_outputs(report, json_path=json_path, markdown_path=md_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["dataset_family"] == "bfcl"
    assert "External Panel: Bfcl" in md_path.read_text()


def test_load_jsonl_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        "# comment\n"
        '{"prompt_id":"x1","category":"t","prompt":"hello"}\n',
        encoding="utf-8",
    )
    rows = load_jsonl_prompts(path, dataset_family="test", default_category="t")
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "x1"


def test_pilot_paths_exist() -> None:
    for family, path in PILOT_PATHS.items():
        assert path.is_file(), f"missing pilot file for {family}: {path}"


def test_kivi_offline_degrades_gracefully_when_unavailable() -> None:
    """kivi_offline must produce skipped cells (not crash) when models.utils_quant is absent.

    Note: smoke=True overrides the compressors arg; use deterministic_mode without smoke
    and pass minimal buckets/mnt so no GPU is needed.
    """
    import importlib

    kivi_available = (
        importlib.util.find_spec("models") is not None
        and importlib.util.find_spec("models.utils_quant") is not None
    )
    if kivi_available:
        pytest.skip("KIVI is available; graceful-degradation test is for environments without it")

    report = run_external_panel(
        "longbench",
        deterministic_mode=True,
        smoke=False,
        compressors=["kivi_offline"],
        context_buckets=[512],
        max_new_tokens_values=[16],
        max_prompts=2,
    )
    assert report["status"] == "benchmark_complete"
    cells = report.get("cells", [])
    assert len(cells) > 0, "expected at least one cell (skipped)"
    skipped = [c for c in cells if c.get("status") == "skipped"]
    assert len(skipped) > 0, "kivi_offline should produce skipped cells when KIVI is unavailable"
    for c in skipped:
        assert c.get("skip_reason") or c.get("compressor_name") == "kivi_offline"


def test_kivi_offline_deterministic_smoke_with_builtins() -> None:
    """Deterministic panel with noop+kivi_offline: noop runs, kivi_offline skips gracefully."""
    report = run_external_panel(
        "longbench",
        deterministic_mode=True,
        smoke=False,
        compressors=["noop", "kivi_offline"],
        context_buckets=[512],
        max_new_tokens_values=[16],
        max_prompts=2,
    )
    cells = report.get("cells", [])
    ok_cells = [c for c in cells if c.get("status") == "ok"]
    # noop cells must have run
    assert any(c.get("compressor_name") == "noop" for c in ok_cells), \
        "noop cells should succeed in deterministic mode"
    # kivi_offline cells should be skipped without crashing
    kivi_cells = [c for c in cells if c.get("compressor_name") == "kivi_offline"]
    assert len(kivi_cells) > 0
    for c in kivi_cells:
        assert c.get("status") in ("skipped", "ok"), f"unexpected status: {c.get('status')}"


def test_build_external_context_prompt_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(range(len(text)))

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return "x" * len(ids)

    class _Runtime:
        tokenizer = _Tok()

    base = {"prompt_id": "p1", "category": "t", "prompt": "abcdefghij"}
    entry = build_external_context_prompt(_Runtime(), base, 5)
    assert entry["prefill_tokens"] == 5
    assert entry["context_bucket"] == 5


def test_external_panel_resume_skips_completed_cells(tmp_path: Path) -> None:
    json_path = tmp_path / "longbench_resume.json"
    first = run_external_panel(
        "longbench",
        deterministic_mode=True,
        smoke=True,
        checkpoint_json=json_path,
    )
    write_external_panel_outputs(first, json_path=json_path)
    assert first["status"] == "benchmark_complete"
    assert json_path.is_file()

    second = run_external_panel(
        "longbench",
        deterministic_mode=True,
        smoke=True,
        resume_json=json_path,
        checkpoint_json=json_path,
    )
    assert second["status"] == "benchmark_complete"
    assert second["cells_run"] == first["cells_run"]
    assert load_external_panel_resume_index(json_path)
