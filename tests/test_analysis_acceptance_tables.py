"""Tests for exactkv.analysis.acceptance_tables.

Gate: acceptance table gate.

Uses synthetic report dicts for fast unit tests, plus one real sweep for
integration coverage.

Verifies:
  * build_acceptance_table produces one row per (compressor, draft_len) pair.
  * group_acceptance_by_compressor collapses draft_len correctly.
  * group_acceptance_by_draft_len collapses compressor correctly.
  * group_acceptance_by_category groups by category.
  * total_drafted == total_accepted + total_rejected in every row.
  * exactkv_failures are counted correctly.
  * write_acceptance_table_csv writes a valid CSV (creates dirs automatically).
  * No forbidden performance fields in any table.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_by_two_report():
    """2 compressors × 2 draft_lengths × 1 prompt = 4 results."""
    return make_report(
        make_result(compressor_name="noop",  draft_len=4, category="nl"),
        make_result(compressor_name="noop",  draft_len=8, category="nl"),
        make_result(compressor_name="int8",  draft_len=4, category="code"),
        make_result(compressor_name="int8",  draft_len=8, category="code"),
    )


@pytest.fixture
def failure_report():
    """Report containing one ExactKV failure."""
    return make_report(
        make_result(compressor_name="noop", exactkv_failure=False),
        make_result(compressor_name="int8", exactkv_failure=True),
    )


# ---------------------------------------------------------------------------
# build_acceptance_table
# ---------------------------------------------------------------------------

def test_build_table_row_count(two_by_two_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    table = build_acceptance_table(two_by_two_report)
    assert len(table) == 4, f"Expected 4 rows, got {len(table)}"


def test_build_table_row_keys(two_by_two_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    table = build_acceptance_table(two_by_two_report)
    required = {
        "compressor_name", "draft_len", "num_runs",
        "mean_acceptance_rate", "mean_average_accepted_length",
        "total_drafted", "total_accepted", "total_rejected",
        "total_corrections", "exactkv_failures",
    }
    for row in table:
        for key in required:
            assert key in row, f"Table row missing key {key!r}"


def test_build_table_count_reconciles(two_by_two_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    for row in build_acceptance_table(two_by_two_report):
        assert row["total_drafted"] == row["total_accepted"] + row["total_rejected"], (
            f"Count mismatch in row: {row}"
        )


def test_build_table_no_forbidden_fields(two_by_two_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    for row in build_acceptance_table(two_by_two_report):
        assert not (set(row.keys()) & _FORBIDDEN), f"Forbidden field in row: {row}"


def test_build_table_exactkv_failures_counted(failure_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    table = build_acceptance_table(failure_report)
    int8_row = next(r for r in table if r["compressor_name"] == "int8")
    noop_row = next(r for r in table if r["compressor_name"] == "noop")
    assert int8_row["exactkv_failures"] == 1
    assert noop_row["exactkv_failures"] == 0


def test_build_table_num_runs(two_by_two_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    for row in build_acceptance_table(two_by_two_report):
        assert row["num_runs"] == 1   # one result per (compressor, draft_len) cell


# ---------------------------------------------------------------------------
# group_acceptance_by_compressor
# ---------------------------------------------------------------------------

def test_group_by_compressor_row_count(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
    rows = group_acceptance_by_compressor(two_by_two_report)
    assert len(rows) == 2   # noop and int8


def test_group_by_compressor_has_key(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
    rows = group_acceptance_by_compressor(two_by_two_report)
    names = {r["compressor_name"] for r in rows}
    assert "noop" in names and "int8" in names


def test_group_by_compressor_num_runs_collapsed(two_by_two_report):
    """Collapsing 2 draft_lens means num_runs == 2 per compressor."""
    from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
    for row in group_acceptance_by_compressor(two_by_two_report):
        assert row["num_runs"] == 2


def test_group_by_compressor_count_reconciles(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
    for row in group_acceptance_by_compressor(two_by_two_report):
        assert row["total_drafted"] == row["total_accepted"] + row["total_rejected"]


def test_group_by_compressor_no_forbidden(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
    for row in group_acceptance_by_compressor(two_by_two_report):
        assert not (set(row.keys()) & _FORBIDDEN)


# ---------------------------------------------------------------------------
# group_acceptance_by_draft_len
# ---------------------------------------------------------------------------

def test_group_by_draft_len_row_count(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_draft_len
    rows = group_acceptance_by_draft_len(two_by_two_report)
    assert len(rows) == 2   # draft_len 4 and 8


def test_group_by_draft_len_has_key(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_draft_len
    rows = group_acceptance_by_draft_len(two_by_two_report)
    lens = {r["draft_len"] for r in rows}
    assert 4 in lens and 8 in lens


def test_group_by_draft_len_num_runs_collapsed(two_by_two_report):
    """Collapsing 2 compressors means num_runs == 2 per draft_len."""
    from exactkv.analysis.acceptance_tables import group_acceptance_by_draft_len
    for row in group_acceptance_by_draft_len(two_by_two_report):
        assert row["num_runs"] == 2


def test_group_by_draft_len_count_reconciles(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_draft_len
    for row in group_acceptance_by_draft_len(two_by_two_report):
        assert row["total_drafted"] == row["total_accepted"] + row["total_rejected"]


# ---------------------------------------------------------------------------
# group_acceptance_by_category
# ---------------------------------------------------------------------------

def test_group_by_category_row_count(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_category
    rows = group_acceptance_by_category(two_by_two_report)
    assert len(rows) == 2   # "nl" and "code"


def test_group_by_category_has_key(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_category
    rows = group_acceptance_by_category(two_by_two_report)
    cats = {r["category"] for r in rows}
    assert "nl" in cats and "code" in cats


def test_group_by_category_count_reconciles(two_by_two_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_category
    for row in group_acceptance_by_category(two_by_two_report):
        assert row["total_drafted"] == row["total_accepted"] + row["total_rejected"]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_write_acceptance_table_csv(two_by_two_report, tmp_path):
    from exactkv.analysis.acceptance_tables import (
        build_acceptance_table,
        write_acceptance_table_csv,
    )
    table = build_acceptance_table(two_by_two_report)
    path = tmp_path / "sub" / "table.csv"
    write_acceptance_table_csv(table, path)
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(table)


def test_write_acceptance_table_csv_creates_dirs(two_by_two_report, tmp_path):
    from exactkv.analysis.acceptance_tables import (
        build_acceptance_table,
        write_acceptance_table_csv,
    )
    nested = tmp_path / "a" / "b" / "c" / "table.csv"
    assert not nested.parent.exists()
    write_acceptance_table_csv(build_acceptance_table(two_by_two_report), nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# Integration: real sweep report
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

@pytest.fixture(scope="module")
def real_sweep_report():
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime

    rt = ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")
    return run_sweep(
        runtime=rt,
        prompts=[{"prompt_id": "atbl_001", "category": "nl", "prompt": "Paris is"}],
        compressor_names=["noop", "int8"],
        draft_lengths=[4],
        max_new_tokens=8,
        prompt_suite="test",
    )


def test_acceptance_table_on_real_sweep(real_sweep_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    table = build_acceptance_table(real_sweep_report)
    assert len(table) == 2   # noop and int8


def test_group_by_compressor_on_real_sweep(real_sweep_report):
    from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
    rows = group_acceptance_by_compressor(real_sweep_report)
    names = {r["compressor_name"] for r in rows}
    assert "noop" in names and "int8" in names


def test_real_sweep_no_exactkv_failures(real_sweep_report):
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    for row in build_acceptance_table(real_sweep_report):
        assert row["exactkv_failures"] == 0
