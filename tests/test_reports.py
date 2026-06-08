"""Tests for exactkv.benchmarks.reports (V2 reporting module).

Verifies:
  * JSON report writes and loads losslessly (round-trip).
  * CSV file writes successfully.
  * CSV has one row per prompt result.
  * CSV contains all required compressor metadata columns.
  * int4_sim rows include is_simulated=True.
  * int4_sim rows include supports_real_bytes_claim=False.
  * int4_sim rows include the memory_claim_note mentioning int8 storage.
  * No forbidden performance fields appear in JSON or CSV output.
  * build_run_manifest returns required provenance keys.
  * validate_report catches missing honesty fields.
  * Existing benchmark runner output is accepted by reports.py.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_TEST_PROMPT = {
    "prompt_id": "rpt_001",
    "category": "test",
    "prompt": "The capital of France is",
}


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def noop_suite_report(runtime):
    """run_suite result with NoOp compressor (1 prompt)."""
    from exactkv.benchmarks.runner import RunConfig, run_suite
    config = RunConfig(compressor_name="noop", draft_len=4, max_new_tokens=12)
    return run_suite(runtime, [_TEST_PROMPT], config)


@pytest.fixture(scope="module")
def int4_suite_report(runtime):
    """run_suite result with int4_sim compressor (1 prompt)."""
    from exactkv.benchmarks.runner import RunConfig, run_suite
    config = RunConfig(compressor_name="int4_sim", draft_len=4, max_new_tokens=12)
    return run_suite(runtime, [_TEST_PROMPT], config)


@pytest.fixture(scope="module")
def manifest():
    from exactkv.benchmarks.reports import build_run_manifest
    return build_run_manifest(
        model_name=MODEL_NAME,
        prompt_suite="smoke",
        compressor_names=["noop", "int4_sim"],
        draft_len=4,
        max_new_tokens=12,
    )


# ---------------------------------------------------------------------------
# build_run_manifest
# ---------------------------------------------------------------------------

def test_manifest_required_keys(manifest):
    required = {
        "model_name", "prompt_suite", "compressor_names", "draft_lengths",
        "max_new_tokens", "seed", "dtype", "device", "timestamp",
        "git_commit", "exactkv_version", "transformers_version", "torch_version",
    }
    for key in required:
        assert key in manifest, f"manifest missing key {key!r}"


def test_manifest_no_performance_fields(manifest):
    forbidden = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}
    for key in forbidden:
        assert key not in manifest, f"Forbidden field {key!r} found in manifest"


def test_manifest_model_name(manifest):
    assert manifest["model_name"] == MODEL_NAME


def test_manifest_draft_lengths_list(manifest):
    """A single draft_len=4 should be stored as [4] in draft_lengths."""
    assert manifest["draft_lengths"] == [4]


def test_manifest_transformers_version_present(manifest):
    assert manifest["transformers_version"] is not None


def test_manifest_torch_version_present(manifest):
    assert manifest["torch_version"] is not None


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_json_write_and_load_lossless(noop_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "report.json"
    write_json_report(noop_suite_report, path, manifest=manifest)

    loaded = load_json_report(path)
    # Round-trip: same prompt_id survives serialization
    assert loaded["results"][0]["prompt_id"] == _TEST_PROMPT["prompt_id"]


def test_json_contains_manifest(noop_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "report.json"
    write_json_report(noop_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    assert "manifest" in loaded, "JSON report must contain 'manifest'"


def test_json_contains_results_and_aggregate(noop_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "report.json"
    write_json_report(noop_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    assert "results" in loaded
    assert "aggregate" in loaded


def test_json_result_compressor_capabilities(noop_suite_report, manifest, tmp_path):
    """Each result in JSON must include compressor_capabilities."""
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "report.json"
    write_json_report(noop_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    result = loaded["results"][0]
    assert "compressor_capabilities" in result, (
        "JSON result must contain 'compressor_capabilities'"
    )


def test_json_memory_honesty_fields(noop_suite_report, manifest, tmp_path):
    """Enriched memory section must include honesty fields."""
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "report.json"
    write_json_report(noop_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    mem = loaded["results"][0]["memory"]
    for key in ("full_kv_bytes", "compressed_kv_bytes", "supports_real_bytes_claim",
                "is_simulated", "memory_claim_note"):
        assert key in mem, f"Enriched memory section missing key {key!r}"


def test_json_no_forbidden_fields_noop(noop_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "report.json"
    write_json_report(noop_suite_report, path, manifest=manifest)
    text = path.read_text()
    for field in ("tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"):
        assert f'"{field}"' not in text, (
            f"Forbidden field {field!r} found in JSON output"
        )


# ---------------------------------------------------------------------------
# int4_sim-specific JSON checks
# ---------------------------------------------------------------------------

def test_int4_sim_json_is_simulated_true(int4_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "int4_report.json"
    write_json_report(int4_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    caps = loaded["results"][0]["compressor_capabilities"]
    assert caps.get("is_simulated") is True


def test_int4_sim_json_supports_real_bytes_claim_false(int4_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "int4_report.json"
    write_json_report(int4_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    caps = loaded["results"][0]["compressor_capabilities"]
    assert caps.get("supports_real_bytes_claim") is False


def test_int4_sim_json_memory_claim_note_present(int4_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "int4_report.json"
    write_json_report(int4_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    note = loaded["results"][0]["memory"].get("memory_claim_note", "")
    assert note, "int4_sim memory_claim_note must be non-empty"


def test_int4_sim_json_memory_claim_note_mentions_int8(int4_suite_report, manifest, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report

    path = tmp_path / "int4_report.json"
    write_json_report(int4_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    note = loaded["results"][0]["memory"]["memory_claim_note"].lower()
    assert "int8" in note, (
        f"memory_claim_note should mention int8 storage for int4_sim, got: {note!r}"
    )


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def test_csv_writes_without_error(noop_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "report.csv"
    write_csv_report(noop_suite_report, path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_csv_one_row_per_prompt(noop_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "report.csv"
    write_csv_report(noop_suite_report, path)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    expected = len(noop_suite_report["results"])
    assert len(rows) == expected, (
        f"Expected {expected} CSV row(s), got {len(rows)}"
    )


def test_csv_required_columns_present(noop_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "report.csv"
    write_csv_report(noop_suite_report, path)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

    required = [
        "prompt_id", "category", "model_name", "compressor_name",
        "compressor_type", "is_simulated", "supports_real_bytes_claim",
        "draft_len", "max_new_tokens",
        "exactkv_token_exact_match", "lossy_token_exact_match",
        "acceptance_rate", "total_drafted", "total_accepted",
        "full_kv_bytes", "compressed_kv_bytes",
        "compression_ratio", "memory_reduction_factor", "memory_claim_note",
    ]
    for col in required:
        assert col in headers, f"CSV missing required column {col!r}"


def test_csv_no_forbidden_columns(noop_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "report.csv"
    write_csv_report(noop_suite_report, path)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
    forbidden = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}
    found = headers & forbidden
    assert not found, f"Forbidden performance columns in CSV: {found}"


def test_csv_int4_sim_is_simulated_true(int4_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "int4.csv"
    write_csv_report(int4_suite_report, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["is_simulated"] == "True"


def test_csv_int4_sim_supports_real_bytes_claim_false(int4_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "int4.csv"
    write_csv_report(int4_suite_report, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["supports_real_bytes_claim"] == "False"


def test_csv_int4_sim_memory_claim_note_present(int4_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "int4.csv"
    write_csv_report(int4_suite_report, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    note = rows[0]["memory_claim_note"]
    assert note, "CSV int4_sim memory_claim_note must not be empty"
    assert "int8" in note.lower(), (
        f"memory_claim_note should mention int8 storage, got: {note!r}"
    )


# ---------------------------------------------------------------------------
# flatten_report_to_rows
# ---------------------------------------------------------------------------

def test_flatten_rows_count(noop_suite_report):
    from exactkv.benchmarks.reports import flatten_report_to_rows
    rows = flatten_report_to_rows(noop_suite_report)
    assert len(rows) == len(noop_suite_report["results"])


def test_flatten_rows_have_required_keys(noop_suite_report):
    from exactkv.benchmarks.reports import flatten_report_to_rows
    rows = flatten_report_to_rows(noop_suite_report)
    for row in rows:
        assert "prompt_id" in row
        assert "compressor_name" in row
        assert "is_simulated" in row
        assert "memory_claim_note" in row


# ---------------------------------------------------------------------------
# validate_report
# ---------------------------------------------------------------------------

def test_validate_report_clean_on_valid(noop_suite_report, manifest, tmp_path):
    """A freshly-written and loaded report should have no validation warnings."""
    from exactkv.benchmarks.reports import (
        load_json_report,
        validate_report,
        write_json_report,
    )
    path = tmp_path / "valid.json"
    write_json_report(noop_suite_report, path, manifest=manifest)
    loaded = load_json_report(path)
    warnings = validate_report(loaded)
    assert not warnings, f"Unexpected validation warnings: {warnings}"


def test_validate_report_catches_forbidden_field():
    from exactkv.benchmarks.reports import validate_report

    bad_report = {
        "results": [{"prompt_id": "x", "tokens_per_second": 9999}],
    }
    warnings = validate_report(bad_report)
    assert any("tokens_per_second" in w for w in warnings), (
        "validate_report should flag 'tokens_per_second'"
    )


# ---------------------------------------------------------------------------
# Existing runner output compatibility
# ---------------------------------------------------------------------------

def test_runner_output_accepted_by_write_json(noop_suite_report, tmp_path):
    """run_suite output can be passed directly to write_json_report."""
    from exactkv.benchmarks.reports import write_json_report
    path = tmp_path / "compat.json"
    write_json_report(noop_suite_report, path)   # no manifest, should still work
    assert path.exists()


def test_runner_output_accepted_by_write_csv(noop_suite_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "compat.csv"
    write_csv_report(noop_suite_report, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# Auto-mkdir: writing to a non-existent nested directory
# ---------------------------------------------------------------------------

def test_write_json_creates_nested_dirs(noop_suite_report, tmp_path):
    """write_json_report must create missing parent directories automatically."""
    from exactkv.benchmarks.reports import write_json_report

    nested_path = tmp_path / "a" / "b" / "c" / "report.json"
    assert not nested_path.parent.exists(), "Pre-condition: directory must not exist yet"

    write_json_report(noop_suite_report, nested_path)

    assert nested_path.exists(), "write_json_report did not create the file"
    assert nested_path.parent.is_dir(), "write_json_report did not create parent dirs"


def test_write_csv_creates_nested_dirs(noop_suite_report, tmp_path):
    """write_csv_report must create missing parent directories automatically."""
    from exactkv.benchmarks.reports import write_csv_report

    nested_path = tmp_path / "x" / "y" / "report.csv"
    assert not nested_path.parent.exists(), "Pre-condition: directory must not exist yet"

    write_csv_report(noop_suite_report, nested_path)

    assert nested_path.exists(), "write_csv_report did not create the file"
    assert nested_path.parent.is_dir(), "write_csv_report did not create parent dirs"


def test_write_json_idempotent_on_existing_dir(noop_suite_report, tmp_path):
    """Calling write_json_report twice to the same directory must not raise."""
    from exactkv.benchmarks.reports import write_json_report

    path = tmp_path / "sub" / "report.json"
    write_json_report(noop_suite_report, path)   # creates dir
    write_json_report(noop_suite_report, path)   # dir already exists — should not raise
    assert path.exists()


def test_write_csv_idempotent_on_existing_dir(noop_suite_report, tmp_path):
    """Calling write_csv_report twice to the same directory must not raise."""
    from exactkv.benchmarks.reports import write_csv_report

    path = tmp_path / "sub" / "report.csv"
    write_csv_report(noop_suite_report, path)
    write_csv_report(noop_suite_report, path)
    assert path.exists()
