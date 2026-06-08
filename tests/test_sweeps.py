"""Tests for exactkv.benchmarks.sweeps (V2 Phase D).

Gate: benchmark sweep gate.

Verifies:
  * Sweep runs over >= 2 compressors and >= 2 draft lengths on a tiny prompt.
  * Total results == prompts × compressors × draft_lengths.
  * exactkv_failures == 0.
  * CSV writes one row per result.
  * Compressor metadata is preserved in every CSV row.
  * int4_sim rows include is_simulated=True and supports_real_bytes_claim=False.
  * Acceptance counts reconcile per result (drafted == accepted + rejected).
  * Aggregate fields are present and correct.
  * No forbidden performance fields in JSON or CSV.
  * Sweep report is compatible with write_json_report and write_csv_report.
  * run_sweep raises ValueError on invalid inputs.
"""
from __future__ import annotations

import csv
import json
import os

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_SWEEP_PROMPT = {
    "prompt_id": "sweep_001",
    "category": "test",
    "prompt": "The capital of France is",
}

_COMPRESSORS = ["noop", "int8"]
_DRAFT_LENGTHS = [4, 8]
_MAX_NEW = 8


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def sweep_report(runtime):
    from exactkv.benchmarks.sweeps import run_sweep
    return run_sweep(
        runtime=runtime,
        prompts=[_SWEEP_PROMPT],
        compressor_names=_COMPRESSORS,
        draft_lengths=_DRAFT_LENGTHS,
        max_new_tokens=_MAX_NEW,
        prompt_suite="test",
    )


@pytest.fixture(scope="module")
def int4_sweep_report(runtime):
    """Minimal sweep with int4_sim to verify metadata handling."""
    from exactkv.benchmarks.sweeps import run_sweep
    return run_sweep(
        runtime=runtime,
        prompts=[_SWEEP_PROMPT],
        compressor_names=["int4_sim"],
        draft_lengths=[4],
        max_new_tokens=_MAX_NEW,
        prompt_suite="test",
    )


# ---------------------------------------------------------------------------
# Result count
# ---------------------------------------------------------------------------

def test_sweep_total_results_count(sweep_report):
    """total results == prompts × compressors × draft_lengths."""
    expected = 1 * len(_COMPRESSORS) * len(_DRAFT_LENGTHS)
    assert len(sweep_report["results"]) == expected, (
        f"Expected {expected} results, got {len(sweep_report['results'])}"
    )


def test_sweep_covers_all_compressors(sweep_report):
    compressors_seen = {r["compressor_name"] for r in sweep_report["results"]}
    for name in _COMPRESSORS:
        assert name in compressors_seen, f"Compressor {name!r} missing from sweep results"


def test_sweep_covers_all_draft_lengths(sweep_report):
    draft_lens_seen = {r["draft_len"] for r in sweep_report["results"]}
    for dl in _DRAFT_LENGTHS:
        assert dl in draft_lens_seen, f"draft_len={dl} missing from sweep results"


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------

def test_sweep_exactkv_failures_zero(sweep_report):
    assert sweep_report["aggregate"]["exactkv_failures"] == 0, (
        f"Expected 0 ExactKV failures, got "
        f"{sweep_report['aggregate']['exactkv_failures']}"
    )


def test_sweep_each_result_exactkv_matches_full(sweep_report):
    for i, r in enumerate(sweep_report["results"]):
        assert r["exactkv"]["token_exact_match"] is True, (
            f"Result[{i}] (compressor={r['compressor_name']}, "
            f"draft_len={r['draft_len']}): ExactKV did not match full output"
        )


# ---------------------------------------------------------------------------
# Acceptance count reconciliation
# ---------------------------------------------------------------------------

def test_sweep_acceptance_reconciles(sweep_report):
    for i, r in enumerate(sweep_report["results"]):
        acc = r["exactkv"]["acceptance"]
        drafted = acc["total_drafted"]
        accepted = acc["total_accepted"]
        rejected = acc["total_rejected"]
        assert drafted == accepted + rejected, (
            f"Result[{i}]: drafted={drafted} != accepted({accepted}) + rejected({rejected})"
        )


# ---------------------------------------------------------------------------
# Aggregate fields
# ---------------------------------------------------------------------------

def test_sweep_aggregate_keys_present(sweep_report):
    agg = sweep_report["aggregate"]
    required = {
        "total_runs", "total_prompts", "compressor_names", "draft_lengths",
        "exactkv_failures", "lossy_divergence_count", "mean_acceptance_rate",
        "mean_average_accepted_length", "total_drafted", "total_accepted",
        "total_rejected", "total_corrections",
    }
    for key in required:
        assert key in agg, f"Aggregate missing key {key!r}"


def test_sweep_aggregate_total_runs(sweep_report):
    agg = sweep_report["aggregate"]
    assert agg["total_runs"] == len(sweep_report["results"])


def test_sweep_aggregate_total_prompts(sweep_report):
    agg = sweep_report["aggregate"]
    assert agg["total_prompts"] == 1


def test_sweep_aggregate_compressor_names(sweep_report):
    agg = sweep_report["aggregate"]
    assert set(agg["compressor_names"]) == set(_COMPRESSORS)


def test_sweep_aggregate_draft_lengths(sweep_report):
    agg = sweep_report["aggregate"]
    assert set(agg["draft_lengths"]) == set(_DRAFT_LENGTHS)


def test_sweep_aggregate_acceptance_rate_valid(sweep_report):
    rate = sweep_report["aggregate"]["mean_acceptance_rate"]
    assert 0.0 <= rate <= 1.0, f"mean_acceptance_rate out of range: {rate}"


def test_sweep_aggregate_drafted_equals_accepted_plus_rejected(sweep_report):
    agg = sweep_report["aggregate"]
    assert agg["total_drafted"] == agg["total_accepted"] + agg["total_rejected"]


def test_sweep_aggregate_no_forbidden_fields(sweep_report):
    forbidden = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}
    agg = sweep_report["aggregate"]
    found = set(agg.keys()) & forbidden
    assert not found, f"Forbidden performance fields in aggregate: {found}"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_sweep_manifest_present(sweep_report):
    assert "manifest" in sweep_report


def test_sweep_manifest_compressor_names(sweep_report):
    manifest = sweep_report["manifest"]
    assert "compressor_names" in manifest
    assert set(manifest["compressor_names"]) == set(_COMPRESSORS)


def test_sweep_manifest_draft_lengths(sweep_report):
    manifest = sweep_report["manifest"]
    assert "draft_lengths" in manifest
    assert set(manifest["draft_lengths"]) == set(_DRAFT_LENGTHS)


def test_sweep_manifest_no_timing_fields(sweep_report):
    manifest = sweep_report["manifest"]
    forbidden = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}
    assert not (set(manifest.keys()) & forbidden)


# ---------------------------------------------------------------------------
# Compressor capabilities in results
# ---------------------------------------------------------------------------

def test_sweep_results_have_capabilities(sweep_report):
    for i, r in enumerate(sweep_report["results"]):
        assert "compressor_capabilities" in r, (
            f"Result[{i}] missing 'compressor_capabilities'"
        )
        caps = r["compressor_capabilities"]
        assert "is_simulated" in caps
        assert "supports_real_bytes_claim" in caps


def test_sweep_int4_sim_is_simulated_true(int4_sweep_report):
    result = int4_sweep_report["results"][0]
    caps = result["compressor_capabilities"]
    assert caps.get("is_simulated") is True


def test_sweep_int4_sim_supports_real_bytes_claim_false(int4_sweep_report):
    result = int4_sweep_report["results"][0]
    caps = result["compressor_capabilities"]
    assert caps.get("supports_real_bytes_claim") is False


# ---------------------------------------------------------------------------
# CSV compatibility
# ---------------------------------------------------------------------------

def test_sweep_csv_one_row_per_result(sweep_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "sweep.csv"
    write_csv_report(sweep_report, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(sweep_report["results"]), (
        f"Expected {len(sweep_report['results'])} CSV rows, got {len(rows)}"
    )


def test_sweep_csv_compressor_metadata_columns(sweep_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "sweep_meta.csv"
    write_csv_report(sweep_report, path)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    for col in ("compressor_name", "compressor_type", "is_simulated", "supports_real_bytes_claim"):
        assert col in headers, f"CSV missing column {col!r}"


def test_sweep_csv_int4_sim_metadata(int4_sweep_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "sweep_int4.csv"
    write_csv_report(int4_sweep_report, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["is_simulated"] == "True"
    assert rows[0]["supports_real_bytes_claim"] == "False"
    assert rows[0]["memory_claim_note"]   # non-empty


def test_sweep_csv_no_forbidden_columns(sweep_report, tmp_path):
    from exactkv.benchmarks.reports import write_csv_report
    path = tmp_path / "sweep_nf.csv"
    write_csv_report(sweep_report, path)
    with path.open(encoding="utf-8") as f:
        headers = set(csv.DictReader(f).fieldnames or [])
    forbidden = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}
    assert not (headers & forbidden), f"Forbidden columns in CSV: {headers & forbidden}"


# ---------------------------------------------------------------------------
# JSON compatibility
# ---------------------------------------------------------------------------

def test_sweep_json_write_and_load(sweep_report, tmp_path):
    from exactkv.benchmarks.reports import load_json_report, write_json_report
    path = tmp_path / "sweep.json"
    write_json_report(sweep_report, path)
    loaded = load_json_report(path)
    assert len(loaded["results"]) == len(sweep_report["results"])
    assert loaded["aggregate"]["exactkv_failures"] == 0


def test_sweep_json_no_forbidden_fields(sweep_report, tmp_path):
    from exactkv.benchmarks.reports import write_json_report
    path = tmp_path / "sweep_nf.json"
    write_json_report(sweep_report, path)
    text = path.read_text()
    for field in ("tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"):
        assert f'"{field}"' not in text, f"Forbidden field {field!r} found in JSON"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_run_sweep_raises_on_empty_compressors(runtime):
    from exactkv.benchmarks.sweeps import run_sweep
    with pytest.raises(ValueError, match="compressor_names"):
        run_sweep(runtime, [_SWEEP_PROMPT], compressor_names=[], draft_lengths=[4])


def test_run_sweep_raises_on_empty_draft_lengths(runtime):
    from exactkv.benchmarks.sweeps import run_sweep
    with pytest.raises(ValueError, match="draft_lengths"):
        run_sweep(runtime, [_SWEEP_PROMPT], compressor_names=["noop"], draft_lengths=[])


def test_run_sweep_raises_on_zero_draft_len(runtime):
    from exactkv.benchmarks.sweeps import run_sweep
    with pytest.raises(ValueError, match="draft_lengths"):
        run_sweep(runtime, [_SWEEP_PROMPT], compressor_names=["noop"], draft_lengths=[0])


def test_run_sweep_raises_on_zero_max_new_tokens(runtime):
    from exactkv.benchmarks.sweeps import run_sweep
    with pytest.raises(ValueError, match="max_new_tokens"):
        run_sweep(runtime, [_SWEEP_PROMPT], compressor_names=["noop"],
                  draft_lengths=[4], max_new_tokens=0)
