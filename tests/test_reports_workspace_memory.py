"""Tests: V5 workspace-aware memory fields in JSON/CSV reports (V5 Phase B).

Verifies:
  * run_one result includes all V5 workspace memory fields.
  * JSON report includes all workspace memory fields in memory sub-dict.
  * CSV report includes all five new workspace columns.
  * flatten_report_to_rows includes the new fields.
  * Old synthetic reports missing workspace fields still flatten and validate.
  * int8 CSV row: stored_kv_bytes < full_kv_bytes, materialized == full.
  * int4_sim CSV row: supports_real_bytes_claim=False, simulation warning.
  * Asymmetric simulated rows: supports_real_bytes_claim=False, workspace fields.
  * k8_v_full / k_full_v8 rows: supports_real_bytes_claim=True, workspace fields.
  * total_kv_footprint_bytes reconciliation: stored + materialized + metadata + temp.
  * total_kv_footprint_bytes is NOT described as measured peak GPU memory.
  * No forbidden performance fields appear in JSON or CSV output.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_FORBIDDEN_PERF_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})

_V5_WORKSPACE_FIELDS = (
    "stored_kv_bytes",
    "materialized_working_kv_bytes",
    "metadata_bytes",
    "temporary_workspace_bytes",
    "total_kv_footprint_bytes",
)

_V4_MEMORY_FIELDS = (
    "full_kv_bytes",
    "compressed_kv_bytes",
    "compression_ratio",
    "memory_reduction_factor",
    "supports_real_bytes_claim",
    "is_simulated",
    "memory_claim_note",
)

_TEST_PROMPT = {
    "prompt_id": "wm_001",
    "category": "test",
    "prompt": "The speed of light is",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_forbidden_in(obj: Any) -> None:
    """Recursively assert no forbidden performance field names are present."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            assert key not in _FORBIDDEN_PERF_FIELDS, (
                f"Forbidden performance field {key!r} found in report"
            )
            _no_forbidden_in(val)
    elif isinstance(obj, list):
        for item in obj:
            _no_forbidden_in(item)


def _csv_rows_from_report(report: dict) -> list[dict]:
    """Write report to CSV and read back as list of dicts."""
    from exactkv.benchmarks.reports import write_csv_report
    buf = io.StringIO()
    rows_raw = []
    import csv as _csv
    from exactkv.benchmarks.reports import flatten_report_to_rows, _CSV_COLUMNS
    rows_raw = flatten_report_to_rows(report)
    # Write to string buffer
    writer = _csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore",
                             lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows_raw)
    buf.seek(0)
    return list(_csv.DictReader(buf))


def _reconciles(mem: dict) -> bool:
    """Check total_kv_footprint_bytes == sum of component fields."""
    expected = (
        mem.get("stored_kv_bytes", 0)
        + mem.get("materialized_working_kv_bytes", 0)
        + mem.get("metadata_bytes", 0)
        + mem.get("temporary_workspace_bytes", 0)
    )
    return mem.get("total_kv_footprint_bytes", -1) == expected


# ---------------------------------------------------------------------------
# Shared runtime fixture (module-scoped to avoid repeated model loads)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


# ---------------------------------------------------------------------------
# Per-compressor report fixtures
# ---------------------------------------------------------------------------

def _make_report(runtime, compressor_name: str) -> dict:
    from exactkv.benchmarks.runner import RunConfig, run_suite
    config = RunConfig(
        compressor_name=compressor_name,
        draft_len=3,
        max_new_tokens=8,
    )
    return run_suite(runtime, [_TEST_PROMPT], config)


@pytest.fixture(scope="module")
def int8_report(runtime):
    return _make_report(runtime, "int8")


@pytest.fixture(scope="module")
def int4_report(runtime):
    return _make_report(runtime, "int4_sim")


@pytest.fixture(scope="module")
def k8v_full_report(runtime):
    return _make_report(runtime, "k8_v_full")


@pytest.fixture(scope="module")
def k_full_v8_report(runtime):
    return _make_report(runtime, "k_full_v8")


@pytest.fixture(scope="module")
def k8v4_sim_report(runtime):
    return _make_report(runtime, "k8_v4_sim")


# ---------------------------------------------------------------------------
# Synthetic (no-model) fixtures for backward-compat tests
# ---------------------------------------------------------------------------

def _make_legacy_result(
    compressor_name: str = "int8",
    full_bytes: int = 4096,
    compressed_bytes: int = 1024,
    supports_real: bool = True,
    is_simulated: bool = False,
) -> dict:
    """Build a minimal V1–V4 result dict (no V5 workspace fields)."""
    return {
        "prompt_id": "legacy_001",
        "category": "test",
        "model_name": MODEL_NAME,
        "compressor_name": compressor_name,
        "compressor_capabilities": {
            "compressor_type": "quantize",
            "is_simulated": is_simulated,
            "supports_real_bytes_claim": supports_real,
            "key_bit_width": 8,
            "value_bit_width": 8,
            "asymmetric": False,
        },
        "draft_len": 4,
        "max_new_tokens": 16,
        "full": {"output_ids": [1, 2, 3], "output_text": "abc"},
        "lossy": {
            "output_ids": [1, 2, 3],
            "output_text": "abc",
            "token_exact_match": True,
            "first_divergence_idx": None,
        },
        "exactkv": {
            "output_ids": [1, 2, 3],
            "output_text": "abc",
            "token_exact_match": True,
            "acceptance": {
                "acceptance_rate": 1.0,
                "avg_accepted_per_round": 3.0,
                "total_drafted": 3,
                "total_accepted": 3,
                "total_rejected": 0,
                "total_corrections": 0,
            },
        },
        # V1–V4 style memory dict: no V5 workspace fields
        "memory": {
            "full_bytes": full_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": compressed_bytes / max(full_bytes, 1),
            "memory_reduction_factor": full_bytes / max(compressed_bytes, 1),
        },
        "exactkv_failure": False,
    }


def _make_legacy_report(
    compressor_name: str = "int8",
    full_bytes: int = 4096,
    compressed_bytes: int = 1024,
) -> dict:
    return {
        "results": [
            _make_legacy_result(compressor_name, full_bytes, compressed_bytes)
        ],
        "aggregate": {
            "total_prompts": 1,
            "compressor_name": compressor_name,
            "exactkv_failures": 0,
            "exactkv_pass_rate": 1.0,
        },
    }


def _make_v5_result(
    compressor_name: str = "int8",
    full_bytes: int = 4096,
    stored_kv: int = 1024,
    materialized: int = 4096,
    metadata: int = 32,
    temporary: int = 1024,
    supports_real: bool = True,
    is_simulated: bool = False,
) -> dict:
    """Build a V5-style result dict with all workspace fields."""
    total_footprint = stored_kv + materialized + metadata + temporary
    compressed_bytes = stored_kv + metadata
    result = _make_legacy_result(
        compressor_name=compressor_name,
        full_bytes=full_bytes,
        compressed_bytes=compressed_bytes,
        supports_real=supports_real,
        is_simulated=is_simulated,
    )
    # Overwrite memory with full V5 fields
    result["memory"] = {
        "full_bytes": full_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compressed_bytes / max(full_bytes, 1),
        "memory_reduction_factor": full_bytes / max(compressed_bytes, 1),
        "stored_kv_bytes": stored_kv,
        "materialized_working_kv_bytes": materialized,
        "metadata_bytes": metadata,
        "temporary_workspace_bytes": temporary,
        "total_kv_footprint_bytes": total_footprint,
        "supports_real_bytes_claim": supports_real,
        "is_simulated": is_simulated,
        "memory_claim_note": (
            f"Simulated compressor ({compressor_name!r}): int8 containers."
            if is_simulated else
            f"Real-storage compressor ({compressor_name!r})."
        ),
    }
    return result


# ===========================================================================
# SECTION A: Unit tests (no model required)
# ===========================================================================

class TestEnrichResultWorkspaceFields:
    """_enrich_result propagates V5 workspace fields from memory dict."""

    def test_workspace_fields_pass_through(self):
        from exactkv.benchmarks.reports import _enrich_result
        result = _make_v5_result(
            stored_kv=1024, materialized=4096, metadata=32, temporary=1024
        )
        enriched = _enrich_result(result)
        mem = enriched["memory"]
        assert mem["stored_kv_bytes"] == 1024
        assert mem["materialized_working_kv_bytes"] == 4096
        assert mem["metadata_bytes"] == 32
        assert mem["temporary_workspace_bytes"] == 1024
        assert mem["total_kv_footprint_bytes"] == 1024 + 4096 + 32 + 1024

    def test_workspace_fields_default_zero_for_legacy(self):
        """Legacy results without workspace fields get 0 defaults."""
        from exactkv.benchmarks.reports import _enrich_result
        result = _make_legacy_result()
        enriched = _enrich_result(result)
        mem = enriched["memory"]
        for f in _V5_WORKSPACE_FIELDS:
            assert mem[f] == 0, f"{f} should default to 0 for legacy results"

    def test_v4_honesty_fields_are_present(self):
        from exactkv.benchmarks.reports import _enrich_result
        result = _make_legacy_result()
        enriched = _enrich_result(result)
        mem = enriched["memory"]
        for f in _V4_MEMORY_FIELDS:
            assert f in mem, f"V4 honesty field {f!r} missing"

    def test_full_kv_bytes_alias(self):
        from exactkv.benchmarks.reports import _enrich_result
        result = _make_legacy_result(full_bytes=8192, compressed_bytes=2048)
        enriched = _enrich_result(result)
        assert enriched["memory"]["full_kv_bytes"] == 8192
        assert enriched["memory"]["compressed_kv_bytes"] == 2048

    def test_memory_claim_note_not_overridden_when_present(self):
        """An existing note from MemorySummary is not clobbered by _enrich_result."""
        from exactkv.benchmarks.reports import _enrich_result
        result = _make_v5_result(is_simulated=True, compressor_name="int4_sim")
        custom_note = "Existing detailed note from MemorySummary."
        result["memory"]["memory_claim_note"] = custom_note
        enriched = _enrich_result(result)
        assert enriched["memory"]["memory_claim_note"] == custom_note

    def test_memory_claim_note_generated_for_legacy(self):
        """Legacy results without a note get a generated note."""
        from exactkv.benchmarks.reports import _enrich_result
        result = _make_legacy_result(compressor_name="int4_sim", is_simulated=True)
        result["memory"]["memory_claim_note"] = ""
        enriched = _enrich_result(result)
        note = enriched["memory"]["memory_claim_note"]
        assert note != "", "Generated note must not be empty for simulated compressor"

    def test_no_forbidden_fields_in_enriched(self):
        from exactkv.benchmarks.reports import _enrich_result
        result = _make_v5_result()
        enriched = _enrich_result(result)
        _no_forbidden_in(enriched)


class TestCsvColumnsWorkspace:
    """CSV schema includes all five V5 workspace columns."""

    def test_csv_columns_include_v5_fields(self):
        from exactkv.benchmarks.reports import _CSV_COLUMNS
        for col in _V5_WORKSPACE_FIELDS:
            assert col in _CSV_COLUMNS, f"V5 column {col!r} missing from _CSV_COLUMNS"

    def test_csv_columns_retain_v4_fields(self):
        from exactkv.benchmarks.reports import _CSV_COLUMNS
        for col in (
            "full_kv_bytes", "compressed_kv_bytes", "compression_ratio",
            "memory_reduction_factor", "memory_claim_note",
        ):
            assert col in _CSV_COLUMNS, f"V4 column {col!r} missing from _CSV_COLUMNS"

    def test_csv_columns_no_forbidden_fields(self):
        from exactkv.benchmarks.reports import _CSV_COLUMNS
        for col in _CSV_COLUMNS:
            assert col not in _FORBIDDEN_PERF_FIELDS, (
                f"Forbidden performance field {col!r} in _CSV_COLUMNS"
            )

    def test_csv_columns_no_active_gpu_bytes(self):
        """active_gpu_kv_bytes is explicitly deferred; must not appear."""
        from exactkv.benchmarks.reports import _CSV_COLUMNS
        assert "active_gpu_kv_bytes" not in _CSV_COLUMNS

    def test_flatten_report_v5_fields_present(self):
        """flatten_report_to_rows includes V5 workspace fields for V5 results."""
        from exactkv.benchmarks.reports import flatten_report_to_rows
        report = {"results": [_make_v5_result(stored_kv=512, materialized=4096,
                                               metadata=16, temporary=512)]}
        rows = flatten_report_to_rows(report)
        assert len(rows) == 1
        row = rows[0]
        for col in _V5_WORKSPACE_FIELDS:
            assert col in row, f"V5 column {col!r} missing from flattened row"

    def test_flatten_legacy_report_workspace_fields_default_zero(self):
        """Legacy V1–V4 reports without workspace fields flatten to 0."""
        from exactkv.benchmarks.reports import flatten_report_to_rows
        report = _make_legacy_report()
        rows = flatten_report_to_rows(report)
        row = rows[0]
        for col in _V5_WORKSPACE_FIELDS:
            assert col in row, f"Column {col!r} missing from legacy flattened row"
            assert row[col] == 0, (
                f"Column {col!r} should be 0 for legacy result, got {row[col]!r}"
            )


class TestValidateReportWorkspace:
    """validate_report handles V5 fields and legacy reports correctly."""

    def test_validate_v5_report_no_warnings(self):
        from exactkv.benchmarks.reports import validate_report, _enrich_result
        result = _enrich_result(_make_v5_result())
        report = {"results": [result]}
        warnings = validate_report(report)
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_validate_legacy_report_no_warnings(self):
        """V1–V4 reports without V5 fields should pass validation."""
        from exactkv.benchmarks.reports import validate_report, _enrich_result
        result = _enrich_result(_make_legacy_result())
        report = {"results": [result]}
        warnings = validate_report(report)
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_validate_warns_on_negative_workspace_field(self):
        from exactkv.benchmarks.reports import validate_report, _enrich_result
        result = _enrich_result(_make_v5_result())
        result["memory"]["stored_kv_bytes"] = -1  # invalid
        report = {"results": [result]}
        warnings = validate_report(report)
        assert any("stored_kv_bytes" in w for w in warnings), (
            "Expected warning about negative stored_kv_bytes"
        )

    def test_validate_warns_on_total_mismatch(self):
        """Validation warns when total_kv_footprint_bytes doesn't reconcile."""
        from exactkv.benchmarks.reports import validate_report, _enrich_result
        result = _enrich_result(_make_v5_result())
        result["memory"]["total_kv_footprint_bytes"] = 99999  # wrong
        report = {"results": [result]}
        warnings = validate_report(report)
        assert any("total_kv_footprint_bytes" in w for w in warnings), (
            "Expected warning about total_kv_footprint_bytes mismatch"
        )

    def test_validate_no_forbidden_fields(self):
        from exactkv.benchmarks.reports import validate_report, _enrich_result
        result = _enrich_result(_make_v5_result())
        report = {"results": [result]}
        warnings = validate_report(report)
        # validate_report would flag forbidden fields in its warnings
        forbidden_warnings = [w for w in warnings if any(
            f in w for f in _FORBIDDEN_PERF_FIELDS
        )]
        assert forbidden_warnings == []

    def test_validate_all_zero_workspace_fields_no_reconcile_warning(self):
        """Legacy results enriched to all-zero V5 fields should pass reconciliation.

        _enrich_result fills missing workspace fields with 0, so 0+0+0+0 == 0
        and the reconciliation check passes without warnings.
        """
        from exactkv.benchmarks.reports import validate_report, _enrich_result
        result = _enrich_result(_make_legacy_result())
        # All workspace fields are 0 after enrichment; 0+0+0+0 == 0 reconciles.
        report = {"results": [result]}
        warnings = validate_report(report)
        recon_warnings = [w for w in warnings if "total_kv_footprint_bytes" in w]
        assert recon_warnings == [], (
            f"All-zero workspace fields should reconcile trivially; got: {recon_warnings}"
        )


class TestTotalFootprintNotMeasuredPeak:
    """memory_claim_note must not claim total_kv_footprint_bytes is measured."""

    def test_generated_note_explicitly_disclaims_measured_peak(self):
        """Note must say total is NOT a measured peak value (not silently imply it)."""
        from exactkv.benchmarks.reports import _memory_claim_note
        for compressor_name, supports_real in [
            ("int8", True), ("int4_sim", False), ("k8_v_full", True),
        ]:
            caps = {"supports_real_bytes_claim": supports_real,
                    "is_simulated": not supports_real}
            note = _memory_claim_note(compressor_name, caps)
            # The note must explicitly negate "measured peak", not silently
            # present bytes as if they were profiled GPU measurements.
            note_lower = note.lower()
            has_disclaimer = (
                "not a measured" in note_lower
                or "is not measured" in note_lower
                or "accounting" in note_lower
            )
            assert has_disclaimer, (
                f"Note for {compressor_name!r} should explicitly disclaim "
                f"measured-peak interpretation: {note}"
            )

    def test_generated_note_says_accounting_total(self):
        """Note must clearly explain that total is an accounting sum."""
        from exactkv.benchmarks.reports import _memory_claim_note
        note = _memory_claim_note("int8", {"supports_real_bytes_claim": True})
        assert "accounting" in note.lower() or "sum" in note.lower(), (
            f"Note should describe total as accounting sum: {note}"
        )

    def test_simulated_note_mentions_int8_container(self):
        from exactkv.benchmarks.reports import _memory_claim_note
        note = _memory_claim_note("int4_sim",
                                  {"supports_real_bytes_claim": False,
                                   "is_simulated": True})
        assert "int8" in note.lower(), (
            f"Simulated note should mention int8 containers: {note}"
        )

    def test_note_says_deferred(self):
        """Note must say active GPU measurement is deferred."""
        from exactkv.benchmarks.reports import _memory_claim_note
        note = _memory_claim_note("int8", {"supports_real_bytes_claim": True})
        assert "deferred" in note.lower() or "later" in note.lower(), (
            f"Note should say active GPU measurement is deferred: {note}"
        )


# ===========================================================================
# SECTION B: Model-backed tests (require Qwen/Qwen2.5-0.5B)
# ===========================================================================

class TestRunOneWorkspaceFields:
    """run_one result includes all V5 workspace memory fields."""

    def test_run_one_int8_has_workspace_fields(self, runtime):
        from exactkv.benchmarks.runner import RunConfig, run_one
        config = RunConfig(compressor_name="int8", draft_len=3, max_new_tokens=8)
        result = run_one(runtime, _TEST_PROMPT, config)
        mem = result["memory"]
        for f in _V5_WORKSPACE_FIELDS:
            assert f in mem, f"run_one result missing workspace field {f!r}"

    def test_run_one_int4sim_has_workspace_fields(self, runtime):
        from exactkv.benchmarks.runner import RunConfig, run_one
        config = RunConfig(compressor_name="int4_sim", draft_len=3, max_new_tokens=8)
        result = run_one(runtime, _TEST_PROMPT, config)
        mem = result["memory"]
        for f in _V5_WORKSPACE_FIELDS:
            assert f in mem, f"run_one result missing workspace field {f!r}"

    def test_run_one_workspace_fields_positive(self, runtime):
        from exactkv.benchmarks.runner import RunConfig, run_one
        config = RunConfig(compressor_name="int8", draft_len=3, max_new_tokens=8)
        result = run_one(runtime, _TEST_PROMPT, config)
        mem = result["memory"]
        assert mem["stored_kv_bytes"] > 0
        assert mem["materialized_working_kv_bytes"] > 0
        assert mem["total_kv_footprint_bytes"] > 0

    def test_run_one_no_forbidden_fields(self, runtime):
        from exactkv.benchmarks.runner import RunConfig, run_one
        config = RunConfig(compressor_name="int8", draft_len=3, max_new_tokens=8)
        result = run_one(runtime, _TEST_PROMPT, config)
        _no_forbidden_in(result)


class TestJsonReportWorkspaceFields:
    """JSON report includes all workspace memory fields."""

    def test_json_report_int8_workspace_fields(self, int8_report):
        from exactkv.benchmarks.reports import write_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            write_json_report(int8_report, path)
            loaded = json.loads(path.read_text())
            mem = loaded["results"][0]["memory"]
            for f in _V5_WORKSPACE_FIELDS:
                assert f in mem, f"JSON memory missing V5 field {f!r}"
        finally:
            path.unlink(missing_ok=True)

    def test_json_report_int4sim_workspace_fields(self, int4_report):
        from exactkv.benchmarks.reports import write_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            write_json_report(int4_report, path)
            loaded = json.loads(path.read_text())
            mem = loaded["results"][0]["memory"]
            for f in _V5_WORKSPACE_FIELDS:
                assert f in mem, f"JSON memory missing V5 field {f!r}"
        finally:
            path.unlink(missing_ok=True)

    def test_json_report_retains_v4_memory_fields(self, int8_report):
        from exactkv.benchmarks.reports import write_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            write_json_report(int8_report, path)
            loaded = json.loads(path.read_text())
            mem = loaded["results"][0]["memory"]
            for f in _V4_MEMORY_FIELDS:
                assert f in mem, f"JSON memory missing V4 field {f!r}"
        finally:
            path.unlink(missing_ok=True)

    def test_json_round_trip_workspace_fields(self, int8_report):
        from exactkv.benchmarks.reports import write_json_report, load_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            write_json_report(int8_report, path)
            loaded = load_json_report(path)
            mem = loaded["results"][0]["memory"]
            for f in _V5_WORKSPACE_FIELDS:
                assert f in mem
                assert isinstance(mem[f], (int, float))
        finally:
            path.unlink(missing_ok=True)

    def test_json_no_forbidden_fields(self, int8_report):
        from exactkv.benchmarks.reports import write_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            write_json_report(int8_report, path)
            loaded = json.loads(path.read_text())
            _no_forbidden_in(loaded)
        finally:
            path.unlink(missing_ok=True)


class TestCsvReportWorkspaceFields:
    """CSV report includes all five new workspace columns."""

    def test_csv_int8_has_workspace_columns(self, int8_report):
        rows = _csv_rows_from_report(int8_report)
        assert len(rows) == 1
        row = rows[0]
        for col in _V5_WORKSPACE_FIELDS:
            assert col in row, f"CSV row missing V5 column {col!r}"

    def test_csv_int4sim_has_workspace_columns(self, int4_report):
        rows = _csv_rows_from_report(int4_report)
        row = rows[0]
        for col in _V5_WORKSPACE_FIELDS:
            assert col in row, f"CSV row missing V5 column {col!r}"

    def test_csv_k8v_full_has_workspace_columns(self, k8v_full_report):
        rows = _csv_rows_from_report(k8v_full_report)
        row = rows[0]
        for col in _V5_WORKSPACE_FIELDS:
            assert col in row, f"CSV row missing V5 column {col!r}"

    def test_csv_k_full_v8_has_workspace_columns(self, k_full_v8_report):
        rows = _csv_rows_from_report(k_full_v8_report)
        row = rows[0]
        for col in _V5_WORKSPACE_FIELDS:
            assert col in row, f"CSV row missing V5 column {col!r}"

    def test_csv_no_forbidden_columns(self, int8_report):
        rows = _csv_rows_from_report(int8_report)
        if rows:
            for key in rows[0]:
                assert key not in _FORBIDDEN_PERF_FIELDS, (
                    f"Forbidden column {key!r} in CSV row"
                )

    def test_csv_no_active_gpu_bytes_column(self, int8_report):
        rows = _csv_rows_from_report(int8_report)
        if rows:
            assert "active_gpu_kv_bytes" not in rows[0]


class TestInt8MemoryValues:
    """int8 compressor workspace values have expected relationships."""

    def test_int8_stored_less_than_full(self, int8_report):
        """int8 stored bytes < full bytes (quantised tensors are smaller)."""
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(int8_report)
        row = rows[0]
        full = int(row["full_kv_bytes"])
        stored = int(row["stored_kv_bytes"])
        assert stored < full, (
            f"int8 stored_kv_bytes ({stored}) should be < full_kv_bytes ({full})"
        )

    def test_int8_materialized_equals_full(self, int8_report):
        """int8 materialized_working_kv_bytes == full_kv_bytes (dequant for attention)."""
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(int8_report)
        row = rows[0]
        full = int(row["full_kv_bytes"])
        materialized = int(row["materialized_working_kv_bytes"])
        assert materialized == full, (
            f"int8 materialized ({materialized}) should == full ({full})"
        )

    def test_int8_supports_real_bytes_claim(self, int8_report):
        rows = _csv_rows_from_report(int8_report)
        assert rows[0]["supports_real_bytes_claim"].lower() in ("true", "1", "yes")

    def test_int8_total_reconciles(self, int8_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(int8_report)
        row = rows[0]
        stored = int(row["stored_kv_bytes"])
        materialized = int(row["materialized_working_kv_bytes"])
        metadata = int(row["metadata_bytes"])
        temporary = int(row["temporary_workspace_bytes"])
        total = int(row["total_kv_footprint_bytes"])
        assert total == stored + materialized + metadata + temporary, (
            f"int8 total_kv_footprint_bytes ({total}) != "
            f"stored+materialized+metadata+temporary "
            f"({stored}+{materialized}+{metadata}+{temporary})"
        )


class TestInt4SimMemoryValues:
    """int4_sim workspace values reflect int8-container reality."""

    def test_int4sim_not_real_bytes_claim(self, int4_report):
        rows = _csv_rows_from_report(int4_report)
        assert rows[0]["supports_real_bytes_claim"].lower() in ("false", "0", "no")

    def test_int4sim_simulation_warning_in_note(self, int4_report):
        rows = _csv_rows_from_report(int4_report)
        note = rows[0]["memory_claim_note"].lower()
        assert "simulated" in note or "int8" in note or "simulation" in note, (
            f"int4_sim note should mention simulated/int8: {note}"
        )

    def test_int4sim_total_reconciles(self, int4_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(int4_report)
        row = rows[0]
        stored = int(row["stored_kv_bytes"])
        materialized = int(row["materialized_working_kv_bytes"])
        metadata = int(row["metadata_bytes"])
        temporary = int(row["temporary_workspace_bytes"])
        total = int(row["total_kv_footprint_bytes"])
        assert total == stored + materialized + metadata + temporary

    def test_int4sim_workspace_fields_positive(self, int4_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(int4_report)
        row = rows[0]
        assert int(row["total_kv_footprint_bytes"]) > 0


class TestAsymmetricSimMemoryValues:
    """Asymmetric simulated compressor workspace fields."""

    def test_k8v4sim_not_real_bytes_claim(self, k8v4_sim_report):
        rows = _csv_rows_from_report(k8v4_sim_report)
        assert rows[0]["supports_real_bytes_claim"].lower() in ("false", "0", "no")

    def test_k8v4sim_has_workspace_fields(self, k8v4_sim_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(k8v4_sim_report)
        row = rows[0]
        for col in _V5_WORKSPACE_FIELDS:
            assert col in row

    def test_k8v4sim_total_reconciles(self, k8v4_sim_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(k8v4_sim_report)
        row = rows[0]
        stored = int(row["stored_kv_bytes"])
        materialized = int(row["materialized_working_kv_bytes"])
        metadata = int(row["metadata_bytes"])
        temporary = int(row["temporary_workspace_bytes"])
        total = int(row["total_kv_footprint_bytes"])
        assert total == stored + materialized + metadata + temporary


class TestRealAsymmetricMemoryValues:
    """Real-bytes asymmetric compressors (k8_v_full, k_full_v8) workspace fields."""

    def test_k8v_full_real_bytes_claim(self, k8v_full_report):
        rows = _csv_rows_from_report(k8v_full_report)
        assert rows[0]["supports_real_bytes_claim"].lower() in ("true", "1", "yes")

    def test_k_full_v8_real_bytes_claim(self, k_full_v8_report):
        rows = _csv_rows_from_report(k_full_v8_report)
        assert rows[0]["supports_real_bytes_claim"].lower() in ("true", "1", "yes")

    def test_k8v_full_total_reconciles(self, k8v_full_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(k8v_full_report)
        row = rows[0]
        stored = int(row["stored_kv_bytes"])
        materialized = int(row["materialized_working_kv_bytes"])
        metadata = int(row["metadata_bytes"])
        temporary = int(row["temporary_workspace_bytes"])
        total = int(row["total_kv_footprint_bytes"])
        assert total == stored + materialized + metadata + temporary

    def test_k_full_v8_total_reconciles(self, k_full_v8_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(k_full_v8_report)
        row = rows[0]
        stored = int(row["stored_kv_bytes"])
        materialized = int(row["materialized_working_kv_bytes"])
        metadata = int(row["metadata_bytes"])
        temporary = int(row["temporary_workspace_bytes"])
        total = int(row["total_kv_footprint_bytes"])
        assert total == stored + materialized + metadata + temporary

    def test_k8v_full_workspace_fields_positive(self, k8v_full_report):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(k8v_full_report)
        row = rows[0]
        assert int(row["stored_kv_bytes"]) > 0
        assert int(row["materialized_working_kv_bytes"]) > 0
        assert int(row["total_kv_footprint_bytes"]) > 0


class TestValidateReportModelBacked:
    """validate_report passes for model-generated reports with V5 workspace fields."""

    def test_validate_int8_report_no_warnings(self, int8_report):
        from exactkv.benchmarks.reports import validate_report, write_json_report, \
            load_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            write_json_report(int8_report, path)
            loaded = load_json_report(path)
            warnings = validate_report(loaded)
            assert warnings == [], f"Unexpected warnings: {warnings}"
        finally:
            path.unlink(missing_ok=True)

    def test_validate_int4sim_report_no_warnings(self, int4_report):
        from exactkv.benchmarks.reports import validate_report, write_json_report, \
            load_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            write_json_report(int4_report, path)
            loaded = load_json_report(path)
            warnings = validate_report(loaded)
            assert warnings == [], f"Unexpected warnings: {warnings}"
        finally:
            path.unlink(missing_ok=True)
