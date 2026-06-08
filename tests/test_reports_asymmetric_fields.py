"""Tests: asymmetric K/V metadata in JSON/CSV reports (V4 Phase D).

Verifies:
  * run_one result includes key_bit_width, value_bit_width, asymmetric
    inside compressor_capabilities.
  * JSON report includes these fields in compressor_capabilities.
  * CSV report includes key_bit_width, value_bit_width, asymmetric columns.
  * int8 row has K=8, V=8, asymmetric=False.
  * int4_sim row has K=4, V=4, asymmetric=False.
  * k8_v4_sim row has K=8, V=4, asymmetric=True.
  * k8_v_full row has K=8, V=None (csv: empty), asymmetric=True, is_simulated=False.
  * Synthetic (legacy) reports without those fields render safely.
  * No forbidden performance fields appear in JSON or CSV output.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_TEST_PROMPT = {
    "prompt_id": "asym_rpt_001",
    "category": "test",
    "prompt": "Asymmetric report test prompt",
}

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})


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

def _make_report(runtime, compressor_name: str):
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
def k8v4_report(runtime):
    return _make_report(runtime, "k8_v4_sim")


@pytest.fixture(scope="module")
def k8vfull_report(runtime):
    return _make_report(runtime, "k8_v_full")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_caps(report: dict) -> dict:
    return report["results"][0]["compressor_capabilities"]


def _csv_rows(report: dict) -> list[dict]:
    from exactkv.benchmarks.reports import flatten_report_to_rows
    rows = flatten_report_to_rows(report)
    assert rows, "Expected at least one CSV row"
    return rows


def _assert_no_forbidden_fields(obj, path=""):
    """Recursively ensure no forbidden performance keys appear."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in _FORBIDDEN_FIELDS, (
                f"Forbidden field {k!r} found at {path or 'root'}"
            )
            _assert_no_forbidden_fields(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, path=f"{path}[{i}]")


# ===========================================================================
# TestRunOneCapabilities: run_one result includes K/V metadata
# ===========================================================================

class TestRunOneCapabilities:
    """run_one results must carry K/V capability fields."""

    def test_int8_has_k8_v8_symmetric(self, int8_report):
        caps = _get_caps(int8_report)
        assert caps.get("key_bit_width") == 8
        assert caps.get("value_bit_width") == 8
        assert caps.get("asymmetric") is False

    def test_int4_has_k4_v4_symmetric(self, int4_report):
        caps = _get_caps(int4_report)
        assert caps.get("key_bit_width") == 4
        assert caps.get("value_bit_width") == 4
        assert caps.get("asymmetric") is False

    def test_k8v4_has_k8_v4_asymmetric(self, k8v4_report):
        caps = _get_caps(k8v4_report)
        assert caps.get("key_bit_width") == 8
        assert caps.get("value_bit_width") == 4
        assert caps.get("asymmetric") is True

    def test_k8vfull_has_k8_vnone_asymmetric(self, k8vfull_report):
        caps = _get_caps(k8vfull_report)
        assert caps.get("key_bit_width") == 8
        assert caps.get("value_bit_width") is None
        assert caps.get("asymmetric") is True
        assert caps.get("is_simulated") is False


# ===========================================================================
# TestJsonReport: JSON reports carry K/V capability metadata
# ===========================================================================

class TestJsonReport:
    """JSON round-trip must preserve K/V capability fields."""

    def _json_roundtrip(self, report: dict) -> dict:
        from exactkv.benchmarks.reports import write_json_report, load_json_report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_json_report(report, path)
            return load_json_report(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_int8_json_has_k8_v8(self, int8_report):
        loaded = self._json_roundtrip(int8_report)
        caps = loaded["results"][0]["compressor_capabilities"]
        assert caps.get("key_bit_width") == 8
        assert caps.get("value_bit_width") == 8
        assert caps.get("asymmetric") is False

    def test_int4_json_has_k4_v4(self, int4_report):
        loaded = self._json_roundtrip(int4_report)
        caps = loaded["results"][0]["compressor_capabilities"]
        assert caps.get("key_bit_width") == 4
        assert caps.get("value_bit_width") == 4

    def test_k8v4_json_has_k8_v4(self, k8v4_report):
        loaded = self._json_roundtrip(k8v4_report)
        caps = loaded["results"][0]["compressor_capabilities"]
        assert caps.get("key_bit_width") == 8
        assert caps.get("value_bit_width") == 4
        assert caps.get("asymmetric") is True

    def test_k8vfull_json_has_vnone(self, k8vfull_report):
        loaded = self._json_roundtrip(k8vfull_report)
        caps = loaded["results"][0]["compressor_capabilities"]
        assert caps.get("key_bit_width") == 8
        assert caps.get("value_bit_width") is None

    def test_no_forbidden_fields_in_json(self, k8v4_report):
        loaded = self._json_roundtrip(k8v4_report)
        _assert_no_forbidden_fields(loaded)


# ===========================================================================
# TestCsvReport: CSV columns include key_bit_width, value_bit_width, asymmetric
# ===========================================================================

class TestCsvReport:
    """CSV rows must contain the three new K/V columns."""

    def test_csv_has_key_bit_width_column_int8(self, int8_report):
        rows = _csv_rows(int8_report)
        assert "key_bit_width" in rows[0], "CSV row missing key_bit_width column"
        assert str(rows[0]["key_bit_width"]) == "8"

    def test_csv_has_value_bit_width_column_int8(self, int8_report):
        rows = _csv_rows(int8_report)
        assert str(rows[0]["value_bit_width"]) == "8"

    def test_csv_asymmetric_false_for_int8(self, int8_report):
        rows = _csv_rows(int8_report)
        assert str(rows[0]["asymmetric"]).lower() in ("false", "0")

    def test_csv_k8v4_key_bit_width_8(self, k8v4_report):
        rows = _csv_rows(k8v4_report)
        assert str(rows[0]["key_bit_width"]) == "8"

    def test_csv_k8v4_value_bit_width_4(self, k8v4_report):
        rows = _csv_rows(k8v4_report)
        assert str(rows[0]["value_bit_width"]) == "4"

    def test_csv_k8v4_asymmetric_true(self, k8v4_report):
        rows = _csv_rows(k8v4_report)
        assert str(rows[0]["asymmetric"]).lower() in ("true", "1")

    def test_csv_k8vfull_value_bit_width_empty_or_none(self, k8vfull_report):
        rows = _csv_rows(k8vfull_report)
        # None serialises as empty string in CSV helper (caps.get returns None → "")
        assert str(rows[0].get("value_bit_width", "")) in ("", "None")

    def test_csv_columns_present_in_written_file(self, k8v4_report, tmp_path):
        from exactkv.benchmarks.reports import write_csv_report
        out = tmp_path / "asym_test.csv"
        write_csv_report(k8v4_report, str(out))
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        assert "key_bit_width" in headers
        assert "value_bit_width" in headers
        assert "asymmetric" in headers

    def test_no_forbidden_columns(self, k8v4_report, tmp_path):
        from exactkv.benchmarks.reports import write_csv_report
        out = tmp_path / "no_perf.csv"
        write_csv_report(k8v4_report, str(out))
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
        for field in _FORBIDDEN_FIELDS:
            assert field not in headers, f"Forbidden column {field!r} found in CSV"


# ===========================================================================
# TestLegacyReportSafety: synthetic reports missing K/V fields render safely
# ===========================================================================

class TestLegacyReportSafety:
    """Reports without K/V capability fields must not crash flatten or render."""

    def _make_legacy_result(self) -> dict:
        """Synthetic result dict without key_bit_width / value_bit_width / asymmetric."""
        return {
            "prompt_id": "legacy_001",
            "category": "test",
            "model_name": MODEL_NAME,
            "compressor_name": "noop",
            "draft_len": 4,
            "max_new_tokens": 12,
            "compressor_capabilities": {
                "name": "noop",
                "compressor_type": "noop",
                "is_simulated": False,
                "supports_real_bytes_claim": True,
                # deliberately omit key_bit_width, value_bit_width, asymmetric
            },
            "exactkv": {
                "token_exact_match": True,
                "acceptance": {
                    "acceptance_rate": 1.0,
                    "avg_accepted_per_round": 4.0,
                    "total_drafted": 4,
                    "total_accepted": 4,
                    "total_rejected": 0,
                    "total_corrections": 0,
                },
            },
            "lossy": {
                "token_exact_match": True,
                "first_divergence_idx": None,
            },
            "memory": {
                "full_kv_bytes": 1000,
                "compressed_kv_bytes": 1000,
                "compression_ratio": 1.0,
                "memory_reduction_factor": 1.0,
                "memory_claim_note": "",
                "supports_real_bytes_claim": True,
            },
        }

    def _legacy_report(self) -> dict:
        return {
            "manifest": {},
            "results": [self._make_legacy_result()],
        }

    def test_flatten_legacy_does_not_raise(self):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(self._legacy_report())
        assert len(rows) == 1

    def test_legacy_csv_key_bit_width_defaults_empty(self):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(self._legacy_report())
        # caps.get("key_bit_width", "") returns "" when field is absent
        assert str(rows[0].get("key_bit_width", "")) == ""

    def test_legacy_csv_asymmetric_defaults_empty(self):
        from exactkv.benchmarks.reports import flatten_report_to_rows
        rows = flatten_report_to_rows(self._legacy_report())
        assert str(rows[0].get("asymmetric", "")) == ""

    def test_legacy_write_csv_does_not_raise(self, tmp_path):
        from exactkv.benchmarks.reports import write_csv_report
        out = tmp_path / "legacy.csv"
        write_csv_report(self._legacy_report(), str(out))
        assert out.exists()

    def test_legacy_write_json_does_not_raise(self, tmp_path):
        from exactkv.benchmarks.reports import write_json_report
        out = tmp_path / "legacy.json"
        write_json_report(self._legacy_report(), str(out))
        assert out.exists()
