"""Tests: leaderboard and markdown rendering for asymmetric K/V compressors (V4 Phase D).

Verifies:
  * average_effective_bit_width returns expected values.
  * render_compressor_leaderboard includes K bits, V bits, avg eff bits columns
    when compressor_caps is provided.
  * Full-precision sides render as "full" in leaderboard.
  * render_compressor_x_draft_leaderboard includes K/V columns when caps provided.
  * Markdown report includes K/V metadata section when asymmetric compressors exist.
  * Markdown report does NOT include K/V metadata section for symmetric-only reports.
  * list-compressors CLI output shows key_bit_width, value_bit_width, asymmetric.
  * No forbidden performance fields in leaderboard output or markdown output.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_forbidden(text: str) -> None:
    """Assert no forbidden performance field names appear as data keys."""
    import re
    for field in _FORBIDDEN:
        # Match as table column header or key-value pattern, not prose negation.
        patterns = [
            rf"\|\s*{re.escape(field)}\s*\|",    # table column
            rf"^\s*{re.escape(field)}\s*:",        # YAML / key-value
        ]
        for pat in patterns:
            assert not re.search(pat, text, re.MULTILINE), (
                f"Forbidden performance field {field!r} found as data key in output"
            )


def _make_acceptance_table(*compressor_names: str) -> list[dict]:
    """Build a synthetic acceptance table suitable for leaderboard rendering."""
    rows = []
    for name in compressor_names:
        rows.append({
            "compressor_name": name,
            "mean_acceptance_rate": 0.75,
            "mean_average_accepted_length": 2.5,
            "total_drafted": 100,
            "total_accepted": 75,
            "total_rejected": 25,
            "total_corrections": 25,
            "num_runs": 5,
            "exactkv_failures": 0,
        })
    return rows


def _make_compressor_caps_map(*names: str) -> dict[str, dict]:
    """Return a synthetic compressor_caps mapping from the live registry."""
    from dataclasses import asdict
    from exactkv.compressors import get_compressor
    result = {}
    for name in names:
        comp = get_compressor(name)
        result[name] = asdict(comp.capabilities)
    return result


def _make_cross_table(*compressor_names: str, draft_lens=(3, 5)) -> list[dict]:
    rows = []
    for comp in compressor_names:
        for dl in draft_lens:
            rows.append({
                "compressor_name": comp,
                "draft_len": dl,
                "mean_acceptance_rate": 0.8,
                "mean_average_accepted_length": 2.0,
                "total_drafted": 50,
                "total_accepted": 40,
                "total_rejected": 10,
                "total_corrections": 10,
                "num_runs": 2,
                "exactkv_failures": 0,
            })
    return rows


# ===========================================================================
# TestAverageEffectiveBitWidth
# ===========================================================================

class TestAverageEffectiveBitWidth:
    """Unit tests for the average_effective_bit_width helper."""

    def _avg(self, k, v, full=32):
        from exactkv.reporting.leaderboard import average_effective_bit_width
        return average_effective_bit_width(k, v, full_bit_width=full)

    def test_int8_symmetric(self):
        assert self._avg(8, 8) == 8.0

    def test_int4_symmetric(self):
        assert self._avg(4, 4) == 4.0

    def test_int2_symmetric(self):
        assert self._avg(2, 2) == 2.0

    def test_k8_v4(self):
        assert self._avg(8, 4) == 6.0

    def test_k8_v2(self):
        assert self._avg(8, 2) == 5.0

    def test_k4_v8(self):
        assert self._avg(4, 8) == 6.0

    def test_k_full_v4_default_full(self):
        # full precision = 32 bits by default
        assert self._avg(None, 4) == 18.0

    def test_k8_v_full_default_full(self):
        assert self._avg(8, None) == 20.0

    def test_both_full(self):
        assert self._avg(None, None) == 32.0

    def test_custom_full_bit_width(self):
        assert self._avg(None, 4, full=16) == 10.0

    def test_returns_float(self):
        result = self._avg(8, 8)
        assert isinstance(result, float)

    def test_exported_from_reporting_package(self):
        from exactkv.reporting import average_effective_bit_width
        assert average_effective_bit_width(8, 4) == 6.0


# ===========================================================================
# TestCompressorLeaderboardWithCaps
# ===========================================================================

class TestCompressorLeaderboardWithCaps:
    """render_compressor_leaderboard with caps shows K/V metadata columns."""

    def test_columns_present_with_caps(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        table = _make_acceptance_table("int8", "k8_v4_sim")
        caps = _make_compressor_caps_map("int8", "k8_v4_sim")
        md = render_compressor_leaderboard(table, compressor_caps=caps)
        assert "K bits" in md
        assert "V bits" in md
        assert "avg eff bits" in md

    def test_no_kv_columns_without_caps(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        table = _make_acceptance_table("int8")
        md = render_compressor_leaderboard(table)
        assert "K bits" not in md
        assert "V bits" not in md
        assert "avg eff bits" not in md

    def test_int8_renders_k8_v8(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        table = _make_acceptance_table("int8")
        caps = _make_compressor_caps_map("int8")
        md = render_compressor_leaderboard(table, compressor_caps=caps)
        assert "| 8 " in md or "| 8|" in md or " 8 " in md

    def test_full_precision_renders_as_full(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        table = _make_acceptance_table("k8_v_full")
        caps = _make_compressor_caps_map("k8_v_full")
        md = render_compressor_leaderboard(table, compressor_caps=caps)
        assert "full" in md

    def test_k8v4_avg_eff_bits_6(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        table = _make_acceptance_table("k8_v4_sim")
        caps = _make_compressor_caps_map("k8_v4_sim")
        md = render_compressor_leaderboard(table, compressor_caps=caps)
        assert "6.0" in md

    def test_simulated_column_present(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        table = _make_acceptance_table("k8_v4_sim")
        caps = _make_compressor_caps_map("k8_v4_sim")
        md = render_compressor_leaderboard(table, compressor_caps=caps)
        assert "simulated" in md

    def test_no_forbidden_fields(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        table = _make_acceptance_table("int8", "k8_v4_sim")
        caps = _make_compressor_caps_map("int8", "k8_v4_sim")
        md = render_compressor_leaderboard(table, compressor_caps=caps)
        _no_forbidden(md)

    def test_empty_table_returns_no_data(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        md = render_compressor_leaderboard([])
        assert "_No data._" in md


# ===========================================================================
# TestCrossLeaderboardWithCaps
# ===========================================================================

class TestCrossLeaderboardWithCaps:
    """render_compressor_x_draft_leaderboard with caps shows K/V columns."""

    def test_kv_columns_present_with_caps(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        table = _make_cross_table("int8", "k8_v4_sim")
        caps = _make_compressor_caps_map("int8", "k8_v4_sim")
        md = render_compressor_x_draft_leaderboard(table, compressor_caps=caps)
        assert "K bits" in md
        assert "V bits" in md
        assert "avg eff bits" in md

    def test_no_kv_columns_without_caps(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        table = _make_cross_table("int8")
        md = render_compressor_x_draft_leaderboard(table)
        assert "K bits" not in md
        assert "V bits" not in md

    def test_full_renders_as_full_in_cross_table(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        table = _make_cross_table("k_full_v8")
        caps = _make_compressor_caps_map("k_full_v8")
        md = render_compressor_x_draft_leaderboard(table, compressor_caps=caps)
        assert "full" in md

    def test_no_forbidden_fields_cross_table(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        table = _make_cross_table("int8", "k8_v4_sim")
        caps = _make_compressor_caps_map("int8", "k8_v4_sim")
        md = render_compressor_x_draft_leaderboard(table, compressor_caps=caps)
        _no_forbidden(md)


# ===========================================================================
# TestMarkdownReportKvSection
# ===========================================================================

class TestMarkdownReportKvSection:
    """render_markdown_report includes K/V metadata section for asymmetric reports."""

    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    _TEST_PROMPT = {
        "prompt_id": "asym_md_001",
        "category": "test",
        "prompt": "Asymmetric markdown test",
    }

    @pytest.fixture(scope="class")
    def runtime(self):
        from exactkv.runtime.model_runtime import ModelRuntime
        return ModelRuntime(self.MODEL_NAME, device="cpu", dtype="float32")

    @pytest.fixture(scope="class")
    def asym_report(self, runtime):
        from exactkv.benchmarks.runner import RunConfig, run_suite
        config = RunConfig(compressor_name="k8_v4_sim", draft_len=3, max_new_tokens=8)
        return run_suite(runtime, [self._TEST_PROMPT], config)

    @pytest.fixture(scope="class")
    def sym_report(self, runtime):
        from exactkv.benchmarks.runner import RunConfig, run_suite
        config = RunConfig(compressor_name="int8", draft_len=3, max_new_tokens=8)
        return run_suite(runtime, [self._TEST_PROMPT], config)

    def test_asym_report_has_kv_metadata_section(self, asym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(asym_report)
        assert "K/V Compression Metadata" in md

    def test_sym_report_has_no_kv_metadata_section(self, sym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(sym_report)
        assert "K/V Compression Metadata" not in md

    def test_kv_section_mentions_full_precision(self, asym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(asym_report)
        # The notes under the KV metadata section explain "full" means full-precision
        assert "full" in md
        assert "full-precision" in md.lower() or "full precision" in md.lower()

    def test_kv_section_mentions_avg_eff_bits_not_memory(self, asym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(asym_report)
        # Must warn it is a comparison aid, not real memory
        assert "comparison aid" in md.lower() or "not a real memory" in md.lower()

    def test_kv_section_mentions_simulated(self, asym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(asym_report)
        assert "simulated" in md.lower()

    def test_no_forbidden_fields_in_asym_markdown(self, asym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(asym_report)
        _no_forbidden(md)

    def test_what_not_proves_mentions_sub_int8(self, asym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(asym_report)
        # WHAT_NOT_PROVES now mentions sub-INT8 asymmetric simulation
        assert "Sub-INT8" in md or "sub-INT8" in md

    def test_what_not_proves_mentions_avg_eff_bits(self, asym_report):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(asym_report)
        assert "Average effective bit width" in md or "average effective bit" in md.lower()


# ===========================================================================
# TestCliListCompressors
# ===========================================================================

class TestCliListCompressors:
    """list-compressors output includes K bits, V bits, asymmetric."""

    def _run_list(self) -> str:
        """Invoke _cmd_list_compressors and capture printed output."""
        import io
        import sys
        import exactkv.compressors  # register built-ins
        from exactkv.cli import _cmd_list_compressors
        import argparse
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _cmd_list_compressors(argparse.Namespace())
        finally:
            sys.stdout = old
        return buf.getvalue()

    def test_output_contains_key_bit_width(self):
        out = self._run_list()
        assert "key_bit_width" in out

    def test_output_contains_value_bit_width(self):
        out = self._run_list()
        assert "value_bit_width" in out

    def test_output_contains_asymmetric(self):
        out = self._run_list()
        assert "asymmetric" in out

    def test_int8_shows_key_8(self):
        out = self._run_list()
        # Find int8 section and verify K=8 appears
        lines = out.splitlines()
        in_int8 = False
        found = False
        for line in lines:
            if line.strip() == "int8":
                in_int8 = True
            elif in_int8:
                if "key_bit_width" in line:
                    assert "8" in line
                    found = True
                    break
                elif line.strip() == "":
                    in_int8 = False
        assert found, "key_bit_width line not found in int8 section"

    def test_k8v4_shows_asymmetric_true(self):
        out = self._run_list()
        lines = out.splitlines()
        in_k8v4 = False
        found = False
        for line in lines:
            if line.strip() == "k8_v4_sim":
                in_k8v4 = True
            elif in_k8v4:
                if "asymmetric" in line:
                    assert "True" in line
                    found = True
                    break
                elif line.strip() == "":
                    in_k8v4 = False
        assert found, "asymmetric line not found in k8_v4_sim section"

    def test_k8vfull_shows_value_full(self):
        out = self._run_list()
        lines = out.splitlines()
        in_k8vfull = False
        found = False
        for line in lines:
            if line.strip() == "k8_v_full":
                in_k8vfull = True
            elif in_k8vfull:
                if "value_bit_width" in line:
                    assert "full" in line
                    found = True
                    break
                elif line.strip() == "":
                    in_k8vfull = False
        assert found, "value_bit_width line not found in k8_v_full section"

    def test_noop_shows_key_full(self):
        out = self._run_list()
        lines = out.splitlines()
        in_noop = False
        found = False
        for line in lines:
            if line.strip() == "noop":
                in_noop = True
            elif in_noop:
                if "key_bit_width" in line:
                    assert "full" in line
                    found = True
                    break
                elif line.strip() == "":
                    in_noop = False
        assert found, "key_bit_width line not found in noop section"

    def test_no_forbidden_fields_in_output(self):
        out = self._run_list()
        _no_forbidden(out)
