"""Tests for exactkv.reporting.leaderboard."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}


def _no_forbidden(text: str):
    for f in _FORBIDDEN:
        assert f not in text, f"Forbidden field {f!r} found in rendered output"


def _compressor_report():
    return make_report(
        make_result(compressor_name="noop", draft_len=4,
                    total_drafted=8, total_accepted=8, total_rejected=0,
                    total_corrections=0, acceptance_rate=1.0,
                    avg_accepted_per_round=4.0),
        make_result(compressor_name="int8", draft_len=4,
                    total_drafted=8, total_accepted=6, total_rejected=2,
                    total_corrections=2, acceptance_rate=0.75,
                    avg_accepted_per_round=3.0),
        make_result(compressor_name="int4_sim", draft_len=4,
                    total_drafted=8, total_accepted=4, total_rejected=4,
                    total_corrections=4, acceptance_rate=0.5,
                    avg_accepted_per_round=2.0, is_simulated=True,
                    supports_real_bytes_claim=False),
    )


# ──────────────────────────────────────────────────────────────
# render_compressor_leaderboard
# ──────────────────────────────────────────────────────────────

class TestCompressorLeaderboard:
    def test_returns_string(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
        report = _compressor_report()
        table = group_acceptance_by_compressor(report)
        result = render_compressor_leaderboard(table)
        assert isinstance(result, str)

    def test_contains_compressor_names(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
        table = group_acceptance_by_compressor(_compressor_report())
        md = render_compressor_leaderboard(table)
        assert "noop" in md
        assert "int8" in md
        assert "int4_sim" in md

    def test_contains_acceptance_rate(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
        table = group_acceptance_by_compressor(_compressor_report())
        md = render_compressor_leaderboard(table)
        assert "accept_rate" in md

    def test_contains_exactkv_fail_column(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
        table = group_acceptance_by_compressor(_compressor_report())
        md = render_compressor_leaderboard(table)
        assert "exactkv_fail" in md

    def test_with_caps_shows_simulated_column(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
        table = group_acceptance_by_compressor(_compressor_report())
        caps = {
            "noop": {"is_simulated": False, "supports_real_bytes_claim": True},
            "int4_sim": {"is_simulated": True, "supports_real_bytes_claim": False},
        }
        md = render_compressor_leaderboard(table, compressor_caps=caps)
        assert "simulated" in md

    def test_empty_table_returns_no_data_string(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        md = render_compressor_leaderboard([])
        assert "No data" in md

    def test_no_forbidden_fields(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
        table = group_acceptance_by_compressor(_compressor_report())
        _no_forbidden(render_compressor_leaderboard(table))


# ──────────────────────────────────────────────────────────────
# render_draft_len_leaderboard
# ──────────────────────────────────────────────────────────────

class TestDraftLenLeaderboard:
    def _sweep_report(self):
        return make_report(
            make_result(draft_len=4, compressor_name="int8",
                        total_drafted=8, total_accepted=6,
                        total_rejected=2, acceptance_rate=0.75),
            make_result(draft_len=8, compressor_name="int8",
                        total_drafted=16, total_accepted=14,
                        total_rejected=2, acceptance_rate=0.875),
        )

    def test_returns_string(self):
        from exactkv.reporting.leaderboard import render_draft_len_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_draft_len
        table = group_acceptance_by_draft_len(self._sweep_report())
        assert isinstance(render_draft_len_leaderboard(table), str)

    def test_contains_draft_lens(self):
        from exactkv.reporting.leaderboard import render_draft_len_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_draft_len
        table = group_acceptance_by_draft_len(self._sweep_report())
        md = render_draft_len_leaderboard(table)
        assert "4" in md
        assert "8" in md

    def test_no_forbidden_fields(self):
        from exactkv.reporting.leaderboard import render_draft_len_leaderboard
        from exactkv.analysis.acceptance_tables import group_acceptance_by_draft_len
        table = group_acceptance_by_draft_len(self._sweep_report())
        _no_forbidden(render_draft_len_leaderboard(table))


# ──────────────────────────────────────────────────────────────
# render_compressor_x_draft_leaderboard
# ──────────────────────────────────────────────────────────────

class TestCompressorXDraftLeaderboard:
    def _sweep_report(self):
        return make_report(
            make_result(compressor_name="noop", draft_len=4),
            make_result(compressor_name="noop", draft_len=8),
            make_result(compressor_name="int8", draft_len=4),
            make_result(compressor_name="int8", draft_len=8),
        )

    def test_returns_string(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        from exactkv.analysis.acceptance_tables import build_acceptance_table
        table = build_acceptance_table(self._sweep_report())
        assert isinstance(render_compressor_x_draft_leaderboard(table), str)

    def test_contains_all_combinations(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        from exactkv.analysis.acceptance_tables import build_acceptance_table
        table = build_acceptance_table(self._sweep_report())
        md = render_compressor_x_draft_leaderboard(table)
        assert "noop" in md
        assert "int8" in md

    def test_has_draft_len_column(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        from exactkv.analysis.acceptance_tables import build_acceptance_table
        table = build_acceptance_table(self._sweep_report())
        md = render_compressor_x_draft_leaderboard(table)
        assert "draft_len" in md

    def test_no_forbidden_fields(self):
        from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
        from exactkv.analysis.acceptance_tables import build_acceptance_table
        table = build_acceptance_table(self._sweep_report())
        _no_forbidden(render_compressor_x_draft_leaderboard(table))


# ──────────────────────────────────────────────────────────────
# Real sweep
# ──────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def real_sweep():
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime
    rt = ModelRuntime(MODEL_NAME, device="auto", dtype="float32")
    prompts = [{"prompt_id": "lb_t1", "category": "test",
                "prompt": "The capital of France is"}]
    return run_sweep(rt, prompts=prompts,
                     compressor_names=["noop", "int8"],
                     draft_lengths=[4, 8], max_new_tokens=8)


def test_real_sweep_compressor_leaderboard(real_sweep):
    from exactkv.reporting.leaderboard import render_compressor_leaderboard
    from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
    table = group_acceptance_by_compressor(real_sweep)
    md = render_compressor_leaderboard(table)
    assert "noop" in md
    assert "int8" in md
    _no_forbidden(md)


class TestMixedPrecisionKvMetadata:
    """Layer-aware boundary compressors must not render as full-V / 20.0 avg bits."""

    _BOUNDARY_CAPS = {
        "k8_v4_boundary_v8_sim": {
            "is_simulated": True,
            "supports_real_bytes_claim": False,
            "key_bit_width": 8,
            "value_bit_width": None,
            "value_bit_width_label": "mixed 8/4-sim",
            "asymmetric": True,
        },
        "k8_v4_sim": {
            "is_simulated": True,
            "key_bit_width": 8,
            "value_bit_width": 4,
            "asymmetric": True,
        },
        "k_full_v4_sim": {
            "is_simulated": True,
            "key_bit_width": None,
            "value_bit_width": 4,
            "asymmetric": True,
        },
    }

    def _table(self):
        from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
        report = make_report(
            make_result(compressor_name="k8_v4_boundary_v8_sim", draft_len=4),
            make_result(compressor_name="k8_v4_sim", draft_len=4),
            make_result(compressor_name="k_full_v4_sim", draft_len=4),
        )
        return group_acceptance_by_compressor(report)

    def test_boundary_renders_mixed_v_not_full(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        md = render_compressor_leaderboard(
            self._table(), compressor_caps=self._BOUNDARY_CAPS,
        )
        assert "mixed 8/4-sim" in md
        assert "k8_v4_boundary_v8_sim" in md
        # boundary row must not pair with V bits = full in the metadata columns
        for line in md.splitlines():
            if "k8_v4_boundary_v8_sim" in line:
                assert "| full |" not in line.replace("k_full_v4_sim", "")
                assert "20.0" not in line
                break
        else:
            raise AssertionError("boundary compressor row not found")

    def test_k8_v4_sim_still_renders_k8_v4(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        md = render_compressor_leaderboard(
            self._table(), compressor_caps=self._BOUNDARY_CAPS,
        )
        for line in md.splitlines():
            if "k8_v4_sim" in line and "boundary" not in line:
                assert "| 8 |" in line or "| 8 " in line
                assert "| 4 |" in line or "| 4 " in line
                assert "6.0" in line
                break

    def test_k_full_v4_sim_still_renders_full_k_v4(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        md = render_compressor_leaderboard(
            self._table(), compressor_caps=self._BOUNDARY_CAPS,
        )
        for line in md.splitlines():
            if "k_full_v4_sim" in line:
                assert "full" in line
                assert "| 4 " in line or "| 4 |" in line
                assert "18.0" in line
                break

    def test_registry_enrichment_for_old_stored_caps(self):
        from exactkv.reporting.leaderboard import enrich_caps_from_registry
        import exactkv.compressors  # noqa: F401

        old_caps = {
            "is_simulated": True,
            "key_bit_width": 8,
            "value_bit_width": None,
            "asymmetric": True,
        }
        enriched = enrich_caps_from_registry("k8_v4_boundary4_v8_sim", old_caps)
        assert enriched.get("value_bit_width_label") == "mixed 8/4-sim"

    def test_no_forbidden_fields(self):
        from exactkv.reporting.leaderboard import render_compressor_leaderboard
        md = render_compressor_leaderboard(
            self._table(), compressor_caps=self._BOUNDARY_CAPS,
        )
        _no_forbidden(md)


def test_real_sweep_x_draft_leaderboard(real_sweep):
    from exactkv.reporting.leaderboard import render_compressor_x_draft_leaderboard
    from exactkv.analysis.acceptance_tables import build_acceptance_table
    table = build_acceptance_table(real_sweep)
    md = render_compressor_x_draft_leaderboard(table)
    # 2 compressors × 2 draft_lens = 4 rows
    assert md.count("noop") >= 1
    assert md.count("int8") >= 1
    _no_forbidden(md)
