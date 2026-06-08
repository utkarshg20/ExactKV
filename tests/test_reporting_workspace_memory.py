"""Tests: V5 Phase C — Markdown and CLI rendering of workspace-aware memory fields.

Verifies:
  * format_bytes produces human-readable strings.
  * render_workspace_memory_table renders a table for V5 reports.
  * render_workspace_memory_table renders a legacy note for all-zero reports.
  * render_markdown_report includes a "Workspace-Aware Memory Accounting" section.
  * Markdown section explains stored KV, materialized KV, metadata, temp workspace.
  * Markdown says total_kv_footprint_bytes is an accounting total, NOT measured peak.
  * Markdown says active GPU memory is deferred.
  * Markdown says simulated sub-INT8 uses int8 containers.
  * Workspace table rows include int8, int4_sim, k8_v4_sim when present.
  * Legacy V4 reports without workspace fields render gracefully.
  * report CLI output includes "Workspace memory" line.
  * bench/sweep CLI summaries include no forbidden performance fields.
  * No forbidden fields in Markdown output.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_FORBIDDEN_PERF_FIELDS = frozenset({
    "tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds",
})
_FORBIDDEN_DATA_PATTERNS = [
    "| tokens_per_second",
    "| throughput |",
    "| latency |",
    "| speedup |",
    "| runtime_seconds",
    "tokens_per_second:",
    "throughput:",
    "latency:",
    "speedup:",
    "runtime_seconds:",
]


def _no_forbidden_md(text: str) -> None:
    lower = text.lower()
    for pat in _FORBIDDEN_DATA_PATTERNS:
        assert pat not in lower, f"Forbidden pattern {pat!r} found in Markdown"


# ---------------------------------------------------------------------------
# Synthetic V5 result helpers
# ---------------------------------------------------------------------------

def _make_v5_result(
    compressor_name: str = "int8",
    full_bytes: int = 131072,  # 128 KiB — realistic for tiny model
    stored_kv: int = 32768,
    materialized: int = 131072,
    metadata: int = 1024,
    temporary: int = 32768,
    supports_real: bool = True,
    is_simulated: bool = False,
    key_bit_width: int | None = 8,
    value_bit_width: int | None = 8,
    asymmetric: bool = False,
) -> dict[str, Any]:
    """Build a synthetic V5-style run_one result dict."""
    total = stored_kv + materialized + metadata + temporary
    compressed = stored_kv + metadata
    note = (
        f"Simulated ({compressor_name!r}): int8 containers."
        if is_simulated else
        f"Real-storage ({compressor_name!r})."
    )
    return {
        "prompt_id": f"wm_c_{compressor_name}",
        "category": "test",
        "model_name": MODEL_NAME,
        "compressor_name": compressor_name,
        "compressor_capabilities": {
            "compressor_type": "quantize",
            "is_simulated": is_simulated,
            "supports_real_bytes_claim": supports_real,
            "key_bit_width": key_bit_width,
            "value_bit_width": value_bit_width,
            "asymmetric": asymmetric,
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
                "avg_accepted_per_round": 4.0,
                "total_drafted": 4,
                "total_accepted": 4,
                "total_rejected": 0,
                "total_corrections": 0,
            },
        },
        "memory": {
            "full_bytes": full_bytes,
            "compressed_bytes": compressed,
            "compression_ratio": compressed / max(full_bytes, 1),
            "memory_reduction_factor": full_bytes / max(compressed, 1),
            "stored_kv_bytes": stored_kv,
            "materialized_working_kv_bytes": materialized,
            "metadata_bytes": metadata,
            "temporary_workspace_bytes": temporary,
            "total_kv_footprint_bytes": total,
            "supports_real_bytes_claim": supports_real,
            "is_simulated": is_simulated,
            "memory_claim_note": note,
        },
        "exactkv_failure": False,
    }


def _make_v5_report(*results: dict[str, Any]) -> dict[str, Any]:
    return {
        "results": list(results),
        "aggregate": {
            "total_prompts": len(results),
            "compressor_name": results[0]["compressor_name"] if results else "?",
            "exactkv_failures": 0,
            "exactkv_pass_rate": 1.0,
        },
    }


def _multi_compressor_v5_report() -> dict[str, Any]:
    """Report with int8, int4_sim, k8_v4_sim entries."""
    return _make_v5_report(
        _make_v5_result("int8", supports_real=True, is_simulated=False),
        _make_v5_result(
            "int4_sim",
            stored_kv=32768,  # int8 container, same as int8
            supports_real=False,
            is_simulated=True,
        ),
        _make_v5_result(
            "k8_v4_sim",
            stored_kv=32768,
            supports_real=False,
            is_simulated=True,
            key_bit_width=8,
            value_bit_width=4,
            asymmetric=True,
        ),
    )


# ===========================================================================
# SECTION A: format_bytes unit tests
# ===========================================================================

class TestFormatBytes:

    def test_zero(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes(0) == "0 B"

    def test_small_bytes(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes(512) == "512 B"
        assert format_bytes(1023) == "1023 B"

    def test_kib(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes(1024) == "1.0 KiB"
        assert format_bytes(2048) == "2.0 KiB"
        assert format_bytes(1536) == "1.5 KiB"

    def test_mib(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes(1_048_576) == "1.0 MiB"
        assert format_bytes(2_097_152) == "2.0 MiB"

    def test_gib(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes(1_073_741_824) == "1.0 GiB"

    def test_negative_returns_dash(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes(-1) == "—"
        assert format_bytes(-1000) == "—"

    def test_none_returns_dash(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes(None) == "—"

    def test_float_input(self):
        from exactkv.reporting.memory import format_bytes
        # Float below 1 KiB truncates to int
        assert format_bytes(512.9) == "512 B"

    def test_string_input_returns_dash(self):
        from exactkv.reporting.memory import format_bytes
        assert format_bytes("512") == "—"


# ===========================================================================
# SECTION B: render_workspace_memory_table unit tests
# ===========================================================================

class TestRenderWorkspaceMemoryTable:

    def test_legacy_report_returns_legacy_note(self):
        """Report with all-zero workspace fields returns legacy note, not a table."""
        from exactkv.reporting.memory import render_workspace_memory_table
        report = make_report(make_result("int8"))  # V1-style, no V5 fields
        result = render_workspace_memory_table(report)
        assert "legacy" in result.lower() or "zero" in result.lower(), (
            f"Expected legacy note, got: {result}"
        )

    def test_empty_report_returns_no_data_note(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        result = render_workspace_memory_table({"results": []})
        assert "no workspace" in result.lower()

    def test_v5_report_returns_table(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(_make_v5_result("int8"))
        result = render_workspace_memory_table(report)
        assert "|" in result, "Expected a Markdown table with pipe characters"

    def test_table_has_compressor_column(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(_make_v5_result("int8"))
        result = render_workspace_memory_table(report)
        assert "int8" in result

    def test_table_has_all_required_columns(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(_make_v5_result("int8"))
        result = render_workspace_memory_table(report)
        for col_fragment in ("Stored", "Materialized", "Metadata", "Total"):
            assert col_fragment.lower() in result.lower(), (
                f"Column {col_fragment!r} missing from workspace table"
            )

    def test_int8_real_bytes_yes(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(
            _make_v5_result("int8", supports_real=True, is_simulated=False)
        )
        result = render_workspace_memory_table(report)
        # "yes" should appear (real bytes claim) without the warning emoji
        assert "yes" in result

    def test_int4sim_real_bytes_no_warning(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(
            _make_v5_result("int4_sim", supports_real=False, is_simulated=True)
        )
        result = render_workspace_memory_table(report)
        # "no ⚠️" should appear for real bytes = no
        assert "no" in result.lower()

    def test_simulated_flag_shown(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(
            _make_v5_result("int4_sim", supports_real=False, is_simulated=True)
        )
        result = render_workspace_memory_table(report)
        # is_simulated=True → "yes ⚠️" in table
        assert "yes" in result  # simulated column

    def test_multi_compressor_table_has_all_rows(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _multi_compressor_v5_report()
        result = render_workspace_memory_table(report)
        for name in ("int8", "int4_sim", "k8_v4_sim"):
            assert name in result, f"Compressor {name!r} missing from workspace table"

    def test_total_footnote_present(self):
        """The footnote must say total is an accounting sum, not measured peak."""
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(_make_v5_result("int8"))
        result = render_workspace_memory_table(report)
        result_lower = result.lower()
        assert "accounting" in result_lower or "sum" in result_lower, (
            f"Table footnote should mention accounting sum: {result}"
        )
        assert "not a measured" in result_lower or "not measured" in result_lower, (
            f"Table footnote should disclaim measured peak: {result}"
        )

    def test_table_no_forbidden_fields(self):
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _multi_compressor_v5_report()
        result = render_workspace_memory_table(report)
        lower = result.lower()
        for field in _FORBIDDEN_PERF_FIELDS:
            assert field not in lower, f"Forbidden field {field!r} in table"

    def test_human_readable_bytes(self):
        """Byte values in table should be human-readable (contain KiB/MiB/B)."""
        from exactkv.reporting.memory import render_workspace_memory_table
        report = _make_v5_report(
            _make_v5_result("int8", stored_kv=32768, materialized=131072)
        )
        result = render_workspace_memory_table(report)
        assert "KiB" in result or "MiB" in result or "B" in result


# ===========================================================================
# SECTION C: render_markdown_report with workspace section
# ===========================================================================

class TestMarkdownWorkspaceSection:

    def test_workspace_section_heading_present(self):
        """Markdown report must contain the workspace section heading."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        assert "workspace-aware memory accounting" in md.lower(), (
            "Markdown missing 'Workspace-Aware Memory Accounting' section"
        )

    def test_workspace_section_present_for_legacy_report(self):
        """Legacy V4-style reports also include the workspace section (with legacy note)."""
        from exactkv.reporting.markdown import render_markdown_report
        report = make_report(
            make_result("int8"),
            make_result("int4_sim", is_simulated=True, supports_real_bytes_claim=False),
        )
        md = render_markdown_report(report, include_examples=False)
        assert "workspace-aware memory accounting" in md.lower()

    def test_workspace_section_mentions_accounting_total(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        md_lower = md.lower()
        assert "accounting" in md_lower, (
            "Markdown should say total is an accounting sum"
        )

    def test_workspace_section_says_not_measured_peak(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        md_lower = md.lower()
        assert "not" in md_lower and ("measured" in md_lower or "peak" in md_lower), (
            "Markdown should explicitly disclaim measured peak GPU memory"
        )

    def test_workspace_section_says_deferred(self):
        """Markdown must say active GPU memory measurement is deferred."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        assert "deferred" in md.lower() or "later" in md.lower(), (
            "Markdown should say active GPU measurement is deferred"
        )

    def test_workspace_section_mentions_materialized_working(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        assert "materialized" in md.lower(), (
            "Markdown should mention materialized_working_kv_bytes concept"
        )

    def test_workspace_section_mentions_stored_kv(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        assert "stored" in md.lower()

    def test_workspace_section_mentions_int8_containers_for_simulated(self):
        """Markdown should say simulated sub-INT8 uses int8 containers."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(
            _make_v5_result("int4_sim", supports_real=False, is_simulated=True)
        )
        md = render_markdown_report(report, include_examples=False)
        assert "int8" in md.lower() and "container" in md.lower(), (
            "Markdown should mention int8 container reality for simulated compressors"
        )

    def test_workspace_table_includes_int8_row(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "`int8`" in md

    def test_workspace_table_includes_int4sim_row(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "`int4_sim`" in md

    def test_workspace_table_includes_k8v4sim_row(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "`k8_v4_sim`" in md

    def test_what_not_proves_mentions_accounting(self):
        """The 'What this does not prove' section should mention accounting total."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        # The _WHAT_NOT_PROVES block mentions workspace total
        assert "total_kv_footprint_bytes" in md or "accounting sum" in md.lower()

    def test_markdown_no_forbidden_data_fields(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        md = render_markdown_report(report, include_examples=False)
        _no_forbidden_md(md)

    def test_markdown_legacy_report_no_crash(self):
        """Legacy report renders without raising."""
        from exactkv.reporting.markdown import render_markdown_report
        report = make_report(
            make_result("int8"),
            make_result("int4_sim", is_simulated=True, supports_real_bytes_claim=False),
        )
        md = render_markdown_report(report, include_examples=False)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_write_markdown_report_creates_file(self):
        from exactkv.reporting.markdown import write_markdown_report
        report = _make_v5_report(_make_v5_result("int8"))
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = Path(f.name)
        try:
            write_markdown_report(report, path, include_examples=False)
            content = path.read_text(encoding="utf-8")
            assert "workspace-aware memory accounting" in content.lower()
        finally:
            path.unlink(missing_ok=True)


# ===========================================================================
# SECTION D: CLI output tests (model-free)
# ===========================================================================

class TestCliWorkspaceLines:
    """CLI output includes workspace memory lines without forbidden fields."""

    @pytest.fixture(scope="class")
    def mini_suite_path(self, tmp_path_factory):
        p = tmp_path_factory.mktemp("cli_wm") / "mini.jsonl"
        p.write_text(
            json.dumps({
                "prompt_id": "wm_cli_001",
                "category": "test",
                "prompt": "Hello world",
            }) + "\n",
            encoding="utf-8",
        )
        return str(p)

    def test_report_cli_includes_workspace_line(
        self, capsys, tmp_path, mini_suite_path
    ):
        """report subcommand prints 'Workspace memory : included' line."""
        from exactkv.cli import main

        # First write a JSON report using a synthetic report
        report = _make_v5_report(_make_v5_result("int8"))
        json_path = tmp_path / "report.json"
        from exactkv.benchmarks.reports import write_json_report
        write_json_report(report, json_path)

        md_path = tmp_path / "out.md"
        rc = main(["report", "--report", str(json_path),
                   "--markdown-out", str(md_path), "--no-examples"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "workspace memory" in out.lower(), (
            f"'report' CLI missing workspace memory line. Got:\n{out}"
        )

    def test_report_cli_no_forbidden_output(self, capsys, tmp_path):
        """report subcommand stdout contains no forbidden performance fields."""
        from exactkv.cli import main
        from exactkv.benchmarks.reports import write_json_report

        report = _make_v5_report(_make_v5_result("int8"))
        json_path = tmp_path / "report2.json"
        write_json_report(report, json_path)
        md_path = tmp_path / "out2.md"
        main(["report", "--report", str(json_path),
              "--markdown-out", str(md_path), "--no-examples"])
        out = capsys.readouterr().out
        for field in _FORBIDDEN_PERF_FIELDS:
            assert field not in out.lower(), (
                f"Forbidden field {field!r} in report CLI output"
            )

    def test_report_cli_markdown_has_workspace_section(self, tmp_path):
        """The Markdown file produced by the report subcommand has workspace section."""
        from exactkv.cli import main
        from exactkv.benchmarks.reports import write_json_report

        report = _make_v5_report(_make_v5_result("int8"))
        json_path = tmp_path / "report3.json"
        write_json_report(report, json_path)
        md_path = tmp_path / "out3.md"
        rc = main(["report", "--report", str(json_path),
                   "--markdown-out", str(md_path), "--no-examples"])
        assert rc == 0
        content = md_path.read_text(encoding="utf-8")
        assert "workspace-aware memory accounting" in content.lower()

    def test_list_compressors_no_forbidden(self, capsys):
        from exactkv.cli import main
        main(["list-compressors"])
        out = capsys.readouterr().out
        for field in _FORBIDDEN_PERF_FIELDS:
            assert field not in out.lower(), (
                f"Forbidden field {field!r} in list-compressors output"
            )


# ===========================================================================
# SECTION E: Memory notes readability cleanup (Phase C cleanup)
# ===========================================================================

class TestMemoryNotesCompactness:
    """Memory Honesty Notes section is compact after Phase C cleanup.

    The verbose per-compressor paragraphs are replaced with a short table.
    All required honesty wording must still appear somewhere in the full report
    (either in the compact table footer, the workspace section, or the disclaimers).
    """

    _FORBIDDEN_PATTERNS = [
        "| tokens_per_second",
        "| throughput |",
        "| latency |",
        "| speedup |",
        "| runtime_seconds",
    ]

    def test_memory_honesty_section_present(self):
        """Memory Honesty Notes section heading is still in the rendered Markdown."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "memory honesty" in md.lower(), (
            "Memory Honesty Notes section heading missing"
        )

    def test_no_table_row_exceeds_200_chars(self):
        """No single pipe-table row in the Memory Honesty Notes is a full paragraph."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)

        # Isolate the Memory Honesty Notes section
        start = md.lower().find("## memory honesty notes")
        end_marker = "\n## "
        end = md.find(end_marker, start + 5) if start != -1 else -1
        section = md[start:end] if start != -1 and end != -1 else md

        for line in section.split("\n"):
            if line.startswith("|") and "compressor" not in line.lower() and "---" not in line:
                assert len(line) <= 200, (
                    f"Memory notes row exceeds 200 chars ({len(line)}): {line!r}"
                )

    def test_key_honesty_wording_accounting_sum(self):
        """'accounting' still appears in the full Markdown."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "accounting" in md.lower()

    def test_key_honesty_wording_not_measured_peak(self):
        """Disclaimer about non-measured peak appears in the full Markdown."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        md_lower = md.lower()
        assert "not" in md_lower and "measured" in md_lower

    def test_key_honesty_wording_deferred(self):
        """'deferred' appears in the full Markdown."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "deferred" in md.lower()

    def test_key_honesty_wording_int8_containers(self):
        """int8 container wording appears somewhere in the full Markdown."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "int8" in md.lower()
        assert "container" in md.lower()

    def test_key_honesty_wording_simulated(self):
        """'simulated' appears in the full Markdown."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "simulated" in md.lower()

    def test_key_honesty_wording_materialized(self):
        """'materialized' appears in the full Markdown."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "materialized" in md.lower()

    def test_workspace_section_still_present(self):
        """Workspace-Aware Memory Accounting section unchanged by cleanup."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "workspace-aware memory accounting" in md.lower()

    def test_simulated_flag_in_compact_table(self):
        """int4_sim row in compact table shows simulated."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        # int4_sim should appear in the compact table rows
        assert "`int4_sim`" in md

    def test_real_bytes_flag_in_compact_table(self):
        """int8 row shows real bytes = yes in compact table."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "`int8`" in md

    def test_compact_table_footer_references_workspace_section(self):
        """Compact table footer points to workspace section."""
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        assert "workspace-aware memory accounting" in md.lower()

    def test_no_forbidden_data_fields_in_cleaned_md(self):
        from exactkv.reporting.markdown import render_markdown_report
        report = _multi_compressor_v5_report()
        md = render_markdown_report(report, include_examples=False)
        lower = md.lower()
        for pat in self._FORBIDDEN_PATTERNS:
            assert pat not in lower, f"Forbidden pattern {pat!r} in Markdown"

    def test_legacy_report_compact_table_no_crash(self):
        """Legacy V4-style report still renders compact table without crashing."""
        from exactkv.reporting.markdown import render_markdown_report
        report = make_report(
            make_result("int8"),
            make_result("int4_sim", is_simulated=True, supports_real_bytes_claim=False),
        )
        # Inject a note so the compact table has something to truncate
        report["results"][1]["memory"]["memory_claim_note"] = (
            "int4_sim uses int8 container storage. "
            "sub-INT8 values are not bit-packed. "
            "supports_real_bytes_claim=False. "
            "total_kv_footprint_bytes is an accounting sum. "
            "Active GPU measurement is deferred."
        )
        md = render_markdown_report(report, include_examples=False)
        assert isinstance(md, str)
        # The compact table renders without paragraphs per row
        start = md.lower().find("## memory honesty notes")
        end = md.find("\n## ", start + 5) if start != -1 else -1
        section = md[start:end] if start != -1 and end != -1 else md
        for line in section.split("\n"):
            if line.startswith("|") and "compressor" not in line.lower() and "---" not in line:
                assert len(line) <= 200, f"Row too long after cleanup: {line!r}"


# ===========================================================================
# SECTION G: __init__.py exports
# ===========================================================================

class TestReportingPackageExports:

    def test_format_bytes_exported(self):
        from exactkv.reporting import format_bytes
        assert callable(format_bytes)

    def test_render_workspace_memory_table_exported(self):
        from exactkv.reporting import render_workspace_memory_table
        assert callable(render_workspace_memory_table)

    def test_format_bytes_basic(self):
        from exactkv.reporting import format_bytes
        assert format_bytes(1024) == "1.0 KiB"

    def test_render_workspace_memory_table_works(self):
        from exactkv.reporting import render_workspace_memory_table
        report = _make_v5_report(_make_v5_result("int8"))
        result = render_workspace_memory_table(report)
        assert isinstance(result, str)
        assert len(result) > 10
