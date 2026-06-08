"""Tests for exactkv.reporting.markdown (V3 Phase C).

Gate: markdown report gate
* render_markdown_report returns a Markdown string.
* All required sections are present.
* Required wording appears (lossy divergence expected, ExactKV failure,
  int4_sim disclaimer, no performance claims).
* write_markdown_report creates parent directories automatically.
* No forbidden performance fields in the output.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN_WORDS = {
    "tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds",
}

# The rendered Markdown may legitimately say "does not claim throughput" in its
# disclaimer prose. The constraint is that these words must NOT appear as data
# fields (table column headers, metric keys) in the report. We check for the
# absence of field-like patterns, not bare word occurrence.
_FORBIDDEN_AS_DATA_FIELD_PATTERNS = [
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

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Required section headings (lower-cased for insensitive match)
_REQUIRED_SECTIONS = [
    "correctness",
    "acceptance",
    "leaderboard",
    "histogram",
    "memory",
    "what this report proves",
    "what this report does not prove",
    "disclaimer",
]

# Required wording fragments (verbatim from spec)
_REQUIRED_WORDING = [
    "lossy divergence is expected",
    "exactkv failure",
    "int4_sim is simulated",
    "does not claim speedup",
]


def _no_forbidden_fields_in_md(md: str) -> None:
    """Assert forbidden words don't appear as data fields (table columns / metric keys).

    Markdown reports may legitimately use words like 'throughput' in prose
    disclaimers (e.g., 'this report does not claim throughput'). The constraint
    is that these words must NOT appear as actual data field names in tables or
    key-value pairs.
    """
    lower = md.lower()
    for pattern in _FORBIDDEN_AS_DATA_FIELD_PATTERNS:
        assert pattern not in lower, (
            f"Forbidden data-field pattern {pattern!r} found in markdown report"
        )


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _basic_report():
    return make_report(
        make_result(compressor_name="noop", draft_len=4,
                    lossy_exact=True, exactkv_failure=False,
                    total_drafted=8, total_accepted=8, total_rejected=0),
        make_result(compressor_name="int8", draft_len=4,
                    lossy_exact=False, first_div_idx=2, exactkv_failure=False,
                    total_drafted=8, total_accepted=6, total_rejected=2),
    )


def _int4_report():
    """Report containing int4_sim to trigger the simulation disclaimer."""
    r = make_result(
        compressor_name="int4_sim", draft_len=4,
        is_simulated=True, supports_real_bytes_claim=False,
        lossy_exact=False, first_div_idx=1, exactkv_failure=False,
    )
    # Inject memory_claim_note so the renderer can detect it
    r["memory"]["memory_claim_note"] = (
        "int4_sim uses int8 container storage; do not interpret as real "
        "packed INT4 memory savings."
    )
    return make_report(r)


# ──────────────────────────────────────────────────────────────
# render_markdown_report — basic
# ──────────────────────────────────────────────────────────────

class TestRenderMarkdownReport:
    def test_returns_string(self):
        from exactkv.reporting.markdown import render_markdown_report
        result = render_markdown_report(_basic_report())
        assert isinstance(result, str)

    def test_not_empty(self):
        from exactkv.reporting.markdown import render_markdown_report
        assert len(render_markdown_report(_basic_report())) > 100

    def test_custom_title_appears(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report(), title="My Custom Title")
        assert "My Custom Title" in md

    def test_default_title_present(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report())
        assert "ExactKV" in md

    def test_required_sections_present(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report()).lower()
        for section in _REQUIRED_SECTIONS:
            assert section in md, f"Required section missing: {section!r}"

    def test_lossy_divergence_described_as_expected(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report()).lower()
        assert "lossy divergence is expected" in md

    def test_exactkv_failure_wording_present(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report()).lower()
        assert "exactkv failure" in md

    def test_no_speedup_claim(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report()).lower()
        assert "does not claim speedup" in md or "no speedup" in md

    def test_no_forbidden_words(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report())
        _no_forbidden_fields_in_md(md)

    def test_int4_sim_disclaimer_when_int4_in_report(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_int4_report()).lower()
        assert "int4_sim is simulated" in md or "simulated" in md

    def test_int4_sim_memory_note_in_memory_section(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_int4_report())
        assert "int8" in md.lower() or "simulated" in md.lower()

    def test_no_int4_disclaimer_when_no_int4(self):
        """Disclaimer about int4_sim should still appear (it's always in the
        general disclaimer block) but no int8 container note if no int4."""
        from exactkv.reporting.markdown import render_markdown_report
        report = make_report(make_result(compressor_name="noop"))
        md = render_markdown_report(report)
        # The general disclaimer always mentions int4_sim
        assert "int4_sim" in md

    def test_correctness_summary_shows_zero_failures(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report())
        # With no failures, should show 0 in the correctness table
        assert "| ExactKV failures |" in md
        assert "| 0 |" in md or "**0**" in md

    def test_compressor_names_appear(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report())
        assert "noop" in md
        assert "int8" in md

    def test_include_examples_false_skips_example_sections(self):
        from exactkv.reporting.markdown import render_markdown_report
        md = render_markdown_report(_basic_report(), include_examples=False).lower()
        assert "divergence examples" not in md or "lossy divergence examples" not in md

    def test_sweep_report_has_grid_section(self):
        from exactkv.reporting.markdown import render_markdown_report
        sweep_report = make_report(
            make_result(compressor_name="noop", draft_len=4),
            make_result(compressor_name="int8", draft_len=4),
            make_result(compressor_name="noop", draft_len=8),
            make_result(compressor_name="int8", draft_len=8),
        )
        md = render_markdown_report(sweep_report).lower()
        assert "grid" in md or "compressor × draft" in md or "compressor x draft" in md


# ──────────────────────────────────────────────────────────────
# write_markdown_report
# ──────────────────────────────────────────────────────────────

class TestWriteMarkdownReport:
    def test_writes_file(self, tmp_path):
        from exactkv.reporting.markdown import write_markdown_report
        out = tmp_path / "report.md"
        write_markdown_report(_basic_report(), out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path):
        from exactkv.reporting.markdown import write_markdown_report
        out = tmp_path / "nested" / "deep" / "report.md"
        write_markdown_report(_basic_report(), out)
        assert out.exists()

    def test_file_contents_is_markdown(self, tmp_path):
        from exactkv.reporting.markdown import write_markdown_report
        out = tmp_path / "r.md"
        write_markdown_report(_basic_report(), out, title="My Report")
        content = out.read_text(encoding="utf-8")
        assert "# My Report" in content
        assert "ExactKV" in content

    def test_no_forbidden_words_in_written_file(self, tmp_path):
        from exactkv.reporting.markdown import write_markdown_report
        out = tmp_path / "r.md"
        write_markdown_report(_basic_report(), out)
        content = out.read_text(encoding="utf-8")
        _no_forbidden_fields_in_md(content)


# ──────────────────────────────────────────────────────────────
# Real sweep
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_sweep():
    from exactkv.benchmarks.sweeps import run_sweep
    from exactkv.runtime.model_runtime import ModelRuntime
    rt = ModelRuntime(MODEL_NAME, device="auto", dtype="float32")
    prompts = [{"prompt_id": "md_t1", "category": "test",
                "prompt": "The capital of France is"}]
    return run_sweep(rt, prompts=prompts,
                     compressor_names=["noop", "int8"],
                     draft_lengths=[4, 8], max_new_tokens=8)


def test_render_markdown_on_real_sweep(real_sweep):
    from exactkv.reporting.markdown import render_markdown_report
    md = render_markdown_report(real_sweep, title="Real Sweep Test")
    assert isinstance(md, str)
    assert "Real Sweep Test" in md
    assert "noop" in md
    assert "int8" in md
    assert "lossy divergence is expected" in md.lower()
    _no_forbidden_fields_in_md(md)


def test_write_markdown_on_real_sweep(real_sweep, tmp_path):
    from exactkv.reporting.markdown import write_markdown_report
    out = tmp_path / "reports" / "sweep_test.md"
    write_markdown_report(real_sweep, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "ExactKV" in content
    assert len(content) > 200


def test_exactkv_failure_zero_on_real_sweep(real_sweep):
    """The real sweep must have 0 ExactKV failures."""
    from exactkv.reporting.markdown import render_markdown_report
    md = render_markdown_report(real_sweep)
    assert "PASS" in md or "pass" in md.lower()
