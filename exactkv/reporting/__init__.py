"""ExactKV V3 reporting package.

Converts existing benchmark/sweep report dicts into docs-ready Markdown.
No model re-runs.  No timing, latency, throughput, or speedup fields.

Modules
-------
markdown     Top-level renderer: ``render_markdown_report`` and
             ``write_markdown_report``.
leaderboard  Markdown tables for acceptance by compressor / draft_len / grid.
examples     Markdown renderers for lossy-divergence and rejection examples.
histograms   Text table renderers for accepted-length, divergence-position,
             and rejection-count histograms.

Key distinction (printed in every rendered report)
---------------------------------------------------
*Lossy divergence* is expected and is NOT an ExactKV failure.
*ExactKV failure* means the verified output differed from full-KV greedy
output — this is a correctness bug and must always be zero.
"""
from exactkv.reporting.markdown import render_markdown_report, write_markdown_report
from exactkv.reporting.leaderboard import (
    render_compressor_leaderboard,
    render_draft_len_leaderboard,
    render_compressor_x_draft_leaderboard,
)
from exactkv.reporting.examples import (
    render_lossy_divergence_examples,
    render_exactkv_failure_examples,
    render_rejection_examples,
)
from exactkv.reporting.histograms import (
    render_accepted_length_table,
    render_first_divergence_table,
    render_rejection_count_table,
)

__all__ = [
    "render_markdown_report",
    "write_markdown_report",
    "render_compressor_leaderboard",
    "render_draft_len_leaderboard",
    "render_compressor_x_draft_leaderboard",
    "render_lossy_divergence_examples",
    "render_exactkv_failure_examples",
    "render_rejection_examples",
    "render_accepted_length_table",
    "render_first_divergence_table",
    "render_rejection_count_table",
]
