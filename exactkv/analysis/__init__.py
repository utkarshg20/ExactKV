"""ExactKV analysis package.

Operates on existing benchmark and sweep reports without re-running the model.
No timing, latency, throughput, or speedup fields are produced by any function.

Modules
-------
acceptance_tables   Build and export acceptance-rate summaries grouped by
                    compressor, draft length, or prompt category.
mismatch            Analyse where lossy divergences and ExactKV rejections occur.
failure_report      Classify and report ExactKV failures vs. expected lossy
                    divergences.
histograms          Compute accepted-length, first-divergence, and rejection-count
                    histograms from existing reports.  (V3)
examples            Extract concrete lossy-divergence, ExactKV-failure, and
                    rejection examples from existing reports.  (V3)
attention_weighted  Proxy divergence analysis for V7 Phase A; uses report fields
                    only unless attention weights are logged.  (V7)

Key distinction
---------------
*Lossy divergence* (``lossy.token_exact_match == False``) is expected and
is not a failure.  It proves that the compressor changes the output and
demonstrates why verification is necessary.

*ExactKV failure* (``exactkv_failure == True``) means the ExactKV loop
produced output that did **not** match ``generate_full_greedy``.  This is
a bug and must always be zero in a correct implementation.
"""
from exactkv.analysis.acceptance_tables import (
    build_acceptance_table,
    group_acceptance_by_category,
    group_acceptance_by_compressor,
    group_acceptance_by_draft_len,
    write_acceptance_table_csv,
)
from exactkv.analysis.examples import (
    extract_exactkv_failure_examples,
    extract_lossy_divergence_examples,
    extract_rejection_examples,
)
from exactkv.analysis.failure_report import (
    build_failure_report,
    list_exactkv_failures,
    list_lossy_divergences,
    write_failure_report_json,
)
from exactkv.analysis.histograms import (
    DEFAULT_ACCEPTED_BUCKETS,
    DEFAULT_DIVERGENCE_BUCKETS,
    DEFAULT_REJECTION_BUCKETS,
    accepted_length_histogram,
    first_divergence_histogram,
    rejection_count_histogram,
)
from exactkv.analysis.attention_weighted import (
    acceptance_vs_divergence_summary,
    compare_reports_for_divergence,
    divergence_by_compressor,
    divergence_position_table,
    has_attention_weights,
    proxy_analysis_metadata,
    rejection_by_compressor,
)
from exactkv.analysis.mismatch import (
    first_lossy_divergences,
    mismatch_position_summary,
    rejection_position_summary,
)

__all__ = [
    # acceptance_tables
    "build_acceptance_table",
    "group_acceptance_by_compressor",
    "group_acceptance_by_draft_len",
    "group_acceptance_by_category",
    "write_acceptance_table_csv",
    # mismatch
    "first_lossy_divergences",
    "mismatch_position_summary",
    "rejection_position_summary",
    # failure_report
    "build_failure_report",
    "list_exactkv_failures",
    "list_lossy_divergences",
    "write_failure_report_json",
    # histograms (V3)
    "accepted_length_histogram",
    "first_divergence_histogram",
    "rejection_count_histogram",
    "DEFAULT_ACCEPTED_BUCKETS",
    "DEFAULT_DIVERGENCE_BUCKETS",
    "DEFAULT_REJECTION_BUCKETS",
    # examples (V3)
    "extract_lossy_divergence_examples",
    "extract_exactkv_failure_examples",
    "extract_rejection_examples",
    # attention_weighted (V7)
    "has_attention_weights",
    "proxy_analysis_metadata",
    "divergence_by_compressor",
    "rejection_by_compressor",
    "divergence_position_table",
    "acceptance_vs_divergence_summary",
    "compare_reports_for_divergence",
]
