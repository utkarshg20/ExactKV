"""Shard external-drafter ablation helpers (Experiment 040).

Mode B external probe only — not default registry, not vendored Shard.
"""
from __future__ import annotations

import statistics
from typing import Any

from exactkv.external.shard_probe import CLAIMS_ALLOWED, CLAIMS_FORBIDDEN
from exactkv.external.shard_stress_panel import (
    DEFAULT_DRAFT_LEN,
    DEFAULT_SHARD_SETTINGS,
    build_divergence_examples,
    prefix_distribution,
)

EXPERIMENT_ID = "040_shard_external_ablation"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Documented Shard Cache knobs (README + cache.py). stream_bits=8 is documented lossless.
BASELINE_SHARD_SETTINGS: dict[str, Any] = {
    "streaming": True,
    "stream_bits": 8,
    "stream_qjl": False,
    "k_target_cr": 16.0,
}

SEMANTIC_DIVERGENCE_KINDS = frozenset({
    "semantic_or_token_mismatch",
    "casing",
})

FORMATTING_DIVERGENCE_KINDS = frozenset({
    "formatting_or_punctuation",
    "whitespace_or_decode_artifact",
})

REQUIRED_REPORT_KEYS = frozenset({
    "experiment_id",
    "ablation_status",
    "blocked_reason",
    "shard_repo_path_present",
    "shard_import_success",
    "model_used",
    "draft_len",
    "prompt_count",
    "settings_tested",
    "settings_skipped",
    "setting_results",
    "notes",
    "claims_allowed",
    "claims_forbidden",
    "recommendation",
})

REQUIRED_SETTING_ROW_KEYS = frozenset({
    "setting_name",
    "max_new_tokens",
    "shard_settings",
    "status",
    "blocked_reason",
    "tokenizer_alignment_pass",
    "blocked_prompt_count",
    "prompt_count",
    "exactkv_failures",
    "accepted_prefix_mean",
    "accepted_prefix_median",
    "accepted_prefix_min",
    "accepted_prefix_distribution",
    "divergence_count",
    "divergence_rate",
    "semantic_divergence_count",
    "formatting_divergence_count",
    "divergence_examples",
})

VALID_ABLATION_STATUSES = frozenset({"pass", "blocked", "restricted_no_go"})
VALID_SETTING_STATUSES = frozenset({"pass", "blocked", "skipped", "restricted_no_go"})


def build_ablation_grid() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return (settings_to_test, settings_skipped_with_reason)."""
    tested = [
        {
            "setting_name": "baseline_64tok",
            "max_new_tokens": 64,
            "shard_settings": {
                **BASELINE_SHARD_SETTINGS,
                "label": "Exp039 baseline — documented-lossless streaming (stream_bits=8)",
            },
        },
        {
            "setting_name": "length_128tok",
            "max_new_tokens": 128,
            "shard_settings": {
                **BASELINE_SHARD_SETTINGS,
                "label": "Longer generation; same documented-lossless streaming",
            },
        },
        {
            "setting_name": "stream_bits_4",
            "max_new_tokens": 64,
            "shard_settings": {
                **BASELINE_SHARD_SETTINGS,
                "stream_bits": 4,
                "label": "Lossy decode streaming — stream_bits=4 (8 is documented lossless)",
            },
        },
        {
            "setting_name": "stream_qjl_on",
            "max_new_tokens": 64,
            "shard_settings": {
                **BASELINE_SHARD_SETTINGS,
                "stream_qjl": True,
                "label": "Optional QJL decode quantizer (streaming.py)",
            },
        },
        {
            "setting_name": "k_target_cr_32",
            "max_new_tokens": 64,
            "shard_settings": {
                **BASELINE_SHARD_SETTINGS,
                "k_target_cr": 32.0,
                "label": "Stronger prefill compression target via Cache.from_model",
            },
        },
    ]
    skipped = [
        {
            "setting_name": "stream_bits_2",
            "reason": "Not documented in Shard README; only stream_bits=8 shown as supported lossless streaming.",
        },
        {
            "setting_name": "custom_sink_window",
            "reason": "sink_tokens/residual_window are Cache.__init__ defaults without public from_model override in our probe path.",
        },
    ]
    return tested, skipped


def summarize_setting_result(
    *,
    setting_name: str,
    max_new_tokens: int,
    shard_settings: dict[str, Any],
    prompt_results: list[dict[str, Any]],
    status: str = "pass",
    blocked_reason: str = "",
) -> dict[str, Any]:
    aligned = [r for r in prompt_results if r.get("token_alignment_pass") and r.get("comparison")]
    blocked_prompt_count = sum(1 for r in prompt_results if r.get("blocked"))
    lengths = [r["comparison"]["accepted_prefix_length"] for r in aligned]
    div_rows = [
        r for r in aligned if r["comparison"].get("first_divergence_index") is not None
    ]
    semantic = sum(
        1
        for r in div_rows
        if r["comparison"].get("divergence_kind") in SEMANTIC_DIVERGENCE_KINDS
    )
    formatting = sum(
        1
        for r in div_rows
        if r["comparison"].get("divergence_kind") in FORMATTING_DIVERGENCE_KINDS
    )
    prompt_count = len(prompt_results)
    divergence_count = len(div_rows)
    dist = prefix_distribution(lengths)
    return {
        "setting_name": setting_name,
        "max_new_tokens": max_new_tokens,
        "shard_settings": shard_settings,
        "status": status,
        "blocked_reason": blocked_reason,
        "tokenizer_alignment_pass": len(aligned) == prompt_count and status == "pass",
        "blocked_prompt_count": blocked_prompt_count,
        "prompt_count": prompt_count,
        "exactkv_failures": sum(1 for r in aligned if r.get("exactkv_failure")),
        "accepted_prefix_mean": round(statistics.mean(lengths), 2) if lengths else None,
        "accepted_prefix_median": statistics.median(lengths) if lengths else None,
        "accepted_prefix_min": min(lengths) if lengths else None,
        "accepted_prefix_distribution": dist,
        "divergence_count": divergence_count,
        "divergence_rate": round(divergence_count / prompt_count, 4) if prompt_count else None,
        "semantic_divergence_count": semantic,
        "formatting_divergence_count": formatting,
        "divergence_examples": build_divergence_examples(prompt_results, limit=5),
        "prompt_results": prompt_results,
    }


def build_ablation_report(
    *,
    ablation_status: str,
    blocked_reason: str,
    shard_repo_path_present: bool,
    shard_import_success: bool,
    model_used: str | None,
    draft_len: int,
    prompt_count: int,
    settings_tested: list[str],
    settings_skipped: list[dict[str, str]],
    setting_results: list[dict[str, Any]],
    notes: list[str],
    recommendation: str | None = None,
) -> dict[str, Any]:
    if ablation_status not in VALID_ABLATION_STATUSES:
        raise ValueError(f"invalid ablation_status: {ablation_status}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_class": "shard_external_drafter_ablation",
        "integration_mode": "mode_b_external_drafter",
        "not_default_registry": True,
        "not_kvcompressor_backend": True,
        "ablation_status": ablation_status,
        "blocked_reason": blocked_reason,
        "shard_repo_path_present": shard_repo_path_present,
        "shard_import_success": shard_import_success,
        "model_used": model_used,
        "draft_len": draft_len,
        "prompt_count": prompt_count,
        "settings_tested": settings_tested,
        "settings_skipped": settings_skipped,
        "setting_results": setting_results,
        "notes": notes,
        "claims_allowed": list(CLAIMS_ALLOWED) + [
            "Ablation may compare draft divergence across documented Shard settings.",
            "stream_bits=8 is documented-lossless streaming; lower bits are lossy streaming.",
            "exactkv_failures counts verified output mismatches vs full-KV greedy.",
        ],
        "claims_forbidden": list(CLAIMS_FORBIDDEN) + [
            "Do not call stream_bits=8 lossy compression — Shard documents it as lossless streaming.",
        ],
        "recommendation": recommendation,
    }
    validate_ablation_report(report)
    return report


def validate_ablation_report(report: dict[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise ValueError(f"report missing keys: {sorted(missing)}")
    if report["ablation_status"] not in VALID_ABLATION_STATUSES:
        raise ValueError(f"invalid ablation_status: {report['ablation_status']}")
    if report["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must be 040_shard_external_ablation")
    for row in report["setting_results"]:
        validate_setting_row(row)


def validate_setting_row(row: dict[str, Any]) -> None:
    missing = REQUIRED_SETTING_ROW_KEYS - row.keys()
    if missing:
        raise ValueError(f"setting row missing keys: {sorted(missing)}")
    if row["status"] not in VALID_SETTING_STATUSES:
        raise ValueError(f"invalid setting status: {row['status']}")
    if "exactkv_failures" not in row:
        raise ValueError("exactkv_failures required on setting row")
