"""Shard external-drafter combined stress helpers (Experiment 041).

Mode B external probe only — not default registry, not vendored Shard.
"""
from __future__ import annotations

from typing import Any

from exactkv.external.shard_ablation import (
    FORMATTING_DIVERGENCE_KINDS,
    SEMANTIC_DIVERGENCE_KINDS,
    summarize_setting_result,
)
from exactkv.external.shard_probe import CLAIMS_ALLOWED, CLAIMS_FORBIDDEN
from exactkv.external.shard_stress_panel import DEFAULT_DRAFT_LEN

EXPERIMENT_ID = "041_shard_combined_stress"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 128

COMBINED_SHARD_SETTINGS: dict[str, Any] = {
    "streaming": True,
    "stream_bits": 4,
    "stream_qjl": False,
    "k_target_cr": 16.0,
    "label": "Combined lossy streaming (stream_bits=4) + long generation (128 tok)",
}

# Exp 040 reference rows for comparison (from RunPod ablation report).
EXP040_COMPARISON_BASELINE: dict[str, Any] = {
    "setting_name": "baseline_64tok",
    "max_new_tokens": 64,
    "stream_bits": 8,
    "divergence_count": 6,
    "divergence_rate": 0.1875,
    "accepted_prefix_mean": 58.22,
    "exactkv_failures": 0,
}

EXP040_COMPARISON_LENGTH_128: dict[str, Any] = {
    "setting_name": "length_128tok",
    "max_new_tokens": 128,
    "stream_bits": 8,
    "divergence_count": 10,
    "divergence_rate": 0.3125,
    "accepted_prefix_mean": 104.62,
    "exactkv_failures": 0,
}

EXP040_COMPARISON_STREAM_BITS_4: dict[str, Any] = {
    "setting_name": "stream_bits_4",
    "max_new_tokens": 64,
    "stream_bits": 4,
    "divergence_count": 8,
    "divergence_rate": 0.25,
    "accepted_prefix_mean": 56.59,
    "exactkv_failures": 0,
}

EXP040_COMPARISONS = (
    EXP040_COMPARISON_BASELINE,
    EXP040_COMPARISON_LENGTH_128,
    EXP040_COMPARISON_STREAM_BITS_4,
)

REQUIRED_REPORT_KEYS = frozenset({
    "experiment_id",
    "combined_status",
    "blocked_reason",
    "shard_repo_path_present",
    "shard_import_success",
    "model_used",
    "max_new_tokens",
    "draft_len",
    "shard_settings",
    "prompt_count",
    "blocked_prompt_count",
    "tokenizer_alignment_pass",
    "exactkv_failures",
    "accepted_prefix_lengths",
    "accepted_prefix_mean",
    "accepted_prefix_median",
    "accepted_prefix_min",
    "accepted_prefix_distribution",
    "divergence_count",
    "divergence_rate",
    "semantic_divergence_count",
    "formatting_divergence_count",
    "divergence_examples",
    "exp040_comparison",
    "combined_vs_exp040",
    "notes",
    "claims_allowed",
    "claims_forbidden",
    "recommendation",
})

VALID_COMBINED_STATUSES = frozenset({"pass", "blocked", "restricted_no_go"})


def build_exp040_comparison_summary(result: dict[str, Any]) -> dict[str, Any]:
    div_rate = result.get("divergence_rate")
    div_count = result.get("divergence_count")
    mean_prefix = result.get("accepted_prefix_mean")
    return {
        "combined_divergence_rate": div_rate,
        "combined_divergence_count": div_count,
        "combined_accepted_prefix_mean": mean_prefix,
        "vs_baseline_64tok": {
            "divergence_rate_delta": (
                None if div_rate is None else round(div_rate - EXP040_COMPARISON_BASELINE["divergence_rate"], 4)
            ),
            "divergence_count_delta": (
                None if div_count is None else div_count - EXP040_COMPARISON_BASELINE["divergence_count"]
            ),
        },
        "vs_length_128tok": {
            "divergence_rate_delta": (
                None if div_rate is None else round(div_rate - EXP040_COMPARISON_LENGTH_128["divergence_rate"], 4)
            ),
            "divergence_count_delta": (
                None if div_count is None else div_count - EXP040_COMPARISON_LENGTH_128["divergence_count"]
            ),
        },
        "vs_stream_bits_4": {
            "divergence_rate_delta": (
                None if div_rate is None else round(div_rate - EXP040_COMPARISON_STREAM_BITS_4["divergence_rate"], 4)
            ),
            "divergence_count_delta": (
                None if div_count is None else div_count - EXP040_COMPARISON_STREAM_BITS_4["divergence_count"]
            ),
        },
        "increased_vs_all_exp040_singles": (
            div_rate is not None
            and div_rate > EXP040_COMPARISON_BASELINE["divergence_rate"]
            and div_rate > EXP040_COMPARISON_LENGTH_128["divergence_rate"]
            and div_rate > EXP040_COMPARISON_STREAM_BITS_4["divergence_rate"]
        ),
        "increased_vs_length_128_or_stream_bits_4": (
            div_rate is not None
            and (
                div_rate > EXP040_COMPARISON_LENGTH_128["divergence_rate"]
                or div_rate > EXP040_COMPARISON_STREAM_BITS_4["divergence_rate"]
            )
        ),
    }


def build_combined_report(
    *,
    combined_status: str,
    blocked_reason: str,
    shard_repo_path_present: bool,
    shard_import_success: bool,
    model_used: str | None,
    max_new_tokens: int,
    draft_len: int,
    shard_settings: dict[str, Any],
    result: dict[str, Any],
    notes: list[str],
    recommendation: str | None = None,
) -> dict[str, Any]:
    if combined_status not in VALID_COMBINED_STATUSES:
        raise ValueError(f"invalid combined_status: {combined_status}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_class": "shard_external_drafter_combined_stress",
        "integration_mode": "mode_b_external_drafter",
        "not_default_registry": True,
        "not_kvcompressor_backend": True,
        "combined_status": combined_status,
        "blocked_reason": blocked_reason,
        "shard_repo_path_present": shard_repo_path_present,
        "shard_import_success": shard_import_success,
        "model_used": model_used,
        "max_new_tokens": max_new_tokens,
        "draft_len": draft_len,
        "shard_settings": shard_settings,
        "prompt_count": result.get("prompt_count", 0),
        "blocked_prompt_count": result.get("blocked_prompt_count", 0),
        "tokenizer_alignment_pass": result.get("tokenizer_alignment_pass", False),
        "exactkv_failures": result.get("exactkv_failures"),
        "accepted_prefix_lengths": [
            r.get("comparison", {}).get("accepted_prefix_length")
            for r in result.get("prompt_results", [])
            if r.get("comparison")
        ],
        "accepted_prefix_mean": result.get("accepted_prefix_mean"),
        "accepted_prefix_median": result.get("accepted_prefix_median"),
        "accepted_prefix_min": result.get("accepted_prefix_min"),
        "accepted_prefix_distribution": result.get("accepted_prefix_distribution", {}),
        "divergence_count": result.get("divergence_count", 0),
        "divergence_rate": result.get("divergence_rate"),
        "semantic_divergence_count": result.get("semantic_divergence_count", 0),
        "formatting_divergence_count": result.get("formatting_divergence_count", 0),
        "divergence_examples": result.get("divergence_examples", []),
        "exp040_comparison": list(EXP040_COMPARISONS),
        "combined_vs_exp040": build_exp040_comparison_summary(result),
        "notes": notes,
        "claims_allowed": list(CLAIMS_ALLOWED) + [
            "Combined stress may report draft divergence without treating it as probe failure.",
            "exactkv_failures counts verified output mismatches vs full-KV greedy.",
        ],
        "claims_forbidden": list(CLAIMS_FORBIDDEN) + [
            "Do not call stream_bits=4 lossless — 8 is documented lossless streaming in Shard.",
        ],
        "recommendation": recommendation,
    }
    validate_combined_report(report)
    return report


def validate_combined_report(report: dict[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise ValueError(f"report missing keys: {sorted(missing)}")
    if report["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must be 041_shard_combined_stress")
    if report["combined_status"] not in VALID_COMBINED_STATUSES:
        raise ValueError(f"invalid combined_status: {report['combined_status']}")
    if "exactkv_failures" not in report:
        raise ValueError("exactkv_failures required")
    if not isinstance(report["exp040_comparison"], list) or len(report["exp040_comparison"]) != 3:
        raise ValueError("exp040_comparison must list three Exp040 reference settings")
    vs = report.get("combined_vs_exp040") or {}
    for key in ("vs_baseline_64tok", "vs_length_128tok", "vs_stream_bits_4"):
        if key not in vs:
            raise ValueError(f"combined_vs_exp040 missing {key}")


def summarize_combined_from_prompt_results(
    prompt_results: list[dict[str, Any]],
    *,
    status: str = "pass",
    blocked_reason: str = "",
) -> dict[str, Any]:
    row = summarize_setting_result(
        setting_name="combined_stream_bits_4_128tok",
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        shard_settings=COMBINED_SHARD_SETTINGS,
        prompt_results=prompt_results,
        status=status,
        blocked_reason=blocked_reason,
    )
    row["prompt_results"] = prompt_results
    return row
