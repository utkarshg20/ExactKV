"""Shard external-drafter stress panel helpers (Experiment 039).

Mode B external probe only — not default registry, not vendored Shard.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from exactkv.external.shard_probe import CLAIMS_ALLOWED, CLAIMS_FORBIDDEN

EXPERIMENT_ID = "039_shard_external_stress_panel"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_DRAFT_LEN = 4

# Documented Shard Cache knobs (from krish1905/shard Cache / Cache.from_model).
DEFAULT_SHARD_SETTINGS: dict[str, Any] = {
    "streaming": True,
    "stream_bits": 8,
    "stream_qjl": False,
    "k_target_cr": 16.0,
    "notes": (
        "stream_bits=8 is documented as lossless streaming in Shard; "
        "k_target_cr passed to Cache.from_model; prefill PCA+VQ uses defaults."
    ),
}

STRESS_CATEGORY_ORDER = (
    "structured_json",
    "retrieval_copy",
    "long_context_summary",
    "code_completion",
    "longbench_style_qa",
    "tool_call_json",
    "instruction_constraints",
    "multilingual",
)

REQUIRED_REPORT_KEYS = frozenset({
    "experiment_id",
    "panel_status",
    "blocked_reason",
    "shard_repo_path_present",
    "shard_import_success",
    "model_used",
    "max_new_tokens",
    "draft_len",
    "shard_settings",
    "tokenizer_alignment_pass",
    "prompt_count",
    "blocked_prompt_count",
    "divergence_count",
    "exactkv_failures",
    "accepted_prefix_lengths",
    "first_divergence_indices",
    "accepted_prefix_distribution",
    "divergence_examples",
    "no_divergence_observed",
    "prompt_results",
    "notes",
    "claims_allowed",
    "claims_forbidden",
    "recommendation",
})

VALID_PANEL_STATUSES = frozenset({"pass", "blocked", "restricted_no_go"})

_PROMPTS_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "prompts"

_LONGBENCH_STYLE_QA: list[dict[str, str]] = [
    {
        "prompt_id": "lb_sp_001",
        "category": "longbench_style_qa",
        "prompt": (
            "Document excerpt:\n"
            "Shard compresses Llama KV caches using PCA on keys and VQ on values. "
            "ExactKV verifies lossy draft tokens against a full-KV greedy reference.\n"
            "Question: What compression methods does Shard use on keys and values? "
            "Answer in one sentence."
        ),
    },
    {
        "prompt_id": "lb_sp_002",
        "category": "longbench_style_qa",
        "prompt": (
            "Policy excerpt:\n"
            "Enterprise refunds are allowed within 30 days of purchase for annual plans. "
            "After 30 days only exchange credits apply.\n"
            "Question: How many days after purchase can an annual plan customer request a refund?\n"
        ),
    },
    {
        "prompt_id": "lb_sp_003",
        "category": "longbench_style_qa",
        "prompt": (
            "Meeting notes:\n"
            "- Billing migration incomplete\n"
            "- Launch owner: Priya\n"
            "- Next checkpoint: September 10\n"
            "Question: Who owns the launch? Answer with the name only."
        ),
    },
    {
        "prompt_id": "lb_sp_004",
        "category": "longbench_style_qa",
        "prompt": (
            "Support transcript:\n"
            "Customer reports SSO failures in staging. Renewal risk is medium. "
            "Maya will send the runbook.\n"
            "Question: What is the renewal risk level? Answer with one word."
        ),
    },
]

_INSTRUCTION_CONSTRAINTS: list[dict[str, str]] = [
    {
        "prompt_id": "ic_001",
        "category": "instruction_constraints",
        "prompt": (
            "Reply with exactly three bullet points. Each bullet must start with '- '. "
            "Topic: why exact token verification matters for lossy KV caches. "
            "Do not add a title or preamble."
        ),
    },
    {
        "prompt_id": "ic_002",
        "category": "instruction_constraints",
        "prompt": (
            "Output only a JSON object with keys action, model, and max_tokens. "
            "Use string values. No markdown fences."
        ),
    },
    {
        "prompt_id": "ic_003",
        "category": "instruction_constraints",
        "prompt": (
            "Write exactly two sentences. First sentence must mention 'full-KV verifier'. "
            "Second sentence must mention 'external drafter'. No extra sentences."
        ),
    },
    {
        "prompt_id": "ic_004",
        "category": "instruction_constraints",
        "prompt": (
            "Complete the list using lowercase words only, comma-separated, no spaces after commas: "
            "acceptance,first_divergence,"
        ),
    },
]


def _normalize_entry(raw: dict[str, Any], *, default_category: str) -> dict[str, str]:
    prompt_id = str(raw.get("prompt_id") or raw.get("id") or "")
    if not prompt_id:
        raise ValueError("prompt entry missing prompt_id/id")
    category = str(raw.get("panel_category") or raw.get("category") or raw.get("primary_category") or default_category)
    prompt = str(raw.get("prompt") or "")
    if not prompt:
        raise ValueError(f"{prompt_id}: missing prompt text")
    return {"prompt_id": prompt_id, "category": category, "prompt": prompt}


def _load_jsonl(path: Path, *, default_category: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(_normalize_entry(json.loads(line), default_category=default_category))
    return rows


def _take_sorted(rows: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: r["prompt_id"])[:n]


def build_stress_prompt_panel(*, per_category: int = 4, max_prompts: int = 48) -> list[dict[str, str]]:
    """Deterministic stress panel across eight categories (default 4 each → 32 prompts)."""
    structured = _load_jsonl(_PROMPTS_ROOT / "structured.jsonl", default_category="structured_json")
    structured_json_only = [
        r for r in structured
        if "json" in r["category"].lower() and "function" not in r["category"].lower()
    ]
    function_calls = [r for r in structured if "function" in r["category"].lower()]

    retrieval = _load_jsonl(_PROMPTS_ROOT / "retrieval_copy.jsonl", default_category="retrieval_copy")
    long_ctx = (
        _load_jsonl(_PROMPTS_ROOT / "long_context.jsonl", default_category="long_context_summary")
        + [r for r in _load_jsonl(_PROMPTS_ROOT / "stress.jsonl", default_category="long_context_summary")
           if "long_context" in r["category"]]
    )
    code = _load_jsonl(_PROMPTS_ROOT / "code.jsonl", default_category="code_completion")
    tool = _load_jsonl(_PROMPTS_ROOT / "tool_json.jsonl", default_category="tool_call_json") + function_calls
    multilingual = _load_jsonl(_PROMPTS_ROOT / "multilingual.jsonl", default_category="multilingual")

    buckets: dict[str, list[dict[str, str]]] = {
        "structured_json": _take_sorted(structured_json_only, per_category),
        "retrieval_copy": _take_sorted(retrieval, per_category),
        "long_context_summary": _take_sorted(long_ctx, per_category),
        "code_completion": _take_sorted(code, per_category),
        "longbench_style_qa": _LONGBENCH_STYLE_QA[:per_category],
        "tool_call_json": _take_sorted(tool, per_category),
        "instruction_constraints": _INSTRUCTION_CONSTRAINTS[:per_category],
        "multilingual": _take_sorted(multilingual, per_category),
    }

    panel: list[dict[str, str]] = []
    seen: set[str] = set()
    for cat in STRESS_CATEGORY_ORDER:
        for row in buckets.get(cat, []):
            if row["prompt_id"] in seen:
                continue
            seen.add(row["prompt_id"])
            panel.append({**row, "panel_category": cat})
            if len(panel) >= max_prompts:
                return panel
    return panel


def classify_divergence_kind(
    *,
    draft_text: str,
    verifier_text: str,
    draft_token_id: int | None,
    verifier_token_id: int | None,
) -> str:
    """Coarse divergence label for reporting (not semantic evaluation)."""
    if draft_token_id is None and verifier_token_id is None:
        return "none"
    if draft_text.strip() == verifier_text.strip():
        return "whitespace_or_decode_artifact"
    punct = set('{}[]":,._-\\/ \t\n')
    d0 = draft_text[:1]
    v0 = verifier_text[:1]
    if d0 in punct or v0 in punct:
        return "formatting_or_punctuation"
    if draft_text.lower() == verifier_text.lower():
        return "casing"
    return "semantic_or_token_mismatch"


def prefix_distribution(lengths: list[int | None]) -> dict[str, Any]:
    valid = [x for x in lengths if x is not None]
    if not valid:
        return {"count": 0, "min": None, "max": None, "mean": None, "histogram": {}}
    hist = dict(sorted(Counter(valid).items()))
    return {
        "count": len(valid),
        "min": min(valid),
        "max": max(valid),
        "mean": round(sum(valid) / len(valid), 2),
        "histogram": hist,
    }


def build_divergence_examples(
    prompt_results: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in prompt_results:
        cmp = row.get("comparison") or {}
        div_idx = cmp.get("first_divergence_index")
        if div_idx is None:
            continue
        examples.append({
            "prompt_id": row.get("prompt_id"),
            "panel_category": row.get("panel_category"),
            "first_divergence_index": div_idx,
            "accepted_prefix_length": cmp.get("accepted_prefix_length"),
            "draft_token_id": cmp.get("draft_token_id"),
            "verifier_token_id": cmp.get("verifier_token_id"),
            "draft_token_text": cmp.get("draft_token_text"),
            "verifier_token_text": cmp.get("verifier_token_text"),
            "divergence_kind": cmp.get("divergence_kind"),
            "decoded_draft_prefix": cmp.get("decoded_draft_prefix"),
            "decoded_verifier_prefix": cmp.get("decoded_verifier_prefix"),
        })
        if len(examples) >= limit:
            break
    return examples


def build_panel_report(
    *,
    panel_status: str,
    blocked_reason: str,
    shard_repo_path_present: bool,
    shard_import_success: bool,
    model_used: str | None,
    max_new_tokens: int,
    draft_len: int,
    shard_settings: dict[str, Any],
    tokenizer_alignment_pass: bool,
    prompt_results: list[dict[str, Any]],
    notes: list[str],
    recommendation: str | None = None,
) -> dict[str, Any]:
    if panel_status not in VALID_PANEL_STATUSES:
        raise ValueError(f"invalid panel_status: {panel_status}")

    blocked_prompt_count = sum(1 for r in prompt_results if r.get("blocked"))
    aligned = [r for r in prompt_results if r.get("token_alignment_pass") and r.get("comparison")]
    accepted = [r["comparison"]["accepted_prefix_length"] for r in aligned]
    divs = [r["comparison"]["first_divergence_index"] for r in aligned]
    divergence_count = sum(1 for d in divs if d is not None)
    exactkv_failures = sum(1 for r in aligned if r.get("exactkv_failure"))

    report = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_class": "shard_external_drafter_stress_panel",
        "integration_mode": "mode_b_external_drafter",
        "not_default_registry": True,
        "not_kvcompressor_backend": True,
        "panel_status": panel_status,
        "blocked_reason": blocked_reason,
        "shard_repo_path_present": shard_repo_path_present,
        "shard_import_success": shard_import_success,
        "model_used": model_used,
        "max_new_tokens": max_new_tokens,
        "draft_len": draft_len,
        "shard_settings": shard_settings,
        "tokenizer_alignment_pass": tokenizer_alignment_pass,
        "prompt_count": len(prompt_results),
        "blocked_prompt_count": blocked_prompt_count,
        "divergence_count": divergence_count,
        "exactkv_failures": exactkv_failures if aligned else None,
        "accepted_prefix_lengths": accepted,
        "first_divergence_indices": divs,
        "accepted_prefix_distribution": prefix_distribution(accepted),
        "divergence_examples": build_divergence_examples(prompt_results),
        "no_divergence_observed": divergence_count == 0 and len(aligned) > 0,
        "prompt_results": prompt_results,
        "notes": notes,
        "claims_allowed": list(CLAIMS_ALLOWED) + [
            "Stress panel may report divergence without treating it as probe failure.",
            "exactkv_failures counts verified output mismatches vs full-KV greedy, not draft drift.",
        ],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": recommendation,
    }
    validate_panel_report(report)
    return report


def validate_panel_report(report: dict[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise ValueError(f"report missing keys: {sorted(missing)}")
    if report["panel_status"] not in VALID_PANEL_STATUSES:
        raise ValueError(f"invalid panel_status: {report['panel_status']}")
    if report["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must be 039_shard_external_stress_panel")
