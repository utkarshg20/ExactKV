#!/usr/bin/env python3
"""Experiment 034b: semantic correction demo search (V13 Phase 8e).

Searches for human-obvious structured-output corrections where lossy KV proposes a
wrong semantic token and ExactKV commits the full-KV correction with exact match.

Correctness search only — not a timing or GPU memory benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.runtime.model_runtime import ModelRuntime
from scripts.run_experiment_034_killer_correction_demo import (
    EXPERIMENT_CLASS,
    SEARCH_COMPRESSORS,
    SNAPKV_NAME,
    _decode_token,
    _kvpress_available,
    _strip_internal,
    enrich_demo_cell,
    run_search_cell,
)

MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B"
DRAFT_LENS = (4, 8)
MAX_NEW_TOKENS = 64
EXPERIMENT_CLASS_034B = "v13_semantic_correction_search"

SEARCH_SUITES = (
    "tool_json",
    "code_structured",
    "retrieval_copy",
)

CRAFTED_PROMPTS: list[dict[str, Any]] = [
    {
        "prompt_id": "ord_001",
        "v10_suite": "crafted_order",
        "category": "tool_schema",
        "prompt": (
            "You are an ordering agent. Return only a JSON tool call.\n\n"
            "User wants:\n- vegan burger\n- no onions\n- quantity 1\n- pickup\n\n"
            "Available item IDs:\nvegan_burger\nbeef_burger\nchicken_wrap\n\n"
            'Return:\n{"tool":"add_item","item_id":'
        ),
    },
    {
        "prompt_id": "ord_002",
        "v10_suite": "crafted_order",
        "category": "tool_schema",
        "prompt": (
            'Return JSON only: {"tool":"add_item","item_id":"vegan_burger",'
            '"quantity":1,"modifiers":["no_onions"],"fulfillment":"pickup","notes":'
        ),
    },
    {
        "prompt_id": "ord_003",
        "v10_suite": "crafted_order",
        "category": "tool_schema",
        "prompt": (
            "Complete tool call JSON for vegan burger pickup order:\n"
            '{"name":"place_order","arguments":{"item_id":'
        ),
    },
    {
        "prompt_id": "pharm_001",
        "v10_suite": "crafted_pharmacy",
        "category": "tool_schema",
        "prompt": (
            'JSON tool call only: {"tool":"refill_prescription","drug":"ibuprofen",'
            '"quantity":30,"pickup":'
        ),
    },
    {
        "prompt_id": "bank_001",
        "v10_suite": "crafted_banking",
        "category": "tool_schema",
        "prompt": (
            'Return JSON: {"action":"transfer","from_account":"checking",'
            '"to_account":"savings","amount":'
        ),
    },
    {
        "prompt_id": "cal_001",
        "v10_suite": "crafted_calendar",
        "category": "tool_schema",
        "prompt": (
            'JSON only: {"tool":"schedule_meeting","title":"Review","date":"2026-06-09",'
            '"time":"14:00","timezone":"America/New_York","attendees":["alice@corp.com"],'
            '"location":'
        ),
    },
]

_PUNCT_CHARS = frozenset('{}[]":,._-\\/ \t\n')

_SEMANTIC_PAIR_BONUS = (
    ("beef", "vegan"),
    ("delivery", "pickup"),
    ("dropoff", "pickup"),
    ("drop", "pickup"),
    ("london", "paris"),
    ("south", "north"),
    ("charge", "refund"),
    ("blue", "red"),
)

_FORBIDDEN = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden(obj: Any, path: str = "artifact") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")


def _core_token(text: str) -> str:
    return text.strip().strip('"').strip("'").strip()


def _is_punct_only(text: str) -> bool:
    core = _core_token(text)
    if not core:
        return True
    return all(c in _PUNCT_CHARS for c in core)


def _suite_priority(suite: str) -> float:
    return {
        "crafted_order": 1200.0,
        "crafted_pharmacy": 1150.0,
        "crafted_banking": 1100.0,
        "crafted_calendar": 1050.0,
        "tool_json": 1000.0,
        "code_structured": 800.0,
        "retrieval_copy": 700.0,
    }.get(suite, 0.0)


def score_semantic_candidate(cell: dict[str, Any]) -> float:
    """Higher = more human-obvious semantic correction."""
    if not cell.get("is_demo_candidate") or not cell.get("exactkv_exact_match"):
        return 0.0

    hr = cell.get("highlight_round") or {}
    rej_id = hr.get("first_rejected_token")
    corr_id = hr.get("correction_token")
    if rej_id is None or corr_id is None or rej_id == corr_id:
        return 0.0

    tokenizer = cell.get("_tokenizer")
    rej_text = hr.get("first_rejected_text")
    corr_text = hr.get("correction_text")
    if tokenizer is not None:
        if rej_text is None:
            rej_text = _decode_token(tokenizer, rej_id)
        if corr_text is None:
            corr_text = _decode_token(tokenizer, corr_id)
    rej_text = rej_text or ""
    corr_text = corr_text or ""

    score = _suite_priority(cell.get("v10_suite", ""))

    rej_core = _core_token(rej_text).lower()
    corr_core = _core_token(corr_text).lower()

    if _is_punct_only(rej_text) or _is_punct_only(corr_text):
        score -= 200.0
    else:
        score += 80.0

    if len(rej_core) >= 3 and rej_core.isalpha():
        score += 60.0
    if len(corr_core) >= 3 and corr_core.isalpha():
        score += 60.0
    if rej_core.isdigit() or corr_core.isdigit():
        score += 50.0

    for a, b in _SEMANTIC_PAIR_BONUS:
        if a in rej_core or b in corr_core:
            score += 90.0

    if rej_core and corr_core and rej_core != corr_core:
        score += 30.0

    div_idx = cell.get("lossy_first_divergence_idx")
    if div_idx is not None:
        if div_idx <= 3:
            score += 25.0
        elif div_idx <= 8:
            score += 10.0
        else:
            score -= 15.0

    lossy_text = cell.get("lossy_output_text", "")
    full_text = cell.get("full_output_text", "")
    if lossy_text and full_text and len(lossy_text) >= 20:
        score += 15.0

    if hr.get("round_idx") == 0:
        score += 20.0
    if cell.get("draft_len") == 4:
        score += 5.0

    return score


def load_search_prompts(*, include_v10: bool = True) -> list[dict[str, Any]]:
    prompts = [dict(p) for p in CRAFTED_PROMPTS]
    if include_v10:
        for suite in SEARCH_SUITES:
            for row in load_v10_suite(suite):
                prompts.append(row)
    return prompts


def rescore_from_exp034_json(
    report_path: Path,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cells: list[dict[str, Any]] = []
    for raw in report.get("all_cells", []):
        cell = dict(raw)
        cell["_tokenizer"] = tokenizer
        cell["semantic_score"] = score_semantic_candidate(cell)
        if cell["semantic_score"] > 0:
            cells.append(cell)
    return cells


def run_live_search(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    compressors: list[str],
    *,
    verification_method: str = "sequential",
) -> list[dict[str, Any]]:
    cache: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []
    for prompt_entry in prompts:
        for comp_name in compressors:
            if comp_name == SNAPKV_NAME:
                from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

                compressor = create_snapkv_experimental_adapter(runtime, compression_ratio=0.5)
            else:
                compressor = get_compressor(comp_name)
            cache[comp_name] = compressor
            for draft_len in DRAFT_LENS:
                cell = run_search_cell(
                    runtime,
                    prompt_entry,
                    compressor,
                    draft_len=draft_len,
                    verification_method=verification_method,
                )
                cell["semantic_score"] = score_semantic_candidate(cell)
                cells.append(cell)
    return cells


def select_best(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [c for c in cells if c.get("semantic_score", 0) > 0]
    if not candidates:
        return None
    candidates.sort(
        key=lambda c: (
            c.get("semantic_score", 0),
            -int(c.get("lossy_first_divergence_idx") or 99),
        ),
        reverse=True,
    )
    return candidates[0]


def _drift_explanation(demo: dict[str, Any]) -> str:
    rej = (demo.get("highlight_round") or {}).get("first_rejected_text", "")
    corr = (demo.get("highlight_round") or {}).get("correction_text", "")
    rej_c = _core_token(rej).lower()
    corr_c = _core_token(corr).lower()
    if "drop" in rej_c and "pickup" in corr_c:
        return "compressed KV tried to use dropoff instead of pickup"
    if "south" in rej_c and "north" in corr_c:
        return "compressed KV corrupted the entity name (SOUTH instead of NORTH)"
    if "blue" in rej_c and "red" in corr_c:
        return "compressed KV repeated the wrong color token"
    return "compressed KV changed the tool call"


def write_report_md(
    report: dict[str, Any],
    path: Path,
    *,
    exp034_fallback: dict[str, Any] | None,
) -> None:
    demo = report.get("selected_demo")
    summary = report.get("search_summary", {})
    lines = [
        "# Experiment 034b: Semantic Correction Search",
        "",
        "_Generated by `scripts/search_experiment_034b_semantic_correction_demo.py`. "
        "V13 Phase 8e — correctness search only._",
        "",
        "**Status:** PASS" if demo else "**Status:** NO SEMANTIC CANDIDATE",
        "",
        "> This is a **correctness search**, not a timing benchmark.",
        "> No speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, "
        "production serving, or model accuracy improvement claim is made.",
        "> ExactKV preserves full-greedy output while using lossy KV only as a draft.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Find a human-obvious semantic correction trace for the terminal-native crash-test demo.",
        "Avoid punctuation-only corrections like `'}}' → 'metric'`.",
        "",
        "## 2. Search summary",
        "",
        f"| Cells searched | {summary.get('cells_searched', 0)} |",
        f"| Semantic candidates | {summary.get('semantic_candidates', 0)} |",
        f"| Better than Exp 034 punctuation trace | {report.get('better_than_exp034', False)} |",
        "",
    ]

    if demo:
        hr = demo.get("highlight_round") or {}
        lines.extend([
            "## 3. Selected semantic demo",
            "",
            f"| Field | Value |",
            f"| --- | --- |",
            f"| prompt_id | `{demo.get('prompt_id')}` |",
            f"| suite | `{demo.get('v10_suite')}` |",
            f"| compressor | `{demo.get('compressor_name')}` |",
            f"| draft_len | {demo.get('draft_len')} |",
            f"| semantic_score | {demo.get('semantic_score', 0):.1f} |",
            f"| rejected token | `{hr.get('first_rejected_text')}` |",
            f"| correction token | `{hr.get('correction_text')}` |",
            f"| drift | {_drift_explanation(demo)} |",
            f"| exactkv_failures | {0 if demo.get('exactkv_exact_match') else 1} |",
            f"| final output match | {str(demo.get('exactkv_exact_match', False)).lower()} |",
            "",
            "## 4. Prompt",
            "",
            "```",
            demo.get("prompt", ""),
            "```",
            "",
            "## 5. Output comparison",
            "",
            "| Mode | Output |",
            "| --- | --- |",
            f"| Full KV | `{demo.get('full_output_text', '')[:180]}` |",
            f"| Lossy KV only | `{demo.get('lossy_output_text', '')[:180]}` |",
            f"| ExactKV | `{demo.get('exactkv_output_text', '')[:180]}` |",
            "",
            "## 6. Why this beats `'}}' → 'metric'`",
            "",
            _why_better_than_exp034(demo),
            "",
        ])
    else:
        lines.extend([
            "## 3. Result",
            "",
            "No stronger semantic trace found. Terminal demo falls back to Exp 034 `tj_002` trace.",
            "",
        ])

    if exp034_fallback:
        lines.extend([
            "## 7. Exp 034 fallback (if needed)",
            "",
            f"- prompt_id: `{exp034_fallback.get('prompt_id')}`",
            f"- rejected: `{(exp034_fallback.get('highlight_round') or {}).get('first_rejected_text')}`",
            f"- correction: `{(exp034_fallback.get('highlight_round') or {}).get('correction_text')}`",
            "",
        ])

    lines.extend([
        "## 8. Reproduction",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/search_experiment_034b_semantic_correction_demo.py \\",
        "  --device cpu --dtype float32",
        "```",
        "",
        "Raw JSON under `reports/` (gitignored).",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _why_better_than_exp034(demo: dict[str, Any]) -> str:
    hr = demo.get("highlight_round") or {}
    rej = hr.get("first_rejected_text", "")
    corr = hr.get("correction_text", "")
    if _is_punct_only(rej):
        return "Still punctuation-heavy; prefer a different candidate."
    return (
        f"The lossy draft proposed `{rej}` but the verifier committed `{corr}` — a "
        f"human-readable semantic change in structured output, not a JSON bracket tokenization "
        f"artifact. Viewers can see *what went wrong* without understanding BPE merges."
    )


def build_report(
    all_cells: list[dict[str, Any]],
    *,
    runtime: ModelRuntime | None,
    exp034_rescored: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    for cell in all_cells:
        if cell.get("semantic_score") is None:
            cell["semantic_score"] = score_semantic_candidate(cell)

    selected = select_best(all_cells)
    exp034_punct_score = 0.0
    if exp034_rescored:
        for c in exp034_rescored:
            if c.get("prompt_id") == "tj_002" and c.get("compressor_name") == "int4_sim":
                exp034_punct_score = c.get("semantic_score", 0.0)
                break

    better = bool(
        selected
        and selected.get("semantic_score", 0) > max(exp034_punct_score, 0) + 50
    )

    serializable = [_strip_internal(c) for c in all_cells if c.get("semantic_score", 0) > 0]
    serializable.sort(key=lambda c: c.get("semantic_score", 0), reverse=True)

    selected_out = None
    if selected and better:
        selected_out = _strip_internal(selected)
        if runtime is not None:
            selected_out = enrich_demo_cell(selected_out, runtime.tokenizer)
        hr = selected_out.get("highlight_round") or {}
        rej = hr.get("first_rejected_text", "")
        corr = hr.get("correction_text", "")
        selected_out["drift_message"] = _drift_explanation(selected_out)
        lossy = selected_out.get("lossy_output_text", "")
        if rej and rej in lossy:
            hr = dict(hr)
            hr["lossy_draft_fragment"] = lossy[: lossy.index(rej) + len(rej)]
            selected_out["highlight_round"] = hr
        if selected_out.get("v10_suite") == "crafted_pharmacy":
            selected_out["prompt_label"] = "PHARMACY TOOL CALL PROMPT"

    return {
        "experiment": "034b",
        "experiment_class": EXPERIMENT_CLASS_034B,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name if runtime else MODEL_DEFAULT,
        "search_summary": {
            "cells_searched": len(all_cells),
            "semantic_candidates": len(serializable),
            "exactkv_failures": sum(1 for c in all_cells if c.get("exactkv_failure")),
        },
        "better_than_exp034": better,
        "exp034_punctuation_score": exp034_punct_score,
        "selected_demo": selected_out,
        "top_semantic_candidates": serializable[:15],
        "fallback_demo_id": "tj_002_int4_sim" if not better else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 034b semantic correction search")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--skip-v10", action="store_true", help="Crafted prompts only")
    parser.add_argument("--try-snapkv", action="store_true")
    parser.add_argument(
        "--rescore-exp034",
        default=str(_ROOT / "reports" / "experiment_034_killer_correction_demo.json"),
        help="Rescore Exp 034 saved search (fast; no GPU cells)",
    )
    parser.add_argument(
        "--report-json",
        default=str(_ROOT / "reports" / "experiment_034b_semantic_correction_search.json"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_034B_SEMANTIC_CORRECTION_SEARCH.md"),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    exp034_path = Path(args.rescore_exp034)
    exp034_cells: list[dict[str, Any]] = []
    if exp034_path.is_file():
        print(f"Rescoring semantic candidates from {exp034_path}")
        exp034_cells = rescore_from_exp034_json(exp034_path, tokenizer)

    compressors = list(SEARCH_COMPRESSORS)
    if args.try_snapkv and _kvpress_available():
        compressors.append(SNAPKV_NAME)

    prompts = load_search_prompts(include_v10=not args.skip_v10)
    print(f"Live search: {len(prompts)} prompts × {len(compressors)} compressors × {len(DRAFT_LENS)} draft_lens")
    runtime = ModelRuntime(model_name=args.model, device=args.device, dtype=args.dtype)
    live_cells = run_live_search(runtime, prompts, compressors)

    all_cells = exp034_cells + live_cells
    report = build_report(all_cells, runtime=runtime, exp034_rescored=exp034_cells)
    _assert_no_forbidden(report)

    json_path = Path(args.report_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    exp034_fallback = None
    if exp034_path.is_file():
        exp034_report = json.loads(exp034_path.read_text(encoding="utf-8"))
        exp034_fallback = exp034_report.get("selected_demo")

    write_report_md(report, Path(args.report_md), exp034_fallback=exp034_fallback)

    demo = report.get("selected_demo")
    print(f"semantic_candidates={report['search_summary']['semantic_candidates']}")
    print(f"better_than_exp034={report['better_than_exp034']}")
    if demo:
        hr = demo.get("highlight_round") or {}
        print(
            f"selected: {demo['prompt_id']} × {demo['compressor_name']} "
            f"rej={hr.get('first_rejected_text')!r} corr={hr.get('correction_text')!r} "
            f"score={demo.get('semantic_score', 0):.1f}"
        )
    else:
        print("No semantic winner; terminal demo will use Exp 034 fallback.")
    print(f"Wrote {args.report_md}")
    return 0 if demo or report["search_summary"]["semantic_candidates"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
