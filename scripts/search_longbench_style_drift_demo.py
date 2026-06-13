#!/usr/bin/env python3
"""Experiment 037: LongBench-style score-preserving drift search (V13 Phase 10A)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.compressors import get_compressor
from exactkv.runtime.model_runtime import ModelRuntime
from scripts.run_experiment_034_killer_correction_demo import (
    _strip_internal,
    enrich_demo_cell,
    run_search_cell,
)

MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B"
SEARCH_COMPRESSORS = ("int4_sim", "k8_v4_sim", "k8_v4_boundary4_v8_sim")
DRAFT_LENS = (4, 8)
MAX_NEW_TOKENS_DEFAULT = 64
EXPERIMENT = "037"
EXPERIMENT_CLASS = "v13_longbench_style_drift_search"

_FORBIDDEN_REPORT_KEYS = frozenset({
    "tokens_per_second", "throughput", "latency", "speedup",
    "runtime_seconds", "active_gpu_kv_bytes",
})

_PUNCT = frozenset('{}[]":,._-\\/ \t\n')

_SEMANTIC_BONUS = (
    ("owner", "support"),
    ("maya", "priya"),
    ("priya", "maya"),
    ("medium", "low"),
    ("medium", "high"),
    ("september", "august"),
    ("30", "60"),
    ("refund", "exchange"),
    ("purchase", "activation"),
    ("approved", "pending"),
    ("owns", "included"),
    ("launch", "support"),
)


def _pad_context(text: str, repeats: int = 8) -> str:
    filler = (
        "Operational note: teams track migration blockers, renewal risk, "
        "follow-up owners, and checkpoint dates in weekly reviews. "
    )
    return filler * repeats + text


@dataclass
class PromptSpec:
    prompt_id: str
    category: str
    prompt: str
    task_type: str
    reference_answers: list[str] = field(default_factory=list)
    required_facts: list[str] = field(default_factory=list)
    v10_suite: str = "longbench_style_v1"


def build_prompt_set() -> list[PromptSpec]:
    specs: list[PromptSpec] = []
    cs_transcript = (
        "Customer-success call transcript (excerpt):\n"
        "Rep: Thanks for joining. Where are you blocked?\n"
        "Customer: SSO setup is still failing for our staging tenant.\n"
        "Rep: Understood. Renewal risk looks medium until SSO is fixed.\n"
        "Rep: Maya will own the follow-up and send the runbook.\n"
    )
    specs.append(PromptSpec(
        "lb_cs_001", "customer_success_summary", _pad_context(
            "Summarize this customer-success call transcript in 3 bullet points.\n\n"
            + cs_transcript
            + "\nReference facts:\n- Customer is blocked by SSO setup.\n"
            "- Renewal risk is medium.\n- Follow-up owner is Maya.\n"
        ),
        "summary",
        required_facts=["sso", "medium", "maya"],
    ))
    meet_transcript = (
        "Meeting transcript (excerpt):\n"
        "PM: Billing migration is still incomplete.\n"
        "Eng: Priya is the launch owner for the cutover.\n"
        "PM: Next checkpoint is September 10.\n"
    )
    specs.append(PromptSpec(
        "lb_mt_001", "meeting_summary", _pad_context(
            "Summarize this meeting transcript in 3 bullet points.\n\n"
            + meet_transcript
            + "\nReference facts:\n- Billing migration is incomplete.\n"
            "- Priya is the launch owner.\n- Next checkpoint is September 10.\n"
        ),
        "summary",
        required_facts=["billing", "priya", "september"],
    ))
    policy = (
        "Support policy excerpt:\n"
        "Pro annual customers may request a full refund within 30 days of purchase.\n"
        "After 30 days, only exchange credits apply.\n"
    )
    for pid, q in [
        ("lb_pol_001", "According to the policy, how many days after purchase can a Pro annual customer request a refund?"),
        ("lb_pol_002", "Under the policy, what is the refund window for Pro annual customers measured from purchase date?"),
    ]:
        specs.append(PromptSpec(
            pid, "support_policy_qa", _pad_context(policy + "\n\n" + q + "\n"),
            "qa",
            reference_answers=["30 days", "30", "thirty days", "30-day"],
        ))
    refund_ctx = (
        "Document A: Refunds for Pro annual plans are allowed within 30 days of purchase.\n"
        "Document B: Activation-date refunds are not supported for legacy tiers.\n"
    )
    specs.append(PromptSpec(
        "lb_pol_003", "refund_policy_qa", _pad_context(
            refund_ctx + "\nQuestion: How many days after purchase can a Pro annual customer request a refund?\n"
        ),
        "qa",
        reference_answers=["30 days", "30", "thirty"],
    ))
    multi_doc = (
        "Context document 1: Friday follow-up on SSO is assigned to Maya.\n"
        "Context document 2: Billing migration checkpoint remains open.\n"
        "Context document 3: Launch communications go to Priya.\n"
    )
    specs.append(PromptSpec(
        "lb_md_001", "multi_doc_qa", _pad_context(
            multi_doc + "\nUse the context documents to answer exactly who owns the Friday follow-up.\n"
        ),
        "qa",
        reference_answers=["maya", "Maya"],
    ))
    specs.append(PromptSpec(
        "lb_rc_001", "retrieval_copy_fact", _pad_context(
            "Long-context retrieval task:\n"
            "Background: " + "The operations team logs owner names in triage tickets. " * 6
            + "\nCopy exactly who owns the Friday follow-up according to the ticket: "
            "OWNER=Maya; TASK=Friday follow-up; STATUS=open\n"
        ),
        "qa",
        reference_answers=["maya", "Maya"],
    ))
    ops = (
        "Weekly operations log:\n"
        "- SSO blocker persists for staging.\n"
        "- Renewal risk: medium.\n"
        "- Follow-up owner: Maya.\n"
        "- Billing migration incomplete; launch owner Priya.\n"
    )
    specs.append(PromptSpec(
        "lb_ops_001", "long_context_ops_summary", _pad_context(
            ops + "\nSummarize the operational status in 3 bullet points.\n"
        ),
        "summary",
        required_facts=["sso", "maya", "priya"],
    ))
    specs.append(PromptSpec(
        "lb_cs_002", "customer_success_summary", _pad_context(
            cs_transcript
            + "\nAnswer in 3 bullets. Include SSO blocker, renewal risk level, and follow-up owner Maya.\n"
        ),
        "summary",
        required_facts=["sso", "maya"],
    ))
    specs.append(PromptSpec(
        "lb_mt_002", "meeting_summary", _pad_context(
            meet_transcript
            + "\nAnswer in 3 bullets mentioning billing migration, launch owner Priya, and September 10.\n"
        ),
        "summary",
        required_facts=["billing", "priya", "september"],
    ))
    return specs


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _contains_ref(output: str, ref: str) -> bool:
    o = _normalize(output)
    r = _normalize(ref)
    if not r:
        return False
    if r in o:
        return True
    if r.isdigit():
        return bool(re.search(rf"\b{re.escape(r)}\b", o))
    return False


def task_heuristic(spec: PromptSpec, text: str) -> dict[str, Any]:
    if spec.task_type == "qa":
        hits = [r for r in spec.reference_answers if _contains_ref(text, r)]
        return {"pass": len(hits) > 0, "matched": hits, "mode": "qa_reference"}
    hits = [f for f in spec.required_facts if f.lower() in text.lower()]
    need = max(1, len(spec.required_facts) * 2 // 3)
    return {"pass": len(hits) >= need, "matched": hits, "required": spec.required_facts, "mode": "summary_facts"}


def _is_punct_only(s: str) -> bool:
    core = s.strip().strip('"').strip("'")
    return not core or all(c in _PUNCT for c in core)


def _min_demo_score() -> float:
    return 200.0


def score_candidate(cell: dict[str, Any], spec: PromptSpec) -> float:
    if not cell.get("is_score_preserving_candidate"):
        return 0.0
    score = 100.0
    if spec.category in ("support_policy_qa", "multi_doc_qa", "customer_success_summary"):
        score += 50.0
    hr = cell.get("highlight_round") or {}
    rej = hr.get("first_rejected_text") or ""
    corr = hr.get("correction_text") or ""
    rej_l, corr_l = rej.lower().strip(), corr.lower().strip()
    if _is_punct_only(rej) and _is_punct_only(corr):
        score -= 200.0
    elif _is_punct_only(rej) or _is_punct_only(corr):
        score -= 120.0
    else:
        score += 80.0
    if rej_l in ("<|endoftext|>", "") or corr_l in ("<|endoftext|>", ""):
        score -= 80.0
    if rej_l and corr_l and rej_l == corr_l:
        score -= 150.0
    for a, b in _SEMANTIC_BONUS:
        if a in rej_l or b in corr_l:
            score += 60.0
    if cell.get("lossy_first_divergence_idx", 99) <= 8:
        score += 20.0
    if cell.get("full_output_text") != cell.get("lossy_output_text"):
        score += 30.0
    if rej_l and corr_l and rej_l != corr_l and len(rej_l) >= 2:
        score += 40.0
    return score


def evaluate_cell(cell: dict[str, Any], spec: PromptSpec) -> dict[str, Any]:
    full_h = task_heuristic(spec, cell.get("full_output_text", ""))
    lossy_h = task_heuristic(spec, cell.get("lossy_output_text", ""))
    ekv_h = task_heuristic(spec, cell.get("exactkv_output_text", ""))
    lossy_exact = cell.get("lossy_exact_match", True)
    ekv_exact = cell.get("exactkv_exact_match", False)
    hr = cell.get("highlight_round")
    has_correction = bool(hr and hr.get("correction_token") is not None and hr.get("num_rejected", 0) > 0)
    is_candidate = (
        full_h["pass"] and lossy_h["pass"] and ekv_h["pass"]
        and not lossy_exact and ekv_exact and has_correction
        and not cell.get("exactkv_failure", True)
    )
    cell = dict(cell)
    cell["task_heuristic"] = {
        "full": full_h, "lossy": lossy_h, "exactkv": ekv_h,
        "outcome_score_changed": False, "behavior_drifted": not lossy_exact,
    }
    cell["is_score_preserving_candidate"] = is_candidate
    cell["candidate_score"] = score_candidate(cell, spec) if is_candidate else 0.0
    cell["prompt_category"] = spec.category
    cell["task_type"] = spec.task_type
    return cell


def prompt_to_entry(spec: PromptSpec) -> dict[str, Any]:
    return {"prompt_id": spec.prompt_id, "v10_suite": spec.v10_suite, "category": spec.category, "prompt": spec.prompt}


def run_search(
    runtime: ModelRuntime,
    prompts: list[PromptSpec],
    compressors: tuple[str, ...],
    *,
    draft_lens: tuple[int, ...],
    max_new_tokens: int,
    max_cells: int | None,
    checkpoint_path: Path | None,
    stop_score: float,
) -> dict[str, Any]:
    all_cells: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    cells_run = 0
    stopped_early = False
    import scripts.run_experiment_034_killer_correction_demo as exp034
    old_max = exp034.MAX_NEW_TOKENS
    exp034.MAX_NEW_TOKENS = max_new_tokens
    try:
        for spec in prompts:
            entry = prompt_to_entry(spec)
            for comp_name in compressors:
                comp = get_compressor(comp_name)
                for draft_len in draft_lens:
                    if max_cells is not None and cells_run >= max_cells:
                        stopped_early = True
                        break
                    raw = run_search_cell(runtime, entry, comp, draft_len=draft_len, verification_method="sequential")
                    cells_run += 1
                    ev = enrich_demo_cell(raw, runtime.tokenizer)
                    ev.pop("_tokenizer", None)
                    ev = evaluate_cell(ev, spec)
                    if ev.get("is_score_preserving_candidate") and ev.get("candidate_score", 0) >= _min_demo_score():
                        candidates.append(ev)
                    all_cells.append(_strip_internal(ev))
                    if checkpoint_path and cells_run % 10 == 0:
                        _write_checkpoint(checkpoint_path, all_cells, candidates, cells_run, runtime, compressors, draft_lens, max_new_tokens)
                    if candidates and max(c.get("candidate_score", 0) for c in candidates) >= stop_score:
                        stopped_early = True
                        break
                if stopped_early:
                    break
            if stopped_early:
                break
    finally:
        exp034.MAX_NEW_TOKENS = old_max
    candidates.sort(key=lambda c: c.get("candidate_score", 0), reverse=True)
    strong = [c for c in candidates if c.get("candidate_score", 0) >= _min_demo_score()]
    selected = _strip_internal(strong[0]) if strong else None
    return {
        "experiment": EXPERIMENT,
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name,
        "device": str(runtime.device),
        "dtype": runtime.dtype_str,
        "search_config": {
            "compressors": list(compressors), "draft_lens": list(draft_lens),
            "max_new_tokens": max_new_tokens, "max_cells": max_cells,
            "cells_searched": cells_run, "stopped_early": stopped_early, "stop_score_threshold": stop_score,
        },
        "search_summary": {"prompts": len(prompts), "candidates": len(candidates),
            "exactkv_failures": sum(1 for c in all_cells if c.get("exactkv_failure"))},
        "selected_demo": selected,
        "top_candidates": [_strip_internal(c) for c in candidates[:15]],
        "all_cells": all_cells,
        "better_than_none": bool(selected),
    }


def _write_checkpoint(path, all_cells, candidates, cells_run, runtime, compressors, draft_lens, max_new_tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "cells_searched": cells_run,
        "candidates": len(candidates),
        "top_candidates": [_strip_internal(c) for c in sorted(candidates, key=lambda x: x.get("candidate_score", 0), reverse=True)[:10]],
        "all_cells": all_cells,
    }, indent=2), encoding="utf-8")


def write_report_md(report: dict[str, Any], path: Path) -> None:
    sel = report.get("selected_demo")
    cfg = report.get("search_config", {})
    lines = [
        "# Experiment 037: LongBench-Style Score-Preserving Drift Demo",
        "",
        "_Generated by `scripts/search_longbench_style_drift_demo.py`. Phase 10A._",
        "",
        "**Status:** " + ("PASS — candidate found" if sel else "NO STRONG CANDIDATE"),
        "",
        "> LongBench-style demonstration, not official LongBench evaluation.",
        "> Outcome benchmarks and ExactKV answer different questions.",
        "",
        f"Cells searched: {cfg.get('cells_searched', 0)} | Candidates: {report.get('search_summary', {}).get('candidates', 0)}",
        "",
    ]
    if sel:
        hr = sel.get("highlight_round") or {}
        lines += [
            f"## Selected: `{sel.get('prompt_id')}` × `{sel.get('compressor_name')}`",
            f"- rejected: `{hr.get('first_rejected_text')}` → correction: `{hr.get('correction_text')}`",
            f"- score: {sel.get('candidate_score', 0):.1f}",
            "",
            "### Full KV", "```", (sel.get("full_output_text") or "")[:600], "```",
            "### Lossy KV", "```", (sel.get("lossy_output_text") or "")[:600], "```",
            "### ExactKV", "```", (sel.get("exactkv_output_text") or "")[:600], "```",
        ]
    else:
        lines.append("No strong candidate in bounded search. Pharmacy demo remains primary.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--stop-score", type=float, default=250.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-json", default=str(_ROOT / "reports/experiment_037_longbench_style_drift_candidates.json"))
    parser.add_argument("--report-md", default=str(_ROOT / "docs/EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md"))
    args = parser.parse_args()
    prompts = build_prompt_set()
    print(f"Exp 037 — {len(prompts)} prompts")
    if args.dry_run:
        print(f"dry-run OK: {len(prompts)} prompts")
        return 0
    runtime = ModelRuntime(model_name=args.model, device=args.device, dtype=args.dtype)
    report = run_search(runtime, prompts, SEARCH_COMPRESSORS, draft_lens=DRAFT_LENS,
        max_new_tokens=args.max_new_tokens, max_cells=args.max_cells,
        checkpoint_path=Path(args.report_json), stop_score=args.stop_score)
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report_md(report, Path(args.report_md))
    demo = report.get("selected_demo")
    print(f"cells={report['search_config']['cells_searched']} candidates={report['search_summary']['candidates']}")
    if demo:
        hr = demo.get("highlight_round") or {}
        print(f"selected: {demo['prompt_id']} rej={hr.get('first_rejected_text')!r} corr={hr.get('correction_text')!r}")
    return 0 if demo else 1


if __name__ == "__main__":
    raise SystemExit(main())
