#!/usr/bin/env python3
"""Experiment 034: killer correction demo (V13 Phase 7).

Correctness/correction demo only — not a timing or GPU memory benchmark.
Searches for a lossy-divergent cell where ExactKV rejects a wrong draft token,
commits the verifier correction, and final output matches full greedy exactly.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from exactkv.analysis.divergence_autopsy import (
    collect_lossy_divergence_observation,
    structured_output_state,
    token_type_at_id,
)
from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 32
DRAFT_LENS = (4, 8)
EXPERIMENT_CLASS = "v13_killer_correction_demo"

SEARCH_SUITES = (
    "tool_json",
    "code_structured",
    "retrieval_copy",
    "long_context",
)

SEARCH_COMPRESSORS = (
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "int4_sim",
)

SNAPKV_NAME = "snapkv_experimental"

PUBLIC_TAGLINE = (
    "Everyone is racing to shrink KV caches. ExactKV tells you when they start lying."
)

_SUITE_PRIORITY = {
    "tool_json": 1000.0,
    "code_structured": 800.0,
    "retrieval_copy": 600.0,
    "long_context": 400.0,
}

_SCARY_STRUCTURAL_REJECTIONS = frozenset({
    "}}",
    "}",
    "{",
    "]",
    '"]',
    '":',
    '",',
    '"}',
    '"',
})

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


def _kvpress_available() -> bool:
    try:
        return importlib.util.find_spec("kvpress") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def load_search_prompts(*, per_suite: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for suite in SEARCH_SUITES:
        rows = load_v10_suite(suite)
        rows.sort(key=lambda r: r["prompt_id"])
        take = rows if per_suite is None else rows[:per_suite]
        out.extend(take)
    return out


def _resolve_compressor(runtime: ModelRuntime, name: str, cache: dict[str, Any]) -> Any:
    if name in cache:
        return cache[name]
    if name == SNAPKV_NAME:
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

        comp = create_snapkv_experimental_adapter(runtime, compression_ratio=0.5)
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


def _decode_token(tokenizer: Any, token_id: int | None) -> str:
    if token_id is None:
        return ""
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _abbrev_output(text: str, *, max_len: int = 200) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _trace_rounds(ekv_result: Any) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for trace in ekv_result.traces:
        acc = trace.acceptance
        first_rejected = acc.rejected_tokens[0] if acc.rejected_tokens else None
        rounds.append({
            "round_idx": trace.round_idx,
            "draft_tokens": list(trace.draft_tokens),
            "verifier_tokens": list(acc.verifier_tokens),
            "accepted_prefix": list(acc.accepted_tokens),
            "first_rejected_token": first_rejected,
            "rejected_tokens": list(acc.rejected_tokens),
            "correction_token": acc.correction_token,
            "num_accepted": acc.num_accepted,
            "num_rejected": acc.num_rejected,
            "all_matched": acc.all_matched,
        })
    return rounds


def _first_correction_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    for r in rounds:
        if r["correction_token"] is not None and r["num_rejected"] > 0:
            return r
    return None


def _rejection_legibility_score(
    rej_text: str,
    corr_text: str,
    suite: str,
    lossy_struct: dict[str, Any],
) -> float:
    """Higher = scarier, more human-readable wrong draft token."""
    score = 0.0
    if rej_text.strip() == corr_text.strip():
        score -= 50.0

    if any(ch in rej_text for ch in '{}[]":,'):
        score += 35.0
    if rej_text in _SCARY_STRUCTURAL_REJECTIONS or rej_text.strip() in _SCARY_STRUCTURAL_REJECTIONS:
        score += 45.0

    corr_core = corr_text.strip().strip('"').strip()
    if corr_core and (corr_core[0].isalpha() or corr_core.isdigit()):
        score += 25.0

    if suite == "tool_json":
        if lossy_struct.get("malformed_json_prefix"):
            score += 30.0
        if lossy_struct.get("unmatched_brackets"):
            score += 20.0
        if '"' in corr_text or corr_core.isalpha():
            score += 15.0
    elif suite == "code_structured":
        if any(op in rej_text or op in corr_text for op in ("==", "!=", "::", "->", "+=", "-=")):
            score += 25.0
        if rej_text.strip() and rej_text.strip()[0].isalpha():
            score += 15.0
    elif suite == "retrieval_copy":
        if rej_text.strip().isdigit() or corr_text.strip().isdigit():
            score += 30.0
        if len(rej_text.strip()) >= 2 and rej_text.strip().isalpha():
            score += 20.0

    if not rej_text.strip():
        score -= 25.0

    return score


def _rejection_legibility_score_with_tokenizer(
    tokenizer: Any,
    rej_text: str,
    corr_text: str,
    suite: str,
    lossy_struct: dict[str, Any],
    rej_id: int | None,
) -> float:
    score = _rejection_legibility_score(rej_text, corr_text, suite, lossy_struct)
    if rej_id is not None:
        tok_type = token_type_at_id(tokenizer, int(rej_id))
        if tok_type in ("bracket", "quote", "punctuation", "numeric"):
            score += 20.0
    return score


def _score_candidate(cell: dict[str, Any]) -> float:
    """Higher = more compelling public-facing demo."""
    suite = cell.get("v10_suite", "")
    score = _SUITE_PRIORITY.get(suite, 0.0)

    if not cell.get("lossy_exact_match"):
        score += 10.0
    if not cell.get("exactkv_exact_match"):
        return 0.0

    corr_round = cell.get("highlight_round")
    if not corr_round:
        return 0.0

    rej_id = corr_round.get("first_rejected_token")
    corr_id = corr_round.get("correction_token")
    if rej_id is None or corr_id is None or rej_id == corr_id:
        return 0.0

    tokenizer = cell.get("_tokenizer")
    rej_text = corr_round.get("first_rejected_text")
    corr_text = corr_round.get("correction_text")
    if tokenizer is not None:
        if rej_text is None:
            rej_text = _decode_token(tokenizer, rej_id)
        if corr_text is None:
            corr_text = _decode_token(tokenizer, corr_id)
    else:
        rej_text = rej_text or ""
        corr_text = corr_text or ""

    lossy_struct = cell.get("lossy_structured_state") or {}
    score += _rejection_legibility_score_with_tokenizer(
        tokenizer, rej_text, corr_text, suite, lossy_struct, rej_id
    )

    div_idx = cell.get("lossy_first_divergence_idx")
    if div_idx is not None:
        if div_idx <= 1:
            score += 25.0
        elif div_idx <= 3:
            score += 15.0
        elif div_idx < 10:
            score += 5.0

    if corr_round.get("round_idx") == 0:
        score += 20.0
    if cell.get("draft_len") == 4:
        score += 5.0

    return score


def _demo_sort_key(cell: dict[str, Any]) -> tuple[float, float, float, float]:
    """Primary demo_score; tie-break toward scarier structural rejections."""
    hr = cell.get("highlight_round") or {}
    tokenizer = cell.get("_tokenizer")
    rej_text = hr.get("first_rejected_text")
    if rej_text is None and tokenizer is not None:
        rej_text = _decode_token(tokenizer, hr.get("first_rejected_token"))
    rej_text = rej_text or ""
    structural = 0.0
    if rej_text in _SCARY_STRUCTURAL_REJECTIONS:
        structural = 30.0
    elif any(ch in rej_text for ch in '{}[]'):
        structural = 15.0
    early = 10.0 if cell.get("lossy_first_divergence_idx") == 1 else 0.0
    lossy_struct = cell.get("lossy_structured_state") or {}
    malformed = 10.0 if lossy_struct.get("malformed_json_prefix") else 0.0
    return (cell.get("demo_score", 0.0), structural, early, malformed)


def run_search_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    compressor: Any,
    *,
    draft_len: int,
    verification_method: str,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]
    compressor_name = getattr(compressor, "name", "unknown")

    full_res = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
    full_ids = full_res.generated_ids.squeeze(0).tolist()
    prompt_len = int(full_res.prompt_ids.shape[-1])

    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, MAX_NEW_TOKENS)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()
    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    ekv_res = ExactKVGenerator(
        runtime,
        compressor,
        draft_len=draft_len,
        verification_method=verification_method,  # type: ignore[arg-type]
    ).generate(prompt, MAX_NEW_TOKENS)
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)
    rounds = _trace_rounds(ekv_res)
    highlight = _first_correction_round(rounds)

    lossy_obs = None
    if not lossy_exact and lossy_div is not None:
        lossy_obs = collect_lossy_divergence_observation(
            runtime,
            compressor,
            prompt,
            prompt_len,
            full_ids,
            lossy_ids,
        )

    is_demo_candidate = (
        not lossy_exact
        and ekv_exact
        and highlight is not None
        and highlight["first_rejected_token"] is not None
        and highlight["correction_token"] is not None
        and highlight["first_rejected_token"] != highlight["correction_token"]
    )

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "category": prompt_entry.get("category", ""),
        "compressor_name": compressor_name,
        "draft_len": draft_len,
        "verification_method": verification_method,
        "prompt": prompt,
        "prompt_abbrev": prompt[:240] + ("…" if len(prompt) > 240 else ""),
        "full_output_ids": full_ids,
        "full_output_text": full_res.output_text,
        "lossy_output_ids": lossy_ids,
        "lossy_output_text": lossy_res.output_text,
        "exactkv_output_ids": ekv_res.output_ids.squeeze(0).tolist(),
        "exactkv_output_text": ekv_res.output_text,
        "lossy_exact_match": lossy_exact,
        "exactkv_exact_match": ekv_exact,
        "exactkv_failure": not ekv_exact,
        "lossy_first_divergence_idx": lossy_div,
        "acceptance": acceptance.to_dict(),
        "total_corrections": ekv_res.total_corrections,
        "trace_rounds": rounds,
        "highlight_round": highlight,
        "lossy_divergence_observation": lossy_obs,
        "lossy_structured_state": structured_output_state(lossy_res.output_text),
        "is_demo_candidate": is_demo_candidate,
        "demo_score": 0.0,
        "_tokenizer": runtime.tokenizer,
    }


def _strip_internal(cell: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cell.items() if not k.startswith("_")}


def enrich_demo_cell(cell: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Add decoded trace fields for reporting."""
    cell = dict(cell)
    cell.pop("_tokenizer", None)
    hr = cell.get("highlight_round")
    if hr:
        enriched = dict(hr)
        enriched["first_rejected_text"] = _decode_token(
            tokenizer, enriched.get("first_rejected_token")
        )
        enriched["correction_text"] = _decode_token(
            tokenizer, enriched.get("correction_token")
        )
        enriched["draft_tokens_text"] = [
            _decode_token(tokenizer, t) for t in enriched.get("draft_tokens", [])
        ]
        enriched["verifier_tokens_text"] = [
            _decode_token(tokenizer, t) for t in enriched.get("verifier_tokens", [])
        ]
        cell["highlight_round"] = enriched
    return cell


def _public_snapshot_tables(demo: dict[str, Any]) -> list[str]:
    hr = demo.get("highlight_round") or {}
    rej_label = (
        f"{hr.get('first_rejected_text')!r} "
        f"(id {hr.get('first_rejected_token')})"
    )
    corr_label = (
        f"{hr.get('correction_text')!r} "
        f"(id {hr.get('correction_token')})"
    )
    div_idx = demo.get("lossy_first_divergence_idx")
    return [
        f"> **{PUBLIC_TAGLINE}**",
        "",
        "### Output comparison",
        "",
        "| Mode | Output |",
        "| --- | --- |",
        f"| Full KV | `{_abbrev_output(demo.get('full_output_text', ''))}` |",
        f"| Lossy compressed KV draft | `{_abbrev_output(demo.get('lossy_output_text', ''))}` |",
        f"| ExactKV | `{_abbrev_output(demo.get('exactkv_output_text', ''))}` |",
        "",
        "### Correction at a glance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| First divergence | token index {div_idx} |",
        f"| Draft token rejected | {rej_label} |",
        f"| Full-KV correction | {corr_label} |",
        f"| ExactKV failures | {0 if demo.get('exactkv_exact_match') else 1} |",
        f"| Final output match | {str(demo.get('exactkv_exact_match', False)).lower()} |",
        "",
    ]


def build_explanation(cell: dict[str, Any]) -> str:
    hr = cell.get("highlight_round") or {}
    rej = hr.get("first_rejected_text", "")
    corr = hr.get("correction_text", "")
    suite = cell.get("v10_suite", "")
    comp = cell.get("compressor_name", "")
    lines = [
        PUBLIC_TAGLINE,
        f"With compressor `{comp}`, the lossy compressed-KV draft proposed "
        f"{rej!r} (id {hr.get('first_rejected_token')}) at round {hr.get('round_idx')}.",
        f"The full-KV verifier predicted {corr!r} (id {hr.get('correction_token')}) instead.",
        "ExactKV rejected the draft token and committed the verifier correction; "
        "the wrong draft was never written to the authoritative KV cache.",
        f"Final ExactKV output matches full greedy ({cell.get('exactkv_exact_match')}).",
    ]
    if suite in ("tool_json", "code_structured"):
        lines.append(
            "Without verification, lossy greedy output could carry malformed structured "
            "tokens into the continuation."
        )
    elif suite == "retrieval_copy":
        lines.append(
            "Without verification, a wrong copied fact or token from compressed KV "
            "would have been committed."
        )
    return " ".join(lines)


def write_trace_markdown(cell: dict[str, Any], path: Path) -> None:
    hr = cell.get("highlight_round") or {}
    lines = [
        "# Experiment 034 — Correction Trace",
        "",
        f"**Prompt:** `{cell['prompt_id']}` ({cell['v10_suite']})",
        f"**Compressor:** `{cell['compressor_name']}` | **draft_len:** {cell['draft_len']}",
        f"**Verification:** {cell['verification_method']}",
        "",
        *_public_snapshot_tables(cell),
        "## Highlight round",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Round | {hr.get('round_idx')} |",
        f"| Draft tokens | `{hr.get('draft_tokens')}` |",
        f"| Verifier tokens | `{hr.get('verifier_tokens')}` |",
        f"| Accepted prefix | `{hr.get('accepted_prefix')}` |",
        f"| First rejected | `{hr.get('first_rejected_token')}` → {hr.get('first_rejected_text')!r} |",
        f"| Correction | `{hr.get('correction_token')}` → {hr.get('correction_text')!r} |",
        "",
        "## Outputs",
        "",
        "### Full greedy",
        "```",
        cell.get("full_output_text", ""),
        "```",
        "",
        "### Lossy greedy (diverges)",
        "```",
        cell.get("lossy_output_text", ""),
        "```",
        "",
        "### ExactKV (exact)",
        "```",
        cell.get("exactkv_output_text", ""),
        "```",
        "",
        "## Explanation",
        "",
        build_explanation(cell),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(
    report: dict[str, Any],
    path: Path,
) -> None:
    demo = report.get("selected_demo")
    search = report.get("search_summary", {})
    passed = demo is not None and demo.get("exactkv_exact_match")

    lines = [
        "# Experiment 034: Killer Correction Demo",
        "",
        "_Generated by `scripts/run_experiment_034_killer_correction_demo.py`. "
        "V13 Phase 7 — correctness/correction demo only._",
        "",
        f"**Status:** {'PASS' if passed else 'FAIL / NO DEMO FOUND'}",
        "",
        "> This is a **correctness/correction demo**, not a timing benchmark.",
        "> This does **not** claim speedup, throughput, latency, runtime, tokens/sec, "
        "active GPU memory savings, production serving, or model accuracy improvement.",
        "> ExactKV does **not** improve the model's underlying accuracy; it preserves "
        "full-greedy output while using lossy KV only as a draft.",
        "> Full-KV verifier remains authoritative. Rejected draft tokens are never committed.",
        "> External Llama, Shard, SnapKV, SpectralQuant, or kvpress results are not "
        "ExactKV results.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Demonstrate a concrete case where lossy compressed KV proposes a wrong token, "
        "ExactKV rejects it, commits the verifier correction, and final output still "
        "matches full greedy exactly.",
        "",
        f"> **{PUBLIC_TAGLINE}**",
        "",
        "## 2. Why this follows Exp 033",
        "",
        "Exp 033 proved exactness on Llama-3.1-8B at scale. Exp 034 makes the "
        "rejection/correction mechanism legible on a single human-readable example.",
        "",
        "## 3. Model/environment",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Model | `{report['model_name']}` |",
        f"| device | `{report['device']}` |",
        f"| dtype | `{report['dtype']}` |",
        "",
        "## 4. Search space",
        "",
        f"| Suites | {', '.join(f'`{s}`' for s in report['search_suites'])} |",
        f"| Compressors | {', '.join(f'`{c}`' for c in report['search_compressors'])} |",
        f"| draft_lens | {report['draft_lens']} |",
        f"| max_new_tokens | {report['max_new_tokens']} |",
        f"| Cells searched | **{search.get('cells_searched', 0)}** |",
        f"| Demo candidates | **{search.get('demo_candidates', 0)}** |",
        "",
    ]

    if not demo:
        lines.extend([
            "## 5–11. Demo",
            "",
            "**No qualifying demo cell found** in the search panel.",
            "",
            "## 15. Limitations",
            "",
            "- Search did not find lossy divergence + ExactKV correction + exact match.",
            "",
            "## 16. Next steps",
            "",
            "Expand search panel, try crafted prompts, or increase `max_new_tokens`.",
            "",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    hr = demo.get("highlight_round", {})
    lines.extend([
        "## 5. Public demo snapshot",
        "",
        *_public_snapshot_tables(demo),
        "## 6. Selected demo cell",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| prompt_id | `{demo['prompt_id']}` |",
        f"| suite | `{demo['v10_suite']}` |",
        f"| compressor | `{demo['compressor_name']}` |",
        f"| draft_len | {demo['draft_len']} |",
        f"| verification | {demo['verification_method']} |",
        f"| demo_score | {demo.get('demo_score', 0):.1f} |",
        "",
        "## 7. Prompt",
        "",
        "```",
        demo["prompt"],
        "```",
        "",
        "## 8. Full greedy output",
        "",
        "```",
        demo.get("full_output_text", ""),
        "```",
        "",
        "## 9. Lossy draft divergence",
        "",
        f"- **lossy_exact_match:** {demo.get('lossy_exact_match')}",
        f"- **first_divergence_idx:** {demo.get('lossy_first_divergence_idx')}",
        "",
        "```",
        demo.get("lossy_output_text", ""),
        "```",
        "",
        "## 10. ExactKV rejection/correction trace",
        "",
        f"| Round | {hr.get('round_idx')} |",
        f"| Draft tokens | `{hr.get('draft_tokens')}` |",
        f"| Verifier tokens | `{hr.get('verifier_tokens')}` |",
        f"| Accepted prefix | `{hr.get('accepted_prefix')}` |",
        f"| First rejected | `{hr.get('first_rejected_token')}` ({hr.get('first_rejected_text')!r}) |",
        f"| Correction committed | `{hr.get('correction_token')}` ({hr.get('correction_text')!r}) |",
        "",
        "## 11. Final ExactKV output",
        "",
        "```",
        demo.get("exactkv_output_text", ""),
        "```",
        "",
        "## 12. Exactness result",
        "",
        f"- **exactkv_exact_match:** {demo.get('exactkv_exact_match')}",
        f"- **exactkv_failures:** {0 if demo.get('exactkv_exact_match') else 1}",
        f"- **total_corrections:** {demo.get('total_corrections')}",
        "",
        "## 13. Why this demo matters",
        "",
        build_explanation(demo),
        "",
        "## 14. What this proves",
        "",
        "- Lossy KV can propose incorrect draft tokens.",
        "- ExactKV detects mismatch via full-KV verification and commits corrections.",
        "- Final output remains identical to full greedy on this cell.",
        "",
        "## 15. What this does not prove",
        "",
        "- No speedup, throughput, latency, runtime, tokens/sec, or active GPU memory savings.",
        "- No production serving readiness or model accuracy improvement.",
        "- Not universal across all prompts or compressors.",
        "",
        "## 16. Limitations",
        "",
        "- Single selected cell from a finite search panel on one model.",
        "- Demo quality depends on prompt/compressor lottery.",
        "",
        "## 17. Next steps",
        "",
        "**Proceed to Phase 8 visual plot package (Exp 035).** Optional Phase 6b Shard "
        "Llama external-drafter probe remains adjunct.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_034_killer_correction_demo.py \\",
        "  --device cuda --dtype float16",
        "```",
        "",
        "Trace sidecar: [`assets/experiment_034_correction_trace.md`](assets/experiment_034_correction_trace.md). "
        "Raw JSON/CSV under `reports/` (gitignored).",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_summary(candidates: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for c in candidates:
        rows.append({
            "prompt_id": c["prompt_id"],
            "v10_suite": c["v10_suite"],
            "compressor_name": c["compressor_name"],
            "draft_len": c["draft_len"],
            "is_demo_candidate": c["is_demo_candidate"],
            "demo_score": c.get("demo_score", 0),
            "lossy_exact_match": c["lossy_exact_match"],
            "exactkv_exact_match": c["exactkv_exact_match"],
            "total_corrections": c["total_corrections"],
            "lossy_first_divergence_idx": c["lossy_first_divergence_idx"],
        })
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def rescore_report_from_json(report: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Re-select demo from saved search cells using public-legibility scoring."""
    all_cells = [dict(c) for c in report.get("all_cells", [])]
    for cell in all_cells:
        cell["_tokenizer"] = tokenizer
        cell["demo_score"] = _score_candidate(cell) if cell.get("is_demo_candidate") else 0.0

    candidates = [c for c in all_cells if c.get("is_demo_candidate")]
    candidates.sort(key=_demo_sort_key, reverse=True)
    selected = enrich_demo_cell(candidates[0], tokenizer) if candidates else None

    updated = dict(report)
    updated["generated_at"] = datetime.now(timezone.utc).isoformat()
    updated["demo_selection_note"] = (
        "Selected for public legibility: suite priority tool_json > code_structured > "
        "retrieval_copy > long_context; human-readable scary draft rejection preferred."
    )
    updated["selected_demo"] = selected
    updated["top_candidates"] = [_strip_internal(c) for c in candidates[:10]]
    updated["all_cells"] = [_strip_internal(c) for c in all_cells]
    return updated


def write_report_artifacts(
    report: dict[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
    md_path: Path,
    trace_path: Path,
) -> None:
    _assert_no_forbidden(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_csv_summary(report.get("top_candidates", []), csv_path)
    write_markdown_report(report, md_path)
    if report.get("selected_demo"):
        write_trace_markdown(report["selected_demo"], trace_path)


def run_experiment(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    compressors: list[str],
    *,
    verification_method: str = "sequential",
) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    all_cells: list[dict[str, Any]] = []
    total = len(prompts) * len(compressors) * len(DRAFT_LENS)
    idx = 0

    for pe in prompts:
        for comp_name in compressors:
            compressor = _resolve_compressor(runtime, comp_name, cache)
            for draft_len in DRAFT_LENS:
                idx += 1
                print(
                    f"  [{idx}/{total}] {pe['prompt_id']} × {comp_name} × draft_len={draft_len}",
                    flush=True,
                )
                cell = run_search_cell(
                    runtime,
                    pe,
                    compressor,
                    draft_len=draft_len,
                    verification_method=verification_method,
                )
                cell["demo_score"] = _score_candidate(cell) if cell["is_demo_candidate"] else 0
                all_cells.append(cell)

    candidates = [c for c in all_cells if c["is_demo_candidate"]]
    candidates.sort(key=_demo_sort_key, reverse=True)
    selected = candidates[0] if candidates else None
    if selected:
        selected = enrich_demo_cell(selected, runtime.tokenizer)

    serializable_cells = [_strip_internal(c) for c in all_cells]
    serializable_candidates = [_strip_internal(c) for c in candidates]

    return {
        "experiment": "034",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name,
        "device": str(runtime.device),
        "dtype": runtime.dtype_str,
        "search_suites": list(SEARCH_SUITES),
        "search_compressors": compressors,
        "draft_lens": list(DRAFT_LENS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "verification_method": verification_method,
        "search_summary": {
            "cells_searched": len(all_cells),
            "demo_candidates": len(candidates),
            "exactkv_failures": sum(1 for c in all_cells if c["exactkv_failure"]),
        },
        "selected_demo": selected,
        "top_candidates": serializable_candidates[:10],
        "all_cells": serializable_cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 034 killer correction demo")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--per-suite",
        type=int,
        default=None,
        help="Limit prompts per suite (default: all in search suites)",
    )
    parser.add_argument("--try-snapkv", action="store_true")
    parser.add_argument(
        "--rescore-from-json",
        default=None,
        help="Re-select demo and regenerate reports from a saved search JSON (no GPU search)",
    )
    parser.add_argument(
        "--report-json",
        default=str(_ROOT / "reports" / "experiment_034_killer_correction_demo.json"),
    )
    parser.add_argument(
        "--report-csv",
        default=str(_ROOT / "reports" / "experiment_034_killer_correction_demo.csv"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_034_KILLER_CORRECTION_DEMO.md"),
    )
    parser.add_argument(
        "--trace-md",
        default=str(_ROOT / "docs/assets/experiment_034_correction_trace.md"),
    )
    args = parser.parse_args()

    json_path = Path(args.report_json)
    csv_path = Path(args.report_csv)
    md_path = Path(args.report_md)
    trace_path = Path(args.trace_md)

    if args.rescore_from_json:
        from transformers import AutoTokenizer

        src = Path(args.rescore_from_json)
        report = json.loads(src.read_text(encoding="utf-8"))
        model_name = report.get("model_name", args.model)
        print(f"Experiment 034 — rescore from {src} ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        report = rescore_report_from_json(report, tokenizer)
    else:
        compressors = list(SEARCH_COMPRESSORS)
        if args.try_snapkv and _kvpress_available():
            compressors.append(SNAPKV_NAME)

        prompts = load_search_prompts(per_suite=args.per_suite)
        print(
            f"Experiment 034 — search {len(prompts)} prompts × {len(compressors)} "
            f"compressors × {len(DRAFT_LENS)} draft_lens"
        )
        runtime = ModelRuntime(
            model_name=args.model, device=args.device, dtype=args.dtype
        )
        report = run_experiment(runtime, prompts, compressors)

    write_report_artifacts(
        report,
        json_path=json_path,
        csv_path=csv_path,
        md_path=md_path,
        trace_path=trace_path,
    )

    summary = report["search_summary"]
    demo = report.get("selected_demo")
    print(f"cells_searched={summary['cells_searched']} candidates={summary['demo_candidates']}")
    print(f"exactkv_failures={summary['exactkv_failures']}")
    if demo:
        print(
            f"selected: {demo['prompt_id']} × {demo['compressor_name']} "
            f"draft_len={demo['draft_len']} score={demo.get('demo_score', 0):.1f}"
        )
    print(f"Wrote {md_path}")

    if demo is None:
        return 1
    if not demo.get("exactkv_exact_match"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
