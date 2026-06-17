"""Claim-safe demo packaging (Phase 17A / Exp 086).

Builds reusable demo cards and narrative from Phase 16 closeout evidence without
adding runtime functionality or new experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exactkv.attention.phase16_closeout import (
    ALLOWED_CLAIMS,
    FORBIDDEN_CLAIMS,
    FUTURE_DEFERRED_CLAIMS,
)

EXPERIMENT_086_ID = "exp086_claim_safe_demo_packaging"
DEFAULT_EXP086_REPORT = Path("reports/experiment_086_claim_safe_demo_packaging.json")
PHASE_17A = "17A"

DEMO_HOOK = (
    "Everyone is racing to shrink KV caches. ExactKV tells you when they start lying."
)
DEMO_PROBLEM_STATEMENT = (
    "KV compression should not be trusted. It should be crash-tested."
)
BENCHMARK_GAP_LINE = (
    "Outcome benchmarks can tell you whether the answer scored well. ExactKV tells "
    "you whether KV compression changed the model's path before the answer looked fine."
)

RECOMMENDED_NEXT_PHASE = "phase17b_broader_model_validation"

SOURCE_DOCS: tuple[str, ...] = (
    "docs/PHASE_16_CLOSEOUT.md",
    "docs/EXPERIMENT_085_PHASE16_CLOSEOUT_SUMMARY.md",
    "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md",
    "docs/EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md",
    "docs/EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md",
    "docs/EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md",
    "docs/EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md",
    "docs/EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md",
)

CARD_FORBIDDEN_WORDS: tuple[str, ...] = (
    "speedup",
    "throughput",
    "latency",
    "tokens_per_second",
    "runtime_seconds",
    "production-ready",
    "VeriCache reproduction",
)


def _inventory_source_docs(root: Path) -> tuple[list[str], list[str], dict[str, str]]:
    found: list[str] = []
    missing: list[str] = []
    contents: dict[str, str] = {}
    for rel in SOURCE_DOCS:
        path = root / rel
        if path.is_file():
            found.append(rel)
            try:
                contents[rel] = path.read_text()
            except OSError:
                contents[rel] = ""
        else:
            missing.append(rel)
    return found, missing, contents


def _extract_bullet_evidence(text: str, *, max_items: int = 3) -> list[str]:
    """Pull markdown bullet lines as evidence snippets; no invented metrics."""
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "| ")) and len(stripped) > 4:
            if stripped.startswith("|") and "---" in stripped:
                continue
            bullets.append(stripped.lstrip("-* ").strip())
        if len(bullets) >= max_items:
            break
    return bullets


def _doc_excerpt(path_key: str, contents: dict[str, str], fallback: str) -> str:
    text = contents.get(path_key, "")
    if not text:
        return fallback
    bullets = _extract_bullet_evidence(text)
    if bullets:
        return "; ".join(bullets[:2])
    # First non-empty paragraph after front matter
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines[:20]:
        if len(ln) > 40 and not ln.startswith(">"):
            return ln[:240]
    return fallback


def build_demo_cards(*, contents: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Build claim-safe demo cards from source doc inventory."""
    return {
        "attention_drift_card": {
            "title": "Attention drift & multi-layer numerics",
            "one_sentence_claim": (
                "ExactKV can offline-probe how compressed KV attention drifts across "
                "Qwen2.5 layers before you trust a smaller cache."
            ),
            "evidence": _doc_excerpt(
                "docs/EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md",
                contents,
                "Phase 16H Qwen-family divergence panel (Exp 073); multi-layer drift "
                "accumulation (069) and numerics audit (070) per Phase 16 closeout.",
            ),
            "allowed_words": [
                "offline",
                "diagnostic",
                "drift",
                "Qwen2.5",
                "shadow probe",
            ],
            "forbidden_words": list(CARD_FORBIDDEN_WORDS),
            "limitations": (
                "Offline replay only; not streaming-attention token-commit integration; "
                "panel-scoped evidence."
            ),
            "suggested_visual": "Layer-by-layer drift heatmap or prefix-length ladder chart.",
            "source_docs": [
                "docs/EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md",
                "docs/EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md",
                "docs/PHASE_16_CLOSEOUT.md",
            ],
        },
        "tolerance_policy_card": {
            "title": "Diagnostic tolerance policy",
            "one_sentence_claim": (
                "ExactKV classifies when local alignment holds versus when free-running "
                "divergence accumulates — as policy, not as an exactness guarantee."
            ),
            "evidence": _doc_excerpt(
                "docs/EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md",
                contents,
                "AttentionTolerancePolicy panel (Exp 074); depth-aware local-alignment vs "
                "free-running accumulation labels per Phase 16 closeout.",
            ),
            "allowed_words": ["tolerance policy", "diagnostic", "local alignment", "offline"],
            "forbidden_words": list(CARD_FORBIDDEN_WORDS) + ["exactness guarantee"],
            "limitations": (
                "Policy applies to offline shadow cells; top-k agreement is supplementary only."
            ),
            "suggested_visual": "Status timeline: local_alignment_pass → free_running_accumulation.",
            "source_docs": [
                "docs/EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md",
                "docs/PHASE_16_CLOSEOUT.md",
            ],
        },
        "external_generation_shadow_card": {
            "title": "External generation-shadow observer",
            "one_sentence_claim": (
                "ExactKV can replay fixed sequences after generation and compare "
                "streaming vs materialized attention without touching token commit."
            ),
            "evidence": _doc_excerpt(
                "docs/EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md",
                contents,
                "Decode-prefix ladder and expanded generation-shadow panels (079, 078); "
                "post-hoc shadow only per Phase 16 closeout.",
            ),
            "allowed_words": ["post-hoc", "generation-shadow", "diagnostic", "external observer"],
            "forbidden_words": list(CARD_FORBIDDEN_WORDS),
            "limitations": "Shadow runs after generation; cannot affect committed tokens.",
            "suggested_visual": "Prefix ladder with shadow status badges per decode step.",
            "source_docs": [
                "docs/EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md",
                "docs/PHASE_16_CLOSEOUT.md",
            ],
        },
        "live_round_observer_card": {
            "title": "Opt-in live round observer",
            "one_sentence_claim": (
                "ExactKV can record immutable post-commit round snapshots during "
                "opt-in generation with baseline parity in tested panels."
            ),
            "evidence": _doc_excerpt(
                "docs/EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md",
                contents,
                "Live observer + post-hoc shadow panel (Exp 082): 16/16 baseline-vs-observer "
                "token/text parity per documented run summary.",
            ),
            "allowed_words": ["live observer", "opt-in", "snapshot", "parity", "diagnostic"],
            "forbidden_words": list(CARD_FORBIDDEN_WORDS),
            "limitations": (
                "Default runtime unchanged when observer disabled; observer return values ignored."
            ),
            "suggested_visual": "Round timeline: commit → snapshot → post-hoc shadow.",
            "source_docs": [
                "docs/EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md",
                "docs/PHASE_16_CLOSEOUT.md",
            ],
        },
        "guarded_decode_shadow_card": {
            "title": "Guarded decode-time shadow dry-run",
            "one_sentence_claim": (
                "ExactKV can run guarded shadow diagnostics inside an observer callback "
                "without changing generated tokens in tested panels."
            ),
            "evidence": _doc_excerpt(
                "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md",
                contents,
                "Expanded guarded decode-time panel (Exp 084): 32/32 baseline-vs-guarded "
                "parity; 53/53 decode-time callbacks; decode-time vs post-hoc match 32/32.",
            ),
            "allowed_words": [
                "guarded",
                "decode-time",
                "dry-run",
                "diagnostic",
                "callback",
            ],
            "forbidden_words": list(CARD_FORBIDDEN_WORDS),
            "limitations": (
                "Not streaming-attention token-commit integration; shadow cannot affect commits."
            ),
            "suggested_visual": "Side-by-side baseline vs guarded token IDs (match) + shadow panel.",
            "source_docs": [
                "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md",
                "docs/PHASE_16_CLOSEOUT.md",
            ],
        },
        "claim_freeze_card": {
            "title": "Phase 16 claim freeze",
            "one_sentence_claim": (
                "ExactKV's public story is bounded: diagnostics yes, speed/memory/serving no."
            ),
            "evidence": _doc_excerpt(
                "docs/EXPERIMENT_085_PHASE16_CLOSEOUT_SUMMARY.md",
                contents,
                "Exp 085 closeout: 19/19 reports inventoried; allowed/forbidden claims frozen.",
            ),
            "allowed_words": ["claim-safe", "scoped", "panel evidence", "diagnostic"],
            "forbidden_words": list(CARD_FORBIDDEN_WORDS),
            "limitations": "Claims must cite specific experiment scope; Phase 17 does not widen them.",
            "suggested_visual": "Two-column allowed vs forbidden claims table.",
            "source_docs": [
                "docs/EXPERIMENT_085_PHASE16_CLOSEOUT_SUMMARY.md",
                "docs/PHASE_16_CLOSEOUT.md",
            ],
        },
    }


def build_demo_sections(*, contents: dict[str, str]) -> dict[str, str]:
    return {
        "hook": DEMO_HOOK,
        "problem": DEMO_PROBLEM_STATEMENT,
        "why_outcome_benchmarks_are_not_enough": BENCHMARK_GAP_LINE,
        "what_exactkv_checks": (
            "ExactKV crash-tests KV compression paths: offline streaming-attention drift, "
            "generation-shadow replay at round boundaries, and guarded observer diagnostics "
            "that never feed back into token commit."
        ),
        "phase16_attention_shadow_evidence": _doc_excerpt(
            "docs/EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md",
            contents,
            "Phases 16A–16H: tensor feasibility through Qwen-family divergence panels.",
        ),
        "generation_shadow_evidence": _doc_excerpt(
            "docs/EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md",
            contents,
            "Phases 16J–16O: wiring review through round-log shadow observers.",
        ),
        "live_round_observer_evidence": _doc_excerpt(
            "docs/EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md",
            contents,
            "Phases 16P–16Q: opt-in live snapshots with baseline parity and post-hoc shadow.",
        ),
        "guarded_decode_time_shadow_evidence": _doc_excerpt(
            "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md",
            contents,
            "Phases 16R–16S: guarded callback-time shadow with 32/32 parity in documented panel.",
        ),
        "claim_boundaries": (
            "Allowed: offline diagnostics, Qwen probes, tolerance policy, shadow observers, "
            "tested-panel parity and zero exactkv_failures. Forbidden: speed, throughput, latency, "
            "memory savings, serving, VeriCache reproduction, production-ready."
        ),
        "demo_takeaway": (
            "ExactKV is a correctness-first KV compression crash-test lab: it shows when "
            "compressed caches diverge from full attention before benchmarks look fine."
        ),
        "next_work": (
            "Deferred: CUDA/Triton kernels, vLLM/LMCache, active memory validation, broader "
            "models/contexts — only with explicit approval after claim-safe demo."
        ),
    }


def build_q_and_a() -> list[dict[str, str]]:
    return [
        {
            "question": "Are you claiming speedups?",
            "answer": "No. ExactKV does not claim speed, throughput, or latency improvements.",
        },
        {
            "question": "Are you claiming memory savings?",
            "answer": (
                "No active GPU or production memory savings claim. Phase 16 used theoretical/"
                "diagnostic memory accounting only where documented."
            ),
        },
        {
            "question": "Did you reproduce VeriCache serving?",
            "answer": (
                "No. ExactKV does not reproduce VeriCache throughput or serving results."
            ),
        },
        {
            "question": "Is streaming attention used for token commit?",
            "answer": (
                "No. Streaming attention is not integrated into token commit. Shadow work is "
                "diagnostic-only observer instrumentation in tested paths."
            ),
        },
        {
            "question": "What did Phase 16 prove?",
            "answer": (
                "Diagnostic guarded shadow infrastructure works in tested panels without "
                "changing generated tokens; exactkv_failures remained zero on cited panels."
            ),
        },
        {
            "question": "What remains?",
            "answer": (
                "CUDA/Triton kernels, vLLM/LMCache integration, active memory validation, "
                "broader model and longer-context validation, and any real compressed-attention "
                "token-commit path — deferred pending explicit approval."
            ),
        },
        {
            "question": "Does top-k agreement prove exactness?",
            "answer": (
                "No. Top-k agreement is supplementary only and is not an exactness guarantee."
            ),
        },
    ]


def run_exp086_claim_safe_demo_packaging(*, root: Path | None = None) -> dict[str, Any]:
    """Build Experiment 086 claim-safe demo packaging report."""
    repo_root = root or Path(".")
    found, missing, contents = _inventory_source_docs(repo_root)
    cards = build_demo_cards(contents=contents)
    sections = build_demo_sections(contents=contents)
    q_and_a = build_q_and_a()

    status = "complete" if not missing else "complete_with_missing_docs"

    return {
        "experiment_id": EXPERIMENT_086_ID,
        "status": status,
        "phase": PHASE_17A,
        "source_docs_found": found,
        "source_docs_missing": missing,
        "demo_hook": DEMO_HOOK,
        "demo_problem_statement": DEMO_PROBLEM_STATEMENT,
        "benchmark_gap_line": BENCHMARK_GAP_LINE,
        "demo_sections": sections,
        "demo_cards": cards,
        "allowed_claims": list(ALLOWED_CLAIMS),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "deferred_work": list(FUTURE_DEFERRED_CLAIMS),
        "q_and_a": q_and_a,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE,
        "limitations": [
            "Claim-safe demo packaging only; no new runtime functionality.",
            "Evidence drawn from local Phase 16 docs; missing docs listed not invented.",
            "Top-k agreement is supplementary only.",
            "ExactKV does not reproduce VeriCache throughput or serving results.",
            "No speed, throughput, latency, serving, or GPU memory claims.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
    }


def validate_exp086_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "phase",
        "source_docs_found",
        "source_docs_missing",
        "demo_hook",
        "demo_problem_statement",
        "benchmark_gap_line",
        "demo_sections",
        "demo_cards",
        "allowed_claims",
        "forbidden_claims",
        "deferred_work",
        "q_and_a",
        "recommended_next_phase",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_086_ID:
        errors.append("experiment_id mismatch")

    if report.get("demo_hook") != DEMO_HOOK:
        errors.append("demo_hook mismatch")

    if report.get("demo_problem_statement") != DEMO_PROBLEM_STATEMENT:
        errors.append("demo_problem_statement mismatch")

    if report.get("benchmark_gap_line") != BENCHMARK_GAP_LINE:
        errors.append("benchmark_gap_line mismatch")

    for claim in ALLOWED_CLAIMS:
        if claim not in (report.get("allowed_claims") or []):
            errors.append(f"missing allowed claim: {claim}")

    for claim in FORBIDDEN_CLAIMS:
        if claim not in (report.get("forbidden_claims") or []):
            errors.append(f"missing forbidden claim: {claim}")

    card_keys = (
        "attention_drift_card",
        "tolerance_policy_card",
        "external_generation_shadow_card",
        "live_round_observer_card",
        "guarded_decode_shadow_card",
        "claim_freeze_card",
    )
    cards = report.get("demo_cards") or {}
    for ck in card_keys:
        if ck not in cards:
            errors.append(f"missing demo card: {ck}")
            continue
        card = cards[ck]
        for field in ("title", "one_sentence_claim", "evidence", "limitations", "source_docs"):
            if not card.get(field):
                errors.append(f"demo_cards.{ck} missing {field}")

    for qa in report.get("q_and_a") or []:
        ans = (qa.get("answer") or "").lower()
        if "speed" in (qa.get("question") or "").lower() and "no" not in ans:
            errors.append("Q&A must reject speed claims")

    return errors
