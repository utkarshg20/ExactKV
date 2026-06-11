"""Repair-policy pilot helpers for Experiment 020 (V11 Phase 5b).

Policy selection lives in the analysis/experiment layer only.  It chooses an
existing built-in compressor and ``draft_len`` per prompt; it does **not**
modify :class:`~exactkv.runtime.exactkv_generator.ExactKVGenerator` or
verification logic.

No timing, throughput, latency, speedup, or active_gpu_kv_bytes fields.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from exactkv.analysis.divergence_autopsy import load_autopsy_prompt_subset
from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime

FORBIDDEN_POLICY_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

EXP014_HARD_SUITES = (
    "long_context",
    "retrieval_copy",
    "tool_json",
    "code_structured",
)

POLICY_BASELINE_K8 = "baseline_k8_v4"
POLICY_BASELINE_BOUNDARY4 = "baseline_boundary4"
POLICY_FALLBACK_INT8_HARD = "fallback_int8_for_hard_categories"
POLICY_STRUCTURED_SAFE = "structured_safe_mode"
POLICY_CATEGORY_ADAPTIVE = "category_adaptive_policy"
POLICY_DRAFT_LEN_ADAPTIVE = "draft_len_adaptive_policy"

ALL_POLICIES = (
    POLICY_BASELINE_K8,
    POLICY_BASELINE_BOUNDARY4,
    POLICY_FALLBACK_INT8_HARD,
    POLICY_STRUCTURED_SAFE,
    POLICY_CATEGORY_ADAPTIVE,
    POLICY_DRAFT_LEN_ADAPTIVE,
)

_HARD_SUITES = frozenset({"long_context", "retrieval_copy"})
_STRUCTURED_SUITES = frozenset({"tool_json", "code_structured"})


@dataclass(frozen=True)
class PolicyCellSpec:
    """Resolved compressor and draft length for one prompt under a policy."""

    policy_name: str
    compressor_name: str
    draft_len: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_policy_artifact_safe(obj: Any, path: str = "artifact") -> None:
    if isinstance(obj, dict):
        hits = FORBIDDEN_POLICY_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            assert_policy_artifact_safe(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_policy_artifact_safe(item, f"{path}[{i}]")


def load_exp014_hard_subset(per_suite: int = 10) -> list[dict[str, Any]]:
    """Deterministic Exp-014-style hard-category subset (40 prompts at N=10)."""
    out: list[dict[str, Any]] = []
    for suite in EXP014_HARD_SUITES:
        rows = load_v10_suite(suite)
        rows.sort(key=lambda r: r["prompt_id"])
        for row in rows[:per_suite]:
            entry = dict(row)
            entry["v10_panel"] = "exp020_hard_panel"
            out.append(entry)
    return out


def load_pilot_prompts(*, panel: str = "exp019") -> list[dict[str, Any]]:
    """Return prompt list for Experiment 020."""
    if panel == "exp019":
        prompts = load_autopsy_prompt_subset(per_suite=5)
        for p in prompts:
            p["v10_panel"] = "exp020_exp019_panel"
        return prompts
    if panel == "exp014_hard":
        return load_exp014_hard_subset(per_suite=10)
    raise ValueError(f"Unknown panel {panel!r}; use 'exp019' or 'exp014_hard'")


def _suite_name(prompt_entry: dict[str, Any]) -> str:
    return str(prompt_entry.get("v10_suite", ""))


def resolve_policy_cell(
    policy_name: str,
    prompt_entry: dict[str, Any],
) -> PolicyCellSpec:
    """Map policy + prompt to compressor and draft_len (analysis layer only)."""
    if policy_name not in ALL_POLICIES:
        raise ValueError(f"Unknown policy {policy_name!r}")

    suite = _suite_name(prompt_entry)

    if policy_name == POLICY_BASELINE_K8:
        return PolicyCellSpec(policy_name, "k8_v4_sim", 4)

    if policy_name == POLICY_BASELINE_BOUNDARY4:
        return PolicyCellSpec(policy_name, "k8_v4_boundary4_v8_sim", 4)

    if policy_name == POLICY_FALLBACK_INT8_HARD:
        comp = "int8" if suite in _HARD_SUITES else "k8_v4_boundary4_v8_sim"
        return PolicyCellSpec(policy_name, comp, 4)

    if policy_name == POLICY_STRUCTURED_SAFE:
        comp = "int8" if suite in _STRUCTURED_SUITES else "k8_v4_boundary4_v8_sim"
        return PolicyCellSpec(policy_name, comp, 4)

    if policy_name == POLICY_CATEGORY_ADAPTIVE:
        # Exp 019/014: int8 wins on hard + structured suites; boundary4 on core_v2.
        if suite in _HARD_SUITES | _STRUCTURED_SUITES:
            comp = "int8"
        elif suite == "core_v2":
            comp = "k8_v4_boundary4_v8_sim"
        else:
            comp = "k8_v4_boundary4_v8_sim"
        return PolicyCellSpec(policy_name, comp, 4)

    if policy_name == POLICY_DRAFT_LEN_ADAPTIVE:
        if suite in _HARD_SUITES | {"tool_json"}:
            comp = "int8"
            dl = 4
        elif suite == "code_structured":
            comp = "int8"
            dl = 8
        elif suite == "core_v2":
            comp = "k8_v4_boundary4_v8_sim"
            dl = 8
        else:
            comp = "k8_v4_boundary4_v8_sim"
            dl = 4
        return PolicyCellSpec(policy_name, comp, dl)

    raise ValueError(f"Unhandled policy {policy_name!r}")


def run_policy_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    policy_name: str,
    *,
    max_new_tokens: int,
    compressor_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one policy cell: select compressor/draft_len, then standard ExactKV."""
    spec = resolve_policy_cell(policy_name, prompt_entry)
    cache = compressor_cache if compressor_cache is not None else {}
    if spec.compressor_name not in cache:
        cache[spec.compressor_name] = get_compressor(spec.compressor_name)
    compressor = cache[spec.compressor_name]

    prompt = prompt_entry["prompt"]
    full_res = generate_full_greedy(runtime, prompt, max_new_tokens)
    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)
    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    ekv_res = ExactKVGenerator(
        runtime, compressor, draft_len=spec.draft_len
    ).generate(prompt, max_new_tokens)
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)

    result: dict[str, Any] = {
        "prompt_id": prompt_entry["prompt_id"],
        "category": prompt_entry.get("category", "unknown"),
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "v10_primary_category": prompt_entry.get(
            "v10_primary_category", prompt_entry.get("category", "")
        ),
        "model_name": runtime.model_name,
        "policy_name": policy_name,
        "compressor_name": spec.compressor_name,
        "draft_len": spec.draft_len,
        "max_new_tokens": max_new_tokens,
        "policy_cell_spec": spec.to_dict(),
        "exactkv_failure": not ekv_exact,
        "lossy": {
            "token_exact_match": lossy_exact,
            "first_divergence_idx": lossy_div,
            "lossy_diverged": not lossy_exact,
        },
        "exactkv": {
            "token_exact_match": ekv_exact,
            "acceptance": acceptance.to_dict(),
        },
    }
    for key in ("v10_panel", "v10_id", "v10_suite_version", "v10_secondary_tags"):
        if key in prompt_entry:
            result[key] = prompt_entry[key]
    return result


def _mean_accept(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(r["exactkv"]["acceptance"]["acceptance_rate"] for r in results) / len(
        results
    )


def aggregate_policy_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize policy pilot cells."""
    total = len(results)
    failures = sum(1 for r in results if r.get("exactkv_failure"))

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_policy_suite: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_policy[r["policy_name"]].append(r)
        by_policy_suite[(r["policy_name"], r.get("v10_suite", ""))].append(r)

    global_by_policy = {}
    for policy, cells in sorted(by_policy.items()):
        acc = [c["exactkv"]["acceptance"] for c in cells]
        global_by_policy[policy] = {
            "num_cells": len(cells),
            "mean_acceptance_rate": _mean_accept(cells),
            "total_rejected": sum(a.get("total_rejected", 0) for a in acc),
            "total_corrections": sum(a.get("total_corrections", 0) for a in acc),
            "lossy_divergence_count": sum(
                1 for c in cells if c["lossy"].get("lossy_diverged")
            ),
            "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
        }

    per_category: list[dict[str, Any]] = []
    for (policy, suite), cells in sorted(by_policy_suite.items()):
        per_category.append({
            "policy_name": policy,
            "v10_suite": suite,
            "num_cells": len(cells),
            "mean_acceptance_rate": _mean_accept(cells),
            "total_rejected": sum(
                c["exactkv"]["acceptance"].get("total_rejected", 0) for c in cells
            ),
            "total_corrections": sum(
                c["exactkv"]["acceptance"].get("total_corrections", 0) for c in cells
            ),
        })

    models = sorted({r["model_name"] for r in results})

    def _prompt_lookup(policy: str, model_name: str) -> dict[str, float]:
        return {
            r["prompt_id"]: r["exactkv"]["acceptance"]["acceptance_rate"]
            for r in by_policy.get(policy, [])
            if r["model_name"] == model_name
        }

    def _compare_to_baseline(
        baseline_policy: str,
        policy: str,
    ) -> dict[str, Any]:
        wins = losses = ties = 0
        rows = []
        for model_name in models:
            baseline = _prompt_lookup(baseline_policy, model_name)
            for r in by_policy.get(policy, []):
                if r["model_name"] != model_name:
                    continue
                pid = r["prompt_id"]
                if pid not in baseline:
                    continue
                acc = r["exactkv"]["acceptance"]["acceptance_rate"]
                base = baseline[pid]
                delta = acc - base
                if delta > 1e-9:
                    wins += 1
                    outcome = "win"
                elif delta < -1e-9:
                    losses += 1
                    outcome = "loss"
                else:
                    ties += 1
                    outcome = "tie"
                rows.append({
                    "model_name": model_name,
                    "prompt_id": pid,
                    "v10_suite": r.get("v10_suite", ""),
                    "policy_acceptance": acc,
                    "baseline_acceptance": base,
                    "delta": delta,
                    "outcome": outcome,
                })
        return {
            "policy_name": policy,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "prompt_rows": rows,
        }

    comparisons_k8 = [
        _compare_to_baseline(POLICY_BASELINE_K8, p)
        for p in ALL_POLICIES
        if p != POLICY_BASELINE_K8
    ]
    comparisons_b4 = [
        _compare_to_baseline(POLICY_BASELINE_BOUNDARY4, p)
        for p in ALL_POLICIES
        if p != POLICY_BASELINE_BOUNDARY4
    ]

    best_vs_k8 = max(
        comparisons_k8,
        key=lambda x: (x["wins"] - x["losses"], x["wins"]),
        default=None,
    )
    best_vs_b4 = max(
        comparisons_b4,
        key=lambda x: (x["wins"] - x["losses"], x["wins"]),
        default=None,
    )

    return {
        "total_cells": total,
        "exactkv_failures": failures,
        "global_by_policy": global_by_policy,
        "per_category_by_policy": per_category,
        "comparisons_vs_baseline_k8_v4": comparisons_k8,
        "comparisons_vs_baseline_boundary4": comparisons_b4,
        "best_policy_vs_baseline_k8_v4": best_vs_k8,
        "best_policy_vs_baseline_boundary4": best_vs_b4,
        "hypothesis_evaluation": evaluate_repair_hypotheses(
            global_by_policy, per_category, comparisons_k8, comparisons_b4
        ),
    }


def evaluate_repair_hypotheses(
    global_by_policy: dict[str, dict[str, Any]],
    per_category: list[dict[str, Any]],
    comparisons_k8: list[dict[str, Any]],
    comparisons_b4: list[dict[str, Any]],
) -> dict[str, Any]:
    """Map Experiment 019 hypotheses to supported / rejected / inconclusive."""
    supported: list[str] = []
    rejected: list[str] = []
    inconclusive: list[str] = []

    def _cat_accept(policy: str, suite: str) -> float | None:
        for row in per_category:
            if row["policy_name"] == policy and row["v10_suite"] == suite:
                return row["mean_acceptance_rate"]
        return None

    b4_g = global_by_policy.get(POLICY_BASELINE_BOUNDARY4, {}).get(
        "mean_acceptance_rate"
    )
    hard = global_by_policy.get(POLICY_FALLBACK_INT8_HARD, {}).get(
        "mean_acceptance_rate"
    )
    struct = global_by_policy.get(POLICY_STRUCTURED_SAFE, {}).get(
        "mean_acceptance_rate"
    )
    cat = global_by_policy.get(POLICY_CATEGORY_ADAPTIVE, {}).get(
        "mean_acceptance_rate"
    )
    draft = global_by_policy.get(POLICY_DRAFT_LEN_ADAPTIVE, {}).get(
        "mean_acceptance_rate"
    )

    if hard is not None and b4_g is not None:
        if hard > b4_g + 1e-9:
            supported.append(
                "dynamic_fallback_int8: fallback_int8_for_hard_categories beats "
                "baseline_boundary4 globally"
            )
        else:
            rejected.append(
                "dynamic_fallback_int8: fallback_int8_for_hard_categories did not "
                "beat baseline_boundary4 globally"
            )

    lc_int8 = _cat_accept(POLICY_FALLBACK_INT8_HARD, "long_context")
    lc_b4 = _cat_accept(POLICY_BASELINE_BOUNDARY4, "long_context")
    if lc_int8 is not None and lc_b4 is not None:
        if lc_int8 > lc_b4 + 1e-9:
            supported.append(
                "dynamic_fallback_int8: int8 improves long_context vs boundary4"
            )
        else:
            inconclusive.append(
                "dynamic_fallback_int8: long_context int8 vs boundary4 inconclusive "
                f"({lc_int8:.3f} vs {lc_b4:.3f})"
            )

    if struct is not None and b4_g is not None:
        tj = _cat_accept(POLICY_STRUCTURED_SAFE, "tool_json")
        cs = _cat_accept(POLICY_STRUCTURED_SAFE, "code_structured")
        if struct > b4_g + 1e-9:
            supported.append(
                "structured_output_safe_mode: structured_safe_mode beats "
                "baseline_boundary4 globally"
            )
        if tj is not None and cs is not None and b4_g is not None:
            if tj >= _cat_accept(POLICY_BASELINE_BOUNDARY4, "tool_json") - 1e-9 and cs >= _cat_accept(POLICY_BASELINE_BOUNDARY4, "code_structured") - 1e-9:
                supported.append(
                    "structured_output_safe_mode: structured suites not hurt vs boundary4"
                )
            else:
                inconclusive.append(
                    "structured_output_safe_mode: mixed structured-suite deltas"
                )

    if cat is not None and b4_g is not None and cat > b4_g + 1e-9:
        supported.append(
            "category_adaptive_policy: beats baseline_boundary4 globally"
        )
    elif cat is not None and b4_g is not None:
        inconclusive.append("category_adaptive_policy: no global gain vs boundary4")

    if draft is not None and b4_g is not None:
        if draft > b4_g + 1e-9:
            supported.append(
                "lower_draft_len_on_low_margin: draft_len_adaptive_policy beats "
                "baseline_boundary4 globally"
            )
        else:
            inconclusive.append(
                "lower_draft_len_on_low_margin: draft_len_adaptive did not beat "
                "baseline_boundary4 globally"
            )

    k8_comp = next(
        (c for c in comparisons_k8 if c["policy_name"] == POLICY_CATEGORY_ADAPTIVE),
        None,
    )
    if k8_comp and k8_comp["wins"] > k8_comp["losses"]:
        supported.append(
            "category_adaptive_policy: more prompt wins than losses vs baseline_k8_v4"
        )

    return {
        "supported": supported,
        "rejected": rejected,
        "inconclusive": inconclusive,
    }
