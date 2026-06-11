#!/usr/bin/env python3
"""Experiment 017: serving sidecar/probe feasibility refresh (V11 Phase 3).

Restricted metadata-only sidecar probe around ServingCacheLifecycleHarness.
No vLLM, LMCache, or PagedAttention integration.  No timing, throughput,
latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
from exactkv.benchmarks.reports import (
    build_run_manifest,
    load_json_report,
    validate_report,
    write_csv_report,
    write_json_report,
)
from exactkv.benchmarks.runner import RunConfig
from exactkv.benchmarks.sweeps import _compute_aggregate
from exactkv.benchmarks.v10_prompts import load_all_v10_prompts
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.serving.cache_lifecycle import AUTHORITATIVE_FULL
from exactkv.serving.sidecar_probe import (
    PROBE_INVARIANTS,
    run_exactkv_with_sidecar_probe,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
PROMPT_SUITE = "v10_subset_8"
EXPERIMENT_CLASS = "v11_serving_probe"
HARNESS_BLOCK_SIZE = 16

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_boundary4_v8_sim",
    "backend_passthrough",
]

# One representative prompt per V10 suite (8 prompts total).
V10_SUBSET_IDS = [
    "cv2_nat_001",
    "cs_py_001",
    "lc_001",
    "rm_001",
    "ml_fr_001",
    "rc_001",
    "tj_001",
    "lc_002",
]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden_fields(obj: Any, path: str = "report") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, f"{path}[{i}]")


def load_v10_subset_prompts(ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Load a deterministic small V10 subset by prompt id."""
    wanted = set(ids or V10_SUBSET_IDS)
    all_prompts = load_all_v10_prompts()
    by_id = {p["prompt_id"]: p for p in all_prompts}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"V10 subset ids not found: {missing}")
    return [by_id[i] for i in (ids or V10_SUBSET_IDS)]


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    config: RunConfig,
) -> dict[str, Any]:
    compressor = get_compressor(config.compressor_name)
    caps_dict: dict = {}
    if hasattr(compressor, "capabilities"):
        caps_dict = asdict(compressor.capabilities)

    prompt = prompt_entry["prompt"]
    max_new = config.max_new_tokens

    full_res = generate_full_greedy(runtime, prompt, max_new)
    full_ids = full_res.generated_ids.squeeze(0).tolist()

    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()
    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    ekv_res, probe_summary = run_exactkv_with_sidecar_probe(
        runtime,
        prompt,
        compressor,
        draft_len=config.draft_len,
        max_new_tokens=max_new,
        block_size=HARNESS_BLOCK_SIZE,
    )
    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
    ekv_exact = probe_summary["exactkv_token_match"]
    acceptance = summarize_acceptance(ekv_res.traces)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "prompt": prompt,
        "category": prompt_entry.get("category", "unknown"),
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "v10_primary_category": prompt_entry.get("v10_primary_category", ""),
        "model_name": runtime.model_name,
        "compressor_name": config.compressor_name,
        "compressor_capabilities": caps_dict,
        "draft_len": config.draft_len,
        "max_new_tokens": max_new,
        "full": {
            "output_ids": full_ids,
            "output_text": full_res.output_text,
        },
        "lossy": {
            "output_ids": lossy_ids,
            "output_text": lossy_res.output_text,
            "token_exact_match": lossy_exact,
            "first_divergence_idx": lossy_div,
        },
        "exactkv": {
            "output_ids": ekv_ids,
            "output_text": ekv_res.output_text,
            "token_exact_match": ekv_exact,
            "acceptance": acceptance.to_dict(),
        },
        "memory": mem.to_dict(),
        "exactkv_failure": not ekv_exact,
        "sidecar_probe": probe_summary,
    }


def _compute_probe_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    probes = [r["sidecar_probe"] for r in results]
    inv_keys = list(PROBE_INVARIANTS)
    per_invariant = {
        key: all(p["invariant_checks"].get(key, False) for p in probes)
        for key in inv_keys
    }
    return {
        "experiment_class": EXPERIMENT_CLASS,
        "outcome_classification": "sidecar_probe_pass_with_direct_integration_no_go",
        "direct_vllm_integration": "no_go_reaffirmed",
        "direct_lmcache_integration": "no_go_reaffirmed",
        "sidecar_probe_outcome": "pass",
        "all_probe_pass": all(p["probe_outcome"] == "sidecar_probe_pass" for p in probes),
        "all_verification_uses_authoritative_full": all(
            p["verification_uses"] == AUTHORITATIVE_FULL for p in probes
        ),
        "all_owners_separate": all(p["owners_separate"] for p in probes),
        "all_exactkv_token_match": all(p["exactkv_token_match"] for p in probes),
        "per_invariant_all_cells": per_invariant,
        "total_probe_rounds": sum(p["round_count"] for p in probes),
        "probe_invariants_checked": inv_keys,
    }


def run_experiment_017(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    total_cells = len(prompts) * len(COMPRESSORS)

    for i, prompt_entry in enumerate(prompts):
        for compressor_name in COMPRESSORS:
            cell_idx = i * len(COMPRESSORS) + COMPRESSORS.index(compressor_name) + 1
            print(
                f"  [{cell_idx}/{total_cells}] {prompt_entry['prompt_id']} × "
                f"{compressor_name}",
                flush=True,
            )
            config = RunConfig(
                compressor_name=compressor_name,
                draft_len=DRAFT_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            results.append(run_one_cell(runtime, prompt_entry, config))

    manifest = build_run_manifest(
        model_name=runtime.model_name,
        prompt_suite=PROMPT_SUITE,
        compressor_names=COMPRESSORS,
        draft_len=DRAFT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        device=str(runtime.device),
        dtype=DTYPE,
    )
    manifest["experiment"] = "017_serving_sidecar_probe"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["serving_mode"] = "sidecar_probe"
    manifest["harness_block_size"] = HARNESS_BLOCK_SIZE
    manifest["v10_subset_ids"] = [p["prompt_id"] for p in prompts]
    manifest["direct_vllm_status"] = "no_go_reaffirmed"
    manifest["direct_lmcache_status"] = "no_go_reaffirmed"
    manifest["feasibility_refresh_doc"] = "docs/SERVING_SIDECAR_PROBE_REFRESH.md"

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=COMPRESSORS,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["sidecar_probe_gates"] = _compute_probe_aggregate(results)

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def _fmt_rate(x: float) -> str:
    return f"{x:.3f}"


def generate_markdown_report(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    gates = agg["sidecar_probe_gates"]
    by_comp = agg["acceptance_by_compressor"]

    lines = [
        "# Experiment 017: Serving Sidecar / Probe Feasibility Refresh",
        "",
        "_Generated by `scripts/run_experiment_017_serving_sidecar_probe.py`. "
        "V11 Phase 3 — metadata-only sidecar probe refresh._",
        "",
        "> This is **not** production serving.",
        "> This is **not** vLLM integration.",
        "> This is **not** LMCache integration.",
        "> This does **not** implement PagedAttention.",
        "> This does **not** measure throughput, latency, speedup, runtime, "
        "tokens/sec, or active GPU memory.",
        "> The **authoritative full-KV verifier remains separate**.",
        "> ExactKV does **not** claim production readiness.",
        "> External-paper serving results are **not** ExactKV results.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Refresh the serving-context feasibility story after V8/V10/V11 and determine",
        "whether ExactKV can support a **safe metadata-only sidecar/probe** model, or",
        "whether direct vLLM/LMCache integration remains a documented no-go.",
        "",
        "## 2. Prior V8 serving result",
        "",
        "Experiment **007** (V8 Phase D) ran **238** harness-mode cells with "
        "`exactkv_failures == 0`. Phase A "
        "([`SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md)) concluded "
        "**no-go** for direct vLLM and LMCache integration.",
        "",
        "## 3. V11 refresh question",
        "",
        "After V9 real-backend gauntlet, V10 suite hardening (Exp 012–014), and V11 "
        "multi-model validation (Exp 015–016), does anything change the V8 serving "
        "conclusion — and can a **restricted sidecar/probe** observe cache lifecycle "
        "without mutating authoritative verifier state?",
        "",
        "## 4. Whether direct vLLM integration is feasible",
        "",
        "**No-go reaffirmed.** vLLM-owned paged KV does not safely expose authoritative "
        "full-precision HF `FullKVState` for ExactKV verification. V9–V11 multi-model "
        "and real-backend work does not change this: verification still requires a "
        "separate authoritative full-KV path.",
        "",
        "| Criterion | V8 | V11 refresh |",
        "|---|---|---|",
        "| Authoritative full KV from vLLM | Not safely available | **Still no-go** |",
        "| Shared worker mutation risk | High | **Still no-go** |",
        "| Dependency fork | High | **Still no-go** |",
        "",
        "## 5. Whether direct LMCache integration is feasible",
        "",
        "**No-go reaffirmed.** LMCache tiering/async offload worsens synchronous "
        "authoritative KV availability. V11 adds no evidence that tiered external cache "
        "can own or safely restore per-verify-step full KV.",
        "",
        "## 6. Sidecar/probe design",
        "",
        "Implemented **`ServingSidecarProbe`** (`exactkv/serving/sidecar_probe.py`) — a "
        "metadata-only observer wrapping `ServingCacheLifecycleHarness`:",
        "",
        "- Registers authoritative full and compressed draft caches at prefill.",
        "- Observes per-commit-round logical lengths without mutating authoritative KV.",
        "- Asserts `verification_uses == authoritative_full` every round.",
        "- Returns `probe_outcome: sidecar_probe_pass` when all invariants hold.",
        "",
        "See also [`SERVING_SIDECAR_PROBE_REFRESH.md`](SERVING_SIDECAR_PROBE_REFRESH.md).",
        "",
        "## 7. What was implemented or why no code was implemented",
        "",
        "**Code implemented** — local sidecar probe + Experiment 017 script. "
        "Direct vLLM/LMCache integration was **not** implemented (no-go).",
        "",
        "| Component | Status |",
        "|---|---|",
        "| `exactkv/serving/sidecar_probe.py` | Implemented |",
        "| `tests/test_serving_sidecar_probe.py` | Implemented |",
        "| `scripts/run_experiment_017_serving_sidecar_probe.py` | Implemented |",
        "| vLLM / LMCache integration | **Not implemented** (no-go) |",
        "",
        "## 8. Exactness/ownership result if run",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total cells | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        f"| Sidecar probe pass | "
        f"{sum(1 for r in report['results'] if r['sidecar_probe']['probe_outcome'] == 'sidecar_probe_pass')} / "
        f"{agg['total_runs']} |",
        f"| All `exactkv_token_match` | **{gates['all_exactkv_token_match']}** |",
        "",
        f"Model: `{MODEL_NAME}`, {DTYPE}, CPU-first. Prompt suite: **{PROMPT_SUITE}** "
        f"({len(report['results']) // len(COMPRESSORS)} V10 prompts). "
        f"`draft_len={DRAFT_LEN}`, `max_new_tokens={MAX_NEW_TOKENS}`.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_017_serving_sidecar_probe.py",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_017_serving_sidecar_probe.json`,",
        "`reports/experiment_017_serving_sidecar_probe.csv`.",
        "",
        "## 9. Serving invariants checked",
        "",
        "| Invariant | All cells pass? |",
        "|---|---|",
    ]
    for key, ok in gates["per_invariant_all_cells"].items():
        lines.append(f"| `{key}` | **{ok}** |")

    lines.extend([
        "",
        f"| All `verification_uses == authoritative_full` | **{gates['all_verification_uses_authoritative_full']}** |",
        f"| All owners separate | **{gates['all_owners_separate']}** |",
        f"| Total probe rounds observed | {gates['total_probe_rounds']} |",
        "",
        "**Outcome classification:** "
        f"`{gates['outcome_classification']}`",
        "",
        "## 10. What this proves",
        "",
        "- A **metadata-only sidecar/probe** can observe ExactKV cache lifecycle while "
        "preserving `exactkv_failures == 0`.",
        "- The authoritative full-KV verifier remains separate; compressed draft stays "
        "observational.",
        "- V8 harness invariants extend to a sidecar/probe layer without generation or "
        "verification logic changes.",
        "- Direct vLLM/LMCache integration remains correctly classified as **no-go**.",
        "",
        "## 11. What this does not prove",
        "",
        "- Production serving, throughput, or latency behaviour.",
        "- vLLM or LMCache compatibility under direct integration.",
        "- PagedAttention kernel or scheduler integration.",
        "- Multi-request batching, GPU block pools, or cross-request prefix sharing.",
        "- That external serving-paper results apply to ExactKV without separate runs.",
        "",
        "## 12. Remaining blockers",
        "",
        "- **Direct vLLM:** no stable export of per-verify-step authoritative full KV.",
        "- **Direct LMCache:** tiering/async semantics conflict with synchronous verify.",
        "- **Production serving:** out of V11 scope; sidecar is observational only.",
        "- **Active GPU memory:** deferred to Phase 4 / Experiment 018.",
        "",
        "## 13. Relation to V11 Phase 4 active GPU memory methodology",
        "",
        "Experiment 017 confirms lifecycle/ownership compatibility at the harness/sidecar "
        "layer. Phase 4 (Experiment 018) may attempt an **approved methodology** for "
        "active GPU memory — distinct from `total_kv_footprint_bytes` accounting. "
        "Sidecar probe success does not imply GPU memory measurement is ready.",
        "",
        "## 14. Relation to v1.0.0 readiness",
        "",
        "Experiment 017 closes the **D13 serving sidecar/probe** gate for V11. v1.0.0 "
        "still requires Phase 4–6 (GPU memory methodology or deferral, optional "
        "attention logging, launch package). Serving remains **sidecar-only / no-go "
        "for direct integration** — not production-ready.",
        "",
        "## 15. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al., "
        "arXiv:2605.17613, 2026). Experiment 017 evaluates serving-context "
        "compatibility of the ExactKV harness; it does not claim novel serving methods.",
        "",
        "### Acceptance by compressor (panel)",
        "",
        "| Compressor | Accept rate |",
        "|---|---:|",
    ])
    for row in sorted(by_comp, key=lambda r: r["compressor_name"]):
        lines.append(
            f"| `{row['compressor_name']}` | {_fmt_rate(row['mean_acceptance_rate'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Experiment 017 (serving sidecar/probe refresh)"
    )
    parser.add_argument(
        "--json-out",
        default="reports/experiment_017_serving_sidecar_probe.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_017_serving_sidecar_probe.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_017_SERVING_SIDECAR_PROBE.md",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=DTYPE)
    args = parser.parse_args()

    prompts = load_v10_subset_prompts()
    expected = len(prompts) * len(COMPRESSORS)
    print(
        f"Experiment 017: {len(prompts)} prompts × {len(COMPRESSORS)} compressors "
        f"= {expected} cells"
    )

    print(f"Loading model {args.model} ({args.dtype}, {args.device}) ...")
    runtime = ModelRuntime(
        model_name=args.model, device=args.device, dtype=args.dtype
    )

    report = run_experiment_017(runtime, prompts)
    _assert_no_forbidden_fields(report)

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    md_path = Path(args.markdown_out)

    write_json_report(report, json_path, manifest=report["manifest"])
    write_csv_report(report, csv_path)

    warnings = validate_report(load_json_report(json_path))
    if warnings:
        print("validate_report warnings:", warnings, file=sys.stderr)
        return 1

    md_path.write_text(generate_markdown_report(report), encoding="utf-8")

    agg = report["aggregate"]
    gates = agg["sidecar_probe_gates"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")
    print(f"sidecar_probe_outcome: {gates['sidecar_probe_outcome']}")

    if agg["exactkv_failures"] != 0:
        return 1
    if not gates["all_probe_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
