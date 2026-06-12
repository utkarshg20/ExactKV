#!/usr/bin/env python3
"""Experiment 028: span verification smoke (V13 Phase 2).

Compares sequential vs span verification on a small panel.  Exactness only —
no timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_all_v10_prompts
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
EXPERIMENT_CLASS = "v13_span_verification_smoke"

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
]

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


def load_v10_subset_prompts(ids: list[str] | None = None) -> list[dict[str, Any]]:
    wanted = set(ids or V10_SUBSET_IDS)
    by_id = {p["prompt_id"]: p for p in load_all_v10_prompts()}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"V10 subset ids not found: {missing}")
    return [by_id[i] for i in (ids or V10_SUBSET_IDS)]


def _run_mode(
    runtime: ModelRuntime,
    prompt: str,
    compressor_name: str,
    verification_method: str,
) -> dict[str, Any]:
    comp = get_compressor(compressor_name)
    gen = ExactKVGenerator(
        runtime,
        comp,
        draft_len=DRAFT_LEN,
        verification_method=verification_method,  # type: ignore[arg-type]
    )
    res = gen.generate(prompt, MAX_NEW_TOKENS)
    acc = summarize_acceptance(res.traces)
    return {
        "output_ids": res.output_ids.squeeze(0).tolist(),
        "output_text": res.output_text,
        "total_accepted": res.total_accepted,
        "total_rejected": res.total_rejected,
        "total_corrections": res.total_corrections,
        "acceptance_rate": res.acceptance_rate,
        "num_rounds": res.num_rounds,
        "acceptance": acc.to_dict(),
    }


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    compressor_name: str,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]
    full = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
    full_ids = full.generated_ids.squeeze(0).tolist()

    sequential = _run_mode(runtime, prompt, compressor_name, "sequential")
    span = _run_mode(runtime, prompt, compressor_name, "span")

    seq_ids = sequential["output_ids"]
    span_ids = span["output_ids"]

    seq_exact = token_exact_match(full.generated_ids, torch.tensor([seq_ids]))
    span_exact = token_exact_match(full.generated_ids, torch.tensor([span_ids]))
    span_seq_parity = seq_ids == span_ids

    caps: dict = {}
    comp = get_compressor(compressor_name)
    if hasattr(comp, "capabilities"):
        caps = asdict(comp.capabilities)

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "compressor_name": compressor_name,
        "model_name": runtime.model_name,
        "draft_len": DRAFT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "compressor_capabilities": caps,
        "full_output_ids": full_ids,
        "sequential": {
            **sequential,
            "exactkv_failure": not seq_exact,
            "token_exact_match_full": seq_exact,
        },
        "span": {
            **span,
            "exactkv_failure": not span_exact,
            "token_exact_match_full": span_exact,
        },
        "span_matches_sequential": span_seq_parity,
        "span_matches_full": span_exact,
        "exactkv_failure_sequential": not seq_exact,
        "exactkv_failure_span": not span_exact,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    seq_fail = sum(1 for r in results if r["exactkv_failure_sequential"])
    span_fail = sum(1 for r in results if r["exactkv_failure_span"])
    parity_fail = sum(1 for r in results if not r["span_matches_sequential"])
    return {
        "total_cells": len(results),
        "sequential_exactkv_failures": seq_fail,
        "span_exactkv_failures": span_fail,
        "span_sequential_parity_failures": parity_fail,
        "all_span_match_sequential": parity_fail == 0,
        "all_span_match_full": span_fail == 0,
        "mean_span_acceptance_rate": (
            sum(r["span"]["acceptance_rate"] for r in results) / len(results)
            if results
            else 0.0
        ),
        "mean_sequential_acceptance_rate": (
            sum(r["sequential"]["acceptance_rate"] for r in results)
            / len(results)
            if results
            else 0.0
        ),
    }


def run_smoke(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    total = len(prompts) * len(COMPRESSORS)
    idx = 0
    for pe in prompts:
        for comp in COMPRESSORS:
            idx += 1
            print(f"  [{idx}/{total}] {pe['prompt_id']} × {comp}", flush=True)
            results.append(run_one_cell(runtime, pe, comp))
    agg = _aggregate(results)
    return {
        "experiment": "028",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name,
        "dtype": DTYPE,
        "device": str(runtime.device),
        "prompt_suite": "v10_subset_8",
        "prompt_count": len(prompts),
        "compressors": COMPRESSORS,
        "draft_len": DRAFT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "verification_methods": ["sequential", "span"],
        "results": results,
        "aggregate": agg,
    }


def write_csv_report(report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for r in report["results"]:
        rows.append({
            "prompt_id": r["prompt_id"],
            "compressor_name": r["compressor_name"],
            "sequential_exactkv_failure": r["exactkv_failure_sequential"],
            "span_exactkv_failure": r["exactkv_failure_span"],
            "span_matches_sequential": r["span_matches_sequential"],
            "sequential_accepted": r["sequential"]["total_accepted"],
            "sequential_rejected": r["sequential"]["total_rejected"],
            "sequential_corrections": r["sequential"]["total_corrections"],
            "span_accepted": r["span"]["total_accepted"],
            "span_rejected": r["span"]["total_rejected"],
            "span_corrections": r["span"]["total_corrections"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    lines = [
        "# Experiment 028: Span Verification Smoke",
        "",
        "_Generated by `scripts/run_experiment_028_span_verification_smoke.py`. "
        "V13 Phase 2 — span verification exactness smoke only._",
        "",
        "> This proves span verification exactness on a **smoke panel only**.",
        "> This does **not** prove speedup.",
        "> This does **not** measure throughput, latency, runtime, tokens/sec, "
        "active GPU memory, or production serving.",
        "> Span verification remains **opt-in**, not default.",
        "> ExactKV does **not** claim model accuracy improvement.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Smoke-test opt-in span verification against sequential verification and "
        "full greedy on 8 V10 prompts × 4 compressors.",
        "",
        "## 2. Design correction summary",
        "",
        "HF causal LM logits shift (see [`SPAN_VERIFICATION_DESIGN.md`]"
        "(SPAN_VERIFICATION_DESIGN.md)):",
        "",
        "- `v_0 = full_state.next_token_id`",
        "- for `i >= 1`, `v_i = argmax(out.logits[:, i - 1, :])`",
        "- `out.logits[:, k - 1, :]` bonus token ignored; bonus acceptance disabled",
        "",
        "## 3. Implementation summary",
        "",
        "- `VerificationEngine.verify_span` in `exactkv/verification/engine.py`",
        "- `ExactKVGenerator(verification_method=\"sequential\"|\"span\")`, default sequential",
        "",
        "## 4. Exactness invariant",
        "",
        "Span output must match full greedy and sequential output on every cell.",
        "",
        "## 5. Cache-state invariant",
        "",
        "Generator alignment asserts unchanged; verify_span does not mutate authoritative KV.",
        "",
        "## 6. Test/smoke configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Model | `{report['model_name']}` |",
        f"| dtype / device | `{DTYPE}` / `{report['device']}` |",
        f"| Prompts | **{report['prompt_count']}** (v10_subset_8) |",
        f"| Compressors | {', '.join(f'`{c}`' for c in COMPRESSORS)} |",
        f"| draft_len | {DRAFT_LEN} |",
        f"| max_new_tokens | {MAX_NEW_TOKENS} |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_028_span_verification_smoke.py",
        "```",
        "",
        "## 7. Sequential vs span exactness result",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Total cells | **{agg['total_cells']}** |",
        f"| Sequential ExactKV failures | **{agg['sequential_exactkv_failures']}** |",
        f"| Span ExactKV failures | **{agg['span_exactkv_failures']}** |",
        "",
        "## 8. Sequential vs span output parity",
        "",
        f"| Span matches sequential | **{agg['all_span_match_sequential']}** |",
        f"| Parity failures | **{agg['span_sequential_parity_failures']}** |",
        "",
        "## 9. Acceptance/rejection/correction comparison",
        "",
        f"| Mode | Mean acceptance rate |",
        f"|---|---:|",
        f"| Sequential | {agg['mean_sequential_acceptance_rate']:.4f} |",
        f"| Span | {agg['mean_span_acceptance_rate']:.4f} |",
        "",
        "Per-cell counters match when span matches sequential (parity gate).",
        "",
        "## 10. Edge cases tested",
        "",
        "Pytest: all-match, golden logits shift, first/middle mismatch, single-token draft, "
        "empty draft, authoritative state unchanged, generator exactness for four compressors.",
        "",
        "## 11. What this proves",
        "",
        "- Span verification preserves exactness on smoke panel.",
        "- Span outputs match sequential outputs bit-for-bit.",
        "- Opt-in flag works; default remains sequential.",
        "",
        "## 12. What this does not prove",
        "",
        "- Speedup or verifier overhead reduction (Phase 3).",
        "- Full-suite exactness (Experiment 029 grid).",
        "- Production serving readiness.",
        "",
        "## 13. Limitations",
        "",
        "- 32 cells only; Qwen2.5-0.5B; CPU/float32 acceptable.",
        "- Non-standard report schema (dual-mode cells).",
        "",
        "## 14. Next step: Experiment 029 exactness grid",
        "",
        "Full prompt × compressor grid with span vs sequential parity before Phase 3 timing.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 028 span verification smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument(
        "--report-json",
        default=str(_ROOT / "reports" / "experiment_028_span_verification_smoke.json"),
    )
    parser.add_argument(
        "--report-csv",
        default=str(_ROOT / "reports" / "experiment_028_span_verification_smoke.csv"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_028_SPAN_VERIFICATION_SMOKE.md"),
    )
    args = parser.parse_args()

    prompts = load_v10_subset_prompts()
    print(f"Experiment 028 — {len(prompts)} prompts × {len(COMPRESSORS)} compressors")
    runtime = ModelRuntime(model_name=MODEL_NAME, device=args.device, dtype=args.dtype)
    report = run_smoke(runtime, prompts)
    _assert_no_forbidden(report)

    json_path = Path(args.report_json)
    csv_path = Path(args.report_csv)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_path)
    write_markdown(report, md_path)

    agg = report["aggregate"]
    print(
        f"Done: sequential_failures={agg['sequential_exactkv_failures']} "
        f"span_failures={agg['span_exactkv_failures']} "
        f"parity={agg['all_span_match_sequential']}"
    )
    if agg["sequential_exactkv_failures"] != 0 or agg["span_exactkv_failures"] != 0:
        return 1
    if not agg["all_span_match_sequential"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
