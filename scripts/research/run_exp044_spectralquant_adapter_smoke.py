#!/usr/bin/env python3
"""Experiment 044: SpectralQuant experimental adapter ExactKV smoke (Phase 10F).

Factory-only ``spectralquant_experimental`` adapter — NOT default registry.
Tiny prompt panel; deterministic greedy; reports exactkv_failures and acceptance.

Requires Exp 043 real-KV path (calibration + tensor API) to succeed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from exactkv.external.spectralquant_adapter import (  # noqa: E402
    MEMORY_CLAIM_NOTE,
    create_spectralquant_experimental_adapter,
)
from exactkv.external.spectralquant_real_kv import (  # noqa: E402
    CLAIMS_FORBIDDEN,
    DEFAULT_MODEL,
    CalibrationConfig,
    EXPERIMENT_044_ID,
    ensure_spectralquant_path,
    validate_044_report,
)
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

DEFAULT_JSON = _ROOT / "reports" / "experiment_044_spectralquant_adapter_smoke.json"

MODEL_NAME = DEFAULT_MODEL
MAX_NEW_TOKENS = 16
DRAFT_LEN = 4
ADAPTER_NAME = "spectralquant_experimental"

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})


def _assert_no_forbidden_fields(obj: Any, path: str = "report") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden performance fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, f"{path}[{i}]")


def load_smoke_prompt_panel() -> list[dict[str, str]]:
    try:
        from exactkv.benchmarks.v10_prompts import load_v10_suite

        suites = ("core_v2", "long_context", "retrieval_copy", "tool_json")
        out: list[dict[str, str]] = []
        for suite_name in suites:
            suite = load_v10_suite(suite_name)
            if suite:
                out.append(suite[0])
        if out:
            return out[:4]
    except Exception:
        pass
    return [
        {"prompt_id": "sq_smoke_0", "category": "smoke", "prompt": "What is 2+2? One word."},
        {"prompt_id": "sq_smoke_1", "category": "smoke", "prompt": "Name the capital of France."},
    ]


def run_one_cell(
    runtime: ModelRuntime,
    adapter: Any,
    prompt_entry: dict[str, str],
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]
    full_res = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
    ekv_res = ExactKVGenerator(
        runtime,
        adapter,
        draft_len=DRAFT_LEN,
    ).generate(prompt, MAX_NEW_TOKENS)

    exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)
    div_idx = first_divergence_idx(full_res.generated_ids, ekv_res.output_ids)

    return {
        "prompt_id": prompt_entry.get("prompt_id", "unknown"),
        "category": prompt_entry.get("category", "unknown"),
        "exactkv_failures": 0 if exact else 1,
        "token_exact_match": exact,
        "acceptance_rate": acceptance.acceptance_rate,
        "acceptance": acceptance.to_dict(),
        "accepted_prefix_lengths": [t.acceptance.num_accepted for t in ekv_res.traces],
        "first_divergence_idx": div_idx,
        "n_output_tokens": len(ekv_res.output_ids),
        "supports_real_bytes_claim": adapter.capabilities.supports_real_bytes_claim,
    }


def blocked_report(reason: str) -> dict[str, Any]:
    report = {
        "experiment_id": EXPERIMENT_044_ID,
        "status": "blocked",
        "adapter_name": ADAPTER_NAME,
        "not_default_registry": True,
        "model": MODEL_NAME,
        "prompt_panel": [],
        "exactkv_failures": None,
        "acceptance_summary": None,
        "per_prompt": [],
        "memory_claim_note": MEMORY_CLAIM_NOTE,
        "blocked_reason": reason,
        "limitations": ["Adapter smoke not attempted."],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_044_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 044 SpectralQuant adapter smoke")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        ensure_spectralquant_path()
    except FileNotFoundError as exc:
        report = blocked_report(str(exc))
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Exp 044: blocked — {exc}")
        return 0

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_str = "float16" if dev.type == "cuda" else "float32"

    try:
        runtime = ModelRuntime(MODEL_NAME, device=dev, dtype=dtype_str)
        adapter = create_spectralquant_experimental_adapter(
            runtime,
            calibration_config=CalibrationConfig(),
        )
    except Exception as exc:  # noqa: BLE001
        report = blocked_report(f"adapter init failed: {type(exc).__name__}: {exc}")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Exp 044: blocked — {exc}")
        return 0

    from exactkv.compressors import list_compressors

    if ADAPTER_NAME in list_compressors():
        raise RuntimeError(f"{ADAPTER_NAME} must not be in default registry")

    panel = load_smoke_prompt_panel()
    per_prompt: list[dict[str, Any]] = []
    failures = 0
    acceptances: list[float] = []

    for entry in panel:
        try:
            cell = run_one_cell(runtime, adapter, entry)
            per_prompt.append(cell)
            failures += cell["exactkv_failures"]
            acceptances.append(cell["acceptance_rate"])
        except Exception as exc:  # noqa: BLE001
            per_prompt.append({
                "prompt_id": entry.get("prompt_id", "?"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "exactkv_failures": 1,
            })
            failures += 1

    mean_acc = sum(acceptances) / len(acceptances) if acceptances else None
    report = {
        "experiment_id": EXPERIMENT_044_ID,
        "status": "pass" if failures == 0 else "failed",
        "adapter_name": ADAPTER_NAME,
        "not_default_registry": True,
        "factory_only": True,
        "model": MODEL_NAME,
        "draft_len": DRAFT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_panel": [p.get("prompt_id") for p in panel],
        "exactkv_failures": failures,
        "acceptance_summary": {
            "mean_acceptance": mean_acc,
            "n_prompts": len(panel),
        },
        "per_prompt": per_prompt,
        "memory_claim_note": MEMORY_CLAIM_NOTE,
        "limitations": [
            "Experimental factory-only adapter — not default registry.",
            "Minimal calibration (2 prompts) — not paper-scale.",
            "Compress then materialise full dequant K/V — no active memory savings.",
            f"{len(panel)}-prompt smoke — not full-panel benchmark.",
        ],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": "restricted_adapter_go" if failures == 0 else "restricted_no_go",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _assert_no_forbidden_fields(report)
    validate_044_report(report)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Exp 044: {report['status']} exactkv_failures={failures} "
        f"mean_accept={mean_acc} recommendation={report['recommendation']}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
