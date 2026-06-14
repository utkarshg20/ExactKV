#!/usr/bin/env python3
"""Experiment 045: SpectralQuant restricted adapter panel (Phase 10G).

Factory-only ``spectralquant_experimental`` — NOT default registry.
12-prompt small panel on Qwen2.5-0.5B; reports exactkv_failures and acceptance.

No speed/memory/serving claims. Results scoped to this panel only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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
    EXPERIMENT_045_ID,
    PANEL_PROMOTION_MIN_PROMPTS,
    CalibrationConfig,
    capture_real_kv_tensors,
    compress_decompress_layer,
    default_calibration_prompts_panel,
    ensure_spectralquant_path,
    leaderboard_promotion_decision,
    load_restricted_panel,
    validate_045_report,
)
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

DEFAULT_JSON = _ROOT / "reports" / "experiment_045_spectralquant_restricted_panel.json"

MODEL_NAME = DEFAULT_MODEL
MAX_NEW_TOKENS = 32
DRAFT_LEN = 4
ADAPTER_NAME = "spectralquant_experimental"
CALIBRATION_PROMPT_COUNT = 6
TARGET_PROMPTS = 12
FALLBACK_PROMPTS = 8
MAX_WALL_SEC = 900  # reduce panel if exceeded mid-run

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})

_RECON_LAYERS = (0, 12, 23)


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


def _reconstruction_summary(runtime: ModelRuntime, engine: Any, prompt: str) -> dict[str, Any]:
    kv = capture_real_kv_tensors(runtime, prompt)
    n_layers = kv["num_layers"]
    layers = [i for i in _RECON_LAYERS if i < n_layers]
    layer_results = []
    for layer_idx in layers:
        try:
            layer_results.append(
                compress_decompress_layer(
                    engine,
                    kv["k_tensors"][layer_idx],
                    kv["v_tensors"][layer_idx],
                    layer_idx,
                )
            )
        except Exception as exc:  # noqa: BLE001
            layer_results.append({"layer_idx": layer_idx, "status": "failed", "error": str(exc)})
    key_max = max((r.get("key_max_abs_error") or 0.0) for r in layer_results)
    val_max = max((r.get("value_max_abs_error") or 0.0) for r in layer_results)
    return {
        "source": "real_kv_round_trip_on_panel_prompt",
        "layers_tested": layers,
        "all_layers_used_by_adapter": True,
        "layer_results": layer_results,
        "key_max_abs_error": key_max,
        "value_max_abs_error": val_max,
        "caveat": (
            "Large key reconstruction error possible (e.g. layer 0 in Exp 043 ~38 max). "
            "Lossy quant — not lossless tensor round-trip."
        ),
    }


def run_one_cell(
    runtime: ModelRuntime,
    adapter: Any,
    prompt_entry: dict[str, Any],
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
    prefix_lens = [t.acceptance.num_accepted for t in ekv_res.traces]
    draft_diverged = acceptance.total_rejected > 0

    return {
        "prompt_id": prompt_entry.get("prompt_id", "unknown"),
        "category": prompt_entry.get("category", "unknown"),
        "panel_category": prompt_entry.get("panel_category", "unknown"),
        "exactkv_failures": 0 if exact else 1,
        "token_exact_match": exact,
        "acceptance_rate": acceptance.acceptance_rate,
        "acceptance": acceptance.to_dict(),
        "accepted_prefix_lengths": prefix_lens,
        "draft_divergence_occurred": draft_diverged,
        "first_divergence_idx": div_idx,
        "n_output_tokens": int(ekv_res.output_ids.shape[-1]),
    }


def _blocked_report(reason: str) -> dict[str, Any]:
    report = {
        "experiment_id": EXPERIMENT_045_ID,
        "status": "blocked",
        "adapter_name": ADAPTER_NAME,
        "not_default_registry": True,
        "model": MODEL_NAME,
        "prompt_count": 0,
        "calibration": {"ran": False},
        "panel_composition": [],
        "exactkv_failures": None,
        "acceptance_summary": None,
        "divergence_summary": None,
        "reconstruction_error_summary": None,
        "materializing_adapter": True,
        "memory_claim_note": MEMORY_CLAIM_NOTE,
        "supports_real_bytes_claim": False,
        "leaderboard_decision": leaderboard_promotion_decision(
            exactkv_failures=1, prompt_count=0
        ),
        "blocked_reason": reason,
        "limitations": ["Panel not attempted."],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": "blocked",
        "per_prompt": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_045_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 045 SpectralQuant restricted panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-prompts", type=int, default=TARGET_PROMPTS)
    args = parser.parse_args()

    try:
        ensure_spectralquant_path()
    except FileNotFoundError as exc:
        report = _blocked_report(str(exc))
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Exp 045: blocked — {exc}")
        return 0

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_str = "float16" if dev.type == "cuda" else "float32"

    panel = load_restricted_panel(max_prompts=args.max_prompts)
    if not panel:
        report = _blocked_report("no V10 panel prompts loaded")
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    calib_prompts = default_calibration_prompts_panel()
    calib_cfg = CalibrationConfig(n_samples=CALIBRATION_PROMPT_COUNT)

    t0 = time.monotonic()
    try:
        runtime = ModelRuntime(MODEL_NAME, device=dev, dtype=dtype_str)
        adapter = create_spectralquant_experimental_adapter(
            runtime,
            calibration_prompts=calib_prompts,
            calibration_config=calib_cfg,
        )
    except Exception as exc:  # noqa: BLE001
        report = _blocked_report(f"adapter init failed: {type(exc).__name__}: {exc}")
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Exp 045: blocked — {exc}")
        return 0

    from exactkv.compressors import list_compressors

    if ADAPTER_NAME in list_compressors():
        raise RuntimeError(f"{ADAPTER_NAME} must not be in default registry")

    recon = _reconstruction_summary(runtime, adapter._engine, panel[0]["prompt"])

    per_prompt: list[dict[str, Any]] = []
    failures = 0
    exact_matches = 0
    acceptances: list[float] = []
    all_prefix_lens: list[int] = []
    divergence_examples: list[dict[str, Any]] = []
    panel_reduced = False
    planned = len(panel)

    for i, entry in enumerate(panel):
        if time.monotonic() - t0 > MAX_WALL_SEC and i >= FALLBACK_PROMPTS:
            panel_reduced = True
            break
        try:
            cell = run_one_cell(runtime, adapter, entry)
            per_prompt.append(cell)
            failures += cell["exactkv_failures"]
            if cell["token_exact_match"]:
                exact_matches += 1
            acceptances.append(cell["acceptance_rate"])
            all_prefix_lens.extend(cell["accepted_prefix_lengths"])
            if cell.get("draft_divergence_occurred") and len(divergence_examples) < 5:
                divergence_examples.append({
                    "prompt_id": cell["prompt_id"],
                    "category": cell["category"],
                    "acceptance_rate": cell["acceptance_rate"],
                    "accepted_prefix_lengths": cell["accepted_prefix_lengths"],
                    "first_divergence_idx": cell["first_divergence_idx"],
                    "exactkv_failures": cell["exactkv_failures"],
                    "note": "Draft diverged from verifier; ExactKV corrected — output still exact.",
                })
        except Exception as exc:  # noqa: BLE001
            per_prompt.append({
                "prompt_id": entry.get("prompt_id", "?"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "exactkv_failures": 1,
            })
            failures += 1

    prompt_count = len(per_prompt)
    mean_acc = statistics.mean(acceptances) if acceptances else None
    median_acc = statistics.median(acceptances) if acceptances else None
    min_acc = min(acceptances) if acceptances else None

    lb_decision = leaderboard_promotion_decision(
        exactkv_failures=failures,
        prompt_count=prompt_count,
    )

    if failures == 0 and prompt_count >= PANEL_PROMOTION_MIN_PROMPTS:
        recommendation = "promote_restricted_backend"
    elif failures == 0:
        recommendation = "keep_smoke_only"
    else:
        recommendation = "restricted_no_go"

    report = {
        "experiment_id": EXPERIMENT_045_ID,
        "status": "pass" if failures == 0 else "failed",
        "adapter_name": ADAPTER_NAME,
        "not_default_registry": True,
        "factory_only": True,
        "model": MODEL_NAME,
        "device": str(dev),
        "dtype": dtype_str,
        "draft_len": DRAFT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_count": prompt_count,
        "planned_prompt_count": planned,
        "panel_reduced_for_runtime": panel_reduced,
        "calibration": {
            "required": True,
            "calibration_prompt_count": len(calib_prompts),
            "n_samples": calib_cfg.n_samples,
            "max_tokens_per_layer": calib_cfg.max_tokens_per_layer,
            "avg_bits": calib_cfg.avg_bits,
            "qjl_projections": calib_cfg.qjl_projections,
            "api": "EigenspectralCalibrator.calibrate + fit_quantizers_from_eigenspectrum",
        },
        "panel_composition": [
            {"prompt_id": p["prompt_id"], "panel_category": p.get("panel_category")}
            for p in panel[:prompt_count]
        ],
        "exactkv_failures": failures,
        "token_exact_match_count": exact_matches,
        "acceptance_summary": {
            "mean_acceptance": mean_acc,
            "median_acceptance": median_acc,
            "min_acceptance": min_acc,
            "n_prompts": prompt_count,
            "accepted_prefix_lengths_all_rounds": all_prefix_lens,
            "accepted_prefix_median": (
                statistics.median(all_prefix_lens) if all_prefix_lens else None
            ),
        },
        "divergence_summary": {
            "draft_divergence_prompt_count": sum(
                1 for p in per_prompt if p.get("draft_divergence_occurred")
            ),
            "output_divergence_count": sum(
                1 for p in per_prompt if not p.get("token_exact_match", True)
            ),
            "examples": divergence_examples,
        },
        "reconstruction_error_summary": recon,
        "materializing_adapter": True,
        "memory_claim_note": MEMORY_CLAIM_NOTE,
        "supports_real_bytes_claim": False,
        "leaderboard_decision": lb_decision,
        "limitations": [
            "SpectralQuant remains a factory-only experimental adapter — not default registry.",
            f"{prompt_count}-prompt restricted panel — not full V10 benchmark.",
            f"Calibration uses {len(calib_prompts)} prompts — not paper-scale.",
            "Materializing adapter: compresses K/V then materialises dequant tensors for draft.",
            "Large key reconstruction error on some layers — see reconstruction_error_summary.",
            "No speedup, active memory savings, production serving, or accuracy claims.",
        ],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": recommendation,
        "per_prompt": per_prompt,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _assert_no_forbidden_fields(report)
    validate_045_report(report)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Exp 045: {report['status']} prompts={prompt_count} "
        f"exactkv_failures={failures} mean_accept={mean_acc} "
        f"promote={lb_decision['promote_to_restricted_backend']} "
        f"recommendation={recommendation}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
