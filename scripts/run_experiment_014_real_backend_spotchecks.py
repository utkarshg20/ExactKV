#!/usr/bin/env python3
"""Experiment 014: V10 real-backend category spot-checks (V10 Phase 4).

Focused 40-prompt subset from harder V10 categories × built-in compressors plus
restricted real backends (factory-only, not default registry).

No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
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
from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
EXPERIMENT_CLASS = "v10_real_backend_spotcheck"

SPOTCHECK_SUITES = (
    "long_context",
    "retrieval_copy",
    "tool_json",
    "code_structured",
)
PROMPTS_PER_SUITE = 10

BUILTIN_COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
]

KVQUANT_NAME = "kvquant_sim_qwen05b"
TURBOQUANT_NAME = "turboquant_python_k3_v3"
KIVI_NAME = "kivi_offline_k2_v2"

REAL_BACKEND_NAMES = (KVQUANT_NAME, TURBOQUANT_NAME, KIVI_NAME)

# Experiment 012 category-suite anchors (draft_len=4, max_new_tokens=16).
_EXP012_CATEGORY_ANCHORS = {
    "long_context": {"int8": 0.933, "k8_v4_sim": 0.900, "k8_v4_boundary4_v8_sim": 0.917},
    "retrieval_copy": {"int8": 0.982, "k8_v4_sim": 0.945, "k8_v4_boundary4_v8_sim": 0.945},
    "tool_json": {"int8": 0.945, "k8_v4_sim": 0.909, "k8_v4_boundary4_v8_sim": 0.927},
    "code_structured": {"int8": 0.957, "k8_v4_sim": 0.914, "k8_v4_boundary4_v8_sim": 0.929},
}

# V9 core-suite real-backend anchors (Experiment 008–010).
_V9_REAL_BACKEND_ANCHORS = {
    KVQUANT_NAME: 0.792,
    TURBOQUANT_NAME: 0.435,
    KIVI_NAME: 0.012,
}

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

_BRACKET_OPEN = "([{"
_BRACKET_CLOSE = ")]}"
_QUOTE_CHARS = "\"'`"


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


def _kvquant_available() -> bool:
    path = os.environ.get("EXACTKV_KVQUANT_QUANTIZERS", "")
    if not path or not os.path.isfile(path):
        return False
    try:
        return importlib.util.find_spec("kvquant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _turboquant_available() -> bool:
    try:
        return importlib.util.find_spec("turboquant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _kivi_available() -> bool:
    try:
        return importlib.util.find_spec("models.utils_quant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def probe_backends() -> dict[str, bool]:
    return {
        KVQUANT_NAME: _kvquant_available(),
        TURBOQUANT_NAME: _turboquant_available(),
        KIVI_NAME: _kivi_available(),
    }


def load_spotcheck_subset() -> list[dict[str, Any]]:
    """Deterministic subset: first N prompt ids per harder V10 suite."""
    out: list[dict[str, Any]] = []
    for suite in SPOTCHECK_SUITES:
        rows = load_v10_suite(suite)
        rows.sort(key=lambda r: r["prompt_id"])
        for row in rows[:PROMPTS_PER_SUITE]:
            entry = dict(row)
            entry["v10_panel"] = "exp014_spotcheck"
            out.append(entry)
    return out


def _compressors_for_panel(panel: str) -> list[str]:
    avail = probe_backends()
    if panel == "builtin":
        return list(BUILTIN_COMPRESSORS)
    if panel == "kvquant":
        if not avail[KVQUANT_NAME]:
            raise SystemExit(
                "KVQuant panel requires kvquant package and EXACTKV_KVQUANT_QUANTIZERS"
            )
        return list(BUILTIN_COMPRESSORS) + [KVQUANT_NAME]
    if panel == "turboquant":
        if not avail[TURBOQUANT_NAME]:
            raise SystemExit(
                "TurboQuant panel requires turboquant (PYTHONPATH=vendor/turboquant_plus)"
            )
        return list(BUILTIN_COMPRESSORS) + [TURBOQUANT_NAME]
    if panel == "kivi":
        if not avail[KIVI_NAME]:
            raise SystemExit(
                "KIVI panel requires models.utils_quant (PYTHONPATH to KIVI repo)"
            )
        return list(BUILTIN_COMPRESSORS) + [KIVI_NAME]
    if panel == "auto":
        compressors = list(BUILTIN_COMPRESSORS)
        for name in REAL_BACKEND_NAMES:
            if avail[name]:
                compressors.append(name)
        return compressors
    raise ValueError(f"Unknown panel {panel!r}")


def _classify_token_text(text: str) -> str:
    if not text:
        return "wordpiece/other"
    if text.isspace():
        return "whitespace"
    if len(text) == 1 and text in _BRACKET_OPEN + _BRACKET_CLOSE:
        return "bracket"
    if len(text) == 1 and text in _QUOTE_CHARS:
        return "quote"
    if len(text) == 1 and text in ".,;:!?-—…":
        return "punctuation"
    if re.fullmatch(r"[\d]+(?:\.[\d]+)?", text.strip()):
        return "numeric"
    if all(c in _BRACKET_OPEN + _BRACKET_CLOSE + _QUOTE_CHARS + ".,;:!?- \t\n" for c in text):
        return "punctuation"
    return "wordpiece/other"


def _token_type_at_id(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return "wordpiece/other"
    return _classify_token_text(text)


def _structured_output_flags(text: str) -> dict[str, Any]:
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    unmatched = False
    for ch in text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                unmatched = True
    if stack:
        unmatched = True
    quote_count = sum(text.count(q) for q in _QUOTE_CHARS)
    quote_imbalance = (quote_count % 2) != 0
    stripped = text.lstrip()
    malformed_json_prefix = bool(
        stripped.startswith("{") or stripped.startswith("[")
    ) and unmatched
    return {
        "unmatched_brackets": unmatched,
        "quote_imbalance": quote_imbalance,
        "malformed_json_prefix": malformed_json_prefix,
    }


def _resolve_compressor(
    runtime: ModelRuntime,
    name: str,
    cache: dict[str, Any],
) -> Any:
    if name in cache:
        return cache[name]
    if name == KVQUANT_NAME:
        from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter

        comp = create_kvquant_sim_adapter(
            runtime,
            quantizers_path=os.environ["EXACTKV_KVQUANT_QUANTIZERS"],
            abits=4,
        )
    elif name == TURBOQUANT_NAME:
        from exactkv.compressors.turboquant_adapter import create_turboquant_python_adapter

        comp = create_turboquant_python_adapter(runtime, k_bits=3, v_bits=3)
    elif name == KIVI_NAME:
        from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter

        comp = create_kivi_offline_adapter(runtime, k_bits=2, v_bits=2, group_size=32)
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    config: RunConfig,
    compressor_cache: dict[str, Any],
) -> dict[str, Any]:
    compressor = _resolve_compressor(runtime, config.compressor_name, compressor_cache)
    caps_dict: dict = {}
    if hasattr(compressor, "capabilities"):
        caps_dict = asdict(compressor.capabilities)

    prompt = prompt_entry["prompt"]
    max_new = config.max_new_tokens
    tokenizer = runtime.tokenizer

    full_res = generate_full_greedy(runtime, prompt, max_new)
    full_ids = full_res.generated_ids.squeeze(0).tolist()
    prompt_len = full_res.prompt_ids.shape[-1]

    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()
    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    ekv_res = ExactKVGenerator(
        runtime, compressor, draft_len=config.draft_len
    ).generate(prompt, max_new)
    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    rejection_positions: list[int] = []
    correction_positions: list[int] = []
    for trace in ekv_res.traces:
        gen_offset = trace.full_seq_len_before - prompt_len
        if trace.acceptance.num_rejected > 0:
            rejection_positions.append(gen_offset + trace.acceptance.num_accepted)
        if trace.acceptance.correction_token is not None:
            correction_positions.append(gen_offset + trace.acceptance.num_accepted)

    div_token_type = None
    if lossy_div is not None and lossy_div < len(full_ids):
        div_token_type = _token_type_at_id(tokenizer, full_ids[lossy_div])

    corr_token_types: list[str] = []
    for trace in ekv_res.traces:
        tok = trace.acceptance.correction_token
        if tok is not None:
            corr_token_types.append(_token_type_at_id(tokenizer, tok))

    primary_cat = prompt_entry.get("v10_primary_category", prompt_entry.get("category", ""))
    structured_flags = None
    if prompt_entry.get("v10_suite") in ("tool_json", "code_structured"):
        structured_flags = _structured_output_flags(lossy_res.output_text)

    forensics = {
        "first_divergence_idx": lossy_div,
        "lossy_diverged": not lossy_exact,
        "rejection_positions": rejection_positions,
        "correction_positions": correction_positions,
        "divergence_token_type": div_token_type,
        "correction_token_types": corr_token_types,
        "structured_output_flags": structured_flags,
        "has_attention_weights": False,
    }

    result = {
        "prompt_id": prompt_entry["prompt_id"],
        "prompt": prompt,
        "category": prompt_entry.get("category", "unknown"),
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
        "forensics": forensics,
    }
    for key in (
        "v10_id",
        "v10_suite",
        "v10_suite_version",
        "v10_primary_category",
        "v10_secondary_tags",
        "v10_panel",
    ):
        if key in prompt_entry:
            result[key] = prompt_entry[key]
    return result


def _aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    acc_blocks = [r.get("exactkv", {}).get("acceptance", {}) for r in results]
    acceptance_rates = [a.get("acceptance_rate", 0.0) for a in acc_blocks]
    avg_lengths = [a.get("avg_accepted_per_round", 0.0) for a in acc_blocks]
    denom = max(n, 1)
    div_idxs = [
        r.get("forensics", {}).get("first_divergence_idx")
        for r in results
        if r.get("forensics", {}).get("lossy_diverged")
        and r.get("forensics", {}).get("first_divergence_idx") is not None
    ]
    return {
        "num_runs": n,
        "mean_acceptance_rate": sum(acceptance_rates) / denom,
        "mean_average_accepted_length": sum(avg_lengths) / denom,
        "total_drafted": sum(a.get("total_drafted", 0) for a in acc_blocks),
        "total_accepted": sum(a.get("total_accepted", 0) for a in acc_blocks),
        "total_rejected": sum(a.get("total_rejected", 0) for a in acc_blocks),
        "total_corrections": sum(a.get("total_corrections", 0) for a in acc_blocks),
        "exactkv_failures": sum(1 for r in results if r.get("exactkv_failure", False)),
        "lossy_divergence_count": sum(
            1 for r in results
            if not r.get("lossy", {}).get("token_exact_match", True)
        ),
        "mean_first_divergence_idx": (
            sum(div_idxs) / len(div_idxs) if div_idxs else None
        ),
    }


def _group_table(
    results: list[dict[str, Any]],
    key_fn,
    key_name: str,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    table = []
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {key_name: key}
        row.update(_aggregate_group(group))
        table.append(row)
    return table


def _build_forensics_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    div_by_comp = Counter()
    token_types_div = Counter()
    token_types_corr = Counter()
    struct_unmatched = struct_malformed = struct_quote = 0
    struct_cells = 0

    for r in results:
        f = r.get("forensics", {})
        if f.get("lossy_diverged") and f.get("first_divergence_idx") is not None:
            div_by_comp[r["compressor_name"]] += 1
            tt = f.get("divergence_token_type")
            if tt:
                token_types_div[tt] += 1
        for tt in f.get("correction_token_types") or []:
            token_types_corr[tt] += 1
        flags = f.get("structured_output_flags")
        if flags is not None:
            struct_cells += 1
            if flags.get("unmatched_brackets"):
                struct_unmatched += 1
            if flags.get("malformed_json_prefix"):
                struct_malformed += 1
            if flags.get("quote_imbalance"):
                struct_quote += 1

    return {
        "divergence_cells_by_compressor": dict(div_by_comp),
        "divergence_token_type_counts": dict(token_types_div),
        "correction_token_type_counts": dict(token_types_corr),
        "structured_output_cells_analyzed": struct_cells,
        "structured_unmatched_brackets": struct_unmatched,
        "structured_malformed_json_prefix": struct_malformed,
        "structured_quote_imbalance": struct_quote,
        "attention_weights_logged": False,
    }


def _win_loss_table(
    results: list[dict[str, Any]],
    comp_a: str,
    comp_b: str,
) -> dict[str, Any]:
    by_prompt: dict[str, dict[str, float]] = defaultdict(dict)
    for r in results:
        by_prompt[r["prompt_id"]][r["compressor_name"]] = (
            r["exactkv"]["acceptance"]["acceptance_rate"]
        )
    wins_a = wins_b = ties = 0
    per_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins_a": 0, "wins_b": 0, "ties": 0}
    )
    for pid, rates in by_prompt.items():
        if comp_a not in rates or comp_b not in rates:
            continue
        a, b = rates[comp_a], rates[comp_b]
        cat = next(
            (r.get("v10_primary_category", "") for r in results if r["prompt_id"] == pid),
            "",
        )
        if a > b + 1e-9:
            wins_a += 1
            per_cat[cat]["wins_a"] += 1
        elif b > a + 1e-9:
            wins_b += 1
            per_cat[cat]["wins_b"] += 1
        else:
            ties += 1
            per_cat[cat]["ties"] += 1
    return {
        "comp_a": comp_a,
        "comp_b": comp_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "per_category": dict(per_cat),
    }


def run_panel(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    compressors: list[str],
    panel_id: str,
    dtype: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    compressor_cache: dict[str, Any] = {}
    total_cells = len(prompts) * len(compressors)
    cell_idx = 0

    for prompt_entry in prompts:
        for compressor_name in compressors:
            cell_idx += 1
            print(
                f"  [{cell_idx}/{total_cells}] panel={panel_id} "
                f"{prompt_entry['prompt_id']} × {compressor_name}",
                flush=True,
            )
            config = RunConfig(
                compressor_name=compressor_name,
                draft_len=DRAFT_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            results.append(
                run_one_cell(runtime, prompt_entry, config, compressor_cache)
            )

    manifest = build_run_manifest(
        model_name=runtime.model_name,
        prompt_suite="v10_exp014_spotcheck",
        compressor_names=compressors,
        draft_lengths=[DRAFT_LEN],
        max_new_tokens=MAX_NEW_TOKENS,
        device=str(runtime.device),
        dtype=dtype,
    )
    manifest["experiment"] = "014_real_backend_spotchecks"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["panel_id"] = panel_id
    manifest["prompt_count"] = len(prompts)
    manifest["spotcheck_suites"] = list(SPOTCHECK_SUITES)
    manifest["prompts_per_suite"] = PROMPTS_PER_SUITE
    manifest["backend_availability"] = probe_backends()
    manifest["real_backends_in_panel"] = [
        c for c in compressors if c in REAL_BACKEND_NAMES
    ]

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=compressors,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["acceptance_by_category"] = _group_table(
        results,
        lambda r: r.get("v10_primary_category", r.get("category", "")),
        "primary_category",
    )
    aggregate["acceptance_by_suite"] = _group_table(
        results, lambda r: r.get("v10_suite", ""), "suite"
    )
    aggregate["acceptance_by_compressor_category"] = _group_table(
        results,
        lambda r: (r["compressor_name"], r.get("v10_primary_category", "")),
        "compressor_category",
    )
    for row in aggregate["acceptance_by_compressor_category"]:
        cc = row.pop("compressor_category")
        row["compressor_name"], row["primary_category"] = cc

    aggregate["forensics"] = _build_forensics_aggregate(results)
    aggregate["boundary4_vs_k8_v4_sim"] = _win_loss_table(
        results, "k8_v4_boundary4_v8_sim", "k8_v4_sim"
    )

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def _infer_panel_id(panel: dict[str, Any], source_path: str = "") -> str:
    manifest = panel.get("manifest") or {}
    pid = manifest.get("panel_id")
    if pid:
        return pid
    name = Path(source_path).stem if source_path else ""
    for token in ("builtin", "kvquant", "turboquant", "kivi", "auto"):
        if token in name:
            return token
    real = [
        c for c in panel.get("aggregate", {}).get("compressor_names", [])
        if c in REAL_BACKEND_NAMES
    ]
    if len(real) == 1:
        return {
            KVQUANT_NAME: "kvquant",
            TURBOQUANT_NAME: "turboquant",
            KIVI_NAME: "kivi",
        }.get(real[0], "unknown")
    return "unknown"


def merge_panel_reports(
    panel_reports: list[tuple[dict[str, Any], str]],
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge split-panel runs; dedupe by (prompt_id, compressor_name)."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    panel_summaries: list[dict[str, Any]] = []
    panel_priority = {"builtin": 0, "kvquant": 1, "turboquant": 2, "kivi": 3, "auto": 4}

    sorted_panels = sorted(
        panel_reports,
        key=lambda item: panel_priority.get(_infer_panel_id(item[0], item[1]), 99),
    )

    for panel, source_path in sorted_panels:
        pid = _infer_panel_id(panel, source_path)
        manifest = panel.get("manifest") or {}
        panel_summaries.append({
            "panel_id": pid,
            "cells": len(panel["results"]),
            "compressors": manifest.get("compressor_names")
            or panel.get("aggregate", {}).get("compressor_names", []),
            "device": manifest.get("device"),
            "dtype": manifest.get("dtype"),
            "real_backends_in_panel": manifest.get("real_backends_in_panel", []),
            "source_json": source_path,
        })
        for r in panel["results"]:
            key = (r["prompt_id"], r["compressor_name"])
            if key not in merged:
                entry = dict(r)
                entry["panel_source"] = pid
                merged[key] = entry

    results = list(merged.values())
    compressors_seen = sorted({r["compressor_name"] for r in results})

    manifest = build_run_manifest(
        model_name=MODEL_NAME,
        prompt_suite="v10_exp014_spotcheck",
        compressor_names=compressors_seen,
        draft_lengths=[DRAFT_LEN],
        max_new_tokens=MAX_NEW_TOKENS,
        device="merged",
        dtype="mixed",
    )
    manifest["experiment"] = "014_real_backend_spotchecks"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["merged_panels"] = panel_summaries
    manifest["prompt_count"] = len(prompts)
    manifest["spotcheck_suites"] = list(SPOTCHECK_SUITES)
    manifest["prompts_per_suite"] = PROMPTS_PER_SUITE
    manifest["backend_availability"] = probe_backends()
    manifest["cross_panel_merge"] = True
    manifest["expected_cells_full_panel"] = len(prompts) * (
        len(BUILTIN_COMPRESSORS) + len(REAL_BACKEND_NAMES)
    )
    manifest["actual_unique_cells"] = len(results)

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=compressors_seen,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["acceptance_by_category"] = _group_table(
        results,
        lambda r: r.get("v10_primary_category", r.get("category", "")),
        "primary_category",
    )
    aggregate["acceptance_by_suite"] = _group_table(
        results, lambda r: r.get("v10_suite", ""), "suite"
    )
    aggregate["acceptance_by_compressor_category"] = _group_table(
        results,
        lambda r: (r["compressor_name"], r.get("v10_primary_category", "")),
        "compressor_category",
    )
    for row in aggregate["acceptance_by_compressor_category"]:
        cc = row.pop("compressor_category")
        row["compressor_name"], row["primary_category"] = cc

    aggregate["forensics"] = _build_forensics_aggregate(results)
    aggregate["boundary4_vs_k8_v4_sim"] = _win_loss_table(
        results, "k8_v4_boundary4_v8_sim", "k8_v4_sim"
    )

    real_rates = {
        r["compressor_name"]: r["mean_acceptance_rate"]
        for r in aggregate["acceptance_by_compressor"]
        if r["compressor_name"] in REAL_BACKEND_NAMES
    }
    if real_rates:
        best_real = max(real_rates, key=real_rates.get)
        aggregate["real_backend_ranking"] = {
            "ranked": sorted(real_rates.items(), key=lambda x: -x[1]),
            "strongest": best_real,
            "kvquant_strongest": best_real == KVQUANT_NAME,
        }

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def _fmt_rate(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def generate_markdown_report(report: dict[str, Any], meta: dict[str, Any]) -> str:
    agg = report["aggregate"]
    forensics = agg["forensics"]
    by_comp = agg["acceptance_by_compressor"]
    by_cat = agg["acceptance_by_category"]
    wl_b4 = agg["boundary4_vs_k8_v4_sim"]
    manifest = report["manifest"]

    real_in_report = [c for c in REAL_BACKEND_NAMES if any(
        r["compressor_name"] == c for r in report["results"]
    )]

    lines = [
        "# Experiment 014: Real-Backend Category Spot-Checks",
        "",
        "_Generated by `scripts/run_experiment_014_real_backend_spotchecks.py`. "
        "V10 Phase 4 — focused category spot-check only._",
        "",
        "> This is a **focused category spot-check**, not a comprehensive benchmark.",
        "> Restricted real backends are **factory-only** and **not** in the default registry.",
        "> TurboQuant Python is **not** production TurboQuant / llama.cpp / MLX.",
        "> KIVI offline is **not** KIVI CUDA/Triton production path.",
        "> KVQuant simquant is **not** KVQuant deployment CUDA.",
        "> Simulated compressors (`_sim`) are **not** real packed-bit backends.",
        "> `total_kv_footprint_bytes` is a conservative accounting sum, "
        "**not** measured peak GPU memory.",
        "> **Active GPU memory is not reported.**",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, "
        "tokens/sec, active GPU memory, or production readiness.",
        "> ExactKV does **not** claim external-paper results as ExactKV results.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Compare restricted real backends against built-in baselines on harder V10 "
        "category prompts (`long_context`, `retrieval_copy`, `tool_json`, "
        "`code_structured`) at anchor settings (`draft_len=4`, `max_new_tokens=16`).",
        "",
        "## 2. Model and environment",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | `{MODEL_NAME}` |",
        f"| `draft_len` | {DRAFT_LEN} |",
        f"| `max_new_tokens` | {MAX_NEW_TOKENS} |",
        f"| Device / dtype | {meta.get('device', '—')} / {meta.get('dtype', '—')} |",
        f"| Total unique cells | **{agg['total_runs']}** |",
        f"| Cross-panel merge | {manifest.get('cross_panel_merge', False)} |",
        "",
    ]

    if manifest.get("merged_panels"):
        lines.extend([
            "### Panel breakdown",
            "",
            "| Panel | Cells | Compressors | Device |",
            "|---|---:|---|---|",
        ])
        for p in manifest["merged_panels"]:
            comps = ", ".join(f"`{c}`" for c in p.get("compressors", []))
            lines.append(
                f"| `{p['panel_id']}` | {p['cells']} | {comps} | {p.get('device', '—')} |"
            )

    lines.extend([
        "",
        "Reproduce:",
        "",
        "```bash",
        "# Built-in panel (any ExactKV env)",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_014_real_backend_spotchecks.py \\",
        "  --panel builtin --device cuda --dtype float16",
        "",
        "# TurboQuant panel",
        "PYTHONPATH=vendor/turboquant_plus TRANSFORMERS_OFFLINE=1 \\",
        "  .venv-turboquant/bin/python scripts/run_experiment_014_real_backend_spotchecks.py \\",
        "  --panel turboquant",
        "",
        "# KIVI panel",
        "PYTHONPATH=/path/to/KIVI TRANSFORMERS_OFFLINE=1 \\",
        "  python3 scripts/run_experiment_014_real_backend_spotchecks.py --panel kivi",
        "",
        "# KVQuant panel (isolated venv + quantizers pickle)",
        "EXACTKV_KVQUANT_QUANTIZERS=/path/to/quantizers.pickle \\",
        "  python3 scripts/run_experiment_014_real_backend_spotchecks.py \\",
        "  --panel kvquant --device cuda --dtype float16",
        "",
        "# Merge panels",
        "python3 scripts/run_experiment_014_real_backend_spotchecks.py --merge-only \\",
        "  --merge-from reports/exp014_panel_builtin.json \\",
        "  --merge-from reports/exp014_panel_turboquant.json \\",
        "  --merge-from reports/exp014_panel_kivi.json \\",
        "  --merge-from reports/exp014_panel_kvquant.json",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_014_real_backend_spotchecks.json`,",
        "`reports/experiment_014_real_backend_spotchecks.csv`.",
        "",
        "## 3. Prompt subset construction",
        "",
        f"Deterministic subset: first **{PROMPTS_PER_SUITE}** prompt ids (sorted) per suite:",
        "",
        "| Suite | Prompts in subset |",
        "|---|---:|",
    ])
    suite_counts: dict[str, int] = Counter(
        r.get("v10_suite", "") for r in report["results"]
    )
    for suite in SPOTCHECK_SUITES:
        n = len({
            r["prompt_id"] for r in report["results"] if r.get("v10_suite") == suite
        })
        lines.append(f"| `{suite}` | {n} |")
    lines.append(f"| **Total** | **{meta['prompt_count']}** |")

    lines.extend([
        "",
        "Selection is deterministic (sorted `prompt_id`), not cherry-picked per compressor.",
        "",
        "## 4. Compressor panel",
        "",
        "| Compressor | Type | In this report |",
        "|---|---|---|",
        "| `noop` | Built-in baseline | "
        + ("yes" if any(r["compressor_name"] == "noop" for r in report["results"]) else "no")
        + " |",
        "| `int8` | Built-in real INT8 | "
        + ("yes" if any(r["compressor_name"] == "int8" for r in report["results"]) else "no")
        + " |",
        "| `k8_v4_sim` | Built-in simulated | "
        + ("yes" if any(r["compressor_name"] == "k8_v4_sim" for r in report["results"]) else "no")
        + " |",
        "| `k8_v4_boundary4_v8_sim` | Built-in simulated | "
        + ("yes" if any(
            r["compressor_name"] == "k8_v4_boundary4_v8_sim" for r in report["results"]
        ) else "no")
        + " |",
        f"| `{KVQUANT_NAME}` | Restricted KVQuant simquant | "
        + ("yes" if KVQUANT_NAME in real_in_report else "no / other panel") + " |",
        f"| `{TURBOQUANT_NAME}` | Restricted TurboQuant Python | "
        + ("yes" if TURBOQUANT_NAME in real_in_report else "no / other panel") + " |",
        f"| `{KIVI_NAME}` | Restricted KIVI offline | "
        + ("yes" if KIVI_NAME in real_in_report else "no / other panel") + " |",
        "",
        "## 5. Exactness result",
        "",
        f"| Total runs | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        "",
        "## 6. Acceptance by compressor",
        "",
        "| Compressor | Accept | Rejected | Corrections | Lossy div |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in sorted(by_comp, key=lambda r: -r["mean_acceptance_rate"]):
        div = sum(
            1 for r in report["results"]
            if r["compressor_name"] == row["compressor_name"]
            and not r.get("lossy", {}).get("token_exact_match", True)
        )
        lines.append(
            f"| `{row['compressor_name']}` | {_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{row['total_rejected']} | {row['total_corrections']} | {div} |"
        )

    lines.extend([
        "",
        "## 7. Acceptance by category",
        "",
        "| Category | Runs | Mean accept | Lossy div |",
        "|---|---:|---:|---:|",
    ])
    for row in by_cat:
        lines.append(
            f"| `{row['primary_category']}` | {row['num_runs']} | "
            f"{_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{row['lossy_divergence_count']} |"
        )

    lines.extend([
        "",
        "## 8. Real-backend comparison",
        "",
        "### KVQuant simquant",
        "",
    ])
    kv_row = next((r for r in by_comp if r["compressor_name"] == KVQUANT_NAME), None)
    if kv_row:
        lines.append(
            f"- `{KVQUANT_NAME}` mean accept **{_fmt_rate(kv_row['mean_acceptance_rate'])}** "
            f"(this report; V9 core anchor {_fmt_rate(_V9_REAL_BACKEND_ANCHORS[KVQUANT_NAME])})"
        )
    else:
        lines.append(f"- `{KVQUANT_NAME}` not in merged report.")

    lines.extend(["", "### TurboQuant Python", ""])
    tq_row = next((r for r in by_comp if r["compressor_name"] == TURBOQUANT_NAME), None)
    if tq_row:
        lines.append(
            f"- `{TURBOQUANT_NAME}` mean accept **{_fmt_rate(tq_row['mean_acceptance_rate'])}** "
            f"(V9 core anchor {_fmt_rate(_V9_REAL_BACKEND_ANCHORS[TURBOQUANT_NAME])})"
        )
    else:
        lines.append(f"- `{TURBOQUANT_NAME}` not in merged report.")

    lines.extend(["", "### KIVI offline", ""])
    kivi_row = next((r for r in by_comp if r["compressor_name"] == KIVI_NAME), None)
    if kivi_row:
        lines.append(
            f"- `{KIVI_NAME}` mean accept **{_fmt_rate(kivi_row['mean_acceptance_rate'])}** "
            f"(V9 core anchor {_fmt_rate(_V9_REAL_BACKEND_ANCHORS[KIVI_NAME])})"
        )
    else:
        lines.append(f"- `{KIVI_NAME}` not in merged report.")

    ranking = agg.get("real_backend_ranking", {})
    if ranking:
        lines.extend([
            "",
            "**Real-backend ranking (this subset):** "
            + " > ".join(
                f"`{c}` ({_fmt_rate(v)})" for c, v in ranking.get("ranked", [])
            ),
        ])

    lines.extend([
        "",
        "## 9. Built-in baseline comparison",
        "",
        "| Compressor | Mean accept |",
        "|---|---:|",
    ])
    for name in BUILTIN_COMPRESSORS:
        row = next((r for r in by_comp if r["compressor_name"] == name), None)
        if row:
            lines.append(f"| `{name}` | {_fmt_rate(row['mean_acceptance_rate'])} |")

    lines.extend([
        "",
        "## 10. Whether real-backend ranking changes on harder V10 categories",
        "",
    ])
    if len(real_in_report) >= 2:
        lines.append(
            "On this harder-category subset, restricted real-backend ordering "
            f"is documented above. Compare to V9 **core** suite anchors (§8) — "
            "category mix differs; ranks may shift."
        )
    else:
        lines.append(
            "_Insufficient real backends in merged report for full ranking comparison._"
        )

    lines.extend([
        "",
        "## 11. Whether KVQuant remains the strongest restricted real backend",
        "",
    ])
    if ranking:
        strongest = ranking.get("strongest")
        lines.append(
            f"Among restricted backends in this report, **`{strongest}`** has highest "
            f"mean acceptance. KVQuant strongest: **{ranking.get('kvquant_strongest')}**."
        )
    elif kv_row and tq_row and kivi_row:
        rates = {
            KVQUANT_NAME: kv_row["mean_acceptance_rate"],
            TURBOQUANT_NAME: tq_row["mean_acceptance_rate"],
            KIVI_NAME: kivi_row["mean_acceptance_rate"],
        }
        best = max(rates, key=rates.get)
        lines.append(f"**`{best}`** leads on this subset.")
    else:
        lines.append("_See merged panel results; not all backends in one environment._")

    lines.extend([
        "",
        "## 12. Whether boundary4 remains stronger than k8_v4_sim on this subset",
        "",
        f"- boundary4 wins: **{wl_b4['wins_a']}**",
        f"- k8_v4_sim wins: **{wl_b4['wins_b']}**",
        f"- ties: **{wl_b4['ties']}**",
        "",
        "| Category | boundary4 wins | k8_v4_sim wins | ties |",
        "|---|---:|---:|---:|",
    ])
    for cat in sorted(wl_b4.get("per_category", {})):
        c = wl_b4["per_category"][cat]
        lines.append(
            f"| `{cat}` | {c['wins_a']} | {c['wins_b']} | {c['ties']} |"
        )

    b4_rate = next(
        (r["mean_acceptance_rate"] for r in by_comp
         if r["compressor_name"] == "k8_v4_boundary4_v8_sim"),
        None,
    )
    k84_rate = next(
        (r["mean_acceptance_rate"] for r in by_comp
         if r["compressor_name"] == "k8_v4_sim"),
        None,
    )
    if b4_rate is not None and k84_rate is not None:
        margin = b4_rate - k84_rate
        lines.append(
            f"\nGlobal margin (boundary4 − k8_v4_sim): **{margin:+.3f}**."
        )

    lines.extend([
        "",
        "## 13. Divergence/rejection/correction summary",
        "",
        f"| Total rejected | {agg['total_rejected']} |",
        f"| Total corrections | {agg['total_corrections']} |",
        f"| Lossy divergence cells | {agg['lossy_divergence_count']} |",
        "",
        "Divergence cells by compressor:",
        "",
        "| Compressor | Divergence cells |",
        "|---|---:|",
    ])
    for comp, n in sorted(forensics["divergence_cells_by_compressor"].items()):
        lines.append(f"| `{comp}` | {n} |")

    lines.extend([
        "",
        "## 14. What this proves",
        "",
        "- Exactness gate holds on harder V10 category subset with factory-only real backends.",
        "- Category-stratified acceptance for built-in and restricted adapters at anchor settings.",
        "- Real-backend behaviour on long-context, retrieval-copy, tool-JSON, and code prompts.",
        "",
        "## 15. What this does not prove",
        "",
        "- Comprehensive benchmark coverage or production serving readiness.",
        "- That external paper results (TurboQuant, KIVI, KVQuant) apply unchanged.",
        "- Single-environment co-installation of all restricted backends.",
        "",
        "## 16. Limitations",
        "",
        f"- **40 prompts** only ({PROMPTS_PER_SUITE} per suite); low-n per category.",
        "- Real backends may require **separate panels** (isolated venvs / PYTHONPATH).",
        "- Cross-panel builtin rows are deduplicated from the dedicated builtin panel when merged.",
        "- Token-type and structured-output flags are heuristics only.",
        "",
        "## 17. Relation to V10 Phase 5 readiness assessment",
        "",
        "Experiment 014 completes V10 Phase 4 optional real-backend spot-checks. "
        "Phase 5 (v1.0.0 readiness assessment, not launch) may proceed per "
        "[`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md). v1.0.0 still requires "
        "V11 substance.",
        "",
        "## 18. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al., "
        "arXiv:2605.17613, 2026). ExactKV implements a compressor-agnostic "
        "evaluation harness; Experiment 014 does not claim novel compression methods.",
        "",
    ])
    return "\n".join(lines)


def _write_outputs(
    report: dict[str, Any],
    meta: dict[str, Any],
    json_out: Path,
    csv_out: Path,
    markdown_out: Path,
) -> None:
    _assert_no_forbidden_fields(report)
    validate_report(report)
    write_json_report(
        {"results": report["results"], "aggregate": report["aggregate"]},
        json_out,
        manifest=report.get("manifest"),
    )
    write_csv_report(report, csv_out)
    markdown_out.write_text(generate_markdown_report(report, meta), encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {markdown_out}")
    print(f"exactkv_failures: {report['aggregate']['exactkv_failures']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 014")
    parser.add_argument(
        "--panel",
        default="auto",
        choices=["auto", "builtin", "kvquant", "turboquant", "kivi"],
        help="Which compressor panel to run in the current environment",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge panel JSON files only (no new runs)",
    )
    parser.add_argument(
        "--merge-from",
        action="append",
        default=[],
        metavar="JSON",
        help="Panel JSON to merge (repeatable); used with --merge-only",
    )
    parser.add_argument(
        "--panel-json-out",
        default="",
        help="Optional separate JSON path for this panel (before merge)",
    )
    parser.add_argument(
        "--json-out",
        default="reports/experiment_014_real_backend_spotchecks.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_014_real_backend_spotchecks.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default=DTYPE,
        choices=["float32", "float16", "bfloat16"],
    )
    args = parser.parse_args()

    prompts = load_spotcheck_subset()

    if args.merge_only:
        if not args.merge_from:
            raise SystemExit("--merge-only requires at least one --merge-from JSON path")
        panel_paths = [Path(p) for p in args.merge_from]
        panels = [(load_json_report(p), str(p)) for p in panel_paths]
        report = merge_panel_reports(panels, prompts)
        meta = {
            "device": "merged",
            "dtype": "mixed",
            "prompt_count": len(prompts),
            "compressors": report["manifest"].get("compressor_names", []),
        }
        _write_outputs(
            report,
            meta,
            Path(args.json_out),
            Path(args.csv_out),
            Path(args.markdown_out),
        )
        return 0 if report["aggregate"]["exactkv_failures"] == 0 else 1

    compressors = _compressors_for_panel(args.panel)
    print(
        f"Experiment 014 panel={args.panel} prompts={len(prompts)} "
        f"compressors={compressors}",
        flush=True,
    )

    runtime = ModelRuntime(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
    )
    report = run_panel(
        runtime,
        prompts,
        compressors,
        panel_id=args.panel,
        dtype=args.dtype,
    )

    meta = {
        "device": str(runtime.device),
        "dtype": args.dtype,
        "prompt_count": len(prompts),
        "compressors": compressors,
        "panel": args.panel,
    }

    if args.panel_json_out:
        panel_path = Path(args.panel_json_out)
        _assert_no_forbidden_fields(report)
        write_json_report(
            {"results": report["results"], "aggregate": report["aggregate"]},
            panel_path,
            manifest=report["manifest"],
        )
        print(f"Wrote panel JSON {panel_path}")
        print(
            "Panel complete. Merge with --merge-only --merge-from … when all panels done.",
            flush=True,
        )
        return 0 if report["aggregate"]["exactkv_failures"] == 0 else 1

    _write_outputs(
        report,
        meta,
        Path(args.json_out),
        Path(args.csv_out),
        Path(args.markdown_out),
    )
    return 0 if report["aggregate"]["exactkv_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
