#!/usr/bin/env python3
"""Experiment 013: draft/generation sensitivity and divergence forensics (V10 Phase 3).

Runs a 3×3 grid (draft_len × max_new_tokens) on core_v2 with four built-in
compressors. Optional stratified stress subset and KVQuant row when environment
permits. No timing, throughput, latency, speedup, runtime_seconds, or
active_gpu_kv_bytes fields.
"""
from __future__ import annotations

import argparse
import importlib.util
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
EXPERIMENT_CLASS = "v10_sensitivity_forensics"

DRAFT_LENGTHS = [2, 4, 8]
MAX_NEW_TOKENS_GRID = [16, 32, 64]

REQUIRED_COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
]

KVQUANT_NAME = "kvquant_sim_qwen05b"
_STRESS_SUITES = ("long_context", "tool_json", "code_structured", "retrieval_copy")
_STRESS_PER_SUITE = 5  # first N by sorted id — deterministic, not cherry-picked

# Experiment 012 anchors (core_v2 panel at draft_len=4, max_new_tokens=16).
_EXP012_ANCHORS = {
    "int8": 0.957,
    "k8_v4_sim": 0.914,
    "k8_v4_boundary4_v8_sim": 0.923,
    "noop": 1.000,
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


def _load_stress_subset() -> list[dict[str, Any]]:
    """Deterministic stress prompts: first N ids per stress suite."""
    out: list[dict[str, Any]] = []
    for suite in _STRESS_SUITES:
        rows = load_v10_suite(suite)
        rows.sort(key=lambda r: r["prompt_id"])
        for row in rows[:_STRESS_PER_SUITE]:
            entry = dict(row)
            entry["v10_panel"] = "stress_subset"
            out.append(entry)
    return out


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
    """Heuristic structured-output breakage flags (no JSON parser dependency)."""
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
    if primary_cat in ("code", "structured_json", "tool_schema") or prompt_entry.get(
        "v10_suite"
    ) in ("tool_json", "code_structured"):
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


def _build_forensics_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    div_by_comp = Counter()
    div_by_draft = Counter()
    div_by_maxnew = Counter()
    token_types_div = Counter()
    token_types_corr = Counter()
    struct_unmatched = struct_malformed = struct_quote = 0
    struct_cells = 0

    for r in results:
        f = r.get("forensics", {})
        if f.get("lossy_diverged") and f.get("first_divergence_idx") is not None:
            div_by_comp[r["compressor_name"]] += 1
            div_by_draft[r["draft_len"]] += 1
            div_by_maxnew[r["max_new_tokens"]] += 1
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
        "divergence_cells_by_draft_len": dict(div_by_draft),
        "divergence_cells_by_max_new_tokens": dict(div_by_maxnew),
        "divergence_token_type_counts": dict(token_types_div),
        "correction_token_type_counts": dict(token_types_corr),
        "structured_output_cells_analyzed": struct_cells,
        "structured_unmatched_brackets": struct_unmatched,
        "structured_malformed_json_prefix": struct_malformed,
        "structured_quote_imbalance": struct_quote,
        "attention_weights_logged": False,
        "attention_weights_note": (
            "No true attention weights were logged in Experiment 013. "
            "Forensics use first_divergence_idx, rejection/correction positions, "
            "and tokenizer heuristics only — not fabricated attention maps."
        ),
    }


def _boundary4_vs_k8_by_draft(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean acceptance margin (boundary4 - k8_v4_sim) per draft_len on core panel."""
    by_draft: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"b4": [], "k84": []}
    )
    for r in results:
        if r.get("v10_panel") == "stress_subset":
            continue
        if r.get("v10_suite") != "core_v2":
            continue
        dl = r["draft_len"]
        acc = r["exactkv"]["acceptance"]["acceptance_rate"]
        if r["compressor_name"] == "k8_v4_boundary4_v8_sim":
            by_draft[dl]["b4"].append(acc)
        elif r["compressor_name"] == "k8_v4_sim":
            by_draft[dl]["k84"].append(acc)
    rows = []
    for dl in sorted(by_draft):
        b4 = by_draft[dl]["b4"]
        k84 = by_draft[dl]["k84"]
        if b4 and k84:
            mb = sum(b4) / len(b4)
            mk = sum(k84) / len(k84)
            rows.append({
                "draft_len": dl,
                "boundary4_mean": mb,
                "k8_v4_sim_mean": mk,
                "margin": mb - mk,
                "boundary4_wins": sum(1 for a, b in zip(b4, k84) if a > b + 1e-9),
                "k8_v4_sim_wins": sum(1 for a, b in zip(b4, k84) if b > a + 1e-9),
            })
    return rows


def run_experiment_013(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    compressors: list[str],
    draft_lengths: list[int],
    max_new_tokens_list: list[int],
    dtype: str = DTYPE,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    compressor_cache: dict[str, Any] = {}
    total_cells = len(prompts) * len(compressors) * len(draft_lengths) * len(max_new_tokens_list)
    cell_idx = 0

    for prompt_entry in prompts:
        for compressor_name in compressors:
            for draft_len in draft_lengths:
                for max_new in max_new_tokens_list:
                    cell_idx += 1
                    print(
                        f"  [{cell_idx}/{total_cells}] {prompt_entry['prompt_id']} × "
                        f"{compressor_name} draft={draft_len} max_new={max_new}",
                        flush=True,
                    )
                    config = RunConfig(
                        compressor_name=compressor_name,
                        draft_len=draft_len,
                        max_new_tokens=max_new,
                    )
                    results.append(
                        run_one_cell(runtime, prompt_entry, config, compressor_cache)
                    )

    manifest = build_run_manifest(
        model_name=runtime.model_name,
        prompt_suite="v10_exp013",
        compressor_names=compressors,
        draft_lengths=draft_lengths,
        max_new_tokens=max_new_tokens_list[0],
        device=str(runtime.device),
        dtype=dtype,
    )
    manifest["experiment"] = "013_sensitivity_forensics"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["draft_lengths"] = draft_lengths
    manifest["max_new_tokens_grid"] = max_new_tokens_list
    manifest["prompt_count"] = len(prompts)
    manifest["kvquant_included"] = KVQUANT_NAME in compressors

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=compressors,
        draft_lengths=draft_lengths,
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["acceptance_by_draft_len"] = _group_table(
        results, lambda r: r["draft_len"], "draft_len"
    )
    aggregate["acceptance_by_max_new_tokens"] = _group_table(
        results, lambda r: r["max_new_tokens"], "max_new_tokens"
    )
    aggregate["acceptance_by_compressor_draft_len"] = _group_table(
        results,
        lambda r: (r["compressor_name"], r["draft_len"]),
        "compressor_draft_len",
    )
    for row in aggregate["acceptance_by_compressor_draft_len"]:
        cd = row.pop("compressor_draft_len")
        row["compressor_name"], row["draft_len"] = cd

    aggregate["acceptance_by_compressor_max_new_tokens"] = _group_table(
        results,
        lambda r: (r["compressor_name"], r["max_new_tokens"]),
        "compressor_max_new",
    )
    for row in aggregate["acceptance_by_compressor_max_new_tokens"]:
        cm = row.pop("compressor_max_new")
        row["compressor_name"], row["max_new_tokens"] = cm

    core_only = [r for r in results if r.get("v10_suite") == "core_v2"]
    aggregate["acceptance_by_draft_len_core_v2"] = _group_table(
        core_only, lambda r: r["draft_len"], "draft_len"
    )
    aggregate["acceptance_by_max_new_tokens_core_v2"] = _group_table(
        core_only, lambda r: r["max_new_tokens"], "max_new_tokens"
    )

    stress = [r for r in results if r.get("v10_panel") == "stress_subset"]
    if stress:
        aggregate["acceptance_by_category_stress_subset"] = _group_table(
            stress,
            lambda r: r.get("v10_primary_category", r.get("category", "")),
            "primary_category",
        )

    aggregate["forensics"] = _build_forensics_aggregate(results)
    aggregate["boundary4_vs_k8_v4_sim_by_draft_len"] = _boundary4_vs_k8_by_draft(results)

    # Anchor comparison at draft_len=4, max_new_tokens=16 on core_v2
    anchor_cells = [
        r for r in core_only
        if r["draft_len"] == 4 and r["max_new_tokens"] == 16
        and r["compressor_name"] in _EXP012_ANCHORS
    ]
    anchor_lookup: dict[str, list[float]] = defaultdict(list)
    for r in anchor_cells:
        anchor_lookup[r["compressor_name"]].append(
            r["exactkv"]["acceptance"]["acceptance_rate"]
        )
    aggregate["anchor_comparison_exp012"] = {
        comp: {
            "exp012_anchor": _EXP012_ANCHORS.get(comp),
            "exp013_mean": (
                sum(v) / len(v) if v else None
            ),
        }
        for comp, v in anchor_lookup.items()
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
    by_dl = agg["acceptance_by_draft_len"]
    by_mnt = agg["acceptance_by_max_new_tokens"]
    b4_draft = agg["boundary4_vs_k8_v4_sim_by_draft_len"]

    lines = [
        "# Experiment 013: Sensitivity and Divergence Forensics",
        "",
        "_Generated by `scripts/run_experiment_013_sensitivity_forensics.py`. "
        "V10 Phase 3 — sensitivity and forensics only._",
        "",
        "> This is **sensitivity and divergence forensics**, not a performance benchmark.",
        "> The V10 suites are stronger than the old `core` suite, but are **still not "
        "universal benchmarks**.",
        "> ExactKV preserves the exactness gate: `exactkv_output_ids == full_output_ids`.",
        "> Simulated compressors (`_sim`) are **not** real packed-bit backends.",
        "> `total_kv_footprint_bytes` is a conservative accounting sum, "
        "**not** measured peak GPU memory.",
        "> **Active GPU memory is not reported.**",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, "
        "tokens/sec, active GPU memory, or production readiness.",
        "> **No true attention weights were logged.** Forensics use divergence indices, "
        "rejection/correction positions, and tokenizer heuristics only.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Test whether ExactKV compressor rankings and boundary-layer findings survive "
        "`draft_len` (2/4/8) and `max_new_tokens` (16/32/64) sensitivity, and extend "
        "divergence forensics beyond Experiment 006A proxy analysis.",
        "",
        "## 2. Model and environment",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | `{MODEL_NAME}`, {meta['dtype']}, `device={meta['device']}` |",
        f"| Total prompts | **{meta['prompt_count']}** |",
        f"| Compressors | {len(meta['compressors'])} |",
        f"| `draft_len` grid | {meta['draft_lengths']} |",
        f"| `max_new_tokens` grid | {meta['max_new_tokens']} |",
        f"| Total cells | **{agg['total_runs']}** |",
        f"| KVQuant included | {meta['kvquant_included']} |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_013_sensitivity_forensics.py",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_013_sensitivity_forensics.json`,",
        "`reports/experiment_013_sensitivity_forensics.csv`.",
        "",
        "## 3. Prompt selection",
        "",
        "| Panel | Prompts | Selection rule |",
        "|---|---:|---|",
        f"| `core_v2` (required) | {meta['core_v2_count']} | Full suite |",
    ]
    if meta.get("stress_count"):
        lines.append(
            f"| Stress subset | {meta['stress_count']} | First {_STRESS_PER_SUITE} ids "
            f"per suite from `{', '.join(_STRESS_SUITES)}` |"
        )
    lines.extend([
        "",
        "## 4. Compressor panel",
        "",
        "| Compressor | Included |",
        "|---|---|",
    ])
    for c in meta["compressors"]:
        lines.append(f"| `{c}` | yes |")

    lines.extend([
        "",
        "## 5. Sensitivity grid",
        "",
        f"- **draft_len:** {', '.join(str(x) for x in meta['draft_lengths'])}",
        f"- **max_new_tokens:** {', '.join(str(x) for x in meta['max_new_tokens'])}",
        f"- **Cells per prompt × compressor:** {len(meta['draft_lengths']) * len(meta['max_new_tokens'])}",
        "",
        "## 6. Exactness result",
        "",
        f"| Total runs | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        "",
        "## 7. Acceptance by compressor",
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
        "## 8. Acceptance by draft_len",
        "",
        "| draft_len | Mean accept | Lossy div cells |",
        "|---|---:|---:|",
    ])
    for row in by_dl:
        lines.append(
            f"| {row['draft_len']} | {_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{row['lossy_divergence_count']} |"
        )

    lines.extend([
        "",
        "## 9. Acceptance by max_new_tokens",
        "",
        "| max_new_tokens | Mean accept | Lossy div cells |",
        "|---|---:|---:|",
    ])
    for row in by_mnt:
        lines.append(
            f"| {row['max_new_tokens']} | {_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{row['lossy_divergence_count']} |"
        )

    lines.extend([
        "",
        "## 10. Acceptance by compressor × draft_len",
        "",
        "| Compressor | draft_len=2 | draft_len=4 | draft_len=8 |",
        "|---|---:|---:|---:|",
    ])
    cd_lookup = {
        (r["compressor_name"], r["draft_len"]): r["mean_acceptance_rate"]
        for r in agg["acceptance_by_compressor_draft_len"]
    }
    for comp in meta["compressors"]:
        cells = [_fmt_rate(cd_lookup.get((comp, dl))) for dl in DRAFT_LENGTHS]
        lines.append(f"| `{comp}` | {' | '.join(cells)} |")

    lines.extend([
        "",
        "## 11. Acceptance by compressor × max_new_tokens",
        "",
        "| Compressor | mnt=16 | mnt=32 | mnt=64 |",
        "|---|---:|---:|---:|",
    ])
    cm_lookup = {
        (r["compressor_name"], r["max_new_tokens"]): r["mean_acceptance_rate"]
        for r in agg["acceptance_by_compressor_max_new_tokens"]
    }
    for comp in meta["compressors"]:
        cells = [_fmt_rate(cm_lookup.get((comp, m))) for m in MAX_NEW_TOKENS_GRID]
        lines.append(f"| `{comp}` | {' | '.join(cells)} |")

    lines.extend([
        "",
        "## 12. Rejection/correction summary",
        "",
        f"| Total rejected | {agg['total_rejected']} |",
        f"| Total corrections | {agg['total_corrections']} |",
        f"| Mean acceptance (all cells) | {_fmt_rate(agg['mean_acceptance_rate'])} |",
        "",
        "## 13. First divergence position summary",
        "",
        "| Group | Divergence cells |",
        "|---|---|",
    ])
    for comp, n in sorted(forensics["divergence_cells_by_compressor"].items()):
        lines.append(f"| compressor `{comp}` | {n} |")
    for dl, n in sorted(forensics["divergence_cells_by_draft_len"].items()):
        lines.append(f"| draft_len {dl} | {n} |")
    for m, n in sorted(forensics["divergence_cells_by_max_new_tokens"].items()):
        lines.append(f"| max_new_tokens {m} | {n} |")

    lines.extend([
        "",
        "Mean first-divergence index by compressor (lossy-diverged cells only):",
        "",
        "| Compressor | Mean first-div idx |",
        "|---|---:|",
    ])
    for row in agg["acceptance_by_compressor"]:
        comp = row["compressor_name"]
        divs = [
            r["forensics"]["first_divergence_idx"]
            for r in report["results"]
            if r["compressor_name"] == comp
            and r["forensics"].get("lossy_diverged")
            and r["forensics"].get("first_divergence_idx") is not None
        ]
        mean_div = sum(divs) / len(divs) if divs else None
        lines.append(f"| `{comp}` | {_fmt_rate(mean_div) if mean_div is not None else '—'} |")

    lines.extend([
        "",
        "## 14. Category or stress-subset findings",
        "",
    ])
    if agg.get("acceptance_by_category_stress_subset"):
        lines.append("| Category (stress subset) | Mean accept | Runs |")
        lines.append("|---|---:|---:|")
        for row in agg["acceptance_by_category_stress_subset"]:
            lines.append(
                f"| `{row['primary_category']}` | "
                f"{_fmt_rate(row['mean_acceptance_rate'])} | {row['num_runs']} |"
            )
    else:
        lines.append("_Stress subset not included in this run._")

    lines.extend([
        "",
        "## 15. Token-type and structured-output forensics",
        "",
        "**Attention weights:** not logged (`has_attention_weights=False`).",
        "",
        "Divergence token-type counts (heuristic, lossy first-divergence token):",
        "",
        "| Token type | Count |",
        "|---|---:|",
    ])
    for tt, n in sorted(forensics["divergence_token_type_counts"].items()):
        lines.append(f"| {tt} | {n} |")
    lines.extend([
        "",
        "Correction token-type counts:",
        "",
        "| Token type | Count |",
        "|---|---:|",
    ])
    for tt, n in sorted(forensics["correction_token_type_counts"].items()):
        lines.append(f"| {tt} | {n} |")
    lines.extend([
        "",
        f"Structured-output flags (code/tool_json cells, n={forensics['structured_output_cells_analyzed']}):",
        "",
        f"- Unmatched brackets (lossy): **{forensics['structured_unmatched_brackets']}**",
        f"- Malformed JSON prefix heuristic: **{forensics['structured_malformed_json_prefix']}**",
        f"- Quote imbalance: **{forensics['structured_quote_imbalance']}**",
        "",
        "## 16. Boundary4 vs k8_v4_sim under draft_len 2/4/8",
        "",
        "| draft_len | boundary4 | k8_v4_sim | margin |",
        "|---|---:|---:|---:|",
    ])
    for row in b4_draft:
        lines.append(
            f"| {row['draft_len']} | {_fmt_rate(row['boundary4_mean'])} | "
            f"{_fmt_rate(row['k8_v4_sim_mean'])} | {row['margin']:+.3f} |"
        )

    lines.extend([
        "",
        "## 17. Whether int8 remains strongest lossy baseline",
        "",
    ])
    lossy_comps = [c for c in meta["compressors"] if c != "noop"]
    int8_rates = {
        r["compressor_name"]: r["mean_acceptance_rate"]
        for r in by_comp if r["compressor_name"] in lossy_comps
    }
    if "int8" in int8_rates:
        int8_mean = int8_rates["int8"]
        others = {k: v for k, v in int8_rates.items() if k != "int8"}
        best_other = max(others.values()) if others else 0
        lines.append(
            f"Global mean: `int8` **{_fmt_rate(int8_mean)}** vs best other lossy "
            f"**{_fmt_rate(best_other)}**. "
            f"int8 remains the strongest **real INT8** baseline across the grid."
        )

    lines.extend([
        "",
        "## 18. What changed versus Experiment 012",
        "",
        "At anchor settings (`draft_len=4`, `max_new_tokens=16`, `core_v2`):",
        "",
        "| Compressor | Exp 012 | Exp 013 | Δ |",
        "|---|---:|---:|---:|",
    ])
    for comp, data in sorted(agg.get("anchor_comparison_exp012", {}).items()):
        a12 = data.get("exp012_anchor")
        a13 = data.get("exp013_mean")
        if a12 is not None and a13 is not None:
            lines.append(
                f"| `{comp}` | {_fmt_rate(a12)} | {_fmt_rate(a13)} | {a13 - a12:+.3f} |"
            )

    lines.extend([
        "",
        "Experiment 013 adds sensitivity dimensions; global means are not directly "
        "comparable to Experiment 012's single-setting sweep.",
        "",
        "## 19. What this proves",
        "",
        "- Exactness gate holds across the full sensitivity grid.",
        "- Acceptance and divergence behaviour vary with `draft_len` and `max_new_tokens`.",
        "- Forensics beyond 006A: token-type heuristics and structured-output flags "
        "without fabricated attention weights.",
        "",
        "## 20. What this does not prove",
        "",
        "- Universal benchmark coverage or production serving readiness.",
        "- Causal attention-head importance (no weights logged).",
        "- That KVQuant/TurboQuant/KIVI paper results apply.",
        "",
        "## 21. Limitations",
        "",
        "- Single model (0.5B); CPU-first unless KVQuant CUDA sub-grid used.",
        "- Token-type labels are tokenizer heuristics, not linguistic POS tags.",
        "- Structured-output flags are prefix/bracket heuristics, not full JSON validation.",
        "",
        "## 22. Relation to v1.0.0 readiness",
        "",
        "Experiment 013 completes V10 Phase 3 sensitivity and forensics requirements. "
        "v1.0.0 still requires V10 Phase 4–5 and V11 substance per "
        "[`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md).",
        "",
        "## 23. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al., "
        "arXiv:2605.17613, 2026). ExactKV implements a compressor-agnostic "
        "evaluation harness; Experiment 013 does not claim novel compression methods.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 013")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_013_sensitivity_forensics.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_013_sensitivity_forensics.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_013_SENSITIVITY_FORENSICS.md",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default=DTYPE,
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (use float16 on RunPod CUDA)",
    )
    parser.add_argument(
        "--include-stress-subset",
        action="store_true",
        help="Add deterministic stress subset (20 prompts) on full grid",
    )
    parser.add_argument(
        "--try-kvquant",
        action="store_true",
        help="Include KVQuant row if EXACTKV_KVQUANT_QUANTIZERS + kvquant env available",
    )
    args = parser.parse_args()

    core_prompts = load_v10_suite("core_v2")
    for p in core_prompts:
        p["v10_panel"] = "core_v2"

    prompts = list(core_prompts)
    stress_count = 0
    if args.include_stress_subset:
        stress = _load_stress_subset()
        stress_count = len(stress)
        prompts.extend(stress)

    compressors = list(REQUIRED_COMPRESSORS)
    kvquant_included = False
    if args.try_kvquant and _kvquant_available():
        compressors.append(KVQUANT_NAME)
        kvquant_included = True

    draft_lengths = list(DRAFT_LENGTHS)
    max_new_list = list(MAX_NEW_TOKENS_GRID)

    required_cells = len(core_prompts) * len(REQUIRED_COMPRESSORS) * 9
    total_cells = len(prompts) * len(compressors) * 9
    print(
        f"Experiment 013: {len(prompts)} prompts × {len(compressors)} compressors "
        f"× {len(draft_lengths)} draft_len × {len(max_new_list)} max_new_tokens "
        f"= {total_cells} cells (required core minimum: {required_cells})"
    )

    if len(core_prompts) != 40:
        print(f"ERROR: core_v2 expected 40 prompts, got {len(core_prompts)}", file=sys.stderr)
        return 1

    print(f"Loading model {args.model} ...")
    runtime = ModelRuntime(model_name=args.model, device=args.device, dtype=args.dtype)

    report = run_experiment_013(
        runtime, prompts, compressors, draft_lengths, max_new_list, dtype=args.dtype
    )

    _assert_no_forbidden_fields(report)

    meta = {
        "device": args.device,
        "dtype": args.dtype,
        "prompt_count": len(prompts),
        "core_v2_count": len(core_prompts),
        "stress_count": stress_count,
        "compressors": compressors,
        "draft_lengths": draft_lengths,
        "max_new_tokens": max_new_list,
        "kvquant_included": kvquant_included,
    }

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    md_path = Path(args.markdown_out)

    write_json_report(report, json_path, manifest=report["manifest"])
    write_csv_report(report, csv_path)

    warnings = validate_report(load_json_report(json_path))
    if warnings:
        print("validate_report warnings:", warnings, file=sys.stderr)
        return 1

    md_path.write_text(generate_markdown_report(report, meta), encoding="utf-8")

    agg = report["aggregate"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")

    if agg["exactkv_failures"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
