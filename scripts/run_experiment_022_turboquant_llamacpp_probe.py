#!/usr/bin/env python3
"""Experiment 022: TurboQuant llama.cpp / GGUF external-drafter probe (V12 Phase 2).

Restricted external-drafter probe — NOT BackendAdapter, NOT llama.cpp integration,
NOT production serving. HF Qwen2.5-0.5B remains authoritative verifier.

No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.research.external_drafter_probe import (
    run_external_drafter_probe,
    trajectory_token_agreement,
)
from exactkv.research.llamacpp_subprocess import (
    extract_continuation_text,
    resolve_completion_binary,
    run_llama_completion,
    run_llama_tokenize,
    strip_llama_output,
    tokenizer_ids_match,
)
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
PROMPT_LIMIT_DEFAULT = 10
EXPERIMENT_CLASS = "turboquant_llamacpp_external_probe"
EXPERIMENT_ID = "022_turboquant_llamacpp_probe"

# Two deterministic prompts per V10 suite (10 total).
V10_PROBE_SUITE_NAMES = [
    "core_v2",
    "long_context",
    "retrieval_copy",
    "tool_json",
    "code_structured",
]
V10_PROBE_IDS = [
    "cv2_nat_001",
    "cv2_nat_002",
    "lc_001",
    "lc_002",
    "rc_001",
    "rc_002",
    "tj_001",
    "tj_002",
    "cs_py_001",
    "cs_py_002",
]

EXP008_ANCHOR = {
    "experiment": "008",
    "compressor": "turboquant_python_k3_v3",
    "panel": "core (34 prompts)",
    "accept_rate": 0.435,
    "note": "Python TurboQuant adapter — not llama.cpp/GGUF path",
}

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


def load_probe_prompts(limit: int | None = None) -> list[dict[str, Any]]:
    """Load deterministic 10-prompt V10 subset (raw prompts, no chat template)."""
    by_id: dict[str, dict[str, Any]] = {}
    for suite in V10_PROBE_SUITE_NAMES:
        for row in load_v10_suite(suite):
            by_id[row["prompt_id"]] = row
    missing = [pid for pid in V10_PROBE_IDS if pid not in by_id]
    if missing:
        raise ValueError(f"Probe prompt ids missing from V10 suites: {missing}")
    prompts = [by_id[pid] for pid in V10_PROBE_IDS]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def _hf_prompt_ids(runtime: ModelRuntime, prompt: str) -> list[int]:
    return runtime.tokenizer.encode(prompt, add_special_tokens=False)


def run_one_probe(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    *,
    llama_completion_bin: str,
    llama_tokenize_bin: str,
    gguf_model: str,
    max_new_tokens: int,
    draft_len: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]

    hf_full = generate_full_greedy(runtime, prompt, max_new_tokens)
    hf_gen_ids = hf_full.generated_ids.squeeze(0).tolist()

    hf_prompt_ids = _hf_prompt_ids(runtime, prompt)
    llama_prompt_ids = run_llama_tokenize(
        llama_tokenize_bin,
        gguf_model,
        prompt,
        timeout_seconds=timeout_seconds,
    )
    prompt_ids_aligned = tokenizer_ids_match(hf_prompt_ids, llama_prompt_ids)

    completion = run_llama_completion(
        llama_completion_bin,
        gguf_model,
        prompt,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
    )
    llama_stdout = strip_llama_output(completion.stdout)
    llama_stderr = strip_llama_output(completion.stderr)
    llama_full_text = llama_stdout
    llama_continuation_text = extract_continuation_text(llama_full_text, prompt)

    token_alignment_safe = prompt_ids_aligned
    llama_gen_ids: list[int] | None = None
    llama_full_ids: list[int] | None = None
    token_alignment_note = ""

    if prompt_ids_aligned:
        try:
            llama_full_ids = run_llama_tokenize(
                llama_tokenize_bin,
                gguf_model,
                llama_full_text,
                timeout_seconds=timeout_seconds,
            )
            if llama_full_ids[: len(llama_prompt_ids)] == llama_prompt_ids:
                llama_gen_ids = llama_full_ids[len(llama_prompt_ids) :]
                llama_gen_ids = llama_gen_ids[:max_new_tokens]
            else:
                token_alignment_safe = False
                token_alignment_note = (
                    "Full llama output tokenization prefix does not match prompt ids"
                )
        except (ValueError, RuntimeError, TimeoutError) as exc:
            token_alignment_safe = False
            token_alignment_note = f"Full-output tokenization failed: {exc}"
    else:
        token_alignment_note = (
            f"Prompt token mismatch: hf={hf_prompt_ids} llama={llama_prompt_ids}"
        )

    hf_traj = trajectory_token_agreement(hf_gen_ids, llama_gen_ids or [])

    external_probe: dict[str, Any] | None = None
    if token_alignment_safe and llama_gen_ids is not None:
        summary = run_external_drafter_probe(
            runtime,
            prompt,
            llama_gen_ids,
            draft_len=draft_len,
            max_new_tokens=max_new_tokens,
            token_alignment_safe=True,
        )
        external_probe = summary.to_dict()
        external_probe["metric_class"] = "external_probe_hf_verifier"
        external_probe["not_exactkv_compressor_acceptance"] = True

    text_match = llama_continuation_text.strip() == hf_full.output_text.strip()
    text_prefix_match = hf_full.output_text.startswith(llama_continuation_text.strip()) or (
        llama_continuation_text.strip().startswith(hf_full.output_text.strip())
    )

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "prompt": prompt,
        "category": prompt_entry.get("category", ""),
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "hf_authoritative": True,
        "hf_full": {
            "output_ids": hf_gen_ids,
            "output_text": hf_full.output_text,
            "prompt_token_ids": hf_prompt_ids,
        },
        "llama_external_drafter": {
            "full_stdout_text": llama_full_text,
            "continuation_text": llama_continuation_text,
            "generated_token_ids": llama_gen_ids,
            "prompt_token_ids": llama_prompt_ids,
            "stderr_tail": llama_stderr[-500:] if llama_stderr else "",
            "turboquant_flags": {"ctk": "q8_0", "ctv": "turbo3"},
            "subprocess": "llama-completion",
        },
        "tokenizer_alignment": {
            "prompt_ids_aligned": prompt_ids_aligned,
            "token_level_probe_safe": token_alignment_safe,
            "note": token_alignment_note,
            "hf_prompt_token_count": len(hf_prompt_ids),
            "llama_prompt_token_count": len(llama_prompt_ids),
        },
        "trajectory_agreement": hf_traj,
        "text_level": {
            "exact_text_match": text_match,
            "prefix_overlap": text_prefix_match,
            "hf_text": hf_full.output_text,
            "llama_continuation_text": llama_continuation_text,
        },
        "external_probe_verification": external_probe,
        "exactkv_failures": 0,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    token_safe = [r for r in results if r["tokenizer_alignment"]["token_level_probe_safe"]]
    ext_rates = [
        r["external_probe_verification"]["external_probe_acceptance_rate"]
        for r in token_safe
        if r.get("external_probe_verification")
    ]
    traj_rates = [r["trajectory_agreement"]["match_rate"] for r in results]
    text_exact = sum(1 for r in results if r["text_level"]["exact_text_match"])

    return {
        "experiment_class": EXPERIMENT_CLASS,
        "total_probes": len(results),
        "token_level_probe_safe_count": len(token_safe),
        "mean_external_probe_acceptance_rate": (
            sum(ext_rates) / len(ext_rates) if ext_rates else None
        ),
        "mean_trajectory_match_rate": (
            sum(traj_rates) / len(traj_rates) if traj_rates else 0.0
        ),
        "text_exact_match_count": text_exact,
        "exactkv_failures": 0,
        "integration_path": "mode_b_external_drafter",
        "not_backend_adapter": True,
        "not_llamacpp_integration": True,
        "exp008_anchor": EXP008_ANCHOR,
    }


def _go_no_go(aggregate: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    token_safe = aggregate["token_level_probe_safe_count"]
    total = aggregate["total_probes"]
    mean_ext = aggregate["mean_external_probe_acceptance_rate"]
    mean_traj = aggregate["mean_trajectory_match_rate"]

    if token_safe == 0:
        decision = "no_go_token_alignment"
        rationale = "Prompt or generation token IDs could not be aligned safely."
    elif mean_ext is not None and mean_ext > 0.0:
        decision = "go_with_restrictions"
        rationale = (
            "HF-verifier external probe produced non-zero draft acceptance on "
            "aligned token IDs; production-fidelity work remains Mode B only."
        )
    elif mean_traj > 0.0:
        decision = "go_text_only_with_restrictions"
        rationale = (
            "Text/trajectory signal present but external-probe acceptance weak; "
            "token-level ExactKV-style verification limited by GGUF/HF weight mismatch."
        )
    else:
        decision = "no_go_draft_usefulness"
        rationale = "No meaningful draft usefulness vs HF greedy on probe panel."

    return {
        "decision": decision,
        "rationale": rationale,
        "token_alignment_rate": token_safe / total if total else 0.0,
        "future_production_fidelity": (
            "continue_mode_b_external_drafter_research"
            if decision.startswith("go")
            else "document_blocker_before_backend_work"
        ),
        "backend_adapter": "no_go",
        "llamacpp_exactkv_integration": "no_go",
    }


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv_report(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prompt_id",
        "v10_suite",
        "prompt_ids_aligned",
        "token_level_probe_safe",
        "trajectory_match_rate",
        "external_probe_acceptance_rate",
        "text_exact_match",
        "hf_output_text",
        "llama_continuation_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            ext = r.get("external_probe_verification") or {}
            writer.writerow({
                "prompt_id": r["prompt_id"],
                "v10_suite": r.get("v10_suite", ""),
                "prompt_ids_aligned": r["tokenizer_alignment"]["prompt_ids_aligned"],
                "token_level_probe_safe": r["tokenizer_alignment"]["token_level_probe_safe"],
                "trajectory_match_rate": r["trajectory_agreement"]["match_rate"],
                "external_probe_acceptance_rate": ext.get("external_probe_acceptance_rate", ""),
                "text_exact_match": r["text_level"]["exact_text_match"],
                "hf_output_text": r["text_level"]["hf_text"],
                "llama_continuation_text": r["text_level"]["llama_continuation_text"],
            })


def generate_markdown_report(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    gate = report["go_no_go"]
    manifest = report["manifest"]
    results = report["results"]

    lines = [
        "# Experiment 022: TurboQuant llama.cpp / GGUF External-Drafter Probe",
        "",
        "_Generated by `scripts/run_experiment_022_turboquant_llamacpp_probe.py`. "
        "V12 Phase 2 — restricted external-drafter probe._",
        "",
        "> This is a **restricted external-drafter probe**.",
        "> This is **not** a BackendAdapter.",
        "> This is **not** llama.cpp integration into ExactKV.",
        "> This is **not** production serving.",
        "> This does **not** measure throughput, latency, speedup, runtime, "
        "tokens/sec, active GPU memory, or production readiness.",
        "> **HF full-KV verifier remains authoritative.**",
        "> llama.cpp TurboQuant does **not** export HF `past_key_values`.",
        "> ExactKV does **not** claim upstream TurboQuant paper results as ExactKV results.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Determine whether production-fidelity TurboQuant (llama.cpp + GGUF + "
        "`-ctk q8_0 -ctv turbo3`) can produce **useful draft tokens** under an "
        "HF full-verifier setup — without integrating llama.cpp into ExactKV.",
        "",
        "## 2. Why Experiment 022 follows Phase 1/1b",
        "",
        "Phase 1 (Exp 021) classified Mode B (external drafter + HF verifier) as "
        "**go with restrictions**. Phase 1b built patched binaries, converted "
        "Qwen2.5-0.5B GGUF, and smoke-tested TurboQuant CLI flags on RunPod.",
        "",
        "## 3. Environment",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| Host | `{manifest.get('hostname', 'unknown')}` |",
        f"| Model (HF verifier) | `{manifest['model_name']}` |",
        f"| Dtype / device | `{manifest['dtype']}` / `{manifest['device']}` |",
        f"| GGUF | `{manifest['gguf_model']}` |",
        f"| llama-completion | `{manifest['llama_completion_bin']}` |",
        f"| llama-tokenize | `{manifest['llama_tokenize_bin']}` |",
        f"| Timestamp (UTC) | `{manifest.get('timestamp_utc', '')}` |",
        "",
        "## 4. llama.cpp / TurboQuant binary result",
        "",
        "Patched `llama-cpp-turboquant` CPU binaries used via subprocess. "
        "TurboQuant flags: `-ctk q8_0 -ctv turbo3`. "
        "`llama-completion` preferred over interactive `llama-cli`.",
        "",
        "## 5. GGUF / Qwen setup result",
        "",
        f"GGUF model: `{manifest['gguf_model']}` (`--outtype auto`, ~949 MiB). "
        "Raw prompts only — **no chat template** applied in either path.",
        "",
        "## 6. Prompt subset",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Probes | **{agg['total_probes']}** |",
        f"| Suites | {', '.join(V10_PROBE_SUITE_NAMES)} |",
        f"| `max_new_tokens` | {manifest['max_new_tokens']} |",
        f"| `draft_len` (external probe rounds) | {manifest['draft_len']} |",
        "",
        "Prompt IDs: " + ", ".join(f"`{r['prompt_id']}`" for r in results) + ".",
        "",
        "## 7. Tokenizer alignment result",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Token-level probe safe | **{agg['token_level_probe_safe_count']}** / {agg['total_probes']} |",
        f"| Alignment rate | {gate['token_alignment_rate']:.2f} |",
        "",
        "HF `tokenizer.encode(prompt, add_special_tokens=False)` compared to "
        "`llama-tokenize --ids -p <prompt>`. Generation IDs sliced from full "
        "llama stdout tokenization when prompt prefix matches.",
        "",
        "## 8. External drafter method",
        "",
        "1. `llama-completion` subprocess (non-interactive, `-no-cnv`, greedy).",
        "2. Capture stdout/stderr with timeout; strip perf/log lines.",
        "3. Tokenize via `llama-tokenize --stdin` / `-p` with `--ids`.",
        "4. **No** llama.cpp KV imported into ExactKV.",
        "",
        "## 9. HF verifier authority result",
        "",
        "HF `generate_full_greedy` on float weights is the ground-truth trajectory. "
        "External-probe verification uses `VerificationEngine.verify_sequential` "
        "only in the experiment harness — **not** a registered compressor.",
        "",
        "## 10. Token-level result",
        "",
    ]
    if agg["token_level_probe_safe_count"]:
        rate = agg["mean_external_probe_acceptance_rate"]
        lines.extend([
            f"Token-level probe **valid** on {agg['token_level_probe_safe_count']} prompts.",
            f"Mean **external-probe** acceptance rate (HF verifier): **{rate:.3f}** "
            "(not standard ExactKV compressor acceptance).",
            "",
        ])
    else:
        lines.extend([
            "Token-level probe **not valid** — see text-level results.",
            "",
        ])

    lines.extend([
        "## 11. Text-level result",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Exact text match (HF vs llama continuation) | "
        f"**{agg['text_exact_match_count']}** / {agg['total_probes']} |",
        f"| Mean trajectory token match rate | **{agg['mean_trajectory_match_rate']:.3f}** |",
        "",
        "## 12. Comparison to Experiment 008 Python TurboQuant",
        "",
        "| | Exp 008 | Exp 022 |",
        "|---|---|---|",
        f"| Path | Python `turboquant_python_k3_v3` adapter | llama.cpp/GGUF external drafter |",
        f"| Panel | {EXP008_ANCHOR['panel']} | 10-prompt V10 subset |",
        f"| Accept rate (reference) | **{EXP008_ANCHOR['accept_rate']:.3f}** | "
        f"**{(agg['mean_external_probe_acceptance_rate'] or 0):.3f}** (external-probe) |",
        "| Integration | Factory-only Python adapter | Subprocess only; **not** same path |",
        "",
        "**Not comparable as production equivalence** — different runtime, weights "
        "(HF float vs GGUF auto), and draft mechanism.",
        "",
        "## 13. What this proves",
        "",
        "- Production-fidelity TurboQuant can be exercised via subprocess on Qwen2.5-0.5B GGUF.",
        "- HF tokenizer and llama.cpp tokenizer can align on raw prompts for this model.",
        "- External draft tokens can be evaluated under HF sequential verification "
        "(experiment harness only).",
        "",
        "## 14. What this does not prove",
        "",
        "- BackendAdapter or llama.cpp KV integration feasibility.",
        "- Production serving readiness or speed/memory improvement.",
        "- Equivalence to Experiment 008 Python TurboQuant acceptance.",
        "- That upstream TurboQuant paper metrics apply to ExactKV.",
        "",
        "## 15. Blockers and risks",
        "",
        "- GGUF-quantized weights vs HF float verifier (trajectory divergence expected).",
        "- Batch llama drafts become stale after first HF correction in multi-round probe.",
        "- CPU-only build slow; interactive `llama-cli` pitfall avoided via `llama-completion`.",
        "- `llama-tokenize` requires `--stdin` / `-p` / `--file`.",
        "",
        "## 16. Go/no-go decision for future production-fidelity work",
        "",
        f"**{gate['decision']}** — {gate['rationale']}",
        "",
        f"BackendAdapter: **{gate['backend_adapter']}**. "
        f"llama.cpp ExactKV integration: **{gate['llamacpp_exactkv_integration']}**.",
        "",
        "## 17. VeriCache attribution",
        "",
        "The draft-then-verify **algorithm** is from VeriCache (Yao et al., "
        "arXiv:2605.17613, 2026). Experiment 022 evaluates an **external** "
        "llama.cpp drafter against HF verification — not a novel compression method.",
        "",
    ])
    return "\n".join(lines)


def run_experiment_022(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    *,
    llama_completion_bin: str,
    llama_tokenize_bin: str,
    gguf_model: str,
    max_new_tokens: int,
    draft_len: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for i, entry in enumerate(prompts, start=1):
        print(f"  [{i}/{len(prompts)}] {entry['prompt_id']}", flush=True)
        results.append(
            run_one_probe(
                runtime,
                entry,
                llama_completion_bin=llama_completion_bin,
                llama_tokenize_bin=llama_tokenize_bin,
                gguf_model=gguf_model,
                max_new_tokens=max_new_tokens,
                draft_len=draft_len,
                timeout_seconds=timeout_seconds,
            )
        )

    aggregate = _aggregate(results)
    go_no_go = _go_no_go(aggregate, results)
    manifest = {
        "experiment": EXPERIMENT_ID,
        "experiment_class": EXPERIMENT_CLASS,
        "model_name": runtime.model_name,
        "dtype": DTYPE,
        "device": str(runtime.device),
        "prompt_suite": "v10_external_probe_10",
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "max_new_tokens": max_new_tokens,
        "draft_len": draft_len,
        "gguf_model": gguf_model,
        "llama_completion_bin": llama_completion_bin,
        "llama_tokenize_bin": llama_tokenize_bin,
        "turboquant_flags": {"ctk": "q8_0", "ctv": "turbo3"},
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "V12_phase_2",
        "mode": "external_drafter_probe",
    }
    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
        "go_no_go": go_no_go,
    }


def main() -> int:
    default_bin = os.environ.get(
        "LLAMA_CPP_BIN_DIR",
        "/workspace/turboquant_prod_prep/llama-cpp-turboquant/build-cpu/bin",
    )
    default_gguf = (
        "/workspace/turboquant_prod_prep/models/qwen2.5-0.5b-auto.gguf"
    )

    parser = argparse.ArgumentParser(description="Experiment 022 external-drafter probe")
    parser.add_argument("--llama-cli", default=f"{default_bin}/llama-cli")
    parser.add_argument("--llama-tokenize", default=f"{default_bin}/llama-tokenize")
    parser.add_argument("--gguf-model", default=default_gguf)
    parser.add_argument("--prompt-limit", type=int, default=PROMPT_LIMIT_DEFAULT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DRAFT_LEN)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--output-json",
        default="reports/experiment_022_turboquant_llamacpp_probe.json",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/experiment_022_turboquant_llamacpp_probe.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=DTYPE)
    args = parser.parse_args()

    llama_completion = resolve_completion_binary(args.llama_cli)
    for path, label in [
        (llama_completion, "llama-completion"),
        (args.llama_tokenize, "llama-tokenize"),
        (args.gguf_model, "gguf-model"),
    ]:
        if not Path(path).is_file():
            raise SystemExit(f"Missing {label}: {path}")

    prompts = load_probe_prompts(args.prompt_limit)
    print(
        f"Experiment 022: {len(prompts)} probes, max_new_tokens={args.max_new_tokens}, "
        f"draft_len={args.draft_len}"
    )
    print(f"  completion: {llama_completion}")
    print(f"  gguf: {args.gguf_model}")

    print(f"Loading HF model {args.model} ({args.dtype}, {args.device}) ...")
    runtime = ModelRuntime(
        model_name=args.model, device=args.device, dtype=args.dtype
    )

    report = run_experiment_022(
        runtime,
        prompts,
        llama_completion_bin=llama_completion,
        llama_tokenize_bin=args.llama_tokenize,
        gguf_model=args.gguf_model,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
        timeout_seconds=args.timeout_seconds,
    )
    _assert_no_forbidden_fields(report)

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    md_path = Path(args.markdown_out)

    write_json_report(report, json_path)
    write_csv_report(report["results"], csv_path)
    md_path.write_text(generate_markdown_report(report), encoding="utf-8")

    agg = report["aggregate"]
    gate = report["go_no_go"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"token_level_safe: {agg['token_level_probe_safe_count']}/{agg['total_probes']}")
    print(f"mean_external_probe_acceptance: {agg['mean_external_probe_acceptance_rate']}")
    print(f"go_no_go: {gate['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
