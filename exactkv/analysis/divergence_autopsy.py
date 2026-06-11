"""Deep divergence autopsy helpers for Experiment 019 (V11 Phase 5).

Collects mechanistic signals at lossy-divergence and ExactKV rejection points
via auxiliary forward passes.  Does **not** modify generation or verification
logic.  No timing, throughput, latency, speedup, or active_gpu_kv_bytes fields.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import extract_kv_tensors
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state

FORBIDDEN_AUTOPSY_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

AUTOPSY_SUITES = (
    "long_context",
    "retrieval_copy",
    "tool_json",
    "code_structured",
    "core_v2",
)

REQUIRED_COMPRESSORS = (
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
)

KVQUANT_NAME = "kvquant_sim_qwen05b"

_BRACKET_OPEN = "([{"
_BRACKET_CLOSE = ")]}"
_QUOTE_CHARS = "\"'`"


@dataclass
class LogitObservation:
    """Top-k and margin at a single mismatch position."""

    gen_offset: int
    round_idx: int
    position_in_round: int
    verifier_top1: int
    drafter_top1: int
    verifier_top_k: list[int] = field(default_factory=list)
    drafter_top_k: list[int] = field(default_factory=list)
    logit_margin_verifier_top1_vs_drafter: float | None = None
    drafter_in_verifier_top_k: bool = False
    token_type: str = "wordpiece/other"
    event_kind: str = "exactkv_rejection"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_autopsy_artifact_safe(obj: Any, path: str = "artifact") -> None:
    """Raise if forbidden performance or active-GPU fields appear."""
    if isinstance(obj, dict):
        hits = FORBIDDEN_AUTOPSY_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            assert_autopsy_artifact_safe(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_autopsy_artifact_safe(item, f"{path}[{i}]")


def load_autopsy_prompt_subset(per_suite: int = 5) -> list[dict[str, Any]]:
    """Deterministic V10 subset: first ``per_suite`` ids per autopsy suite."""
    out: list[dict[str, Any]] = []
    for suite in AUTOPSY_SUITES:
        rows = load_v10_suite(suite)
        rows.sort(key=lambda r: r["prompt_id"])
        for row in rows[:per_suite]:
            entry = dict(row)
            entry["v10_panel"] = "exp019_autopsy"
            out.append(entry)
    return out


def kvquant_available() -> bool:
    path = os.environ.get("EXACTKV_KVQUANT_QUANTIZERS", "")
    if not path or not os.path.isfile(path):
        return False
    try:
        return importlib.util.find_spec("kvquant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def resolve_compressor(
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


def classify_token_text(text: str) -> str:
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


def token_type_at_id(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return "wordpiece/other"
    return classify_token_text(text)


def structured_output_state(text: str) -> dict[str, Any]:
    """Bracket depth, quote balance, and JSON-ish prefix heuristics."""
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    max_depth = 0
    depth = 0
    unmatched = False
    for ch in text:
        if ch in pairs:
            depth += 1
            max_depth = max(max_depth, depth)
            stack.append(pairs[ch])
        elif ch in pairs.values():
            depth = max(depth - 1, 0)
            if not stack or stack.pop() != ch:
                unmatched = True
    if stack:
        unmatched = True
    quote_count = sum(text.count(q) for q in _QUOTE_CHARS)
    quote_imbalance = (quote_count % 2) != 0
    stripped = text.lstrip()
    jsonish_prefix = stripped.startswith("{") or stripped.startswith("[")
    return {
        "bracket_depth": max_depth,
        "unmatched_brackets": unmatched,
        "quote_imbalance": quote_imbalance,
        "jsonish_prefix": jsonish_prefix,
        "malformed_json_prefix": jsonish_prefix and unmatched,
    }


def top_k_token_ids(logits: torch.Tensor, k: int = 5) -> list[int]:
    vec = logits.detach().float().squeeze()
    if vec.ndim != 1:
        vec = vec.view(-1)
    k = min(k, int(vec.numel()))
    if k <= 0:
        return []
    return vec.topk(k).indices.tolist()


def logit_margin(logits: torch.Tensor, token_a: int, token_b: int) -> float | None:
    vec = logits.detach().float().squeeze()
    if vec.ndim != 1:
        vec = vec.view(-1)
    n = int(vec.numel())
    if token_a < 0 or token_a >= n or token_b < 0 or token_b >= n:
        return None
    return float(vec[token_a].item() - vec[token_b].item())


@torch.no_grad()
def _commit_one_token(
    runtime: ModelRuntime,
    full_state: FullKVState,
    token_id: int,
) -> tuple[FullKVState, torch.Tensor]:
    """Commit one token to full state; return updated state and post-commit logits."""
    past_kv = full_state.past_key_values
    new_gen = list(full_state.generated_ids.squeeze(0).tolist())
    new_gen.append(token_id)
    if token_id == runtime.eos_token_id:
        next_id = runtime.eos_token_id
        logits = torch.zeros(runtime.vocab_size, device=runtime.device)
    else:
        tok = torch.tensor([[token_id]], dtype=torch.long, device=runtime.device)
        out = runtime.forward(tok, past_key_values=past_kv)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :].squeeze()
        next_id = int(logits.argmax(dim=-1).item())
    gen_tensor = torch.tensor([new_gen], dtype=torch.long, device=runtime.device)
    full_seq = torch.cat([full_state.prompt_ids, gen_tensor], dim=1)
    new_state = FullKVState(
        past_key_values=past_kv,
        prompt_ids=full_state.prompt_ids,
        generated_ids=gen_tensor,
        full_sequence_ids=full_seq,
        device=full_state.device,
        dtype=full_state.dtype,
        metadata={"next_token_id": next_id},
    )
    return new_state, logits


@torch.no_grad()
def logits_pair_at_gen_prefix(
    runtime: ModelRuntime,
    compressor: Any,
    prompt: str,
    gen_prefix: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verifier and drafter logits predicting the token at ``len(gen_prefix)``."""
    full_state = prefill_to_full_state(runtime, prompt)
    compressed = compressor.compress(full_state)

    prompt_ids = runtime.encode(prompt)
    pf_out = runtime.forward(prompt_ids)
    full_kv = pf_out.past_key_values
    verifier_logits = pf_out.logits[:, -1, :].squeeze()

    draft_kv = copy.deepcopy(compressor.materialize_for_draft(compressed))
    drafter_logits = verifier_logits.clone()

    for tid in gen_prefix:
        tok = torch.tensor([[tid]], dtype=torch.long, device=runtime.device)
        v_step = runtime.forward(tok, past_key_values=full_kv)
        full_kv = v_step.past_key_values
        verifier_logits = v_step.logits[:, -1, :].squeeze()

        d_step = runtime.forward(tok, past_key_values=draft_kv)
        draft_kv = d_step.past_key_values
        drafter_logits = d_step.logits[:, -1, :].squeeze()

    return verifier_logits, drafter_logits


@torch.no_grad()
def compute_kv_layer_errors(
    runtime: ModelRuntime,
    compressor: Any,
    prompt: str,
) -> dict[str, Any]:
    """Cosine / relative error between full and materialized compressed KV after prefill."""
    full_state = prefill_to_full_state(runtime, prompt)
    compressed = compressor.compress(full_state)
    mat = compressor.materialize_for_draft(compressed)

    full_k, full_v, _ = extract_kv_tensors(full_state.past_key_values)
    comp_k, comp_v, _ = extract_kv_tensors(mat)
    num_layers = len(full_k)

    layer_rows: list[dict[str, Any]] = []
    k_cosines: list[float] = []
    v_cosines: list[float] = []
    k_rel: list[float] = []
    v_rel: list[float] = []

    for layer in range(num_layers):
        fk = full_k[layer].float().reshape(-1)
        fv = full_v[layer].float().reshape(-1)
        ck = comp_k[layer].float().reshape(-1)
        cv = comp_v[layer].float().reshape(-1)
        k_cos = float(torch.nn.functional.cosine_similarity(fk, ck, dim=0).item())
        v_cos = float(torch.nn.functional.cosine_similarity(fv, cv, dim=0).item())
        k_r = float((fk - ck).norm() / (fk.norm() + 1e-8))
        v_r = float((fv - cv).norm() / (fv.norm() + 1e-8))
        k_cosines.append(k_cos)
        v_cosines.append(v_cos)
        k_rel.append(k_r)
        v_rel.append(v_r)
        layer_rows.append({
            "layer": layer,
            "k_cosine": k_cos,
            "v_cosine": v_cos,
            "k_relative_l2": k_r,
            "v_relative_l2": v_r,
            "is_boundary": layer in {0, num_layers - 1},
        })

    return {
        "num_layers": num_layers,
        "mean_k_cosine": sum(k_cosines) / max(len(k_cosines), 1),
        "mean_v_cosine": sum(v_cosines) / max(len(v_cosines), 1),
        "mean_k_relative_l2": sum(k_rel) / max(len(k_rel), 1),
        "mean_v_relative_l2": sum(v_rel) / max(len(v_rel), 1),
        "boundary_mean_v_cosine": (
            (layer_rows[0]["v_cosine"] + layer_rows[-1]["v_cosine"]) / 2
            if layer_rows
            else None
        ),
        "interior_mean_v_cosine": (
            sum(r["v_cosine"] for r in layer_rows[1:-1]) / max(len(layer_rows) - 2, 1)
            if len(layer_rows) > 2
            else None
        ),
        "per_layer": layer_rows,
    }


@torch.no_grad()
def try_attention_prefill_snapshot(
    runtime: ModelRuntime,
    prompt: str,
) -> dict[str, Any]:
    """Optional attention on prefill only; honest deferral if unavailable."""
    prompt_ids = runtime.encode(prompt)
    try:
        out = runtime.forward(prompt_ids, output_attentions=True)
    except TypeError:
        return {
            "available": False,
            "reason": "model.forward does not accept output_attentions",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    attentions = getattr(out, "attentions", None)
    if attentions is None:
        return {
            "available": False,
            "reason": (
                "forward returned no attentions (sdpa backend does not support "
                "output_attentions=True on this model)"
            ),
        }

    try:
        layers = len(attentions)
        last = attentions[-1]
        shape = tuple(last.shape)
        mean_attn = float(last.detach().float().mean().item())
    except Exception as exc:
        return {"available": False, "reason": f"attention tensor parse failed: {exc}"}

    return {
        "available": True,
        "num_layers": layers,
        "last_layer_shape": shape,
        "last_layer_mean": mean_attn,
        "note": "Prefill-only snapshot; not used for acceptance decisions.",
    }


def _build_logit_observation(
    runtime: ModelRuntime,
    compressor: Any,
    prompt: str,
    gen_prefix: list[int],
    *,
    round_idx: int,
    position_in_round: int,
    event_kind: str,
    top_k: int = 5,
) -> LogitObservation:
    verifier_logits, drafter_logits = logits_pair_at_gen_prefix(
        runtime, compressor, prompt, gen_prefix
    )
    v_top1 = int(verifier_logits.argmax(dim=-1).item())
    d_top1 = int(drafter_logits.argmax(dim=-1).item())
    v_topk = top_k_token_ids(verifier_logits, top_k)
    margin = logit_margin(verifier_logits, v_top1, d_top1)
    d_tok = d_top1
    if position_in_round >= 0:
        # At rejection, drafter token is the proposed draft token.
        pass
    return LogitObservation(
        gen_offset=len(gen_prefix),
        round_idx=round_idx,
        position_in_round=position_in_round,
        verifier_top1=v_top1,
        drafter_top1=d_tok,
        verifier_top_k=v_topk,
        drafter_top_k=top_k_token_ids(drafter_logits, top_k),
        logit_margin_verifier_top1_vs_drafter=margin,
        drafter_in_verifier_top_k=d_tok in v_topk,
        token_type=token_type_at_id(runtime.tokenizer, d_tok),
        event_kind=event_kind,
    )


def collect_rejection_observations(
    runtime: ModelRuntime,
    compressor: Any,
    prompt: str,
    prompt_len: int,
    ekv_result: Any,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Logit observations at each ExactKV rejection in traces."""
    committed: list[int] = []
    observations: list[dict[str, Any]] = []

    for trace in ekv_result.traces:
        acc = trace.acceptance
        if acc.num_rejected <= 0 and acc.correction_token is None:
            committed.extend(acc.accepted_tokens)
            continue

        round_start_gen = trace.full_seq_len_before - prompt_len
        mismatch_pos = acc.num_accepted
        prefix_at_mismatch = list(committed) + list(trace.draft_tokens[:mismatch_pos])
        draft_tok = (
            trace.draft_tokens[mismatch_pos]
            if mismatch_pos < len(trace.draft_tokens)
            else acc.correction_token
        )

        obs = _build_logit_observation(
            runtime,
            compressor,
            prompt,
            prefix_at_mismatch,
            round_idx=trace.round_idx,
            position_in_round=mismatch_pos,
            event_kind="exactkv_rejection",
            top_k=top_k,
        )
        if draft_tok is not None:
            obs.drafter_top1 = int(draft_tok)
            obs.token_type = token_type_at_id(runtime.tokenizer, int(draft_tok))
            v_logits, _ = logits_pair_at_gen_prefix(
                runtime, compressor, prompt, prefix_at_mismatch
            )
            obs.logit_margin_verifier_top1_vs_drafter = logit_margin(
                v_logits, obs.verifier_top1, int(draft_tok)
            )
            obs.drafter_in_verifier_top_k = int(draft_tok) in obs.verifier_top_k

        row = obs.to_dict()
        row["rejected_token_position"] = round_start_gen + mismatch_pos
        row["correction_token_position"] = (
            round_start_gen + mismatch_pos
            if acc.correction_token is not None
            else None
        )
        row["correction_token"] = acc.correction_token
        observations.append(row)

        committed.extend(acc.accepted_tokens)
        if acc.correction_token is not None:
            committed.append(acc.correction_token)

    return observations


def collect_lossy_divergence_observation(
    runtime: ModelRuntime,
    compressor: Any,
    prompt: str,
    prompt_len: int,
    full_ids: list[int],
    lossy_ids: list[int],
    *,
    top_k: int = 5,
) -> dict[str, Any] | None:
    """Logit observation at first lossy-vs-full divergence index."""
    div_idx = first_divergence_idx(
        torch.tensor([full_ids]),
        torch.tensor([lossy_ids]),
    )
    if div_idx is None:
        return None

    gen_prefix = full_ids[:div_idx] if div_idx < len(full_ids) else full_ids
    obs = _build_logit_observation(
        runtime,
        compressor,
        prompt,
        gen_prefix,
        round_idx=-1,
        position_in_round=div_idx,
        event_kind="lossy_first_divergence",
        top_k=top_k,
    )
    if div_idx < len(lossy_ids):
        obs.drafter_top1 = int(lossy_ids[div_idx])
        obs.token_type = token_type_at_id(runtime.tokenizer, int(lossy_ids[div_idx]))
        v_logits, _ = logits_pair_at_gen_prefix(runtime, compressor, prompt, gen_prefix)
        obs.logit_margin_verifier_top1_vs_drafter = logit_margin(
            v_logits, obs.verifier_top1, int(lossy_ids[div_idx])
        )
        obs.drafter_in_verifier_top_k = int(lossy_ids[div_idx]) in obs.verifier_top_k

    row = obs.to_dict()
    row["first_divergence_idx"] = div_idx
    row["rejected_token_position"] = div_idx
    row["correction_token_position"] = None
    row["correction_token"] = full_ids[div_idx] if div_idx < len(full_ids) else None
    return row


@torch.no_grad()
def run_autopsy_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    compressor: Any,
    *,
    draft_len: int,
    max_new_tokens: int,
    collect_kv_errors: bool = True,
    collect_attention: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run one autopsy cell: ExactKV + forensics without changing core logic."""
    prompt = prompt_entry["prompt"]
    compressor_name = getattr(compressor, "name", "unknown")

    full_res = generate_full_greedy(runtime, prompt, max_new_tokens)
    full_ids = full_res.generated_ids.squeeze(0).tolist()
    prompt_len = full_res.prompt_ids.shape[-1]

    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()
    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    ekv_res = ExactKVGenerator(
        runtime, compressor, draft_len=draft_len
    ).generate(prompt, max_new_tokens)
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)

    primary_cat = prompt_entry.get(
        "v10_primary_category", prompt_entry.get("category", "")
    )

    rejection_obs = collect_rejection_observations(
        runtime,
        compressor,
        prompt,
        prompt_len,
        ekv_res,
        top_k=top_k,
    )
    for row in rejection_obs:
        row["v10_suite"] = prompt_entry.get("v10_suite", "")
        row["v10_primary_category"] = primary_cat

    struct_state = structured_output_state(lossy_res.output_text)
    if prompt_entry.get("v10_suite") in ("tool_json", "code_structured"):
        struct_state["structured_suite"] = True

    lossy_obs = None
    if not lossy_exact:
        lossy_obs = collect_lossy_divergence_observation(
            runtime,
            compressor,
            prompt,
            prompt_len,
            full_ids,
            lossy_ids,
            top_k=top_k,
        )

    kv_errors = None
    if collect_kv_errors and compressor_name != "noop":
        kv_errors = compute_kv_layer_errors(runtime, compressor, prompt)

    attention = None
    if collect_attention:
        attention = try_attention_prefill_snapshot(runtime, prompt)

    div_token_type = None
    if lossy_div is not None and lossy_div < len(full_ids):
        div_token_type = token_type_at_id(runtime.tokenizer, full_ids[lossy_div])

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "category": prompt_entry.get("category", "unknown"),
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "v10_primary_category": primary_cat,
        "model_name": runtime.model_name,
        "compressor_name": compressor_name,
        "draft_len": draft_len,
        "max_new_tokens": max_new_tokens,
        "exactkv_failure": not ekv_exact,
        "lossy": {
            "token_exact_match": lossy_exact,
            "first_divergence_idx": lossy_div,
            "divergence_token_type": div_token_type,
        },
        "exactkv": {
            "token_exact_match": ekv_exact,
            "acceptance": acceptance.to_dict(),
        },
        "autopsy": {
            "rejection_observations": rejection_obs,
            "lossy_divergence_observation": lossy_obs,
            "structured_output_state": struct_state,
            "kv_layer_errors": kv_errors,
            "attention_prefill": attention,
        },
    }


def aggregate_autopsy_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize autopsy cells for report generation."""
    total = len(results)
    failures = sum(1 for r in results if r.get("exactkv_failure"))
    lossy_div_cells = sum(
        1 for r in results if not r["lossy"]["token_exact_match"]
    )

    rejection_events: list[dict[str, Any]] = []
    lossy_events: list[dict[str, Any]] = []
    for r in results:
        rejection_events.extend(r["autopsy"]["rejection_observations"])
        lo = r["autopsy"]["lossy_divergence_observation"]
        if lo:
            lo = dict(lo)
            lo["compressor_name"] = r["compressor_name"]
            lo["draft_len"] = r["draft_len"]
            lo["v10_suite"] = r.get("v10_suite", "")
            lo["v10_primary_category"] = r.get("v10_primary_category", "")
            lossy_events.append(lo)

    def _group_counter(
        events: list[dict[str, Any]],
        key: str,
    ) -> dict[str, int]:
        return dict(Counter(str(e.get(key, "")) for e in events if e.get(key)))

    margins = [
        e["logit_margin_verifier_top1_vs_drafter"]
        for e in rejection_events + lossy_events
        if e.get("logit_margin_verifier_top1_vs_drafter") is not None
    ]
    low_margin = sum(1 for m in margins if m < 1.0)
    drafter_in_topk = sum(
        1 for e in rejection_events + lossy_events if e.get("drafter_in_verifier_top_k")
    )

    first_div_by_suite: Counter[str] = Counter()
    first_div_by_token: Counter[str] = Counter()
    for r in results:
        if r["lossy"]["first_divergence_idx"] is not None:
            first_div_by_suite[r.get("v10_suite", "")] += 1
            tt = r["lossy"].get("divergence_token_type")
            if tt:
                first_div_by_token[tt] += 1

    by_compressor: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cells": 0, "lossy_div": 0, "rejections": 0, "mean_accept": []}
    )
    for r in results:
        c = r["compressor_name"]
        by_compressor[c]["cells"] += 1
        if not r["lossy"]["token_exact_match"]:
            by_compressor[c]["lossy_div"] += 1
        by_compressor[c]["rejections"] += len(r["autopsy"]["rejection_observations"])
        by_compressor[c]["mean_accept"].append(
            r["exactkv"]["acceptance"]["acceptance_rate"]
        )
    for stats in by_compressor.values():
        accs = stats.pop("mean_accept")
        stats["mean_acceptance_rate"] = sum(accs) / max(len(accs), 1)

    draft_len_compare: dict[int, dict[str, float]] = {}
    for dl in sorted({r["draft_len"] for r in results}):
        sub = [r for r in results if r["draft_len"] == dl]
        draft_len_compare[dl] = {
            "mean_acceptance": sum(
                r["exactkv"]["acceptance"]["acceptance_rate"] for r in sub
            )
            / max(len(sub), 1),
            "lossy_div_cells": sum(
                1 for r in sub if not r["lossy"]["token_exact_match"]
            ),
            "rejection_events": sum(
                len(r["autopsy"]["rejection_observations"]) for r in sub
            ),
        }

    boundary_vs_k8: list[dict[str, Any]] = []
    b4 = [r for r in results if r["compressor_name"] == "k8_v4_boundary4_v8_sim"]
    k8 = [r for r in results if r["compressor_name"] == "k8_v4_sim"]
    b4_lookup = {(r["prompt_id"], r["draft_len"]): r for r in b4}
    for r in k8:
        key = (r["prompt_id"], r["draft_len"])
        if key not in b4_lookup:
            continue
        other = b4_lookup[key]
        boundary_vs_k8.append({
            "prompt_id": r["prompt_id"],
            "draft_len": r["draft_len"],
            "k8_rejections": len(r["autopsy"]["rejection_observations"]),
            "boundary4_rejections": len(other["autopsy"]["rejection_observations"]),
            "k8_first_div": r["lossy"]["first_divergence_idx"],
            "boundary4_first_div": other["lossy"]["first_divergence_idx"],
        })

    kv_summaries: list[dict[str, Any]] = []
    for r in results:
        kv = r["autopsy"].get("kv_layer_errors")
        if kv:
            kv_summaries.append({
                "compressor_name": r["compressor_name"],
                "prompt_id": r["prompt_id"],
                "mean_k_cosine": kv["mean_k_cosine"],
                "mean_v_cosine": kv["mean_v_cosine"],
                "boundary_mean_v_cosine": kv.get("boundary_mean_v_cosine"),
                "interior_mean_v_cosine": kv.get("interior_mean_v_cosine"),
            })

    attention_logged = any(
        (r["autopsy"].get("attention_prefill") or {}).get("available")
        for r in results
    )

    return {
        "total_cells": total,
        "exactkv_failures": failures,
        "lossy_divergence_cells": lossy_div_cells,
        "total_rejection_events": len(rejection_events),
        "total_lossy_divergence_events": len(lossy_events),
        "rejection_token_types": _group_counter(rejection_events, "token_type"),
        "lossy_divergence_token_types": _group_counter(lossy_events, "token_type"),
        "rejection_by_suite": _group_counter(rejection_events, "v10_suite"),
        "first_divergence_by_suite": dict(first_div_by_suite),
        "first_divergence_by_token_type": dict(first_div_by_token),
        "logit_margin_count": len(margins),
        "logit_margin_mean": sum(margins) / max(len(margins), 1) if margins else None,
        "logit_margin_low_count": low_margin,
        "drafter_in_verifier_top_k_count": drafter_in_topk,
        "by_compressor": dict(by_compressor),
        "by_draft_len": draft_len_compare,
        "boundary4_vs_k8_v4_sim": boundary_vs_k8,
        "kv_layer_error_summaries": kv_summaries,
        "attention_weights_logged": attention_logged,
        "repair_hypotheses": build_repair_hypotheses(
            results,
            rejection_events,
            lossy_events,
            draft_len_compare,
            kv_summaries,
        ),
    }


def build_repair_hypotheses(
    results: list[dict[str, Any]],
    rejection_events: list[dict[str, Any]],
    lossy_events: list[dict[str, Any]],
    draft_len_compare: dict[int, dict[str, float]],
    kv_summaries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Propose repair policies — hypotheses only, not implemented."""
    hypotheses: list[dict[str, str]] = []

    lc_rc_rejections = sum(
        1
        for e in rejection_events
        if e.get("v10_suite") in ("long_context", "retrieval_copy")
    )
    if lc_rc_rejections > len(rejection_events) * 0.25:
        hypotheses.append({
            "policy": "dynamic_fallback_int8",
            "rationale": (
                f"long_context/retrieval_copy account for {lc_rc_rejections} of "
                f"{len(rejection_events)} rejection events — int8 may be safer on copy-heavy prompts"
            ),
            "status": "hypothesis_only",
        })

    struct_rejections = sum(
        1
        for e in rejection_events
        if e.get("token_type") in ("bracket", "quote", "punctuation")
    )
    if struct_rejections > 0:
        hypotheses.append({
            "policy": "structured_output_safe_mode",
            "rationale": (
                f"{struct_rejections} rejections involve bracket/quote/punctuation tokens"
            ),
            "status": "hypothesis_only",
        })

    if draft_len_compare:
        dl8 = draft_len_compare.get(8)
        dl4 = draft_len_compare.get(4)
        if dl8 and dl4 and dl8.get("rejection_events", 0) > dl4.get("rejection_events", 0):
            hypotheses.append({
                "policy": "lower_draft_len_on_low_margin",
                "rationale": (
                    f"draft_len=8 has {dl8['rejection_events']} rejection events vs "
                    f"{dl4.get('rejection_events', 0)} at draft_len=4"
                ),
                "status": "hypothesis_only",
            })

    margins = [
        e.get("logit_margin_verifier_top1_vs_drafter")
        for e in rejection_events + lossy_events
        if e.get("logit_margin_verifier_top1_vs_drafter") is not None
    ]
    if margins and sum(1 for m in margins if m < 1.0) > len(margins) * 0.3:
        hypotheses.append({
            "policy": "confidence_gated_acceptance",
            "rationale": (
                "Substantial share of divergences occur at low verifier margins "
                "(<1.0 logit gap) — future policy could gate draft acceptance on margin"
            ),
            "status": "hypothesis_only",
        })

    if kv_summaries:
        k_err = sum(s["mean_k_cosine"] for s in kv_summaries) / len(kv_summaries)
        v_err = sum(s["mean_v_cosine"] for s in kv_summaries) / len(kv_summaries)
        if k_err < v_err:
            hypotheses.append({
                "policy": "stricter_k_preservation_retrieval",
                "rationale": (
                    f"Mean K cosine ({k_err:.4f}) < V cosine ({v_err:.4f}) after prefill — "
                    "retrieval/copy may need stricter K preservation"
                ),
                "status": "hypothesis_only",
            })
        hypotheses.append({
            "policy": "boundary_depth_adaptation",
            "rationale": (
                "Layer-wise KV error varies by boundary vs interior — boundary4 policy "
                "may need per-category depth tuning"
            ),
            "status": "hypothesis_only",
        })

    kvquant_cells = [r for r in results if r["compressor_name"] == KVQUANT_NAME]
    if kvquant_cells:
        by_suite: dict[str, list[float]] = defaultdict(list)
        int8_by_suite: dict[str, list[float]] = defaultdict(list)
        for r in kvquant_cells:
            by_suite[r.get("v10_suite", "")].append(
                r["exactkv"]["acceptance"]["acceptance_rate"]
            )
        for r in results:
            if r["compressor_name"] == "int8":
                int8_by_suite[r.get("v10_suite", "")].append(
                    r["exactkv"]["acceptance"]["acceptance_rate"]
                )
        wins = [
            suite
            for suite, accs in by_suite.items()
            if accs
            and sum(accs) / len(accs)
            > sum(int8_by_suite.get(suite, [0])) / max(len(int8_by_suite.get(suite, [1])), 1)
        ]
        if wins:
            hypotheses.append({
                "policy": "kvquant_category_gating",
                "rationale": f"KVQuant mean accept exceeds int8 on suites: {', '.join(wins)}",
                "status": "hypothesis_only",
            })

    if not hypotheses:
        hypotheses.append({
            "policy": "continue_monitoring",
            "rationale": "No dominant correlation crossed heuristic thresholds in this pilot panel",
            "status": "hypothesis_only",
        })

    return hypotheses
