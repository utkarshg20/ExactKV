"""Shard external-drafter feasibility helpers (Experiment 038).

Shard is used only as an external draft source compared against a full-KV HF
verifier. No default registry entry. No vendored Shard code.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CLAIMS_ALLOWED = [
    "Shard may be probed as an external Llama-only lossy drafter under Mode B.",
    "Token-level acceptance and first-divergence may be reported when alignment passes.",
    "Feasibility classification: restricted external-drafter probe only.",
    "Full-KV HF greedy path remains the authoritative verifier.",
]

CLAIMS_FORBIDDEN = [
    "Shard is not integrated as a default ExactKV compressor.",
    "External Shard benchmark or README throughput/memory numbers are not ExactKV results.",
    "No speedup, active memory savings, production serving, or model accuracy improvement claim.",
    "No Qwen direct KVCompressor claim.",
    "No fake token alignment or fabricated ExactKV failures.",
]

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_DRAFT_LEN = 4

MINIMAL_PROBE_PROMPTS: list[dict[str, str]] = [
    {
        "prompt_id": "json_structured",
        "category": "structured_json",
        "prompt": (
            'Return valid JSON only: {"task": "summarize", "topic": "KV cache compression", '
            '"constraints": ["exactness", "no serving claims"]}'
        ),
    },
    {
        "prompt_id": "retrieval_copy",
        "category": "retrieval_copy",
        "prompt": (
            "Document: Shard compresses Llama KV caches with PCA on keys and VQ on values. "
            "Question: What does Shard compress? Answer in one sentence."
        ),
    },
    {
        "prompt_id": "long_context_summary",
        "category": "long_context",
        "prompt": (
            "Background: ExactKV verifies lossy KV draft tokens against a full-KV greedy "
            "reference and reports acceptance, first divergence, and correction need. "
            "This probe does not integrate Shard into the default compressor registry. "
            "Summarize the verifier role in one sentence."
        ),
    },
    {
        "prompt_id": "code_structured",
        "category": "code",
        "prompt": (
            "Write a Python function `accepted_prefix(a: list[int], b: list[int]) -> int` "
            "that returns the length of the matching prefix."
        ),
    },
]

REQUIRED_REPORT_KEYS = frozenset({
    "probe_status",
    "blocked_reason",
    "shard_repo_path_present",
    "shard_import_success",
    "model_used",
    "tokenizer_alignment_pass",
    "prompt_count",
    "exactkv_failures",
    "accepted_prefix_lengths",
    "first_divergence_indices",
    "notes",
    "claims_allowed",
    "claims_forbidden",
})

VALID_PROBE_STATUSES = frozenset({"pass", "blocked", "restricted_no_go"})


@dataclass(frozen=True)
class ShardImportResult:
    success: bool
    reason: str
    repo_path: Path | None
    cache_cls: Any | None
    enable_llama_fused_attention: Callable[..., Any] | None


def resolve_shard_repo_path() -> Path | None:
    import os

    raw = os.environ.get("SHARD_REPO_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def try_import_shard(repo_path: Path) -> ShardImportResult:
    if not repo_path.is_dir():
        return ShardImportResult(
            success=False,
            reason=f"SHARD_REPO_PATH is not a directory: {repo_path}",
            repo_path=repo_path,
            cache_cls=None,
            enable_llama_fused_attention=None,
        )

    candidates = [repo_path / "src", repo_path]
    added: list[str] = []
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            added.append(str(candidate))

    try:
        shard_mod = importlib.import_module("shard")
        cache_cls = getattr(shard_mod, "Cache", None)
        enable_fn = getattr(shard_mod, "enable_llama_fused_attention", None)
        if cache_cls is None or enable_fn is None:
            return ShardImportResult(
                success=False,
                reason="shard module imported but Cache or enable_llama_fused_attention missing",
                repo_path=repo_path,
                cache_cls=None,
                enable_llama_fused_attention=None,
            )
        return ShardImportResult(
            success=True,
            reason="",
            repo_path=repo_path,
            cache_cls=cache_cls,
            enable_llama_fused_attention=enable_fn,
        )
    except ImportError as exc:
        return ShardImportResult(
            success=False,
            reason=f"shard import failed: {exc}",
            repo_path=repo_path,
            cache_cls=None,
            enable_llama_fused_attention=None,
        )
    finally:
        for path in added:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def prompt_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def check_tokenizer_alignment(
    tokenizer_a: Any,
    tokenizer_b: Any,
    prompt: str,
    *,
    generated_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Verify two tokenizers agree on prompt IDs and decode round-trip when provided."""
    ids_a = prompt_token_ids(tokenizer_a, prompt)
    ids_b = prompt_token_ids(tokenizer_b, prompt)
    prompt_aligned = ids_a == ids_b

    decode_ok = True
    decode_note = ""
    if generated_ids is not None:
        try:
            text = tokenizer_a.decode(generated_ids, skip_special_tokens=False)
            roundtrip = prompt_token_ids(tokenizer_a, text)
            if roundtrip != generated_ids:
                decode_ok = False
                decode_note = "decode(encode(text)) != generated_ids"
        except Exception as exc:  # noqa: BLE001 — probe reports alignment failures
            decode_ok = False
            decode_note = f"decode round-trip failed: {exc}"

    return {
        "prompt_token_ids_a": ids_a,
        "prompt_token_ids_b": ids_b,
        "prompt_aligned": prompt_aligned,
        "decode_roundtrip_ok": decode_ok,
        "decode_note": decode_note,
        "alignment_pass": prompt_aligned and decode_ok,
    }


def prompt_ids_comparable(
    hf_ids: list[int],
    shard_ids: list[int],
    tokenizer: Any,
) -> bool:
    """True when prompt IDs match modulo a leading BOS on one path only."""
    if hf_ids == shard_ids:
        return True
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return False
    if len(shard_ids) == len(hf_ids) + 1 and shard_ids[0] == bos and shard_ids[1:] == hf_ids:
        return True
    if len(hf_ids) == len(shard_ids) + 1 and hf_ids[0] == bos and hf_ids[1:] == shard_ids:
        return True
    return False


def compare_token_sequences(
    verifier_ids: list[int],
    draft_ids: list[int],
) -> dict[str, Any]:
    """Compare greedy verifier tokens to external Shard draft tokens."""
    compare_len = min(len(verifier_ids), len(draft_ids))
    first_div: int | None = None
    for i in range(compare_len):
        if verifier_ids[i] != draft_ids[i]:
            first_div = i
            break

    if first_div is None:
        accepted_prefix = compare_len
        exact_match = len(verifier_ids) == len(draft_ids)
    else:
        accepted_prefix = first_div
        exact_match = False

    draft_tok = (
        draft_ids[first_div]
        if first_div is not None and first_div < len(draft_ids)
        else None
    )
    verifier_tok = (
        verifier_ids[first_div]
        if first_div is not None and first_div < len(verifier_ids)
        else None
    )
    correction_needed = not exact_match

    return {
        "accepted_prefix_length": accepted_prefix,
        "first_divergence_index": first_div,
        "draft_token_id": draft_tok,
        "verifier_token_id": verifier_tok,
        "exact_match": exact_match,
        "correction_needed": correction_needed,
        "verifier_len": len(verifier_ids),
        "draft_len": len(draft_ids),
    }


def blocked_report(
    *,
    reason: str,
    shard_repo_path_present: bool,
    shard_import_success: bool = False,
    model_used: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return build_report(
        probe_status="blocked",
        blocked_reason=reason,
        shard_repo_path_present=shard_repo_path_present,
        shard_import_success=shard_import_success,
        model_used=model_used,
        tokenizer_alignment_pass=False,
        prompt_count=0,
        exactkv_failures=None,
        accepted_prefix_lengths=[],
        first_divergence_indices=[],
        prompt_results=[],
        notes=notes or [reason],
    )


def restricted_no_go_report(
    *,
    reason: str,
    shard_repo_path_present: bool,
    shard_import_success: bool,
    model_used: str | None,
    prompt_results: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    results = prompt_results or []
    return build_report(
        probe_status="restricted_no_go",
        blocked_reason=reason,
        shard_repo_path_present=shard_repo_path_present,
        shard_import_success=shard_import_success,
        model_used=model_used,
        tokenizer_alignment_pass=False,
        prompt_count=len(results),
        exactkv_failures=None,
        accepted_prefix_lengths=[
            r.get("comparison", {}).get("accepted_prefix_length")
            for r in results
            if r.get("comparison")
        ],
        first_divergence_indices=[
            r.get("comparison", {}).get("first_divergence_index")
            for r in results
            if r.get("comparison")
        ],
        prompt_results=results,
        notes=notes or [reason],
    )


def build_report(
    *,
    probe_status: str,
    blocked_reason: str,
    shard_repo_path_present: bool,
    shard_import_success: bool,
    model_used: str | None,
    tokenizer_alignment_pass: bool,
    prompt_count: int,
    exactkv_failures: int | None,
    accepted_prefix_lengths: list[int | None],
    first_divergence_indices: list[int | None],
    prompt_results: list[dict[str, Any]],
    notes: list[str],
    recommendation: str | None = None,
) -> dict[str, Any]:
    if probe_status not in VALID_PROBE_STATUSES:
        raise ValueError(f"invalid probe_status: {probe_status}")

    report = {
        "experiment_id": "038_shard_external_drafter_probe",
        "experiment_class": "shard_external_drafter_feasibility",
        "integration_mode": "mode_b_external_drafter",
        "not_default_registry": True,
        "not_kvcompressor_backend": True,
        "probe_status": probe_status,
        "blocked_reason": blocked_reason,
        "shard_repo_path_present": shard_repo_path_present,
        "shard_import_success": shard_import_success,
        "model_used": model_used,
        "tokenizer_alignment_pass": tokenizer_alignment_pass,
        "prompt_count": prompt_count,
        "exactkv_failures": exactkv_failures,
        "accepted_prefix_lengths": accepted_prefix_lengths,
        "first_divergence_indices": first_divergence_indices,
        "prompt_results": prompt_results,
        "notes": notes,
        "claims_allowed": list(CLAIMS_ALLOWED),
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": recommendation,
    }
    validate_report_shape(report)
    return report


def validate_report_shape(report: dict[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise ValueError(f"report missing keys: {sorted(missing)}")
    if report["probe_status"] not in VALID_PROBE_STATUSES:
        raise ValueError(f"invalid probe_status: {report['probe_status']}")
