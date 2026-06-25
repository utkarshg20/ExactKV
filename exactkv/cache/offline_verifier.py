"""Offline verifier restore integration smoke (Phase 12C–12F).

Isolated draft/verify loop where the verifier reads **reloaded** full-KV payloads
from ``KVStorageBackend``. **Not** wired into ``ExactKVGenerator`` or default runtime.

This is an offline verifier restore smoke, not default runtime integration.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch

from exactkv.cache.dual_cache import CacheResidency
from exactkv.cache.full_state import FullKVState
from exactkv.cache.hf_kv_restore import (
    FORBIDDEN_CLAIMS,
    HfKvRestoreError,
    PrefillKVCapture,
    build_storage_payload_from_cache,
    capture_prefill_kv,
    detect_hf_cache_format,
    first_divergence_index,
    restore_cache_from_storage_payload,
    store_prefill_payload,
)
from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.storage import KVStorageBackend, KVStorageHandle
from exactkv.compressors.registry import get_compressor
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.acceptance import AcceptanceResult, compute_acceptance
from exactkv.verification.engine import VerificationEngine

EXPERIMENT_048_ID = "exp048_offline_verifier_restore_smoke"
EXPERIMENT_049_ID = "exp049_offline_verifier_lossy_draft"
EXPERIMENT_050_ID = "exp050_offline_restored_verifier_drift_stress"
EXPERIMENT_051_ID = "exp051_offline_verifier_cuda_drift_panel"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_MAX_NEW_TOKENS = 12
DEFAULT_DRAFT_LEN = 4
DEFAULT_DRIFT_MAX_NEW_TOKENS = 32
DEFAULT_DRIFT_DRAFT_LENS = (4, 8)
DRAFT_SOURCE_TYPE = "controlled_draft_with_injected_mismatch"
VERIFIER_SOURCE = "reloaded_full_kv"

OFFLINE_VERIFIER_CLAIM_NOTE = (
    "Offline verifier restore smoke (Phase 12C). Reloaded full-KV payloads used as "
    "verifier source in an isolated draft/verify loop only — not default runtime, "
    "vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

OFFLINE_LOSSY_CLAIM_NOTE = (
    "Offline verifier lossy-draft smoke (Phase 12D). Reloaded full-KV verifier with "
    "existing built-in lossy compressor draft logic in an isolated loop only — not "
    "default runtime, vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

OFFLINE_DRIFT_STRESS_CLAIM_NOTE = (
    "Offline restored-verifier drift stress (Phase 12E). Reloaded full-KV verifier with "
    "existing lossy compressor drafts on a drift-prone panel in an isolated loop only — "
    "not default runtime, vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

OFFLINE_CUDA_DRIFT_CLAIM_NOTE = (
    "Offline restored-verifier CUDA drift panel (Phase 12F). CUDA exactness check for "
    "reloaded full-KV verifier with lossy drafts in an isolated loop only — not default "
    "runtime, vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

DEFAULT_CUDA_DRIFT_PROMPT_IDS = (
    "drift_001",
    "drift_002",
    "drift_003",
    "drift_005",
    "drift_006",
    "drift_011",
)

DEFAULT_LOSSY_COMPRESSORS = ("int8", "int4_sim", "k8_v4_sim")
DEFAULT_DRIFT_COMPRESSORS = (
    "int4_sim",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "int8",
)
SEMANTIC_DRIFT_CATEGORIES = frozenset(
    {
        "pharmacy_semantic",
        "longbench_style",
        "retrieval_copy",
        "tool_call_json",
        "structured_json",
    }
)


@dataclass
class OfflineVerifierRoundTrace:
    """One draft/verify/commit round in the offline loop."""

    round_idx: int
    draft_tokens: list[int]
    verifier_tokens: list[int]
    accepted_prefix_length: int
    correction_token: int | None
    committed_tokens: list[int]
    all_matched: bool
    num_rejected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OfflineVerifierCellResult:
    """Per prompt×backend offline verifier smoke result."""

    prompt_id: str
    prompt: str
    backend_name: str
    cache_format: str
    draft_source_type: str
    verifier_source: str
    live_reference_token_ids: list[int]
    offline_output_token_ids: list[int]
    token_exact_match: bool
    exactkv_failures: int
    accepted_prefix_lengths: list[int]
    first_divergence_idx: int | None = None
    restore_blocker: str = ""
    verification_blocker: str = ""
    round_traces: list[OfflineVerifierRoundTrace] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.round_traces is not None:
            data["round_traces"] = [t.to_dict() for t in self.round_traces]
        return data


@dataclass
class OfflineLossyCellResult:
    """Per prompt×backend×compressor offline lossy-draft result."""

    prompt_id: str
    prompt: str
    backend_name: str
    compressor_name: str
    cache_format: str
    draft_source: str
    verifier_source: str
    live_reference_token_ids: list[int]
    offline_output_token_ids: list[int]
    token_exact_match: bool
    exactkv_failures: int
    accepted_prefix_lengths: list[int]
    mean_acceptance: float
    first_divergence_idx: int | None = None
    restore_blocker: str = ""
    draft_blocker: str = ""
    verification_blocker: str = ""
    round_traces: list[OfflineVerifierRoundTrace] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.round_traces is not None:
            data["round_traces"] = [t.to_dict() for t in self.round_traces]
        return data


@dataclass
class OfflineDriftStressCellResult:
    """Per prompt×backend×compressor×draft_len drift-stress cell result."""

    prompt_id: str
    prompt: str
    category: str
    backend_name: str
    compressor_name: str
    draft_len: int
    cache_format: str
    draft_source: str
    verifier_source: str
    live_reference_token_ids: list[int]
    offline_output_token_ids: list[int]
    token_exact_match: bool
    exactkv_failures: int
    accepted_prefix_lengths: list[int]
    mean_acceptance: float
    draft_divergence_count: int
    semantic_divergence_count: int
    first_divergence_idx: int | None = None
    restore_blocker: str = ""
    draft_blocker: str = ""
    verification_blocker: str = ""
    round_traces: list[OfflineVerifierRoundTrace] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.round_traces is not None:
            data["round_traces"] = [t.to_dict() for t in self.round_traces]
        return data


@dataclass
class CudaDtypeConfig:
    """CUDA dtype matrix entry for Phase 12F."""

    device: str
    dtype: str
    dtype_supported: bool
    status: str = "pending"  # pending | tested | skipped
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OfflineCudaDriftCellResult:
    """Per CUDA drift-panel cell with exactness metadata."""

    prompt_id: str
    prompt: str
    category: str
    backend_name: str
    compressor_name: str
    draft_len: int
    device: str
    dtype: str
    cache_format: str
    draft_source: str
    verifier_source: str
    live_reference_token_ids: list[int]
    offline_output_token_ids: list[int]
    token_exact_match: bool
    exactkv_failures: int
    accepted_prefix_lengths: list[int]
    mean_acceptance: float
    draft_divergence_count: int
    semantic_divergence_count: int
    first_divergence_idx: int | None = None
    restore_blocker: str = ""
    draft_blocker: str = ""
    verification_blocker: str = ""
    exactness_blocker: str = ""
    round_traces: list[OfflineVerifierRoundTrace] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.round_traces is not None:
            data["round_traces"] = [t.to_dict() for t in self.round_traces]
        return data


def default_offline_prompts() -> list[dict[str, str]]:
    """Deterministic 4–8 prompt panel for offline verifier smoke."""
    return [
        {"prompt_id": "offline_001", "prompt": "What is 2+2? Answer in one word."},
        {"prompt_id": "offline_002", "prompt": "Name the capital of France."},
        {"prompt_id": "offline_003", "prompt": 'Return JSON: {"color":'},
        {"prompt_id": "offline_004", "prompt": "def add(a, b):\n    return"},
        {
            "prompt_id": "offline_005",
            "prompt": "Passage: Water boils at 100C. State the boiling point:",
        },
        {"prompt_id": "offline_006", "prompt": '<tool_call>{"name": "lookup", "id":'},
    ]


def default_lossy_compressors() -> list[str]:
    """Built-in lossy compressors available for Phase 12D without registry changes."""
    available: list[str] = []
    for name in DEFAULT_LOSSY_COMPRESSORS:
        try:
            get_compressor(name)
            available.append(name)
        except ValueError:
            continue
    return available


def default_drift_stress_compressors() -> list[str]:
    """Drift-prone built-in compressors for Phase 12E (registry unchanged)."""
    available: list[str] = []
    for name in DEFAULT_DRIFT_COMPRESSORS:
        try:
            get_compressor(name)
            available.append(name)
        except ValueError:
            continue
    return available


def _drift_context_pad(text: str, repeats: int = 4) -> str:
    filler = (
        "Operational note: teams track migration blockers, renewal risk, "
        "follow-up owners, and checkpoint dates in weekly reviews. "
    )
    return filler * repeats + text


def default_drift_stress_prompts() -> list[dict[str, str]]:
    """Deterministic 8–16 prompt panel targeting lossy draft divergence."""
    return [
        {
            "prompt_id": "drift_001",
            "category": "pharmacy_semantic",
            "prompt": (
                'JSON tool call only: {"tool":"refill_prescription","drug":"ibuprofen",'
                '"quantity":30,"pickup":'
            ),
        },
        {
            "prompt_id": "drift_002",
            "category": "longbench_style",
            "prompt": _drift_context_pad(
                "Context document 1: Friday follow-up on SSO is assigned to Maya.\n"
                "Context document 2: Billing migration checkpoint remains open.\n"
                "Context document 3: Launch communications go to Priya.\n\n"
                "Use the context documents to answer exactly who owns the Friday follow-up.\n"
            ),
        },
        {
            "prompt_id": "drift_003",
            "category": "tool_call_json",
            "prompt": (
                "You are an ordering agent. Return only a JSON tool call.\n\n"
                "User wants:\n- vegan burger\n- no onions\n- quantity 1\n- pickup\n\n"
                'Return:\n{"tool":"add_item","item_id":'
            ),
        },
        {
            "prompt_id": "drift_004",
            "category": "structured_json",
            "prompt": 'Return JSON only: {"order_id":"A-1024","status":',
        },
        {
            "prompt_id": "drift_005",
            "category": "retrieval_copy",
            "prompt": _drift_context_pad(
                "Long-context retrieval task:\n"
                "Background: The operations team logs owner names in triage tickets.\n"
                "Copy exactly who owns the Friday follow-up according to the ticket: "
                "OWNER=Maya; TASK=Friday follow-up; STATUS=open\n"
            ),
        },
        {
            "prompt_id": "drift_006",
            "category": "code_like",
            "prompt": (
                "def validate_order(order):\n"
                "    if order['fulfillment'] == 'pickup':\n"
                "        return"
            ),
        },
        {
            "prompt_id": "drift_007",
            "category": "long_context_summary",
            "prompt": _drift_context_pad(
                "Weekly operations log:\n"
                "- SSO blocker persists for staging.\n"
                "- Renewal risk: medium.\n"
                "- Follow-up owner: Maya.\n"
                "- Billing migration incomplete; launch owner Priya.\n\n"
                "Summarize the operational status in 3 bullet points.\n"
            ),
        },
        {
            "prompt_id": "drift_008",
            "category": "tool_call_json",
            "prompt": (
                'Return JSON: {"action":"transfer","from_account":"checking",'
                '"to_account":"savings","amount":'
            ),
        },
        {
            "prompt_id": "drift_009",
            "category": "structured_json",
            "prompt": 'List as JSON array: ["red", "green", "blue",',
        },
        {
            "prompt_id": "drift_010",
            "category": "retrieval_copy",
            "prompt": (
                "Passage: The refund window for Pro annual customers is 30 days from purchase. "
                "Quote the window exactly:"
            ),
        },
        {
            "prompt_id": "drift_011",
            "category": "longbench_style",
            "prompt": _drift_context_pad(
                "Support policy excerpt:\n"
                "Pro annual customers may request a full refund within 30 days of purchase.\n"
                "After 30 days, only exchange credits apply.\n\n"
                "According to the policy, how many days after purchase can a Pro annual "
                "customer request a refund?\n"
            ),
        },
        {
            "prompt_id": "drift_012",
            "category": "code_like",
            "prompt": (
                'import json\n\npayload = {"tool":"search","query":'
            ),
        },
    ]


def default_cuda_drift_prompts(*, full_panel: bool = False) -> list[dict[str, str]]:
    """Reduced 6-prompt CUDA panel by default; optional full 12-prompt panel."""
    all_prompts = default_drift_stress_prompts()
    if full_panel:
        return all_prompts
    by_id = {p["prompt_id"]: p for p in all_prompts}
    return [by_id[pid] for pid in DEFAULT_CUDA_DRIFT_PROMPT_IDS if pid in by_id]


def resolve_cuda_drift_dtype_configs() -> list[CudaDtypeConfig]:
    """Return CUDA float16 and optional bfloat16 configs when hardware permits."""
    if not torch.cuda.is_available():
        return [
            CudaDtypeConfig(
                device="cuda",
                dtype="float16",
                dtype_supported=False,
                status="skipped",
                skip_reason="CUDA unavailable",
            ),
            CudaDtypeConfig(
                device="cuda",
                dtype="bfloat16",
                dtype_supported=False,
                status="skipped",
                skip_reason="CUDA unavailable",
            ),
        ]
    configs = [
        CudaDtypeConfig(
            device="cuda",
            dtype="float16",
            dtype_supported=True,
            status="pending",
        )
    ]
    if torch.cuda.is_bf16_supported():
        configs.append(
            CudaDtypeConfig(
                device="cuda",
                dtype="bfloat16",
                dtype_supported=True,
                status="pending",
            )
        )
    else:
        configs.append(
            CudaDtypeConfig(
                device="cuda",
                dtype="bfloat16",
                dtype_supported=False,
                status="skipped",
                skip_reason="bfloat16 not supported on this CUDA device",
            )
        )
    return configs


def configure_cuda_determinism() -> None:
    """Best-effort deterministic CUDA settings for greedy exactness checks."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_acceptance_from_traces(traces: list[OfflineVerifierRoundTrace]) -> float:
    """Compute mean acceptance rate across offline verifier rounds."""
    if not traces:
        return 1.0
    accepted = sum(t.accepted_prefix_length for t in traces)
    rejected = sum(t.num_rejected for t in traces)
    denom = accepted + rejected
    return accepted / denom if denom > 0 else 1.0


def draft_divergence_count_from_traces(traces: list[OfflineVerifierRoundTrace]) -> int:
    """Count verify rounds where lossy draft did not fully match verifier."""
    return sum(1 for trace in traces if not trace.all_matched)


def semantic_divergence_count_from_traces(
    traces: list[OfflineVerifierRoundTrace],
    *,
    category: str,
) -> int:
    """Count correction rounds on semantic-tagged prompts (honest classifier)."""
    if category not in SEMANTIC_DRIFT_CATEGORIES:
        return 0
    return sum(
        1
        for trace in traces
        if trace.correction_token is not None and not trace.all_matched
    )


def propose_controlled_draft(
    reference: list[int],
    position: int,
    draft_len: int,
    round_idx: int,
    vocab_size: int,
) -> list[int]:
    """Propose draft tokens from reference with optional injected mismatch.

    Controlled draft source for restore-verifier integration smoke — **not**
    compressor evaluation.
    """
    chunk = reference[position : position + draft_len]
    if not chunk:
        return []
    draft = list(chunk)
    if round_idx % 2 == 1 and len(draft) >= 2:
        wrong = (draft[1] + 1) % max(vocab_size, 1)
        if wrong == draft[1]:
            wrong = (draft[1] + 2) % max(vocab_size, 1)
        draft[1] = wrong
    return draft


def truncate_at_eos(tokens: list[int], eos_token_id: int) -> tuple[list[int], bool]:
    """Return tokens up to and including the first EOS (if any)."""
    result: list[int] = []
    for token in tokens:
        result.append(token)
        if token == eos_token_id:
            return result, True
    return result, False


def reconstruct_output_from_acceptances(
    acceptances: list[tuple[list[int], AcceptanceResult]],
    *,
    eos_token_id: int | None = None,
) -> list[int]:
    """Rebuild final output from per-round draft/acceptance pairs (unit-test helper)."""
    output: list[int] = []
    for _draft, acceptance in acceptances:
        committed = list(acceptance.accepted_tokens)
        if acceptance.correction_token is not None:
            committed.append(acceptance.correction_token)
        if eos_token_id is not None:
            committed, eos_found = truncate_at_eos(committed, eos_token_id)
            output.extend(committed)
            if eos_found:
                break
        else:
            output.extend(committed)
    return output


@torch.no_grad()
def commit_full_state(
    runtime: ModelRuntime,
    full_state: FullKVState,
    committed_tokens: list[int],
) -> FullKVState:
    """Advance authoritative full KV state after a commit (isolated copy)."""
    past_kv = full_state.past_key_values
    current_next_token_id = full_state.next_token_id
    new_gen_ids: list[int] = full_state.generated_ids.squeeze(0).tolist()

    for token_id in committed_tokens:
        new_gen_ids.append(token_id)
        if token_id == runtime.eos_token_id:
            current_next_token_id = runtime.eos_token_id
            break
        tok_tensor = torch.tensor(
            [[token_id]], dtype=torch.long, device=runtime.device
        )
        out = runtime.forward(tok_tensor, past_key_values=past_kv)
        past_kv = out.past_key_values
        current_next_token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())

    gen_tensor = torch.tensor([new_gen_ids], dtype=torch.long, device=runtime.device)
    full_seq = torch.cat([full_state.prompt_ids, gen_tensor], dim=1)
    return FullKVState(
        past_key_values=past_kv,
        prompt_ids=full_state.prompt_ids,
        generated_ids=gen_tensor,
        full_sequence_ids=full_seq,
        device=full_state.device,
        dtype=full_state.dtype,
        metadata={"next_token_id": current_next_token_id},
    )


def full_state_from_prefill_capture(
    capture: PrefillKVCapture,
    *,
    past_key_values: Any,
    next_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> FullKVState:
    """Wrap reloaded prefill KV in a ``FullKVState``."""
    empty_gen = torch.zeros(1, 0, dtype=torch.long, device=device)
    return FullKVState(
        past_key_values=past_key_values,
        prompt_ids=capture.prompt_ids,
        generated_ids=empty_gen,
        full_sequence_ids=capture.prompt_ids,
        device=device,
        dtype=dtype,
        metadata={"next_token_id": next_token_id},
    )


def store_and_reload_prefill_state(
    runtime: ModelRuntime,
    *,
    prompt: str,
    backend: KVStorageBackend,
    prompt_id: str,
    namespace_prefix: str = "exp048",
) -> tuple[FullKVState, PrefillKVCapture, str]:
    """Capture prefill KV, persist via backend, reload into ``FullKVState``."""
    capture = capture_prefill_kv(runtime, prompt)
    payload = build_storage_payload_from_cache(capture)
    backend_name = getattr(backend, "_backend_name", type(backend).__name__)
    handle = KVStorageHandle(
        namespace=f"{namespace_prefix}/{capture.model_name}/{backend_name}",
        key=prompt_id,
        version="1",
    )
    residency = (
        CacheResidency.CPU
        if backend_name == "in_memory_kv_storage"
        else CacheResidency.DISK
    )
    store_prefill_payload(backend, handle, payload, residency=residency)
    loaded = backend.get(handle).payload
    restored_cache, restored_next = restore_cache_from_storage_payload(
        loaded, device=runtime.device
    )
    state = full_state_from_prefill_capture(
        capture,
        past_key_values=restored_cache,
        next_token_id=restored_next,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    cache_format = detect_hf_cache_format(restored_cache)
    return state, capture, cache_format


@torch.no_grad()
def draft_lossy_tokens(
    runtime: ModelRuntime,
    compressor: Any,
    compressed: CompressedKVState,
    n: int,
) -> tuple[list[int], str]:
    """Generate draft tokens from lossy/materialized compressed KV (isolated _draft copy)."""
    try:
        draft_kv: Any = copy.deepcopy(compressor.materialize_for_draft(compressed))
        d_current: int = compressed.next_token_id
        draft_tokens: list[int] = []
        for i in range(n):
            draft_tokens.append(d_current)
            if d_current == runtime.eos_token_id:
                break
            if i < n - 1:
                tok_tensor = torch.tensor(
                    [[d_current]], dtype=torch.long, device=runtime.device
                )
                out = runtime.forward(tok_tensor, past_key_values=draft_kv)
                draft_kv = out.past_key_values
                d_current = int(out.logits[:, -1, :].argmax(dim=-1).item())
        return draft_tokens, ""
    except Exception as exc:  # noqa: BLE001 — report draft blockers
        return [], f"{type(exc).__name__}: {exc}"


def verify_draft_with_compressor(
    engine: VerificationEngine,
    compressor: Any,
    full_state: FullKVState,
    draft_tokens: list[int],
) -> AcceptanceResult:
    """Sequential verify, honoring compressor ``verification_mode`` when present."""
    verify = engine.verify_sequential
    mode = getattr(compressor, "verification_mode", None)
    if callable(mode):
        with mode():
            return verify(full_state, draft_tokens)
    return verify(full_state, draft_tokens)


def _assert_cache_alignment(
    full_state: FullKVState,
    compressed: CompressedKVState,
    *,
    round_idx: int,
) -> None:
    if full_state.seq_len != compressed.logical_seq_len:
        raise RuntimeError(
            f"cache alignment broken after round {round_idx}: "
            f"full={full_state.seq_len}, compressed={compressed.logical_seq_len}"
        )


@torch.no_grad()
def run_offline_verifier_loop(
    runtime: ModelRuntime,
    reloaded_state: FullKVState,
    reference: list[int],
    *,
    max_new_tokens: int,
    draft_len: int = DEFAULT_DRAFT_LEN,
) -> tuple[list[int], list[OfflineVerifierRoundTrace], list[str]]:
    """Run isolated draft/verify/commit loop with reloaded full KV as verifier."""
    engine = VerificationEngine(runtime)
    output: list[int] = []
    traces: list[OfflineVerifierRoundTrace] = []
    verification_blockers: list[str] = []
    state = reloaded_state
    round_idx = 0
    done = False

    while not done and len(output) < max_new_tokens:
        remaining = max_new_tokens - len(output)
        n = min(draft_len, remaining)
        draft = propose_controlled_draft(
            reference, len(output), n, round_idx, runtime.vocab_size
        )
        if not draft:
            break

        try:
            acceptance = engine.verify_sequential(state, draft)
        except Exception as exc:  # noqa: BLE001 — report verification blockers
            verification_blockers.append(f"round {round_idx}: {type(exc).__name__}: {exc}")
            break

        committed = list(acceptance.accepted_tokens)
        if acceptance.correction_token is not None:
            committed.append(acceptance.correction_token)
        committed, eos_found = truncate_at_eos(committed, runtime.eos_token_id)
        if not committed:
            break

        state = commit_full_state(runtime, state, committed)
        output.extend(committed)
        traces.append(
            OfflineVerifierRoundTrace(
                round_idx=round_idx,
                draft_tokens=list(draft),
                verifier_tokens=list(acceptance.verifier_tokens),
                accepted_prefix_length=acceptance.num_accepted,
                correction_token=acceptance.correction_token,
                committed_tokens=committed,
                all_matched=acceptance.all_matched,
                num_rejected=acceptance.num_rejected,
            )
        )
        round_idx += 1
        if eos_found or len(output) >= max_new_tokens:
            done = True

    return output, traces, verification_blockers


@torch.no_grad()
def run_offline_lossy_verifier_loop(
    runtime: ModelRuntime,
    reloaded_state: FullKVState,
    compressor: Any,
    *,
    max_new_tokens: int,
    draft_len: int = DEFAULT_DRAFT_LEN,
) -> tuple[list[int], list[OfflineVerifierRoundTrace], list[str], str]:
    """Run isolated lossy draft / reloaded-KV verify / commit loop."""
    engine = VerificationEngine(runtime)
    try:
        compressed = compressor.compress(reloaded_state)
        _assert_cache_alignment(reloaded_state, compressed, round_idx=-1)
    except Exception as exc:  # noqa: BLE001 — report draft blockers
        return [], [], [], f"compress failed: {type(exc).__name__}: {exc}"

    output: list[int] = []
    traces: list[OfflineVerifierRoundTrace] = []
    verification_blockers: list[str] = []
    state = reloaded_state
    round_idx = 0
    done = False

    while not done and len(output) < max_new_tokens:
        remaining = max_new_tokens - len(output)
        n = min(draft_len, remaining)
        draft, draft_err = draft_lossy_tokens(runtime, compressor, compressed, n)
        if draft_err:
            return output, traces, verification_blockers, draft_err
        if not draft:
            break

        try:
            acceptance = verify_draft_with_compressor(engine, compressor, state, draft)
        except Exception as exc:  # noqa: BLE001 — report verification blockers
            verification_blockers.append(f"round {round_idx}: {type(exc).__name__}: {exc}")
            break

        committed = list(acceptance.accepted_tokens)
        if acceptance.correction_token is not None:
            committed.append(acceptance.correction_token)
        committed, eos_found = truncate_at_eos(committed, runtime.eos_token_id)
        if not committed:
            break

        state = commit_full_state(runtime, state, committed)
        try:
            compressed = compressor.update_after_commit(compressed, state)
            _assert_cache_alignment(state, compressed, round_idx=round_idx)
        except Exception as exc:  # noqa: BLE001 — report draft blockers
            return (
                output,
                traces,
                verification_blockers,
                f"update_after_commit failed: {type(exc).__name__}: {exc}",
            )

        output.extend(committed)
        traces.append(
            OfflineVerifierRoundTrace(
                round_idx=round_idx,
                draft_tokens=list(draft),
                verifier_tokens=list(acceptance.verifier_tokens),
                accepted_prefix_length=acceptance.num_accepted,
                correction_token=acceptance.correction_token,
                committed_tokens=committed,
                all_matched=acceptance.all_matched,
                num_rejected=acceptance.num_rejected,
            )
        )
        round_idx += 1
        if eos_found or len(output) >= max_new_tokens:
            done = True

    return output, traces, verification_blockers, ""


@torch.no_grad()
def run_offline_verifier_cell(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt: str,
    backend: KVStorageBackend,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    draft_len: int = DEFAULT_DRAFT_LEN,
) -> OfflineVerifierCellResult:
    """Run one offline verifier restore smoke cell."""
    backend_name = getattr(backend, "_backend_name", type(backend).__name__)
    try:
        reloaded_state, _capture, cache_format = store_and_reload_prefill_state(
            runtime,
            prompt=prompt,
            backend=backend,
            prompt_id=prompt_id,
        )
    except HfKvRestoreError as exc:
        return OfflineVerifierCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            cache_format="unknown",
            draft_source_type=DRAFT_SOURCE_TYPE,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            restore_blocker=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — smoke must report blockers
        return OfflineVerifierCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            cache_format="unknown",
            draft_source_type=DRAFT_SOURCE_TYPE,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            restore_blocker=f"{type(exc).__name__}: {exc}",
        )

    reference = generate_full_greedy(runtime, prompt, max_new_tokens).generated_ids[
        0
    ].tolist()
    offline_output, traces, verification_blockers = run_offline_verifier_loop(
        runtime,
        reloaded_state,
        reference,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
    )
    div = first_divergence_index(reference, offline_output)
    exact = div is None and reference == offline_output
    verification_blocker = "; ".join(verification_blockers)

    return OfflineVerifierCellResult(
        prompt_id=prompt_id,
        prompt=prompt,
        backend_name=backend_name,
        cache_format=cache_format,
        draft_source_type=DRAFT_SOURCE_TYPE,
        verifier_source=VERIFIER_SOURCE,
        live_reference_token_ids=reference,
        offline_output_token_ids=offline_output,
        token_exact_match=exact and not verification_blocker,
        exactkv_failures=0 if exact and not verification_blocker else 1,
        accepted_prefix_lengths=[t.accepted_prefix_length for t in traces],
        first_divergence_idx=div,
        verification_blocker=verification_blocker,
        round_traces=traces,
    )


@torch.no_grad()
def run_offline_lossy_verifier_cell(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt: str,
    backend: KVStorageBackend,
    compressor_name: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    draft_len: int = DEFAULT_DRAFT_LEN,
) -> OfflineLossyCellResult:
    """Run one offline lossy-draft verifier restore smoke cell."""
    backend_name = getattr(backend, "_backend_name", type(backend).__name__)
    try:
        compressor = get_compressor(compressor_name)
    except ValueError as exc:
        return OfflineLossyCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            compressor_name=compressor_name,
            cache_format="unknown",
            draft_source=compressor_name,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            mean_acceptance=0.0,
            draft_blocker=str(exc),
        )

    try:
        reloaded_state, _capture, cache_format = store_and_reload_prefill_state(
            runtime,
            prompt=prompt,
            backend=backend,
            prompt_id=f"{prompt_id}__{compressor_name}",
            namespace_prefix="exp049",
        )
    except HfKvRestoreError as exc:
        return OfflineLossyCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            compressor_name=compressor_name,
            cache_format="unknown",
            draft_source=compressor_name,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            mean_acceptance=0.0,
            restore_blocker=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — smoke must report blockers
        return OfflineLossyCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            compressor_name=compressor_name,
            cache_format="unknown",
            draft_source=compressor_name,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            mean_acceptance=0.0,
            restore_blocker=f"{type(exc).__name__}: {exc}",
        )

    reference = generate_full_greedy(runtime, prompt, max_new_tokens).generated_ids[
        0
    ].tolist()
    offline_output, traces, verification_blockers, draft_blocker = (
        run_offline_lossy_verifier_loop(
            runtime,
            reloaded_state,
            compressor,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
        )
    )
    div = first_divergence_index(reference, offline_output)
    exact = div is None and reference == offline_output
    verification_blocker = "; ".join(verification_blockers)
    mean_acc = mean_acceptance_from_traces(traces)

    return OfflineLossyCellResult(
        prompt_id=prompt_id,
        prompt=prompt,
        backend_name=backend_name,
        compressor_name=compressor_name,
        cache_format=cache_format,
        draft_source=compressor_name,
        verifier_source=VERIFIER_SOURCE,
        live_reference_token_ids=reference,
        offline_output_token_ids=offline_output,
        token_exact_match=exact
        and not verification_blocker
        and not draft_blocker,
        exactkv_failures=0
        if exact and not verification_blocker and not draft_blocker
        else 1,
        accepted_prefix_lengths=[t.accepted_prefix_length for t in traces],
        mean_acceptance=mean_acc,
        first_divergence_idx=div,
        draft_blocker=draft_blocker,
        verification_blocker=verification_blocker,
        round_traces=traces,
    )


@torch.no_grad()
def run_offline_drift_stress_cell(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt: str,
    category: str,
    backend: KVStorageBackend,
    compressor_name: str,
    draft_len: int,
    max_new_tokens: int = DEFAULT_DRIFT_MAX_NEW_TOKENS,
    namespace_prefix: str = "exp050",
    storage_key: str | None = None,
) -> OfflineDriftStressCellResult:
    """Run one offline restored-verifier drift-stress cell."""
    backend_name = getattr(backend, "_backend_name", type(backend).__name__)
    storage_key = storage_key or f"{prompt_id}__{compressor_name}__dl{draft_len}"
    try:
        compressor = get_compressor(compressor_name)
    except ValueError as exc:
        return OfflineDriftStressCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            category=category,
            backend_name=backend_name,
            compressor_name=compressor_name,
            draft_len=draft_len,
            cache_format="unknown",
            draft_source=compressor_name,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            mean_acceptance=0.0,
            draft_divergence_count=0,
            semantic_divergence_count=0,
            draft_blocker=str(exc),
        )

    try:
        reloaded_state, _capture, cache_format = store_and_reload_prefill_state(
            runtime,
            prompt=prompt,
            backend=backend,
            prompt_id=storage_key,
            namespace_prefix=namespace_prefix,
        )
    except HfKvRestoreError as exc:
        return OfflineDriftStressCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            category=category,
            backend_name=backend_name,
            compressor_name=compressor_name,
            draft_len=draft_len,
            cache_format="unknown",
            draft_source=compressor_name,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            mean_acceptance=0.0,
            draft_divergence_count=0,
            semantic_divergence_count=0,
            restore_blocker=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — smoke must report blockers
        return OfflineDriftStressCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            category=category,
            backend_name=backend_name,
            compressor_name=compressor_name,
            draft_len=draft_len,
            cache_format="unknown",
            draft_source=compressor_name,
            verifier_source=VERIFIER_SOURCE,
            live_reference_token_ids=[],
            offline_output_token_ids=[],
            token_exact_match=False,
            exactkv_failures=1,
            accepted_prefix_lengths=[],
            mean_acceptance=0.0,
            draft_divergence_count=0,
            semantic_divergence_count=0,
            restore_blocker=f"{type(exc).__name__}: {exc}",
        )

    reference = generate_full_greedy(runtime, prompt, max_new_tokens).generated_ids[
        0
    ].tolist()
    offline_output, traces, verification_blockers, draft_blocker = (
        run_offline_lossy_verifier_loop(
            runtime,
            reloaded_state,
            compressor,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
        )
    )
    div = first_divergence_index(reference, offline_output)
    exact = div is None and reference == offline_output
    verification_blocker = "; ".join(verification_blockers)
    mean_acc = mean_acceptance_from_traces(traces)
    draft_div = draft_divergence_count_from_traces(traces)
    semantic_div = semantic_divergence_count_from_traces(traces, category=category)

    return OfflineDriftStressCellResult(
        prompt_id=prompt_id,
        prompt=prompt,
        category=category,
        backend_name=backend_name,
        compressor_name=compressor_name,
        draft_len=draft_len,
        cache_format=cache_format,
        draft_source=compressor_name,
        verifier_source=VERIFIER_SOURCE,
        live_reference_token_ids=reference,
        offline_output_token_ids=offline_output,
        token_exact_match=exact
        and not verification_blocker
        and not draft_blocker,
        exactkv_failures=0
        if exact and not verification_blocker and not draft_blocker
        else 1,
        accepted_prefix_lengths=[t.accepted_prefix_length for t in traces],
        mean_acceptance=mean_acc,
        draft_divergence_count=draft_div,
        semantic_divergence_count=semantic_div,
        first_divergence_idx=div,
        draft_blocker=draft_blocker,
        verification_blocker=verification_blocker,
        round_traces=traces,
    )


def _exactness_blocker_from_drift_result(result: OfflineDriftStressCellResult) -> str:
    """Build exactness blocker text when offline output diverges from live greedy."""
    if result.exactkv_failures == 0:
        return ""
    if result.restore_blocker or result.draft_blocker or result.verification_blocker:
        return ""
    idx = result.first_divergence_idx
    if idx is None:
        return "offline output length or token sequence diverged from live full greedy"
    return f"offline output diverged from live full greedy at token index {idx}"


def drift_stress_to_cuda_cell(
    result: OfflineDriftStressCellResult,
    *,
    device: str,
    dtype: str,
) -> OfflineCudaDriftCellResult:
    """Convert a drift-stress cell result into a CUDA panel cell result."""
    return OfflineCudaDriftCellResult(
        prompt_id=result.prompt_id,
        prompt=result.prompt,
        category=result.category,
        backend_name=result.backend_name,
        compressor_name=result.compressor_name,
        draft_len=result.draft_len,
        device=device,
        dtype=dtype,
        cache_format=result.cache_format,
        draft_source=result.draft_source,
        verifier_source=result.verifier_source,
        live_reference_token_ids=result.live_reference_token_ids,
        offline_output_token_ids=result.offline_output_token_ids,
        token_exact_match=result.token_exact_match,
        exactkv_failures=result.exactkv_failures,
        accepted_prefix_lengths=result.accepted_prefix_lengths,
        mean_acceptance=result.mean_acceptance,
        draft_divergence_count=result.draft_divergence_count,
        semantic_divergence_count=result.semantic_divergence_count,
        first_divergence_idx=result.first_divergence_idx,
        restore_blocker=result.restore_blocker,
        draft_blocker=result.draft_blocker,
        verification_blocker=result.verification_blocker,
        exactness_blocker=_exactness_blocker_from_drift_result(result),
        round_traces=result.round_traces,
    )


@torch.no_grad()
def run_offline_cuda_drift_cell(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt: str,
    category: str,
    backend: KVStorageBackend,
    compressor_name: str,
    draft_len: int,
    dtype: str,
    max_new_tokens: int = DEFAULT_DRIFT_MAX_NEW_TOKENS,
) -> OfflineCudaDriftCellResult:
    """Run one CUDA offline restored-verifier drift panel cell."""
    storage_key = f"{prompt_id}__{compressor_name}__dl{draft_len}__{dtype}"
    result = run_offline_drift_stress_cell(
        runtime,
        prompt_id=prompt_id,
        prompt=prompt,
        category=category,
        backend=backend,
        compressor_name=compressor_name,
        draft_len=draft_len,
        max_new_tokens=max_new_tokens,
        namespace_prefix=f"exp051/{dtype}",
        storage_key=storage_key,
    )
    device = str(runtime.device)
    return drift_stress_to_cuda_cell(result, device=device, dtype=dtype)


def validate_exp051_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 051 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends",
        "compressor_names",
        "draft_len_values",
        "max_new_tokens",
        "verifier_source",
        "cells",
        "exactkv_failures",
        "token_exact_match_count",
        "mean_acceptance",
        "accepted_prefix_lengths",
        "draft_divergence_count",
        "semantic_divergence_count",
        "first_divergences",
        "cuda_available",
        "dtype_supported",
        "skipped_configs",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "exactness_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_051_ID:
        errors.append("experiment_id must be exp051_offline_verifier_cuda_drift_panel")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not isinstance(report.get("cuda_available"), bool):
        errors.append("cuda_available must be a bool")
    if not isinstance(report.get("dtype_supported"), dict):
        errors.append("dtype_supported must be a dict")
    if not isinstance(report.get("skipped_configs"), list):
        errors.append("skipped_configs must be a list")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    cells = report.get("cells", [])
    exact_count = int(report.get("token_exact_match_count", -1))
    failures = int(report.get("exactkv_failures", -1))
    if exact_count >= 0 and failures >= 0 and exact_count + failures != len(cells):
        errors.append(
            "token_exact_match_count + exactkv_failures must equal len(cells)"
        )
    draft_div = report.get("draft_divergence_count")
    if not isinstance(draft_div, int) or draft_div < 0:
        errors.append("draft_divergence_count must be a non-negative int")
    semantic_div = report.get("semantic_divergence_count")
    if not isinstance(semantic_div, int) or semantic_div < 0:
        errors.append("semantic_divergence_count must be a non-negative int")
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    for cell in cells:
        for field in (
            "prompt_id",
            "backend_name",
            "compressor_name",
            "draft_len",
            "device",
            "dtype",
            "draft_source",
            "verifier_source",
            "token_exact_match",
            "exactkv_failures",
            "mean_acceptance",
            "draft_divergence_count",
            "semantic_divergence_count",
        ):
            if field not in cell:
                errors.append(f"cells missing field: {field}")
        div = cell.get("first_divergence_idx")
        if div is not None and not isinstance(div, int):
            errors.append("first_divergence_idx must be int or null")
    return errors


def validate_exp050_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 050 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends",
        "compressor_names",
        "draft_len_values",
        "max_new_tokens",
        "verifier_source",
        "cells",
        "exactkv_failures",
        "token_exact_match_count",
        "mean_acceptance",
        "accepted_prefix_lengths",
        "draft_divergence_count",
        "first_divergences",
        "semantic_divergence_count",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "no_real_drift_observed",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_050_ID:
        errors.append("experiment_id must be exp050_offline_restored_verifier_drift_stress")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not isinstance(report.get("no_real_drift_observed"), bool):
        errors.append("no_real_drift_observed must be a bool")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    cells = report.get("cells", [])
    exact_count = int(report.get("token_exact_match_count", -1))
    failures = int(report.get("exactkv_failures", -1))
    if exact_count >= 0 and failures >= 0 and exact_count + failures != len(cells):
        errors.append(
            "token_exact_match_count + exactkv_failures must equal len(cells)"
        )
    draft_div = report.get("draft_divergence_count")
    if not isinstance(draft_div, int) or draft_div < 0:
        errors.append("draft_divergence_count must be a non-negative int")
    semantic_div = report.get("semantic_divergence_count")
    if not isinstance(semantic_div, int) or semantic_div < 0:
        errors.append("semantic_divergence_count must be a non-negative int")
    if bool(report.get("no_real_drift_observed")) and int(draft_div or -1) > 0:
        errors.append("no_real_drift_observed cannot be true when draft_divergence_count > 0")
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    for cell in cells:
        for field in (
            "prompt_id",
            "backend_name",
            "compressor_name",
            "draft_len",
            "draft_source",
            "verifier_source",
            "token_exact_match",
            "exactkv_failures",
            "mean_acceptance",
            "draft_divergence_count",
            "semantic_divergence_count",
        ):
            if field not in cell:
                errors.append(f"cells missing field: {field}")
        div = cell.get("first_divergence_idx")
        if div is not None and not isinstance(div, int):
            errors.append("first_divergence_idx must be int or null")
    return errors


def validate_exp049_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 049 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends",
        "compressor_names",
        "draft_len",
        "max_new_tokens",
        "verifier_source",
        "cells",
        "exactkv_failures",
        "token_exact_match_count",
        "accepted_prefix_lengths",
        "mean_acceptance",
        "first_divergences",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_049_ID:
        errors.append("experiment_id must be exp049_offline_verifier_lossy_draft")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    cells = report.get("cells", [])
    exact_count = int(report.get("token_exact_match_count", -1))
    failures = int(report.get("exactkv_failures", -1))
    if exact_count >= 0 and failures >= 0 and exact_count + failures != len(cells):
        errors.append(
            "token_exact_match_count + exactkv_failures must equal len(cells)"
        )
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    for cell in cells:
        for field in (
            "prompt_id",
            "backend_name",
            "compressor_name",
            "draft_source",
            "verifier_source",
            "token_exact_match",
            "exactkv_failures",
            "mean_acceptance",
        ):
            if field not in cell:
                errors.append(f"cells missing field: {field}")
        div = cell.get("first_divergence_idx")
        if div is not None and not isinstance(div, int):
            errors.append("first_divergence_idx must be int or null")
    return errors


def validate_exp048_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 048 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends",
        "cache_format",
        "draft_source_type",
        "verifier_source",
        "cells",
        "exactkv_failures",
        "token_exact_match_count",
        "accepted_prefix_lengths",
        "first_divergences",
        "restore_blockers",
        "verification_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_048_ID:
        errors.append("experiment_id must be exp048_offline_verifier_restore_smoke")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    cells = report.get("cells", [])
    if int(report.get("prompt_count", 0)) <= 0 and cells:
        errors.append("prompt_count must be positive when cells present")
    exact_count = int(report.get("token_exact_match_count", -1))
    failures = int(report.get("exactkv_failures", -1))
    if exact_count >= 0 and failures >= 0 and exact_count + failures != len(cells):
        errors.append(
            "token_exact_match_count + exactkv_failures must equal len(cells)"
        )
    for cell in cells:
        for field in (
            "prompt_id",
            "backend_name",
            "draft_source_type",
            "verifier_source",
            "token_exact_match",
            "exactkv_failures",
        ):
            if field not in cell:
                errors.append(f"cells missing field: {field}")
        div = cell.get("first_divergence_idx")
        if div is not None and not isinstance(div, int):
            errors.append("first_divergence_idx must be int or null")
    return errors
