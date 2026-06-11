"""External-drafter verification probe harness (Experiment 022 only).

Uses HF VerificationEngine as authoritative verifier. Does not integrate llama.cpp
into ExactKV runtime. Metrics are labeled external-probe, not standard ExactKV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from exactkv.cache.full_state import FullKVState
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state
from exactkv.verification.engine import VerificationEngine


@dataclass
class ExternalProbeRound:
    round_idx: int
    draft_tokens: list[int]
    accepted_tokens: list[int]
    rejected_tokens: list[int]
    correction_token: int | None
    all_matched: bool
    external_draft_stale_after: bool


@dataclass
class ExternalProbeSummary:
    """HF-verifier external probe over pre-generated llama.cpp draft token IDs."""

    token_alignment_safe: bool
    rounds: list[ExternalProbeRound]
    total_drafted: int
    total_accepted: int
    total_rejected: int
    total_corrections: int
    external_probe_acceptance_rate: float
    trajectory_match_count: int
    trajectory_compare_len: int
    trajectory_match_rate: float
    committed_output_ids: list[int]
    external_draft_stale_after_round: int | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rounds"] = [asdict(r) for r in self.rounds]
        return d


@torch.no_grad()
def _commit_tokens(
    runtime: ModelRuntime,
    full_state: FullKVState,
    committed_tokens: list[int],
) -> FullKVState:
    past_kv = full_state.past_key_values
    current_next = full_state.next_token_id
    new_gen = full_state.generated_ids.squeeze(0).tolist()

    for token_id in committed_tokens:
        new_gen.append(token_id)
        if token_id == runtime.eos_token_id:
            current_next = runtime.eos_token_id
            break
        tok = torch.tensor([[token_id]], dtype=torch.long, device=runtime.device)
        out = runtime.forward(tok, past_key_values=past_kv)
        past_kv = out.past_key_values
        current_next = int(out.logits[:, -1, :].argmax(dim=-1).item())

    gen_tensor = torch.tensor([new_gen], dtype=torch.long, device=runtime.device)
    full_seq = torch.cat([full_state.prompt_ids, gen_tensor], dim=1)
    return FullKVState(
        past_key_values=past_kv,
        prompt_ids=full_state.prompt_ids,
        generated_ids=gen_tensor,
        full_sequence_ids=full_seq,
        device=full_state.device,
        dtype=full_state.dtype,
        metadata={"next_token_id": current_next},
    )


def trajectory_token_agreement(
    hf_generated_ids: list[int],
    external_generated_ids: list[int],
) -> dict[str, Any]:
    compare_len = min(len(hf_generated_ids), len(external_generated_ids))
    matches = sum(
        1
        for i in range(compare_len)
        if hf_generated_ids[i] == external_generated_ids[i]
    )
    rate = matches / compare_len if compare_len else 0.0
    first_div = next(
        (
            i
            for i in range(compare_len)
            if hf_generated_ids[i] != external_generated_ids[i]
        ),
        None,
    )
    return {
        "compare_len": compare_len,
        "match_count": matches,
        "match_rate": rate,
        "first_divergence_idx": first_div,
    }


def run_external_drafter_probe(
    runtime: ModelRuntime,
    prompt: str,
    external_draft_ids: list[int],
    *,
    draft_len: int,
    max_new_tokens: int,
    token_alignment_safe: bool,
) -> ExternalProbeSummary:
    """Simulate HF-verifier rounds over pre-generated external draft tokens."""
    engine = VerificationEngine(runtime)
    full_state = prefill_to_full_state(runtime, prompt)

    rounds: list[ExternalProbeRound] = []
    total_accepted = 0
    total_rejected = 0
    total_corrections = 0
    pos = 0
    round_idx = 0
    stale_after: int | None = None
    active = token_alignment_safe

    while active and len(full_state.generated_ids.squeeze(0)) < max_new_tokens:
        if pos >= len(external_draft_ids):
            break
        remaining = max_new_tokens - len(full_state.generated_ids.squeeze(0))
        n = min(draft_len, remaining, len(external_draft_ids) - pos)
        draft_batch = external_draft_ids[pos : pos + n]
        pos += n

        acceptance = engine.verify_sequential(full_state, draft_batch)
        committed = list(acceptance.accepted_tokens)
        correction = acceptance.correction_token
        if correction is not None:
            committed.append(correction)
            total_corrections += 1

        total_accepted += len(acceptance.accepted_tokens)
        total_rejected += acceptance.num_rejected

        stale = correction is not None
        if stale and stale_after is None:
            stale_after = round_idx

        rounds.append(
            ExternalProbeRound(
                round_idx=round_idx,
                draft_tokens=draft_batch,
                accepted_tokens=list(acceptance.accepted_tokens),
                rejected_tokens=list(acceptance.rejected_tokens),
                correction_token=correction,
                all_matched=acceptance.all_matched,
                external_draft_stale_after=stale,
            )
        )
        full_state = _commit_tokens(runtime, full_state, committed)
        round_idx += 1

        if committed and committed[-1] == runtime.eos_token_id:
            break
        if stale:
            active = False

    committed_ids = full_state.generated_ids.squeeze(0).tolist()
    total_drafted = sum(len(r.draft_tokens) for r in rounds)
    accept_rate = (
        total_accepted / total_drafted if total_drafted else 0.0
    )
    traj = trajectory_token_agreement(committed_ids, external_draft_ids[: len(committed_ids)])

    return ExternalProbeSummary(
        token_alignment_safe=token_alignment_safe,
        rounds=rounds,
        total_drafted=total_drafted,
        total_accepted=total_accepted,
        total_rejected=total_rejected,
        total_corrections=total_corrections,
        external_probe_acceptance_rate=accept_rate,
        trajectory_match_count=traj["match_count"],
        trajectory_compare_len=traj["compare_len"],
        trajectory_match_rate=traj["match_rate"],
        committed_output_ids=committed_ids,
        external_draft_stale_after_round=stale_after,
    )
