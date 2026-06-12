"""Hot-adapter feasibility analysis helpers for Experiment 032 (V13 Phase 5).

Feasibility study only — not a production adapter claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FeasibilityClass(str, Enum):
    """A/B/C classification for hot-adapter candidates."""

    A_FEASIBLE_NOW = "A_feasible_now"
    B_RESTRICTED = "B_restricted_feasibility"
    C_NO_GO = "C_no_go_for_now"


@dataclass
class CompatibilityAnswer:
    """One ExactKV compatibility question and answer."""

    question: str
    answer: str
    blocks_exactkv: bool = False


@dataclass
class CandidateFeasibility:
    """Structured feasibility record for one hot-adapter candidate."""

    name: str
    classification: FeasibilityClass
    summary: str
    literature_boundary: list[str]
    requirements: list[str]
    compatibility: list[CompatibilityAnswer]
    blockers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    exactkv_generator_changes_required: bool = False
    verification_engine_changes_required: bool = False
    custom_cuda_required: bool = False
    attention_weights_required: bool = False
    model_internals_changes_required: bool = False
    span_verification_conflict: bool = False
    offline_compressor_possible: bool = False
    factory_only_recommended: bool = True
    production_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        d["compatibility"] = [asdict(c) for c in self.compatibility]
        return d


def _snapkv_compatibility() -> list[CompatibilityAnswer]:
    return [
        CompatibilityAnswer(
            "Can it produce compressed/dropped KV for draft generation?",
            "Yes — token-retention policy yields a shorter physical DynamicCache "
            "for drafting (same pattern as KVPressKnormAdapter).",
        ),
        CompatibilityAnswer(
            "Can ExactKV maintain full verifier KV separately?",
            "Yes — full state is authoritative; compression replays on an isolated "
            "model copy; verification_mode() asserts hooks inactive on verifier model.",
        ),
        CompatibilityAnswer(
            "Does it require changing model forward internals?",
            "No core ExactKV changes — but replay uses forward hooks via kvpress "
            "(or an inspired-by offline index+slice over past_key_values).",
        ),
        CompatibilityAnswer(
            "Does it require attention weights?",
            "Yes for paper-faithful SnapKV — observation-window attention statistics "
            "vote for prefix retention (kvpress SnapKVPress uses hidden_states context).",
            blocks_exactkv=False,
        ),
        CompatibilityAnswer(
            "Does it need custom CUDA kernels?",
            "No for MVP via kvpress Python hooks; production SnapKV systems may use "
            "fused kernels — out of V13 scope.",
        ),
        CompatibilityAnswer(
            "Does it conflict with span verification?",
            "No inherent conflict — span verify uses full KV only; same as sequential.",
        ),
        CompatibilityAnswer(
            "Does it conflict with recompress after commit?",
            "No — update_after_commit re-runs compress from new full state each round "
            "(V1 recompress policy; expensive but correct).",
        ),
        CompatibilityAnswer(
            "Can it be implemented as offline compressor over past_key_values?",
            "Partially — inspired-by retention indices could slice tensors without hooks, "
            "but that is not paper-exact SnapKV without attention statistics.",
        ),
        CompatibilityAnswer(
            "Can it avoid production serving assumptions?",
            "Yes — single-request replay prefill under factory-only adapter, matching "
            "Exp 005 KnormPress pattern.",
        ),
    ]


def _shardkv_compatibility() -> list[CompatibilityAnswer]:
    return [
        CompatibilityAnswer(
            "Can it produce compressed/dropped KV for draft generation?",
            "Unclear — no ShardKV LLM KV-compression reference exists in this repository.",
            blocks_exactkv=True,
        ),
        CompatibilityAnswer(
            "Can ExactKV maintain full verifier KV separately?",
            "N/A for distributed serving shardkv (MIT 6.824-style) — that shards keys "
            "across servers, not draft-cache compression within one request.",
            blocks_exactkv=True,
        ),
        CompatibilityAnswer(
            "Does it require changing model forward internals?",
            "Distributed shardkv is a serving-layer concern; not a KVCompressor adapter.",
            blocks_exactkv=True,
        ),
        CompatibilityAnswer(
            "Does it require attention weights?",
            "Unknown for unnamed ShardKV hot-adapter target.",
            blocks_exactkv=True,
        ),
        CompatibilityAnswer(
            "Does it need custom CUDA kernels?",
            "Likely for cross-device sharding runtimes — out of V13 non-goals.",
            blocks_exactkv=True,
        ),
        CompatibilityAnswer(
            "Does it conflict with span verification?",
            "No direct conflict if it were a draft compressor — but no viable design.",
        ),
        CompatibilityAnswer(
            "Does it conflict with recompress after commit?",
            "Unknown — no adapter design.",
            blocks_exactkv=True,
        ),
        CompatibilityAnswer(
            "Can it be implemented as offline compressor over past_key_values?",
            "No recognized ShardKV algorithm mapped to past_key_values slicing.",
            blocks_exactkv=True,
        ),
        CompatibilityAnswer(
            "Can it avoid production serving assumptions?",
            "Classic shardkv assumes multi-server coordination — conflicts with V13 "
            "single-request scope.",
            blocks_exactkv=True,
        ),
    ]


def analyze_snapkv() -> CandidateFeasibility:
    """SnapKV feasibility from local ExactKV + kvpress integration notes."""
    return CandidateFeasibility(
        name="SnapKV",
        classification=FeasibilityClass.B_RESTRICTED,
        summary=(
            "SnapKV is a recognizable prompt-KV retention method (observation-window "
            "attention voting). ExactKV already proves token-dropping via "
            "KVPressKnormAdapter (Exp 005). SnapKV is feasible as a **restricted "
            "factory-only** kvpress SnapKVPress adapter following the same replay "
            "pattern — not paper-exact without hidden_states/attention context."
        ),
        literature_boundary=[
            "SnapKV (Li et al., 2024): retain important prefix tokens using an "
            "observation window at the prompt tail; cluster/vote over attention "
            "statistics during prefill.",
            "kvpress provides SnapKVPress (Phase C+ in KVPRESS_INTEGRATION_RESEARCH.md) "
            "— needs hidden_states / attention context, not keys-only like KnormPress.",
            "ChunkKV and other kvpress presses share the NVIDIA kvpress repo but are "
            "separate methods — not SnapKV.",
            "External SnapKV throughput/memory numbers are not ExactKV results.",
        ],
        requirements=[
            "Access to attention scores or hidden_states during compression replay.",
            "Prompt/prefill attention statistics (observation window at sequence tail).",
            "Token selection / clustering policy (SnapKVPress in kvpress).",
            "Per-layer retained token indices → physical KV gather/slice in DynamicCache.",
            "HuggingFace Qwen DynamicCache layout (dynamic_v5) — proven in Exp 005.",
            "logical_seq_len preserved separately from physical_seq_len (kvpress_knorm pattern).",
            "Isolated compression model copy to avoid rotary_emb / hook side effects.",
            "verification_mode() hook guard during verify (existing in KVPressKnormAdapter).",
        ],
        compatibility=_snapkv_compatibility(),
        blockers=[
            "SnapKVPress requires more forward context than KnormPress (hidden_states).",
            "Exp 026: default SDPA prefill does not return attentions; eager or kvpress "
            "hook path required for score extraction.",
            "kvpress global attention patch — must stay isolated to [kvpress] env / factory.",
            "Recompress-after-commit replays full prefill each round — correct but costly.",
        ],
        risks=[
            "Inspired-by index slicing without SnapKVPress is not production SnapKV.",
            "Accept rate may be low under aggressive retention (Exp 005 Knorm ~0.41 accept).",
            "Exp 031: active CUDA peak unchanged at 0.5B — hot adapter does not fix VRAM headline.",
            "Hook leakage into verifier would break exactness — must keep isolation gates.",
        ],
        exactkv_generator_changes_required=False,
        verification_engine_changes_required=False,
        custom_cuda_required=False,
        attention_weights_required=True,
        model_internals_changes_required=False,
        span_verification_conflict=False,
        offline_compressor_possible=True,
        factory_only_recommended=True,
        production_claim_allowed=False,
    )


def analyze_shardkv() -> CandidateFeasibility:
    """ShardKV / Shard-style secondary candidate."""
    return CandidateFeasibility(
        name="ShardKV",
        classification=FeasibilityClass.C_NO_GO,
        summary=(
            "No canonical LLM KV-cache compression method named ShardKV exists in the "
            "ExactKV repository or cited integration research. V13 'Shard/ShardKV' "
            "most likely refers to distributed serving sharding (e.g. MIT 6.824 shardkv) "
            "or an unspecified second hot name — both are **poor fits** for a draft-only "
            "KVCompressor adapter without production serving scope."
        ),
        literature_boundary=[
            "RELATED_WORK lists SnapKV, H2O, StreamingLLM, PyramidKV — not ShardKV.",
            "Distributed 'shardkv' (6.824 / raftkv) partitions keys across replica groups "
            "for throughput — orthogonal to per-request draft compression.",
            "If 'Shard' meant layer-wise budget (PyramidKV-like), that is a different "
            "named method — use PyramidKV feasibility separately, not ShardKV.",
            "No kvpress press class named ShardKV in KVPRESS_INTEGRATION_RESEARCH.md.",
        ],
        requirements=[
            "Unclear target algorithm — cannot list implementation requirements honestly.",
            "Would require serving/multi-request infrastructure if interpreted as shardkv.",
            "No mapped path to past_key_values retention indices in ExactKV compressors/.",
        ],
        compatibility=_shardkv_compatibility(),
        blockers=[
            "No recognized ShardKV hot-adapter algorithm in scope.",
            "Distributed sharding conflicts with V13 single-request, non-serving scope.",
            "No local reference implementation to wrap via BackendAdapter.",
        ],
        risks=[
            "Mislabeling a PyramidKV or ChunkKV adapter as 'ShardKV' would overclaim.",
            "Implementing serving shardkv would violate V13 non-goals (vLLM/LMCache/etc.).",
        ],
        exactkv_generator_changes_required=True,
        verification_engine_changes_required=False,
        custom_cuda_required=True,
        attention_weights_required=True,
        model_internals_changes_required=True,
        span_verification_conflict=False,
        offline_compressor_possible=False,
        factory_only_recommended=True,
        production_claim_allowed=False,
    )


def design_snapkv_experimental_mvp() -> dict[str, Any]:
    """MVP design for Phase 5b — design only, not implemented in Exp 032."""
    return {
        "compressor_name": "snapkv_experimental",
        "registry_policy": "factory-only via create_snapkv_experimental_adapter(); NOT default registry",
        "base_class": "BackendAdapter (replay-from-full-state, mirrors KVPressKnormAdapter)",
        "config_fields": {
            "compression_ratio": "float — retained fraction (kvpress press API)",
            "window_size": "int — SnapKV observation window (press-specific)",
            "isolate_compression_model": "bool — default True (deepcopy verifier model)",
            "backend_env": "requires [kvpress] optional extra / .venv-kvpress",
        },
        "attention_source": (
            "Replay prefill under kvpress SnapKVPress hooks on isolated model; "
            "hidden_states from forward hook context — not output_attentions on SDPA."
        ),
        "token_selection": "SnapKVPress scoring + retain indices per layer (kvpress internal)",
        "kv_slicing": "Pruned DynamicCache in backend_data['dynamic_cache']",
        "logical_seq_len": "FullKVState.seq_len (alignment invariant unchanged)",
        "update_after_commit": "Recompress from authoritative new_full_state (V1 recompress)",
        "memory_accounting": {
            "stored_kv_bytes": "kv_total_bytes(pruned cache)",
            "materialized_working_kv_bytes": "== stored (token-dropping)",
            "supports_real_bytes_claim": True,
            "is_simulated": False,
            "note": "V5 accounting may show MiB-scale reduction; Exp 031 shows active CUDA peak unchanged at 0.5B.",
        },
        "simulation_vs_real": (
            "Real token dropping at fp16/full precision if kvpress SnapKVPress used; "
            "NOT a claim of paper-identical SnapKV without verification against reference."
        ),
        "tests_needed": [
            "Hook isolation: zero forward hooks during verification_mode",
            "exactkv_failures == 0 on 4+ prompt smoke panel",
            "logical_seq_len == full_state.seq_len after each compress",
            "Parity with generate_full_greedy (exactness gate)",
            "Optional: compare retention indices to reference SnapKV on one long prompt",
        ],
        "implementation_status": "design_only — not implemented in Experiment 032",
    }


def build_feasibility_artifact() -> dict[str, Any]:
    """Assemble full Experiment 032 feasibility artifact."""
    snapkv = analyze_snapkv()
    shardkv = analyze_shardkv()
    chosen = "snapkv_restricted_mvp_phase_5b"
    if snapkv.classification == FeasibilityClass.C_NO_GO:
        chosen = "no_adapter_documented_blocker"
    elif shardkv.classification == FeasibilityClass.A_FEASIBLE_NOW:
        chosen = "shardkv_restricted_mvp"

    return {
        "experiment": "032",
        "experiment_class": "v13_hot_adapter_feasibility",
        "candidates": {
            "snapkv": snapkv.to_dict(),
            "shardkv": shardkv.to_dict(),
        },
        "chosen_path": chosen,
        "adapter_implemented": False,
        "adapter_design": design_snapkv_experimental_mvp(),
        "precedent": {
            "kvpress_knorm_adapter": (
                "Exp 005 — KVPressKnormAdapter, factory-only, exactkv_failures == 0, "
                "low accept rate but exact outputs."
            ),
            "backend_adapter_poc": "Exp 006 V6 — BackendAdapter sealed protocol",
            "attention_logging": "Exp 026 — eager prefill can log attentions; SDPA prefill cannot",
            "gpu_memory": "Exp 031 — active CUDA peak dominated by weights; no VRAM savings claim",
        },
        "interpretation": {
            "snapkv_classification": snapkv.classification.value,
            "shardkv_classification": shardkv.classification.value,
            "full_kv_verifier_authoritative": True,
            "active_gpu_memory_savings_claim_allowed": False,
            "speed_claim_allowed": False,
            "production_snapkv_claim_allowed": False,
            "phase5b_recommended": snapkv.classification == FeasibilityClass.B_RESTRICTED,
            "phase6_llama_allowed": True,
        },
    }
