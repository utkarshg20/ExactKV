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
    exactkv_integration_path: str = ""
    integration_path_notes: dict[str, str] = field(default_factory=dict)
    external_repo_url: str = ""

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


def analyze_shard() -> CandidateFeasibility:
    """Shard (krish1905/shard) — real Llama KV compression Cache subclass."""
    return CandidateFeasibility(
        name="Shard",
        classification=FeasibilityClass.B_RESTRICTED,
        external_repo_url="https://github.com/krish1905/shard",
        exactkv_integration_path="C_llama_sidecar_or_external_drafter",
        integration_path_notes={
            "A_direct_kvcompressor": (
                "C — Shard Cache requires fused compressed-K attention and "
                "enable_llama_fused_attention() monkey-patch; not a tensor-only "
                "BackendAdapter over past_key_values."
            ),
            "B_external_drafter": (
                "B — Draft with Shard Cache + patched LlamaAttention on an isolated "
                "model; full-KV verifier on authoritative HF cache (Exp 022 Mode B "
                "pattern)."
            ),
            "C_llama_sidecar_probe": (
                "B — Best fit: Llama-3.1-8B-only legibility probe alongside Exp 033; "
                "not a default-registry compressor."
            ),
            "D_no_go": "For direct KVCompressor without draft-model fork.",
        },
        summary=(
            "Shard (krish1905/shard) is a **real** Llama-3.1 KV-cache compression "
            "system: `transformers.Cache` subclass, PCA+VQ asymmetric compression, "
            "fused Q·K on int4 coefficients without FP16 K materialization. "
            "README reports ~10× memory reduction at 8K but **0.4–0.5× decode "
            "throughput** vs FP16 (external results — not ExactKV). ExactKV can "
            "likely use it only as an **external drafter / Llama-only sidecar probe**, "
            "not as a Qwen KVCompressor backend without Llama + attention monkey-patch."
        ),
        literature_boundary=[
            "Repo: https://github.com/krish1905/shard — `src/shard/cache.py` subclasses "
            "`transformers.cache_utils.Cache`.",
            "`enable_llama_fused_attention()` monkey-patches `LlamaAttention.forward`.",
            "Triton kernels in `triton_kernels.py` with CPU/PyTorch fallbacks.",
            "E2E benchmarks via `benchmarks/benchmark.py` (Modal B200).",
            "Supersedes Exp 032 misread of 'ShardKV' as MIT 6.824 distributed shardkv.",
        ],
        requirements=[
            "Llama model (example: meta-llama/Llama-3.1-8B-Instruct).",
            "Call `enable_llama_fused_attention(model)` before generate.",
            "Pass `shard.Cache` as `past_key_values` to `model.generate`.",
            "Isolated draft model copy for compression/draft path.",
            "Full-precision HF DynamicCache for verifier (no Shard patch on verify model).",
        ],
        compatibility=[
            CompatibilityAnswer(
                "Can it produce compressed KV for draft generation?",
                "Yes — Shard Cache compresses during prefill and streams decode in "
                "compressed format.",
            ),
            CompatibilityAnswer(
                "Can ExactKV maintain full verifier KV separately?",
                "Yes **if** verifier model is unpatched and uses standard full KV; "
                "draft model uses Shard Cache + fused attention only.",
            ),
            CompatibilityAnswer(
                "Does it require changing model forward internals?",
                "Yes — monkey-patches LlamaAttention for fused compressed-K decode.",
                blocks_exactkv=False,
            ),
            CompatibilityAnswer(
                "Does it require custom CUDA/Triton?",
                "Optional — Triton accelerates; PyTorch fallbacks exist.",
            ),
            CompatibilityAnswer(
                "Does it conflict with span verification?",
                "No — verifier uses full KV on unpatched model.",
            ),
            CompatibilityAnswer(
                "Can it be a direct KVCompressor over past_key_values?",
                "No — not without reimplementing fused attention inside materialize_for_draft.",
                blocks_exactkv=True,
            ),
        ],
        blockers=[
            "Llama-only in reference implementation (no Qwen path).",
            "Attention monkey-patch must not leak to verifier model.",
            "Not drop-in for ExactKV BackendAdapter tensor path.",
            "Aligns with Exp 033 Llama panel, not current Qwen 0.5B smoke path.",
        ],
        risks=[
            "External 10× compression / quality numbers are **not** ExactKV results.",
            "Decode throughput below FP16 — must not claim ExactKV speedup.",
            "Overclaiming 'Shard integrated' if only inspired-by slicing.",
        ],
        exactkv_generator_changes_required=False,
        verification_engine_changes_required=False,
        custom_cuda_required=False,
        attention_weights_required=False,
        model_internals_changes_required=True,
        span_verification_conflict=False,
        offline_compressor_possible=False,
        factory_only_recommended=True,
        production_claim_allowed=False,
    )


def analyze_spectralquant() -> CandidateFeasibility:
    """SpectralQuant (Dynamis-Labs/spectralquant) — calibration + tensor compressor."""
    return CandidateFeasibility(
        name="SpectralQuant",
        classification=FeasibilityClass.B_RESTRICTED,
        external_repo_url="https://github.com/Dynamis-Labs/spectralquant",
        exactkv_integration_path="B_offline_calibration_tensor_compressor",
        integration_path_notes={
            "A_direct_kvcompressor": (
                "B — Wrap `spectralquant.SpectralQuantEngine.compress_keys/decompress_*` "
                "in BackendAdapter after calibrating per model; materialize dequantized "
                "K/V for draft forwards."
            ),
            "B_offline_calibration_tensor_compressor": (
                "B — Best fit: EigenspectralCalibrator + per-layer engine; mirrors "
                "TurboQuant Python adapter (Exp 008) pattern."
            ),
            "C_external_drafter": "Possible but unnatural — engine is tensor-level, not HF generate.",
            "D_no_go": "If turboquant_cutile baseline cannot be installed for kernel path.",
        },
        summary=(
            "SpectralQuant is a **real library** (`src/spectralquant/`) with calibration, "
            "spectral rotation, non-uniform Lloyd-Max quantization, and selective QJL. "
            "It **subclasses TurboQuant** for the kernel engine and requires a "
            "`baseline/turboquant_cutile` clone for full reproduction. It operates on "
            "**per-layer K/V tensors**, not arbitrary HF `past_key_values` drop-in. "
            "Paper/repo headline metrics are **external** — not ExactKV results. "
            "Feasible as **offline calibration + tensor compressor** BackendAdapter (B)."
        ),
        literature_boundary=[
            "Repo: https://github.com/Dynamis-Labs/spectralquant",
            "Canonical engine: `spectralquant.spectralquant.SpectralQuantEngine` (pure Python).",
            "Kernel variant: `spectralquant.engine.SpectralQuantEngine` subclasses TurboQuantEngine.",
            "Requires ~15s calibration (`EigenspectralCalibrator`) per model.",
            "21 experiment scripts + frozen `results/` JSON — strong paper artifact.",
            "Qwen/Llama/Mistral/Gemma supported in experiment adapters (llama_like).",
        ],
        requirements=[
            "Calibration pass collecting per-head KV statistics.",
            "Per-layer `compress_keys` / `compress_values` on tensors extracted from full_state.",
            "Dequantized materialization for draft attention (materialized_working_kv_bytes).",
            "Optional: turboquant_cutile for kernel-accelerated path (Modal/CUDA).",
            "Factory-only adapter; `supports_real_bytes_claim` labeling per capabilities.",
        ],
        compatibility=[
            CompatibilityAnswer(
                "Can it produce compressed KV for draft generation?",
                "Yes — after calibration, compress/decompress K/V tensors per layer.",
            ),
            CompatibilityAnswer(
                "Can ExactKV maintain full verifier KV separately?",
                "Yes — verifier uses authoritative full_state; compression is draft-only.",
            ),
            CompatibilityAnswer(
                "Does it integrate with HF past_key_values directly?",
                "No — tensor API; adapter must extract/rebuild via cache/utils.",
                blocks_exactkv=False,
            ),
            CompatibilityAnswer(
                "Does it require calibration artifacts?",
                "Yes — eigenspectral calibration per model (or cached pickle).",
            ),
            CompatibilityAnswer(
                "Does it depend on TurboQuant?",
                "Kernel engine subclasses TurboQuantEngine; pure-Python path testable without.",
            ),
            CompatibilityAnswer(
                "Does it conflict with span verification?",
                "No — verify path uses full KV only.",
            ),
        ],
        blockers=[
            "Extra dependency surface (spectralquant + optional turboquant_cutile).",
            "Calibration step before first compress — not zero-config like noop.",
            "Less public name recognition than SnapKV/Shard for launch legibility.",
            "Exp 031: active CUDA savings unlikely at 0.5B even if V5 accounting improves.",
        ],
        risks=[
            "Treating paper JSON results as ExactKV outcomes.",
            "Claiming production SpectralQuant without calibration parity checks.",
            "Higher implementation time than kvpress SnapKVPress on Qwen.",
        ],
        exactkv_generator_changes_required=False,
        verification_engine_changes_required=False,
        custom_cuda_required=False,
        attention_weights_required=False,
        model_internals_changes_required=False,
        span_verification_conflict=False,
        offline_compressor_possible=True,
        factory_only_recommended=True,
        production_claim_allowed=False,
    )


def analyze_shardkv() -> CandidateFeasibility:
    """Legacy ShardKV label — superseded by analyze_shard() for krish1905/shard (Exp 032 addendum)."""
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


def build_addendum_artifact() -> dict[str, Any]:
    """Experiment 032 addendum — Shard + SpectralQuant repo inspection."""
    snapkv = analyze_snapkv()
    shard = analyze_shard()
    spectralquant = analyze_spectralquant()
    legacy_shardkv = analyze_shardkv()

    ranking = [
        {
            "rank": 1,
            "candidate": "SnapKV",
            "classification": snapkv.classification.value,
            "rationale": (
                "Lowest ExactKV integration risk on Qwen via existing kvpress replay "
                "pattern (Exp 005); no Llama requirement; fastest Phase 5b MVP."
            ),
        },
        {
            "rank": 2,
            "candidate": "Shard",
            "classification": shard.classification.value,
            "rationale": (
                "Highest launch legibility for compressed-KV story, but Llama-only + "
                "attention monkey-patch → external drafter / Phase 6 adjunct, not "
                "KVCompressor preempt."
            ),
        },
        {
            "rank": 3,
            "candidate": "SpectralQuant",
            "classification": spectralquant.classification.value,
            "rationale": (
                "Real tensor compressor with calibration; more deps and setup than "
                "SnapKV; better as Phase 5c or parallel probe after TurboQuant wiring."
            ),
        },
        {
            "rank": 4,
            "candidate": "ShardKV (legacy label)",
            "classification": legacy_shardkv.classification.value,
            "rationale": "Misidentified distributed-systems name — superseded by Shard repo.",
        },
    ]

    return {
        "experiment": "032_addendum",
        "experiment_class": "v13_hot_adapter_feasibility_addendum",
        "supersedes_note": (
            "Revises Exp 032 ShardKV classification. See "
            "EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md."
        ),
        "external_repos": {
            "shard": shard.external_repo_url,
            "spectralquant": spectralquant.external_repo_url,
        },
        "candidates": {
            "snapkv": snapkv.to_dict(),
            "shard": shard.to_dict(),
            "spectralquant": spectralquant.to_dict(),
            "shardkv_legacy": legacy_shardkv.to_dict(),
        },
        "candidate_ranking": ranking,
        "chosen_path": "snapkv_restricted_mvp_phase_5b",
        "phase5b_priority": {
            "primary": "snapkv_experimental_adapter",
            "parallel_optional": [
                "shard_llama_external_drafter_probe_with_exp_033",
            ],
            "deferred": [
                "spectralquant_experimental_adapter_phase_5c",
            ],
        },
        "snapkv_still_recommended_for_5b": True,
        "shard_or_spectralquant_preempt_snapkv": False,
        "adapter_implemented": False,
        "interpretation": {
            "snapkv_classification": snapkv.classification.value,
            "shard_classification": shard.classification.value,
            "spectralquant_classification": spectralquant.classification.value,
            "full_kv_verifier_authoritative": True,
            "external_results_are_not_exactkv": True,
            "active_gpu_memory_savings_claim_allowed": False,
            "speed_claim_allowed": False,
            "phase5b_primary": "snapkv_experimental",
            "phase6_llama_allowed": True,
        },
    }
