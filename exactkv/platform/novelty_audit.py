"""Phase I novelty and prior-art audit data model."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EvidenceStatus = Literal["verified", "source_pending", "ambiguous"]
SourceQuality = Literal["primary", "secondary", "unknown"]
ClaimStatus = Literal["allowed", "allowed_with_qualification", "forbidden", "needs_more_evidence"]
EvidenceStrength = Literal["strong", "moderate", "weak", "missing"]
RiskLevel = Literal["low", "medium", "high"]

REQUIRED_PRIOR_ART_NAMES = (
    "VeriCache",
    "KVQuant",
    "KIVI",
    "TurboQuant",
    "QuantSpec",
    "SparseSpec",
    "SpecAttn",
    "MagicDec",
    "LMCache",
    "CacheGen",
    "ShardCache (shard-kv)",
    "Redis / Valkey cache benchmarks",
    "GPTCache / semantic cache benchmarks",
)

REQUIRED_CANDIDATE_CLAIMS = (
    "ExactKV is a KV-cache compression exactness benchmark.",
    "ExactKV is a compressor-agnostic token-level drift leaderboard.",
    "ExactKV is the first system like this.",
    "ExactKV reproduces VeriCache.",
    "ExactKV invented compressed-KV verification.",
    "ExactKV measures first divergence across compressors.",
    "ExactKV reports acceptance rate and accepted span across compressors.",
    "ExactKV proves end-to-end speedups.",
    "ExactKV proves active GPU memory savings.",
    "ExactKV has a real Triton KV compression kernel path.",
    "ExactKV is production ready.",
    "ExactKV is a research-grade evaluation framework.",
    "ExactKV evaluates real 7B/8B models.",
    "ExactKV compares real SpectralQuant.",
    "ExactKV compares real Shard.",
    "ExactKV includes SpectralQuant fallback/proxy support.",
    "ExactKV includes Shard probe-first analysis.",
    "ExactKV is a public benchmark platform.",
)


@dataclass
class PriorArtSystem:
    system_name: str
    category: str
    source_urls: list[str] = field(default_factory=list)
    source_quality: SourceQuality = "unknown"
    primary_goal: str = ""
    core_method_summary: str = ""
    public_artifact_type: str = "unknown"
    measures_kv_compression: bool = False
    measures_token_level_divergence: bool = False
    measures_first_divergence_index: bool = False
    measures_acceptance_rate: bool = False
    measures_avg_accepted_span: bool = False
    has_full_kv_verifier: bool = False
    supports_compressor_agnostic_comparison: bool = False
    has_public_leaderboard: bool = False
    has_runtime_serving_path: bool = False
    reports_speed_or_throughput: bool = False
    reports_memory_savings: bool = False
    measures_active_gpu_memory: bool = False
    relationship_to_exactkv: str = ""
    overlap_with_exactkv: str = ""
    exactkv_differentiator: str = ""
    claim_risk_level: RiskLevel = "medium"
    notes: str = ""
    evidence_status: EvidenceStatus = "source_pending"
    is_closest_conceptual_prior_art: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NoveltyClaim:
    claim: str
    status: ClaimStatus
    supporting_evidence: str = ""
    missing_evidence: str = ""
    safe_public_wording: str = ""
    unsafe_wording_to_avoid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExactKVCapability:
    capability: str
    evidence_artifact: str
    evidence_strength: EvidenceStrength
    limitations: str = ""
    public_claim_status: ClaimStatus = "allowed_with_qualification"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prior_art_catalog() -> list[PriorArtSystem]:
    """Curated prior-art records (Phase I). URLs from primary sources where verified."""
    return [
        PriorArtSystem(
            system_name="VeriCache",
            category="verifier_mediated_inference_system",
            source_urls=[
                "https://arxiv.org/abs/2605.17613",
                "https://arxiv.org/html/2605.17613",
            ],
            source_quality="primary",
            primary_goal="Lossless LLM inference with lossy KV compression via draft-verify",
            core_method_summary="Compressed KV drafts tokens; full KV verifies; system optimizes swap/offload for serving throughput.",
            public_artifact_type="paper",
            measures_kv_compression=True,
            measures_token_level_divergence=False,
            measures_first_divergence_index=False,
            measures_acceptance_rate=True,
            measures_avg_accepted_span=False,
            has_full_kv_verifier=True,
            supports_compressor_agnostic_comparison=False,
            has_public_leaderboard=False,
            has_runtime_serving_path=True,
            reports_speed_or_throughput=True,
            reports_memory_savings=True,
            measures_active_gpu_memory=True,
            relationship_to_exactkv="Closest conceptual prior art: compressed-KV draft + full-KV verification loop.",
            overlap_with_exactkv="Draft/verify/commit semantics and acceptance-oriented evaluation of compressed KV.",
            exactkv_differentiator="ExactKV is a public exactness benchmark/leaderboard platform, not a serving runtime.",
            claim_risk_level="high",
            notes="Paper reports up to ~4x throughput vs full-KV on tested serving setup; ExactKV does not reproduce these system results.",
            evidence_status="verified",
            is_closest_conceptual_prior_art=True,
        ),
        PriorArtSystem(
            system_name="KVQuant",
            category="kv_compression_method",
            source_urls=[
                "https://arxiv.org/abs/2401.18079",
                "https://github.com/SqueezeAILab/KVQuant",
            ],
            source_quality="primary",
            primary_goal="Low-precision KV cache quantization for long context",
            core_method_summary="Per-channel key / pre-RoPE / NUQ / dense-sparse quantization with CUDA kernels.",
            public_artifact_type="paper",
            measures_kv_compression=True,
            reports_speed_or_throughput=True,
            reports_memory_savings=True,
            relationship_to_exactkv="Compressor baseline / comparison target, not a drift leaderboard platform.",
            overlap_with_exactkv="KV quantization affects token generation; ExactKV can evaluate simquant adapter cells.",
            exactkv_differentiator="ExactKV compares compressors on token drift metrics across models.",
            claim_risk_level="low",
            evidence_status="verified",
        ),
        PriorArtSystem(
            system_name="KIVI",
            category="kv_compression_method",
            source_urls=[
                "https://arxiv.org/abs/2402.02750",
                "https://github.com/jy-yuan/KIVI",
            ],
            source_quality="primary",
            primary_goal="Tuning-free asymmetric 2-bit KV quantization",
            core_method_summary="Per-channel key + per-token value quantization with residual full-precision tail.",
            public_artifact_type="paper",
            measures_kv_compression=True,
            reports_speed_or_throughput=True,
            reports_memory_savings=True,
            relationship_to_exactkv="Compression method baseline; ExactKV has KIVI adapter experiments in repo history.",
            overlap_with_exactkv="Asymmetric KV quantization is directly relevant to ExactKV compressor panels.",
            exactkv_differentiator="ExactKV measures cross-compressor exactness, not KIVI throughput claims.",
            claim_risk_level="low",
            evidence_status="verified",
        ),
        PriorArtSystem(
            system_name="TurboQuant",
            category="kv_compression_method",
            source_urls=["https://github.com/TheTom/turboquant_plus"],
            source_quality="primary",
            primary_goal="KV cache quantization / compression implementation",
            core_method_summary="External TurboQuant adapter evaluated in ExactKV restricted panels.",
            public_artifact_type="repo",
            measures_kv_compression=True,
            relationship_to_exactkv="External compressor adapter target in ExactKV Phase 10.",
            overlap_with_exactkv="Same problem domain (KV compression).",
            exactkv_differentiator="ExactKV provides benchmark harness, not TurboQuant algorithm authorship.",
            claim_risk_level="low",
            evidence_status="verified",
        ),
        PriorArtSystem(
            system_name="QuantSpec",
            category="speculative_decoding_system",
            source_urls=[],
            source_quality="unknown",
            primary_goal="Quantization-aware speculative decoding (pending primary source fetch)",
            core_method_summary="source_pending — search: QuantSpec speculative decoding KV quantization arxiv",
            public_artifact_type="paper",
            measures_acceptance_rate=True,
            relationship_to_exactkv="Adjacent speculative/acceptance literature; not ExactKV's benchmark scope proof.",
            overlap_with_exactkv="Acceptance-style metrics may overlap conceptually.",
            exactkv_differentiator="ExactKV focuses on KV compression exactness across compressors.",
            claim_risk_level="medium",
            evidence_status="source_pending",
            notes="source_pending: add arXiv/GitHub URL after manual verification.",
        ),
        PriorArtSystem(
            system_name="SparseSpec",
            category="speculative_decoding_system",
            source_urls=[],
            source_quality="unknown",
            primary_goal="Sparse speculative decoding (pending primary source)",
            core_method_summary="source_pending — search: SparseSpec speculative decoding paper",
            public_artifact_type="paper",
            measures_acceptance_rate=True,
            relationship_to_exactkv="Adjacent speculative decoding prior art.",
            overlap_with_exactkv="Draft acceptance concepts.",
            exactkv_differentiator="ExactKV is compressor-agnostic KV drift benchmarking.",
            claim_risk_level="medium",
            evidence_status="source_pending",
        ),
        PriorArtSystem(
            system_name="SpecAttn",
            category="speculative_decoding_system",
            source_urls=[],
            source_quality="unknown",
            primary_goal="Speculative attention acceleration (pending source)",
            core_method_summary="source_pending — search: SpecAttn speculative attention LLM",
            public_artifact_type="paper",
            relationship_to_exactkv="May overlap on acceptance/speed; not established as drift leaderboard.",
            evidence_status="source_pending",
        ),
        PriorArtSystem(
            system_name="MagicDec",
            category="speculative_decoding_system",
            source_urls=[],
            source_quality="unknown",
            primary_goal="Speculative decoding variant (pending source)",
            core_method_summary="source_pending — search: MagicDec speculative decoding arxiv",
            public_artifact_type="paper",
            measures_acceptance_rate=True,
            relationship_to_exactkv="Adjacent speculative decoding prior art.",
            evidence_status="source_pending",
        ),
        PriorArtSystem(
            system_name="LMCache",
            category="kv_storage_or_offload_system",
            source_urls=[
                "https://arxiv.org/abs/2510.09665",
                "https://github.com/LMCache/LMCache",
            ],
            source_quality="primary",
            primary_goal="Enterprise KV cache offloading, reuse, and transfer across engines",
            core_method_summary="Tiered KV storage (CPU/disk/remote); vLLM/SGLang connectors; TTFT/throughput focus.",
            public_artifact_type="paper",
            measures_kv_compression=False,
            has_runtime_serving_path=True,
            reports_speed_or_throughput=True,
            reports_memory_savings=True,
            relationship_to_exactkv="Storage/offload system; ExactKV does not replace LMCache.",
            overlap_with_exactkv="Both concern KV caches in LLM inference.",
            exactkv_differentiator="ExactKV measures compression exactness, not prefix-cache hit rate or TTFT.",
            claim_risk_level="medium",
            evidence_status="verified",
        ),
        PriorArtSystem(
            system_name="CacheGen",
            category="kv_storage_or_offload_system",
            source_urls=[],
            source_quality="unknown",
            primary_goal="KV cache compression + storage for serving (pending source)",
            core_method_summary="source_pending — search: CacheGen KV cache compression storage LLM paper",
            public_artifact_type="paper",
            measures_kv_compression=True,
            has_runtime_serving_path=True,
            relationship_to_exactkv="Adjacent KV storage/compression serving work.",
            evidence_status="source_pending",
        ),
        PriorArtSystem(
            system_name="ShardCache (shard-kv)",
            category="cache_database_benchmark_system",
            source_urls=["https://github.com/d-tietjen/shard-kv"],
            source_quality="primary",
            primary_goal="High-throughput sharded KV store + LMCache storage backend benchmarks",
            core_method_summary="Embedded/TCP shardcache engine; Redis/Valkey-compatible server; LMCache plugin benchmarks.",
            public_artifact_type="repo",
            measures_kv_compression=False,
            measures_token_level_divergence=False,
            has_public_leaderboard=False,
            reports_speed_or_throughput=True,
            relationship_to_exactkv="Adjacent cache/storage benchmarking, not transformer token-drift exactness by default.",
            overlap_with_exactkv="Name collision risk: 'shard' in ExactKV is a probe adapter, not ShardCache product.",
            exactkv_differentiator="ExactKV evaluates LLM KV compression token drift; ShardCache benchmarks storage I/O.",
            claim_risk_level="high",
            notes="LMCache plugin rows measure payload GET/SET throughput — not necessarily compressed-attention token equivalence.",
            evidence_status="verified",
        ),
        PriorArtSystem(
            system_name="Redis / Valkey cache benchmarks",
            category="cache_database_benchmark_system",
            source_urls=["https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/benchmarks/"],
            source_quality="secondary",
            primary_goal="General-purpose in-memory datastore throughput/latency benchmarks",
            public_artifact_type="docs",
            measures_kv_compression=False,
            measures_token_level_divergence=False,
            relationship_to_exactkv="Orthogonal general KV datastore benchmarks.",
            overlap_with_exactkv="Both use 'KV' terminology.",
            exactkv_differentiator="ExactKV is transformer KV-cache exactness, not Redis object GET/SET.",
            claim_risk_level="low",
            evidence_status="ambiguous",
            notes="Redis docs are secondary pointer; not LLM-specific.",
        ),
        PriorArtSystem(
            system_name="GPTCache / semantic cache benchmarks",
            category="semantic_cache_benchmark_system",
            source_urls=["https://github.com/zilliztech/GPTCache"],
            source_quality="primary",
            primary_goal="Semantic cache for LLM API responses",
            public_artifact_type="repo",
            measures_kv_compression=False,
            measures_token_level_divergence=False,
            relationship_to_exactkv="Semantic response caching ≠ transformer KV compression drift.",
            overlap_with_exactkv="Both reduce redundant LLM work.",
            exactkv_differentiator="ExactKV measures per-token draft/verify drift under compression.",
            claim_risk_level="low",
            evidence_status="verified",
        ),
        PriorArtSystem(
            system_name="SpectralQuant",
            category="kv_compression_method",
            source_urls=[],
            source_quality="unknown",
            primary_goal="Spectral / frequency-domain KV compression (repo-dependent)",
            core_method_summary="ExactKV spectralquant_real adapter currently uses fallback when dependency unavailable.",
            public_artifact_type="unknown",
            measures_kv_compression=True,
            relationship_to_exactkv="External compressor comparison target via adapter.",
            overlap_with_exactkv="Same evaluation slot in ExactKV leaderboard.",
            exactkv_differentiator="ExactKV documents fallback/proxy mode honestly.",
            claim_risk_level="high",
            evidence_status="ambiguous",
            notes="spectralquant_available=False in current environment per Gate R0.",
        ),
    ]


def _default_claim_decisions() -> list[NoveltyClaim]:
    return [
        NoveltyClaim(
            claim="ExactKV is a KV-cache compression exactness benchmark.",
            status="allowed",
            supporting_evidence="Phase A/H+ panels, scale_7b/raw.json, unified leaderboard schema",
            safe_public_wording="ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression exactness.",
        ),
        NoveltyClaim(
            claim="ExactKV is a compressor-agnostic token-level drift leaderboard.",
            status="allowed",
            supporting_evidence="reports/scale_7b/leaderboard.json, Phase G FirstDivergenceAuthority",
            safe_public_wording="ExactKV publishes a public leaderboard ranking compressors by acceptance, divergence, and verifier agreement.",
        ),
        NoveltyClaim(
            claim="ExactKV is the first system like this.",
            status="forbidden",
            missing_evidence="No exhaustive prior-art survey proving uniqueness.",
            unsafe_wording_to_avoid="first ever, first system like this, nothing like this exists",
            safe_public_wording="ExactKV is a research-grade evaluation framework for cross-compressor KV exactness (not a uniqueness claim).",
        ),
        NoveltyClaim(
            claim="ExactKV reproduces VeriCache.",
            status="forbidden",
            supporting_evidence="docs/VERICACHE_PARITY_CLAIM_GATE.md — full parity forbidden",
            unsafe_wording_to_avoid="reproduces VeriCache, beats VeriCache",
            safe_public_wording="ExactKV is inspired by verifier-mediated compressed-KV ideas; it does not reproduce VeriCache serving throughput.",
        ),
        NoveltyClaim(
            claim="ExactKV invented compressed-KV verification.",
            status="forbidden",
            supporting_evidence="VeriCache arXiv:2605.17613 describes compressed-KV draft + full-KV verify",
            unsafe_wording_to_avoid="invented compressed-KV verification",
            safe_public_wording="ExactKV builds on known draft/verify semantics and adds a public exactness benchmark layer.",
        ),
        NoveltyClaim(
            claim="ExactKV measures first divergence across compressors.",
            status="allowed",
            supporting_evidence="Phase G FirstDivergenceAuthority in scale_7b cells",
            safe_public_wording="ExactKV reports canonical first_divergence_index per compressor and model.",
        ),
        NoveltyClaim(
            claim="ExactKV reports acceptance rate and accepted span across compressors.",
            status="allowed",
            supporting_evidence="Phase A metrics in raw.json cells",
            safe_public_wording="ExactKV reports token-level acceptance rate and accepted-span statistics in benchmark cells.",
        ),
        NoveltyClaim(
            claim="ExactKV proves end-to-end speedups.",
            status="forbidden",
            supporting_evidence="Phase F is kernel microbenchmark only",
            unsafe_wording_to_avoid="end-to-end speedup, faster inference",
            safe_public_wording="Phase F reports kernel microbenchmark speedups only (int8 ~1.63x, int4 ~1.54x on tested shape).",
        ),
        NoveltyClaim(
            claim="ExactKV proves active GPU memory savings.",
            status="forbidden",
            supporting_evidence="Compression ratios are stored-byte ratios unless active GPU memory measured",
            unsafe_wording_to_avoid="active GPU memory savings, VRAM savings",
        ),
        NoveltyClaim(
            claim="ExactKV has a real Triton KV compression kernel path.",
            status="allowed_with_qualification",
            supporting_evidence="reports/phaseF_kernel_benchmark.json — cuda, triton_available=true; block_sparse uses torch execution_backend",
            safe_public_wording="ExactKV includes a CUDA/Triton KV compression kernel microbenchmark path (tested shape/hardware only).",
        ),
        NoveltyClaim(
            claim="ExactKV is production ready.",
            status="forbidden",
            unsafe_wording_to_avoid="production ready, production serving",
            safe_public_wording="ExactKV is a research-grade evaluation framework, not a production serving runtime.",
        ),
        NoveltyClaim(
            claim="ExactKV is a research-grade evaluation framework.",
            status="allowed",
            supporting_evidence="Gate R0 evidence + novelty audit scope",
        ),
        NoveltyClaim(
            claim="ExactKV evaluates real 7B/8B models.",
            status="allowed",
            supporting_evidence="reports/scale_7b/scale_summary.json — 1500 cells, deterministic_mode=false",
            safe_public_wording="Phase H+ scale panel includes real GPU inference on Llama-3.1-8B and Mistral-7B-Instruct-v0.3.",
            missing_evidence="Sequential model execution on 50GB volume; not all compressors are full external integrations.",
        ),
        NoveltyClaim(
            claim="ExactKV compares real SpectralQuant.",
            status="forbidden",
            supporting_evidence="spectralquant_available=False; fallback mode",
            safe_public_wording="ExactKV includes SpectralQuant fallback/proxy adapter cells — not a verified real SpectralQuant integration.",
        ),
        NoveltyClaim(
            claim="ExactKV compares real Shard.",
            status="forbidden",
            supporting_evidence="shard_real probe_only=True",
            safe_public_wording="ExactKV includes Shard probe-first heuristic analysis — not a full Shard implementation.",
        ),
        NoveltyClaim(
            claim="ExactKV includes SpectralQuant fallback/proxy support.",
            status="allowed",
            supporting_evidence="exactkv/adapters/spectralquant_real_adapter.py",
        ),
        NoveltyClaim(
            claim="ExactKV includes Shard probe-first analysis.",
            status="allowed",
            supporting_evidence="exactkv/adapters/shard_real_adapter.py",
        ),
        NoveltyClaim(
            claim="ExactKV is a public benchmark platform.",
            status="allowed_with_qualification",
            supporting_evidence="reports/public_release/, scripts/exactkv.py CLI",
            safe_public_wording="ExactKV publishes reproducible benchmark artifacts and a public leaderboard bundle.",
            missing_evidence="Not a hosted SaaS; artifacts are repo-local reports.",
        ),
    ]


def _load_internal_evidence(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    paths = {
        "scale_summary": root / "reports/scale_7b/scale_summary.json",
        "release_evidence": root / "reports/release_evidence_status.json",
        "phase_f": root / "reports/phaseF_kernel_benchmark.json",
    }
    for key, path in paths.items():
        if path.is_file():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _capabilities_from_evidence(evidence: dict[str, Any]) -> list[ExactKVCapability]:
    scale = evidence.get("scale_summary") or {}
    pf = (evidence.get("release_evidence") or {}).get("phase_f_summary") or {}
    caps = [
        ExactKVCapability(
            capability="Real 7B/8B scale benchmark",
            evidence_artifact="reports/scale_7b/raw.json",
            evidence_strength="strong" if scale.get("total_cells") == 1500 else "moderate",
            limitations="Sequential run; float16; 50GB volume constraint",
            public_claim_status="allowed_with_qualification",
        ),
        ExactKVCapability(
            capability="Zero ExactKV failures on scale panel",
            evidence_artifact="reports/scale_7b/scale_summary.json",
            evidence_strength="strong" if scale.get("exactkv_failures") == 0 else "missing",
            public_claim_status="allowed",
        ),
        ExactKVCapability(
            capability="Triton kernel microbenchmark path",
            evidence_artifact="reports/phaseF_kernel_benchmark.json",
            evidence_strength="strong" if pf.get("int8_speedup_x") else "moderate",
            limitations="Kernel microbenchmark only; block_sparse torch path",
            public_claim_status="allowed_with_qualification",
        ),
        ExactKVCapability(
            capability="SpectralQuant real integration",
            evidence_artifact="spectralquant_available probe",
            evidence_strength="missing",
            limitations="Fallback/proxy mode in current environment",
            public_claim_status="forbidden",
        ),
        ExactKVCapability(
            capability="Shard real integration",
            evidence_artifact="shard_real adapter metadata",
            evidence_strength="weak",
            limitations="Probe-first heuristic only",
            public_claim_status="forbidden",
        ),
        ExactKVCapability(
            capability="Public leaderboard publication",
            evidence_artifact="reports/public_release/leaderboard_final.json",
            evidence_strength="strong",
            public_claim_status="allowed",
        ),
    ]
    return caps


def build_novelty_audit(root: Path | str = ".") -> dict[str, Any]:
    """Assemble full Phase I novelty audit report."""
    root = Path(root)
    prior_art = _prior_art_catalog()
    claims = _default_claim_decisions()
    evidence = _load_internal_evidence(root)
    capabilities = _capabilities_from_evidence(evidence)

    closest = [p for p in prior_art if p.is_closest_conceptual_prior_art]
    verified = [p for p in prior_art if p.evidence_status == "verified"]
    pending = [p for p in prior_art if p.evidence_status == "source_pending"]

    return {
        "phase_id": "phaseI_novelty_audit",
        "status": "novelty_audit_complete",
        "closest_prior_art": [p.system_name for p in closest],
        "prior_art_systems": [p.to_dict() for p in prior_art],
        "novelty_claims": [c.to_dict() for c in claims],
        "exactkv_capabilities": [c.to_dict() for c in capabilities],
        "internal_evidence_refs": list(evidence.keys()),
        "summary": {
            "prior_art_count": len(prior_art),
            "verified_sources": len(verified),
            "source_pending": len(pending),
            "claims_allowed": sum(1 for c in claims if c.status == "allowed"),
            "claims_qualified": sum(1 for c in claims if c.status == "allowed_with_qualification"),
            "claims_forbidden": sum(1 for c in claims if c.status == "forbidden"),
            "claims_needs_evidence": sum(1 for c in claims if c.status == "needs_more_evidence"),
        },
        "recommended_public_positioning": (
            "ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM "
            "KV-cache compression. It measures token-level drift, first divergence, acceptance "
            "rate, verifier agreement, and exactness failures across compressors and models."
        ),
    }


def write_novelty_audit_matrix_csv(prior_art: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "system_name",
        "category",
        "evidence_status",
        "source_quality",
        "measures_kv_compression",
        "measures_token_level_divergence",
        "measures_first_divergence_index",
        "measures_acceptance_rate",
        "has_full_kv_verifier",
        "supports_compressor_agnostic_comparison",
        "has_public_leaderboard",
        "has_runtime_serving_path",
        "reports_speed_or_throughput",
        "is_closest_conceptual_prior_art",
        "relationship_to_exactkv",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in prior_art:
            writer.writerow({k: row.get(k, "") for k in fields})


def render_novelty_audit_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    claims = report.get("novelty_claims") or []
    prior = report.get("prior_art_systems") or []
    closest = report.get("closest_prior_art") or []

    lines = [
        "# ExactKV Novelty Audit (Phase I)",
        "",
        "## 1. Executive summary",
        "",
        report.get("recommended_public_positioning", ""),
        "",
        f"- Prior-art systems catalogued: **{summary.get('prior_art_count', 0)}**",
        f"- Primary sources verified: **{summary.get('verified_sources', 0)}**",
        f"- Sources pending: **{summary.get('source_pending', 0)}**",
        f"- Allowed claims: **{summary.get('claims_allowed', 0)}**; qualified: **{summary.get('claims_qualified', 0)}**; forbidden: **{summary.get('claims_forbidden', 0)}**",
        "",
        "## 2. What ExactKV is",
        "",
        "A research-grade, compressor-agnostic evaluation framework and public leaderboard for LLM KV-cache compression exactness.",
        "",
        "## 3. What ExactKV is not",
        "",
        "- Not a production serving system",
        "- Not a VeriCache reproduction or throughput competitor",
        "- Not proof of end-to-end inference speedups or active GPU memory savings",
        "- Not a verified real SpectralQuant or full Shard integration in the current environment",
        "",
        "## 4. Closest prior art",
        "",
        ", ".join(f"**{n}**" for n in closest) or "VeriCache (pending confirmation)",
        "",
        "## 5. Prior-art matrix",
        "",
        "See `reports/novelty_audit_matrix.csv` for the full table.",
        "",
        "| System | Category | Evidence | Overlap | Differentiator |",
        "|--------|----------|----------|---------|----------------|",
    ]
    for p in prior[:12]:
        lines.append(
            f"| {p.get('system_name')} | {p.get('category')} | {p.get('evidence_status')} | "
            f"{(p.get('overlap_with_exactkv') or '')[:60]} | {(p.get('exactkv_differentiator') or '')[:60]} |",
        )
    lines.extend([
        "",
        "## 6. VeriCache relationship",
        "",
        "VeriCache is the closest conceptual prior art: compressed-KV draft + full-KV verification for lossless inference with serving optimizations. ExactKV must **not** claim to invent this loop or reproduce VeriCache throughput/memory results.",
        "",
        "## 7. ShardCache / shard-kv relationship",
        "",
        "ShardCache (shard-kv) is primarily a **cache database / LMCache storage benchmark** system. It is adjacent but not equivalent to transformer KV-cache token-drift exactness benchmarking unless primary evidence shows otherwise (none verified here).",
        "",
        "## 8. External compressor relationship",
        "",
        "KVQuant, KIVI, TurboQuant, and SpectralQuant are **compression methods** or adapter targets. ExactKV compares them in a benchmark harness; it does not subsume their algorithms.",
        "",
        "## 9. Storage/offload system relationship",
        "",
        "LMCache and CacheGen focus on KV **storage, reuse, and serving**. ExactKV does not replace them; overlap is limited to shared KV terminology.",
        "",
        "## 10. Speculative decoding relationship",
        "",
        "QuantSpec / SparseSpec / MagicDec / SpecAttn are adjacent speculative-decoding literature. ExactKV must not claim invention of acceptance measurement without qualification.",
        "",
        "## 11. ExactKV defensible novelty",
        "",
        "- Public, compressor-agnostic **token-level drift** and **first-divergence** leaderboard",
        "- Phase G canonical divergence authority across cells",
        "- Reproducible artifact pipeline (benchmark → leaderboard → public_release)",
        "- Real 7B/8B scale panel with zero ExactKV failures (current evidence)",
        "",
        "## 12. Claims allowed",
        "",
    ])
    for c in claims:
        if c.get("status") in ("allowed", "allowed_with_qualification"):
            lines.append(f"- **{c.get('claim')}** — {c.get('status')}")
    lines.extend(["", "## 13. Claims requiring qualification", ""])
    for c in claims:
        if c.get("status") == "allowed_with_qualification":
            lines.append(f"- {c.get('claim')}: {c.get('safe_public_wording')}")
    lines.extend(["", "## 14. Claims forbidden", ""])
    for c in claims:
        if c.get("status") == "forbidden":
            lines.append(f"- {c.get('claim')}")
    lines.extend([
        "",
        "## 15. Remaining uncertainty",
        "",
        "- QuantSpec, SparseSpec, MagicDec, SpecAttn, CacheGen: **source_pending**",
        "- SpectralQuant real dependency not available in current environment",
        "- Uniqueness vs all exactness benchmarks: **not established** — do not claim 'first'",
        "",
        "## 16. Recommended public positioning",
        "",
        report.get("recommended_public_positioning", ""),
        "",
    ])
    return "\n".join(lines)
