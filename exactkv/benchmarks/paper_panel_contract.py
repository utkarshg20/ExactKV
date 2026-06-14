"""Paper-like reproduction panel contracts (Phase 11J).

Metadata for a future VeriCache-parity evaluation panel. **Does not** run
experiments or wire into ``ExactKVGenerator``.

This is a panel contract, not a reproduction result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

_CLAIM_NOTE = (
    "Paper-like reproduction panel contract (Phase 11J). Planning metadata only — "
    "not a reproduction result. ExactKV has not reproduced VeriCache throughput, "
    "memory benefits, or production-scale deployment. External paper numbers are "
    "not ExactKV results."
)

_ALLOWED_CLAIMS = (
    "Paper-like reproduction panel contract exists",
    "Panel dimensions and claim gates documented for future VeriCache parity work",
    "ExactKV tested-panel exactness may be cited with panel name — not paper reproduction",
    "Restricted Shard/SpectralQuant rows are not paper-equivalent compressor reproduction",
)

_FORBIDDEN_CLAIMS = (
    "speedup",
    "latency improvement",
    "throughput improvement",
    "memory savings",
    "production serving",
    "vericache reproduction complete",
    "vericache throughput reproduced",
    "vericache memory benefits reproduced",
    "paper numbers as exactkv results",
)

_NEGATION_PREFIXES = ("no ", "not ", "without ", "never ", "non-", "does not ")


class PaperPanelStatus(str, Enum):
    """Lifecycle status for the paper-like reproduction panel."""

    CONTRACT_ONLY = "CONTRACT_ONLY"
    BLOCKED = "BLOCKED"
    READY_TO_RUN = "READY_TO_RUN"
    RUN_COMPLETE_UNVERIFIED = "RUN_COMPLETE_UNVERIFIED"
    CLAIM_ELIGIBLE = "CLAIM_ELIGIBLE"


class PaperPanelDimension(str, Enum):
    """Panel dimension categories tracked by the contract."""

    MODEL = "MODEL"
    COMPRESSOR = "COMPRESSOR"
    WORKLOAD = "WORKLOAD"
    METRIC = "METRIC"
    HARDWARE = "HARDWARE"
    RUNTIME_BACKEND = "RUNTIME_BACKEND"


@dataclass
class PaperPanelModelSpec:
    """Model row in the reproduction panel."""

    model_id: str
    parameter_scale: str
    access_requirement: str
    status: str
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperPanelModelSpec:
        return cls(
            model_id=str(data["model_id"]),
            parameter_scale=str(data.get("parameter_scale", "")),
            access_requirement=str(data.get("access_requirement", "")),
            status=str(data.get("status", "blocked")),
            blocker=str(data.get("blocker", "")),
        )


@dataclass
class PaperPanelCompressorSpec:
    """Compressor row in the reproduction panel."""

    compressor_name: str
    implementation_status: str
    real_or_simulated: str
    claim_note: str
    paper_equivalent_real_backend: bool = False
    paper_equivalent_integrated_compressor: bool = False
    claims_active_memory_savings: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperPanelCompressorSpec:
        return cls(
            compressor_name=str(data["compressor_name"]),
            implementation_status=str(data.get("implementation_status", "not_implemented")),
            real_or_simulated=str(data.get("real_or_simulated", "unknown")),
            claim_note=str(data.get("claim_note", "")),
            paper_equivalent_real_backend=bool(
                data.get("paper_equivalent_real_backend", False)
            ),
            paper_equivalent_integrated_compressor=bool(
                data.get("paper_equivalent_integrated_compressor", False)
            ),
            claims_active_memory_savings=bool(
                data.get("claims_active_memory_savings", False)
            ),
        )


@dataclass
class PaperPanelWorkloadSpec:
    """Workload/benchmark row in the reproduction panel."""

    workload_name: str
    exact_match_to_vericache_paper: bool
    implemented: bool
    blocker: str = ""
    claim_note: str = ""
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperPanelWorkloadSpec:
        return cls(
            workload_name=str(data["workload_name"]),
            exact_match_to_vericache_paper=bool(
                data.get("exact_match_to_vericache_paper", False)
            ),
            implemented=bool(data.get("implemented", False)),
            blocker=str(data.get("blocker", "")),
            claim_note=str(data.get("claim_note", "")),
            required=bool(data.get("required", True)),
        )


@dataclass
class PaperPanelMetricSpec:
    """Metric row required for claim eligibility."""

    metric_name: str
    required: bool
    implemented: bool
    claim_allowed: bool
    claim_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperPanelMetricSpec:
        return cls(
            metric_name=str(data["metric_name"]),
            required=bool(data.get("required", True)),
            implemented=bool(data.get("implemented", False)),
            claim_allowed=bool(data.get("claim_allowed", False)),
            claim_note=str(data.get("claim_note", "")),
        )


@dataclass
class PaperLikeReproductionPanel:
    """Serializable paper-like VeriCache reproduction panel contract."""

    status: PaperPanelStatus
    models: list[PaperPanelModelSpec] = field(default_factory=list)
    compressors: list[PaperPanelCompressorSpec] = field(default_factory=list)
    workloads: list[PaperPanelWorkloadSpec] = field(default_factory=list)
    metrics: list[PaperPanelMetricSpec] = field(default_factory=list)
    hardware_requirements: dict[str, Any] = field(default_factory=dict)
    runtime_requirements: dict[str, Any] = field(default_factory=dict)
    exactness_gate_required: bool = True
    exactness_gate_passed: bool = False
    throughput_gate_required: bool = True
    throughput_gate_passed: bool = False
    memory_gate_required: bool = True
    memory_gate_passed: bool = False
    runtime_gate_passed: bool = False
    allowed_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    claim_note: str = _CLAIM_NOTE
    paper_numbers_as_exactkv_results: bool = False
    claim_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "models": [m.to_dict() for m in self.models],
            "compressors": [c.to_dict() for c in self.compressors],
            "workloads": [w.to_dict() for w in self.workloads],
            "metrics": [m.to_dict() for m in self.metrics],
            "hardware_requirements": dict(self.hardware_requirements),
            "runtime_requirements": dict(self.runtime_requirements),
            "exactness_gate_required": self.exactness_gate_required,
            "exactness_gate_passed": self.exactness_gate_passed,
            "throughput_gate_required": self.throughput_gate_required,
            "throughput_gate_passed": self.throughput_gate_passed,
            "memory_gate_required": self.memory_gate_required,
            "memory_gate_passed": self.memory_gate_passed,
            "runtime_gate_passed": self.runtime_gate_passed,
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "claim_note": self.claim_note,
            "paper_numbers_as_exactkv_results": self.paper_numbers_as_exactkv_results,
            "claim_eligible": self.claim_eligible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperLikeReproductionPanel:
        return cls(
            status=PaperPanelStatus(data["status"]),
            models=[PaperPanelModelSpec.from_dict(m) for m in data.get("models", [])],
            compressors=[
                PaperPanelCompressorSpec.from_dict(c) for c in data.get("compressors", [])
            ],
            workloads=[
                PaperPanelWorkloadSpec.from_dict(w) for w in data.get("workloads", [])
            ],
            metrics=[PaperPanelMetricSpec.from_dict(m) for m in data.get("metrics", [])],
            hardware_requirements=dict(data.get("hardware_requirements", {})),
            runtime_requirements=dict(data.get("runtime_requirements", {})),
            exactness_gate_required=bool(data.get("exactness_gate_required", True)),
            exactness_gate_passed=bool(data.get("exactness_gate_passed", False)),
            throughput_gate_required=bool(data.get("throughput_gate_required", True)),
            throughput_gate_passed=bool(data.get("throughput_gate_passed", False)),
            memory_gate_required=bool(data.get("memory_gate_required", True)),
            memory_gate_passed=bool(data.get("memory_gate_passed", False)),
            runtime_gate_passed=bool(data.get("runtime_gate_passed", False)),
            allowed_claims=list(data.get("allowed_claims", [])),
            forbidden_claims=list(data.get("forbidden_claims", [])),
            claim_note=str(data.get("claim_note", _CLAIM_NOTE)),
            paper_numbers_as_exactkv_results=bool(
                data.get("paper_numbers_as_exactkv_results", False)
            ),
            claim_eligible=bool(data.get("claim_eligible", False)),
        )


def _default_models() -> list[PaperPanelModelSpec]:
    return [
        PaperPanelModelSpec(
            model_id="Qwen/Qwen2.5-0.5B",
            parameter_scale="0.5B",
            access_requirement="HF public weights",
            status="partial_exactness_on_v10_panels",
            blocker="Not paper model set at paper context lengths",
        ),
        PaperPanelModelSpec(
            model_id="meta-llama/Llama-3.1-8B",
            parameter_scale="8B",
            access_requirement="HF gated weights",
            status="partial_small_suite_only",
            blocker="Exp 033 12-prompt suite only — not paper panel scale",
        ),
        PaperPanelModelSpec(
            model_id="paper_scale_model_placeholder",
            parameter_scale="paper_scale",
            access_requirement="TBD",
            status="not_implemented",
            blocker="Paper model/context-length matrix not defined in ExactKV",
        ),
    ]


def _default_compressors() -> list[PaperPanelCompressorSpec]:
    return [
        PaperPanelCompressorSpec(
            compressor_name="int8",
            implementation_status="built_in_default",
            real_or_simulated="real_int8_path",
            claim_note="ExactKV built-in; tested on V10 panels — not paper compressor matrix",
            paper_equivalent_real_backend=False,
        ),
        PaperPanelCompressorSpec(
            compressor_name="int4_sim",
            implementation_status="built_in_default",
            real_or_simulated="simulated",
            claim_note="Simulated INT4 in int8 containers — not paper-equivalent real backend",
            paper_equivalent_real_backend=False,
        ),
        PaperPanelCompressorSpec(
            compressor_name="k8_v4_sim",
            implementation_status="built_in_default",
            real_or_simulated="simulated",
            claim_note="Simulated asymmetric K/V — not paper-equivalent real backend",
            paper_equivalent_real_backend=False,
        ),
        PaperPanelCompressorSpec(
            compressor_name="paper_compressor_set_placeholder",
            implementation_status="not_implemented",
            real_or_simulated="unknown",
            claim_note="VeriCache paper compressor matrix not reproduced in ExactKV",
            paper_equivalent_real_backend=False,
        ),
        PaperPanelCompressorSpec(
            compressor_name="shard_external_drafter",
            implementation_status="restricted_probe",
            real_or_simulated="restricted_external",
            claim_note=(
                "Shard external-drafter restricted probe (Exp 039–041) — "
                "not paper-equivalent integrated compressor"
            ),
            paper_equivalent_real_backend=False,
            paper_equivalent_integrated_compressor=False,
        ),
        PaperPanelCompressorSpec(
            compressor_name="spectralquant_experimental",
            implementation_status="restricted_probe",
            real_or_simulated="materializing_adapter",
            claim_note=(
                "SpectralQuant materializing factory-only adapter (Exp 044–045) — "
                "no active memory savings; not paper-equivalent integrated compressor"
            ),
            paper_equivalent_real_backend=False,
            paper_equivalent_integrated_compressor=False,
            claims_active_memory_savings=False,
        ),
    ]


def _default_workloads() -> list[PaperPanelWorkloadSpec]:
    return [
        PaperPanelWorkloadSpec(
            workload_name="v10_crash_test_suites",
            exact_match_to_vericache_paper=False,
            implemented=True,
            claim_note="ExactKV custom V10 suites — not VeriCache paper workloads",
        ),
        PaperPanelWorkloadSpec(
            workload_name="paper_long_context_workload",
            exact_match_to_vericache_paper=False,
            implemented=False,
            blocker="Paper long-context benchmark not implemented",
            claim_note="Not exact VeriCache paper workload",
        ),
        PaperPanelWorkloadSpec(
            workload_name="paper_remote_prefix_workload",
            exact_match_to_vericache_paper=False,
            implemented=False,
            blocker="Remote prefix runtime not implemented (Phase 11H loopback only)",
            claim_note="Not exact VeriCache paper remote-prefix workload",
        ),
        PaperPanelWorkloadSpec(
            workload_name="paper_quality_workload_panel",
            exact_match_to_vericache_paper=False,
            implemented=False,
            blocker="Official paper QA/benchmark panel not implemented",
            claim_note="LongBench-style demo is not official LongBench — not paper-equivalent",
        ),
    ]


def _default_metrics() -> list[PaperPanelMetricSpec]:
    return [
        PaperPanelMetricSpec(
            metric_name="exactness_exactkv_failures",
            required=True,
            implemented=True,
            claim_allowed=True,
            claim_note="On cited panels only when exactkv_failures == 0",
        ),
        PaperPanelMetricSpec(
            metric_name="acceptance_rate",
            required=True,
            implemented=True,
            claim_allowed=True,
            claim_note="Panel-bound mean acceptance — not universal ranking",
        ),
        PaperPanelMetricSpec(
            metric_name="draft_divergence",
            required=True,
            implemented=True,
            claim_allowed=True,
            claim_note="Forensics/drift metrics on tested panels",
        ),
        PaperPanelMetricSpec(
            metric_name="throughput_tokens_per_second",
            required=True,
            implemented=False,
            claim_allowed=False,
            claim_note="Exp 030 diagnostic only — ExactKV slower; not claim-ready",
        ),
        PaperPanelMetricSpec(
            metric_name="memory_active_gpu_accounting",
            required=True,
            implemented=False,
            claim_allowed=False,
            claim_note="Exp 031 diagnostic — no active VRAM savings at tested scale",
        ),
    ]


def build_default_paper_like_panel() -> PaperLikeReproductionPanel:
    """Factory for Phase 11J conservative CONTRACT_ONLY panel."""
    return PaperLikeReproductionPanel(
        status=PaperPanelStatus.CONTRACT_ONLY,
        models=_default_models(),
        compressors=_default_compressors(),
        workloads=_default_workloads(),
        metrics=_default_metrics(),
        hardware_requirements={
            "gpu_metadata_required": True,
            "dtype_required": True,
            "implemented": False,
            "blocker": "Paper panel hardware matrix not locked",
        },
        runtime_requirements={
            "hf_offline_harness": True,
            "vllm_integrated": False,
            "lmcache_integrated": False,
            "remote_prefix_runtime": False,
            "serving_runtime": False,
            "blocker": "Paper runtime backend (vLLM/LMCache/serving) not implemented",
        },
        exactness_gate_required=True,
        exactness_gate_passed=False,
        throughput_gate_required=True,
        throughput_gate_passed=False,
        memory_gate_required=True,
        memory_gate_passed=False,
        runtime_gate_passed=False,
        allowed_claims=list(_ALLOWED_CLAIMS),
        forbidden_claims=list(_FORBIDDEN_CLAIMS),
        claim_note=_CLAIM_NOTE,
        paper_numbers_as_exactkv_results=False,
        claim_eligible=False,
    )


def _encodes_positive_forbidden_claim(text_lower: str, term: str) -> bool:
    start = 0
    while True:
        pos = text_lower.find(term, start)
        if pos == -1:
            return False
        window = text_lower[max(0, pos - 40):pos]
        if not any(neg in window for neg in _NEGATION_PREFIXES):
            return True
        start = pos + len(term)


def _claim_encodes_performance(text: str) -> bool:
    lower = text.lower()
    terms = (
        "speedup",
        "throughput improvement",
        "latency improvement",
        "memory savings",
        "production serving",
    )
    return any(_encodes_positive_forbidden_claim(lower, t) for t in terms)


def validate_paper_panel_compressor(spec: PaperPanelCompressorSpec) -> list[str]:
    errors: list[str] = []
    if spec.real_or_simulated.lower() in ("simulated", "sim") and spec.paper_equivalent_real_backend:
        errors.append(
            f"{spec.compressor_name}: simulated compressor cannot be paper-equivalent real backend"
        )
    if "shard" in spec.compressor_name.lower() and spec.paper_equivalent_integrated_compressor:
        errors.append(
            f"{spec.compressor_name}: Shard external-drafter cannot be paper-equivalent integrated compressor"
        )
    if (
        "spectralquant" in spec.compressor_name.lower()
        and spec.claims_active_memory_savings
    ):
        errors.append(
            f"{spec.compressor_name}: SpectralQuant materializing adapter cannot claim active memory savings"
        )
    if not spec.claim_note.strip():
        errors.append(f"{spec.compressor_name}: claim_note required")
    return errors


def validate_paper_panel_workload(spec: PaperPanelWorkloadSpec) -> list[str]:
    errors: list[str] = []
    if not spec.exact_match_to_vericache_paper:
        note = spec.claim_note.lower()
        if not note.strip():
            errors.append(
                f"{spec.workload_name}: non-paper workload requires claim_note caveat"
            )
        elif "not" not in note and "paper" not in note:
            errors.append(
                f"{spec.workload_name}: non-paper workload requires not/paper caveat in claim_note"
            )
    return errors


def validate_paper_like_reproduction_panel(panel: PaperLikeReproductionPanel) -> list[str]:
    """Return human-readable panel invariant violations."""
    errors: list[str] = []

    if panel.paper_numbers_as_exactkv_results:
        errors.append("paper_numbers_as_exactkv_results must remain False")

    if not panel.claim_note.strip():
        errors.append("claim_note required on paper-like panel")

    if panel.status is PaperPanelStatus.CLAIM_ELIGIBLE and not panel.claim_eligible:
        errors.append("CLAIM_ELIGIBLE status requires claim_eligible=True")

    if panel.claim_eligible and panel.status is not PaperPanelStatus.CLAIM_ELIGIBLE:
        errors.append("claim_eligible=True requires status CLAIM_ELIGIBLE")

    if panel.status is PaperPanelStatus.CLAIM_ELIGIBLE or panel.claim_eligible:
        if panel.exactness_gate_required and not panel.exactness_gate_passed:
            errors.append("CLAIM_ELIGIBLE requires exactness_gate_passed")
        if panel.throughput_gate_required and not panel.throughput_gate_passed:
            errors.append("CLAIM_ELIGIBLE requires throughput_gate_passed")
        if panel.memory_gate_required and not panel.memory_gate_passed:
            errors.append("CLAIM_ELIGIBLE requires memory_gate_passed")
        if not panel.runtime_gate_passed:
            errors.append("CLAIM_ELIGIBLE requires runtime_gate_passed")

        for metric in panel.metrics:
            if metric.required and not metric.implemented:
                errors.append(
                    f"required metric not implemented: {metric.metric_name}"
                )
            if metric.required and not metric.claim_allowed:
                errors.append(
                    f"required metric not claim-allowed: {metric.metric_name}"
                )

        for workload in panel.workloads:
            if workload.required and not workload.implemented:
                errors.append(
                    f"required workload not implemented: {workload.workload_name}"
                )

        if not panel.hardware_requirements.get("implemented", False):
            errors.append("CLAIM_ELIGIBLE requires hardware_requirements.implemented")

    if panel.status is not PaperPanelStatus.CLAIM_ELIGIBLE:
        for claim in panel.allowed_claims:
            if _claim_encodes_performance(claim):
                errors.append(
                    "speed/throughput/memory/serving claims forbidden unless "
                    "status is CLAIM_ELIGIBLE with gates passed"
                )

    for term in _FORBIDDEN_CLAIMS:
        if term not in panel.forbidden_claims:
            errors.append(f"forbidden_claims must include: {term}")

    note_lower = panel.claim_note.lower()
    for term in _FORBIDDEN_CLAIMS:
        if _encodes_positive_forbidden_claim(note_lower, term):
            errors.append(f"claim_note must not encode positive forbidden claim: {term}")

    for compressor in panel.compressors:
        errors.extend(validate_paper_panel_compressor(compressor))

    for workload in panel.workloads:
        errors.extend(validate_paper_panel_workload(workload))

    return errors
