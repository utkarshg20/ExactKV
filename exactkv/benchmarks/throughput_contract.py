"""Throughput benchmark harness contracts (Phase 11I).

Methodology metadata for diagnostic timing and future throughput claims.
**Does not** run model inference or wire into ``ExactKVGenerator``.

This is a benchmarking methodology and contract layer, not a throughput result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

_CLAIM_NOTE = (
    "Throughput benchmark harness contract (Phase 11I). Methodology metadata only — "
    "not a throughput result. Current ExactKV timing diagnostics do not support a "
    "speedup claim. No performance, deployment, or resource-usage claims."
)

_ALLOWED_CLAIMS = (
    "Throughput benchmark methodology contract exists",
    "Diagnostic timing numbers panel-bound with exactness gate cited",
    "Exactness gate required before any performance claim",
    "Negative or neutral timing results must be reported honestly",
)

_FORBIDDEN_CLAIMS = (
    "speedup",
    "latency improvement",
    "throughput improvement",
    "memory savings",
    "production serving",
    "vericache throughput reproduced",
)

_NEGATION_PREFIXES = ("no ", "not ", "without ", "never ", "non-", "does not ")

_MIN_SAMPLES_FOR_CLAIM = 3


class ThroughputBenchmarkMode(str, Enum):
    """Benchmark execution context (metadata only for placeholders)."""

    OFFLINE_SINGLE_REQUEST = "OFFLINE_SINGLE_REQUEST"
    BATCHED_PLACEHOLDER = "BATCHED_PLACEHOLDER"
    SERVING_PLACEHOLDER = "SERVING_PLACEHOLDER"
    REMOTE_PREFIX_PLACEHOLDER = "REMOTE_PREFIX_PLACEHOLDER"


class TimingMetricKind(str, Enum):
    """Named timing metrics required or reported by the harness."""

    TOKENS_PER_SECOND = "TOKENS_PER_SECOND"
    SECONDS_PER_TOKEN = "SECONDS_PER_TOKEN"
    PREFILL_SECONDS = "PREFILL_SECONDS"
    DECODE_SECONDS = "DECODE_SECONDS"
    VERIFY_SECONDS = "VERIFY_SECONDS"
    TOTAL_SECONDS = "TOTAL_SECONDS"


class ThroughputClaimStatus(str, Enum):
    """What performance language is permitted for a plan or result."""

    NOT_MEASURED = "NOT_MEASURED"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    NEGATIVE_OR_NEUTRAL = "NEGATIVE_OR_NEUTRAL"
    POSITIVE_BUT_UNVERIFIED = "POSITIVE_BUT_UNVERIFIED"
    CLAIM_ALLOWED = "CLAIM_ALLOWED"


@dataclass
class ThroughputBenchmarkPlan:
    """Serializable throughput benchmark methodology plan."""

    mode: ThroughputBenchmarkMode
    metrics_required: list[TimingMetricKind] = field(default_factory=list)
    exactness_gate_required: bool = True
    warmup_required: bool = True
    synchronization_required: bool = True
    sample_count: int = 3
    batch_size: int = 1
    model: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    baseline_arm: str = ""
    allowed_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    claim_status: ThroughputClaimStatus = ThroughputClaimStatus.DIAGNOSTIC_ONLY
    claim_note: str = _CLAIM_NOTE
    runtime_placeholder_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "metrics_required": [m.value for m in self.metrics_required],
            "exactness_gate_required": self.exactness_gate_required,
            "warmup_required": self.warmup_required,
            "synchronization_required": self.synchronization_required,
            "sample_count": self.sample_count,
            "batch_size": self.batch_size,
            "model": self.model,
            "hardware": dict(self.hardware),
            "baseline_arm": self.baseline_arm,
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "claim_status": self.claim_status.value,
            "claim_note": self.claim_note,
            "runtime_placeholder_active": self.runtime_placeholder_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThroughputBenchmarkPlan:
        return cls(
            mode=ThroughputBenchmarkMode(data["mode"]),
            metrics_required=[
                TimingMetricKind(v) for v in data.get("metrics_required", [])
            ],
            exactness_gate_required=bool(data.get("exactness_gate_required", True)),
            warmup_required=bool(data.get("warmup_required", True)),
            synchronization_required=bool(data.get("synchronization_required", True)),
            sample_count=int(data.get("sample_count", 3)),
            batch_size=int(data.get("batch_size", 1)),
            model=str(data.get("model", "")),
            hardware=dict(data.get("hardware", {})),
            baseline_arm=str(data.get("baseline_arm", "")),
            allowed_claims=list(data.get("allowed_claims", [])),
            forbidden_claims=list(data.get("forbidden_claims", [])),
            claim_status=ThroughputClaimStatus(
                data.get("claim_status", ThroughputClaimStatus.DIAGNOSTIC_ONLY.value)
            ),
            claim_note=str(data.get("claim_note", _CLAIM_NOTE)),
            runtime_placeholder_active=bool(data.get("runtime_placeholder_active", False)),
        )


@dataclass
class ThroughputDiagnosticResult:
    """JSON-friendly diagnostic timing result (metadata schema)."""

    plan: ThroughputBenchmarkPlan
    measured: bool
    exactness_passed: bool
    baseline_name: str
    candidate_name: str
    total_seconds: float
    generated_tokens: int
    tokens_per_second: float
    verify_seconds: float = 0.0
    decode_seconds: float = 0.0
    notes: str = ""
    claim_status: ThroughputClaimStatus = ThroughputClaimStatus.DIAGNOSTIC_ONLY
    hide_negative_results: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "measured": self.measured,
            "exactness_passed": self.exactness_passed,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "total_seconds": self.total_seconds,
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": self.tokens_per_second,
            "verify_seconds": self.verify_seconds,
            "decode_seconds": self.decode_seconds,
            "notes": self.notes,
            "claim_status": self.claim_status.value,
            "hide_negative_results": self.hide_negative_results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThroughputDiagnosticResult:
        return cls(
            plan=ThroughputBenchmarkPlan.from_dict(data["plan"]),
            measured=bool(data.get("measured", False)),
            exactness_passed=bool(data.get("exactness_passed", False)),
            baseline_name=str(data.get("baseline_name", "")),
            candidate_name=str(data.get("candidate_name", "")),
            total_seconds=float(data.get("total_seconds", 0.0)),
            generated_tokens=int(data.get("generated_tokens", 0)),
            tokens_per_second=float(data.get("tokens_per_second", 0.0)),
            verify_seconds=float(data.get("verify_seconds", 0.0)),
            decode_seconds=float(data.get("decode_seconds", 0.0)),
            notes=str(data.get("notes", "")),
            claim_status=ThroughputClaimStatus(
                data.get("claim_status", ThroughputClaimStatus.DIAGNOSTIC_ONLY.value)
            ),
            hide_negative_results=bool(data.get("hide_negative_results", False)),
        )

    def baseline_tokens_per_second(self) -> float | None:
        """Parse baseline tok/s from notes when encoded as ``baseline_tps=``."""
        for part in self.notes.split(";"):
            part = part.strip()
            if part.startswith("baseline_tps="):
                try:
                    return float(part.split("=", 1)[1])
                except ValueError:
                    return None
        return None


def _default_metrics() -> list[TimingMetricKind]:
    return [
        TimingMetricKind.TOKENS_PER_SECOND,
        TimingMetricKind.TOTAL_SECONDS,
        TimingMetricKind.VERIFY_SECONDS,
        TimingMetricKind.DECODE_SECONDS,
    ]


def build_default_diagnostic_plan() -> ThroughputBenchmarkPlan:
    """Factory for Phase 11I default diagnostic-only methodology plan."""
    return ThroughputBenchmarkPlan(
        mode=ThroughputBenchmarkMode.OFFLINE_SINGLE_REQUEST,
        metrics_required=_default_metrics(),
        exactness_gate_required=True,
        warmup_required=True,
        synchronization_required=True,
        sample_count=3,
        batch_size=1,
        model="",
        hardware={},
        baseline_arm="full_greedy",
        allowed_claims=list(_ALLOWED_CLAIMS),
        forbidden_claims=list(_FORBIDDEN_CLAIMS),
        claim_status=ThroughputClaimStatus.DIAGNOSTIC_ONLY,
        claim_note=_CLAIM_NOTE,
        runtime_placeholder_active=False,
    )


def build_diagnostic_result_stub(
    *,
    negative_vs_baseline: bool = True,
) -> ThroughputDiagnosticResult:
    """Blocked diagnostic JSON stub — no model inference."""
    plan = build_default_diagnostic_plan()
    # Exp 030 illustrative ordering only; stub is not a live measurement.
    baseline_tps = 54.4
    candidate_tps = 20.4 if negative_vs_baseline else 54.4
    status = (
        ThroughputClaimStatus.NEGATIVE_OR_NEUTRAL
        if candidate_tps < baseline_tps
        else ThroughputClaimStatus.DIAGNOSTIC_ONLY
    )
    return ThroughputDiagnosticResult(
        plan=plan,
        measured=False,
        exactness_passed=True,
        baseline_name="full_greedy",
        candidate_name="exactkv_sequential",
        total_seconds=1.5,
        generated_tokens=32,
        tokens_per_second=candidate_tps,
        verify_seconds=0.8,
        decode_seconds=0.7,
        notes=(
            f"methodology_stub; baseline_tps={baseline_tps}; "
            "Exp 030 panel reference — not a live measurement"
        ),
        claim_status=status,
        hide_negative_results=False,
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


def _claim_encodes_speed_or_throughput(text: str) -> bool:
    lower = text.lower()
    terms = (
        "speedup",
        "faster than",
        "throughput improvement",
        "latency improvement",
        "beats full greedy",
    )
    return any(_encodes_positive_forbidden_claim(lower, t) for t in terms)


def validate_throughput_benchmark_plan(plan: ThroughputBenchmarkPlan) -> list[str]:
    """Return human-readable plan invariant violations."""
    errors: list[str] = []

    if plan.sample_count <= 0:
        errors.append("sample_count must be positive for measured plans")

    if not plan.metrics_required:
        errors.append("metrics_required must list explicit TimingMetricKind values")

    if plan.claim_status is ThroughputClaimStatus.DIAGNOSTIC_ONLY:
        if not plan.claim_note.strip():
            errors.append("diagnostic-only plan requires claim_note caveat")

    if plan.mode is not ThroughputBenchmarkMode.OFFLINE_SINGLE_REQUEST:
        if plan.runtime_placeholder_active:
            errors.append("placeholder modes cannot mark runtime_placeholder_active")
        note = plan.claim_note.lower()
        if "placeholder" not in note and plan.mode.value.lower() not in note:
            errors.append(f"{plan.mode.value} requires placeholder caveat in claim_note")

    if plan.claim_status is ThroughputClaimStatus.CLAIM_ALLOWED:
        if not plan.exactness_gate_required:
            errors.append("CLAIM_ALLOWED requires exactness_gate_required")
        if plan.sample_count < _MIN_SAMPLES_FOR_CLAIM:
            errors.append(
                f"CLAIM_ALLOWED requires sample_count >= {_MIN_SAMPLES_FOR_CLAIM}"
            )
        if not plan.warmup_required:
            errors.append("CLAIM_ALLOWED requires warmup_required")
        if not plan.synchronization_required:
            errors.append("CLAIM_ALLOWED requires synchronization_required")
        if not plan.hardware:
            errors.append("CLAIM_ALLOWED requires hardware metadata")
        if not plan.baseline_arm.strip():
            errors.append("CLAIM_ALLOWED requires baseline_arm for comparison")
        if not plan.model.strip():
            errors.append("CLAIM_ALLOWED requires model identifier")

    if plan.claim_status is not ThroughputClaimStatus.CLAIM_ALLOWED:
        for claim in plan.allowed_claims:
            if _claim_encodes_speed_or_throughput(claim):
                errors.append(
                    "speed/latency/throughput claims forbidden unless "
                    "claim_status is CLAIM_ALLOWED"
                )

    for term in _FORBIDDEN_CLAIMS:
        if term not in plan.forbidden_claims:
            errors.append(f"forbidden_claims must include: {term}")

    for claim in plan.allowed_claims:
        for term in _FORBIDDEN_CLAIMS:
            if _encodes_positive_forbidden_claim(claim.lower(), term):
                errors.append(f"allowed_claims must not encode positive forbidden claim: {term}")

    note_lower = plan.claim_note.lower()
    for term in _FORBIDDEN_CLAIMS:
        if _encodes_positive_forbidden_claim(note_lower, term):
            errors.append(f"claim_note must not encode positive forbidden claim: {term}")

    return errors


def validate_throughput_diagnostic_result(result: ThroughputDiagnosticResult) -> list[str]:
    """Validate diagnostic result schema and claim guards."""
    errors = validate_throughput_benchmark_plan(result.plan)

    if result.hide_negative_results:
        errors.append("hide_negative_results must remain False — negative results must not be hidden")

    if not result.baseline_name.strip():
        errors.append("baseline_name required on diagnostic result")
    if not result.candidate_name.strip():
        errors.append("candidate_name required on diagnostic result")

    if result.measured and not result.exactness_passed:
        errors.append("measured results require exactness_passed before timing interpretation")

    if result.claim_status is ThroughputClaimStatus.CLAIM_ALLOWED:
        claim_plan = ThroughputBenchmarkPlan.from_dict(result.plan.to_dict())
        claim_plan.claim_status = ThroughputClaimStatus.CLAIM_ALLOWED
        errors.extend(validate_throughput_benchmark_plan(claim_plan))
        if not result.exactness_passed:
            errors.append("CLAIM_ALLOWED result requires exactness_passed")
        baseline_tps = result.baseline_tokens_per_second()
        if baseline_tps is None:
            errors.append("CLAIM_ALLOWED requires baseline comparison encoded in notes or fields")

    if result.claim_status is not ThroughputClaimStatus.CLAIM_ALLOWED:
        if _claim_encodes_speed_or_throughput(result.notes):
            errors.append(
                "speed/latency/throughput claims forbidden unless claim_status is CLAIM_ALLOWED"
            )

    if result.claim_status is ThroughputClaimStatus.NEGATIVE_OR_NEUTRAL:
        baseline_tps = result.baseline_tokens_per_second()
        if baseline_tps is not None and result.tokens_per_second > baseline_tps:
            errors.append(
                "NEGATIVE_OR_NEUTRAL status inconsistent with candidate faster than baseline"
            )

    if result.claim_status is ThroughputClaimStatus.DIAGNOSTIC_ONLY and not result.plan.claim_note.strip():
        errors.append("diagnostic result requires plan claim_note caveat")

    return errors
