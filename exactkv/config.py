from __future__ import annotations

from dataclasses import dataclass, field

_VALID_REPORT_FORMATS = frozenset({"json", "csv"})


@dataclass
class ExactKVConfig:
    """Per-request configuration for a single ExactKV generation run.

    dtype defaults to float32 for determinism in tests.
    greedy is always True in V1; sampling is out of scope.
    """

    model_name: str
    compressor: str = "noop"
    draft_len: int = 8
    max_new_tokens: int = 128
    device: str = "auto"
    dtype: str = "float32"
    greedy: bool = True
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.greedy:
            raise ValueError(
                "ExactKV V1 supports greedy decoding only. Set greedy=True."
            )
        if self.draft_len < 1:
            raise ValueError(f"draft_len must be >= 1, got {self.draft_len}")
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )


@dataclass
class BenchmarkConfig:
    """Experiment-level configuration for multi-compressor benchmark sweeps.

    Covers all prompt suites, compressor names, and draft-length combinations
    for a single model.  Use ``to_exactkv_config`` to project one cell of the
    sweep into a per-request ``ExactKVConfig``.

    Relationship to ``ExactKVConfig``
    ---------------------------------
    ``ExactKVConfig`` configures *one* generation request (single compressor,
    single draft_len).  ``BenchmarkConfig`` configures the *experiment* — the
    outer loop that iterates over compressors and draft lengths and writes
    aggregate reports.

    Relationship to ``RunConfig`` (benchmarks/runner.py)
    -----------------------------------------------------
    ``RunConfig`` is the V1 thin config used directly by ``run_one`` / ``run_suite``.
    ``BenchmarkConfig`` is the V2 experiment-level config.  ``RunConfig`` remains
    for backward compatibility; new code should prefer ``BenchmarkConfig``.
    """

    model_name: str
    compressors: list[str] = field(default_factory=lambda: ["int8"])
    draft_lens: list[int] = field(default_factory=lambda: [4])
    max_new_tokens: int = 32
    prompt_suite: str = "smoke"
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 0
    output_dir: str = "reports"
    report_formats: list[str] = field(default_factory=lambda: ["json"])

    def __post_init__(self) -> None:
        if not self.compressors:
            raise ValueError("BenchmarkConfig.compressors must not be empty.")
        if not self.draft_lens:
            raise ValueError("BenchmarkConfig.draft_lens must not be empty.")
        if any(d < 1 for d in self.draft_lens):
            raise ValueError(
                f"All draft_lens must be >= 1, got {self.draft_lens}"
            )
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )
        invalid = set(self.report_formats) - _VALID_REPORT_FORMATS
        if invalid:
            raise ValueError(
                f"Invalid report_formats {invalid}. "
                f"Allowed: {sorted(_VALID_REPORT_FORMATS)}"
            )

    def to_exactkv_config(self, compressor: str, draft_len: int) -> ExactKVConfig:
        """Project one sweep cell into a single-request ``ExactKVConfig``.

        Args:
            compressor: One of the names in ``self.compressors``.
            draft_len:  One of the values in ``self.draft_lens``.

        Returns:
            ``ExactKVConfig`` for that specific (compressor, draft_len) pair.
        """
        return ExactKVConfig(
            model_name=self.model_name,
            compressor=compressor,
            draft_len=draft_len,
            max_new_tokens=self.max_new_tokens,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
        )
