from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExactKVConfig:
    """Top-level configuration for ExactKV Phase 1.

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
