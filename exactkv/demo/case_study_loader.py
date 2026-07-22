"""Load curated divergence case studies for the live terminal demo."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CASE_STUDIES_JSON = Path("site/data/case_studies.json")

# Curated carousel order: clearest visual spread for a ~90s recording.
DEFAULT_CAROUSEL_KEYS: tuple[tuple[str, str], ...] = (
    ("p02_p2_json_tool", "int4_sim"),
    ("p00_p0_capital_france", "int4_sim"),
    ("lb_narrativeqa_000_ctx2048", "int4_sim"),
    ("bfcl_parallel_parallel_6_ctx2048", "int4_sim"),
)

# Shorter single-case default for quick recordings.
DEFAULT_SINGLE_CASE = ("p02_p2_json_tool", "int4_sim")

PUBLIC_TAGLINE = (
    "Benchmarks miss when compressed KV starts lying.\n"
    "ExactKV crash-tests every token vs full precision."
)

# Closing beat removed from live demos (victory / ship comparison is enough).
CLOSING_LINES = ""


@dataclass(frozen=True)
class CaseStudy:
    dataset_family: str
    task_category: str
    prompt_id: str
    model_name: str
    compressor_name: str
    panel: str
    full_snippet: str
    lossy_snippet: str
    exactkv_snippet: str
    context_bucket: int | None = None
    max_new_tokens: int | None = None
    first_divergence_index: int | None = None
    acceptance_rate: float | None = None
    exactkv_failure: bool = False
    interpretation: str = ""

    @property
    def short_model(self) -> str:
        return self.model_name.split("/")[-1]

    @property
    def title(self) -> str:
        return f"{self.dataset_family} · {self.task_category}"

    @property
    def has_drift(self) -> bool:
        return self.lossy_snippet.strip() != self.full_snippet.strip()

    @property
    def exactkv_matches_full(self) -> bool:
        return self.exactkv_snippet.strip() == self.full_snippet.strip()

    @property
    def meta_line(self) -> str:
        parts = [
            self.short_model,
            self.compressor_name,
            self.panel,
        ]
        if self.context_bucket is not None:
            parts.append(f"{self.context_bucket} ctx")
        if self.first_divergence_index is not None:
            parts.append(f"fdi={self.first_divergence_index}")
        if self.acceptance_rate is not None:
            parts.append(f"accept={self.acceptance_rate:.0%}")
        return " · ".join(parts)


def _row_from_dict(raw: dict[str, Any]) -> CaseStudy:
    return CaseStudy(
        dataset_family=str(raw.get("dataset_family", "")),
        task_category=str(raw.get("task_category", "")),
        prompt_id=str(raw.get("prompt_id", "")),
        model_name=str(raw.get("model_name", "")),
        compressor_name=str(raw.get("compressor_name", "")),
        panel=str(raw.get("panel", "")),
        full_snippet=str(raw.get("full_snippet", "")),
        lossy_snippet=str(raw.get("lossy_snippet", "")),
        exactkv_snippet=str(raw.get("exactkv_snippet", "")),
        context_bucket=raw.get("context_bucket"),
        max_new_tokens=raw.get("max_new_tokens"),
        first_divergence_index=raw.get("first_divergence_index"),
        acceptance_rate=raw.get("acceptance_rate"),
        exactkv_failure=bool(raw.get("exactkv_failure", False)),
        interpretation=str(raw.get("interpretation", "")),
    )


def load_case_studies(path: Path | None = None) -> list[CaseStudy]:
    json_path = path or DEFAULT_CASE_STUDIES_JSON
    if not json_path.is_file():
        raise FileNotFoundError(f"case studies JSON not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("case_studies") or []
    out: list[CaseStudy] = []
    for raw in rows:
        if not raw.get("snippets_available", True):
            continue
        out.append(_row_from_dict(raw))
    if not out:
        raise ValueError(f"no case studies in {json_path}")
    return out


def select_cases(
    all_cases: list[CaseStudy],
    *,
    prompt_id: str | None = None,
    compressor: str | None = None,
    carousel_keys: tuple[tuple[str, str], ...] | None = None,
    index: int | None = None,
) -> list[CaseStudy]:
    if index is not None:
        if index < 0 or index >= len(all_cases):
            raise IndexError(f"case index out of range: {index}")
        return [all_cases[index]]

    def _key(c: CaseStudy) -> tuple[str, str]:
        return (c.prompt_id, c.compressor_name)

    by_key = {_key(c): c for c in all_cases}

    if prompt_id:
        comp = compressor or "int4_sim"
        key = (prompt_id, comp)
        if key not in by_key:
            # fallback: any compressor for prompt_id
            matches = [c for c in all_cases if c.prompt_id == prompt_id]
            if not matches:
                raise KeyError(f"unknown prompt_id: {prompt_id}")
            return [matches[0]]
        return [by_key[key]]

    keys = carousel_keys or DEFAULT_CAROUSEL_KEYS
    selected: list[CaseStudy] = []
    for pid, comp in keys:
        if (pid, comp) in by_key:
            selected.append(by_key[(pid, comp)])
    if not selected:
        return all_cases[:4]
    return selected
