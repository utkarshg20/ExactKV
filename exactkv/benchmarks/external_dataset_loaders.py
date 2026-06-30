"""Load established external benchmark prompts for ExactKV drift panels.

Supported families (priority order for paper):
  - longbench  (THUDM/LongBench on Hugging Face, or bundled pilot JSONL)
  - ruler      (bundled RULER-style synthetic tasks, or custom JSONL)
  - bfcl       (bundled pilot, full BFCL via HF when available)
  - humaneval  (openai/openai_humaneval on HF, or bundled pilot)
  - mbpp       (google-research-datasets/mbpp on HF, or bundled pilot)

ExactKV uses these for **token-path drift** (first divergence, acceptance,
exactkv_failure), not official LongBench/RULER leaderboard scores.

Claim boundary: pilot subsets are diagnostic panels unless a full export is
documented in the run manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

_BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "prompts"
PILOT_PATHS: dict[str, Path] = {
    "longbench": _BENCHMARKS_DIR / "longbench_pilot.jsonl",
    "ruler": _BENCHMARKS_DIR / "ruler_pilot.jsonl",
    "bfcl": _BENCHMARKS_DIR / "bfcl_pilot.jsonl",
    "humaneval": _BENCHMARKS_DIR / "humaneval_pilot.jsonl",
    "mbpp": _BENCHMARKS_DIR / "mbpp_pilot.jsonl",
}

LONGBENCH_HF_SUBSETS: tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "trec",
    "samsum",
    "lcc",
    "passage_retrieval_en",
)

RULER_TASK_TYPES: tuple[str, ...] = (
    "niah_single",
    "niah_multi",
    "variable_tracking",
    "common_words_extraction",
)

# BFCL v3 files on gorilla-llm/Berkeley-Function-Calling-Leaderboard (JSONL per file).
# Official BFCL is not compatible with HuggingFace load_dataset; use huggingface_hub instead.
BFCL_HF_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("simple", "BFCL_v3_simple.json"),
    ("parallel", "BFCL_v3_parallel.json"),
    ("multi_turn", "BFCL_v3_multi_turn_base.json"),
    ("ast_eval", "BFCL_v3_java.json"),
)

BFCL_EXPORT_PATH = _BENCHMARKS_DIR / "bfcl_export.jsonl"


def _normalize_row(
    row: Mapping[str, Any],
    *,
    dataset_family: str,
    default_category: str,
) -> dict[str, Any]:
    prompt_id = str(row.get("prompt_id") or row.get("id") or "unknown")
    category = str(
        row.get("category")
        or row.get("task")
        or row.get("dataset")
        or row.get("subtask")
        or default_category,
    )
    prompt = str(row.get("prompt") or row.get("input") or row.get("context") or "")
    if not prompt.strip():
        raise ValueError(f"{dataset_family}:{prompt_id}: empty prompt")
    out: dict[str, Any] = {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": prompt,
        "dataset_family": dataset_family,
    }
    for key in (
        "source_dataset",
        "source",
        "task_type",
        "task_id",
        "test_list",
        "entry_point",
        "declared_context_tokens",
        "language",
        "reference_answer",
        "notes",
    ):
        if key in row and row[key] is not None:
            out[key] = row[key]
    return out


def load_jsonl_prompts(path: Path, *, dataset_family: str, default_category: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = json.loads(line)
            try:
                rows.append(
                    _normalize_row(entry, dataset_family=dataset_family, default_category=default_category),
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return rows


def load_longbench_pilot(*, max_prompts: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_prompts(
        PILOT_PATHS["longbench"],
        dataset_family="longbench",
        default_category="longbench",
    )
    return rows[:max_prompts] if max_prompts else rows


def load_longbench_hf(
    *,
    subsets: Sequence[str] | None = None,
    max_per_subset: int = 2,
    max_total: int | None = None,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load prompts from THUDM/LongBench on Hugging Face (requires network)."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pip install datasets to load LongBench from Hugging Face") from exc

    chosen = list(subsets or LONGBENCH_HF_SUBSETS)
    rows: list[dict[str, Any]] = []
    for subset in chosen:
        ds = load_dataset(
            "THUDM/LongBench",
            subset,
            split=split,
            trust_remote_code=True,
        )
        for i, item in enumerate(ds):
            if i >= max_per_subset:
                break
            context = str(item.get("context") or "")
            input_text = str(item.get("input") or "")
            prompt = f"{context}\n\n{input_text}".strip() if context else input_text
            rows.append(
                _normalize_row(
                    {
                        "prompt_id": f"lb_{subset}_{i:03d}",
                        "category": subset,
                        "prompt": prompt,
                        "source_dataset": f"THUDM/LongBench/{subset}",
                        "source": "hf",
                        "language": item.get("language", "en"),
                        "reference_answer": item.get("answers"),
                    },
                    dataset_family="longbench",
                    default_category=subset,
                ),
            )
            if max_total and len(rows) >= max_total:
                return rows
    return rows


def load_longbench_prompts(
    *,
    source: str = "pilot",
    max_prompts: int = 12,
    subsets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if source == "pilot":
        return load_longbench_pilot(max_prompts=max_prompts)
    if source == "hf":
        return load_longbench_hf(subsets=subsets, max_total=max_prompts)
    raise ValueError(f"unknown longbench source {source!r}")


def load_ruler_pilot(*, max_prompts: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_prompts(
        PILOT_PATHS["ruler"],
        dataset_family="ruler",
        default_category="ruler",
    )
    return rows[:max_prompts] if max_prompts else rows


def load_ruler_prompts(*, max_prompts: int = 12) -> list[dict[str, Any]]:
    """Bundled RULER-style synthetic tasks (full RULER via NVIDIA repo is optional)."""
    return load_ruler_pilot(max_prompts=max_prompts)


def load_bfcl_pilot(*, max_prompts: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_prompts(
        PILOT_PATHS["bfcl"],
        dataset_family="bfcl",
        default_category="tool_call",
    )
    for row in rows:
        row.setdefault("source", "bundled_pilot")
    return rows[:max_prompts] if max_prompts else rows


def _format_bfcl_prompt(item: Mapping[str, Any]) -> str:
    """Render a BFCL v3 entry as a single prompt string for ExactKV drift panels."""
    parts = ["You are a helpful assistant with access to tools."]
    functions = item.get("function") or []
    if functions:
        parts.append("\nAvailable tools:\n" + json.dumps(functions, indent=2, ensure_ascii=False))
    initial = item.get("initial_config")
    if initial:
        parts.append(
            "\nInitial environment:\n"
            + json.dumps(initial, indent=2, ensure_ascii=False)[:4000],
        )
    parts.append("\nConversation:")
    for turn_group in item.get("question") or []:
        if not isinstance(turn_group, list):
            continue
        for msg in turn_group:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").capitalize()
            content = str(msg.get("content") or "")
            if content.strip():
                parts.append(f"\n{role}: {content}")
    parts.append("\nRespond with valid JSON tool call(s) only.")
    return "\n".join(parts)


def _bfcl_item_to_row(item: Mapping[str, Any], *, category: str) -> dict[str, Any]:
    task_id = str(item.get("id") or "unknown")
    return _normalize_row(
        {
            "prompt_id": f"bfcl_{category}_{task_id}",
            "task_id": task_id,
            "category": category,
            "task_type": "tool_call",
            "prompt": _format_bfcl_prompt(item),
            "source": "hf",
            "source_dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        },
        dataset_family="bfcl",
        default_category=category,
    )


def load_bfcl_hf(
    *,
    categories: Sequence[tuple[str, str]] | None = None,
    max_per_category: int = 13,
    max_total: int | None = None,
) -> list[dict[str, Any]]:
    """Load BFCL v3 prompts via huggingface_hub (not datasets.load_dataset)."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub to load BFCL from Hugging Face") from exc

    chosen = list(categories or BFCL_HF_CATEGORIES)
    rows: list[dict[str, Any]] = []
    for category, filename in chosen:
        path = hf_hub_download(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            filename,
            repo_type="dataset",
        )
        with Path(path).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_per_category:
                    break
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rows.append(_bfcl_item_to_row(item, category=category))
                if max_total and len(rows) >= max_total:
                    return rows
    return rows


def load_bfcl_export(*, max_prompts: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_prompts(
        BFCL_EXPORT_PATH,
        dataset_family="bfcl",
        default_category="tool_call",
    )
    return rows[:max_prompts] if max_prompts else rows


def load_bfcl_prompts(
    *,
    source: str = "pilot",
    max_prompts: int = 50,
    max_per_category: int | None = None,
) -> list[dict[str, Any]]:
    if source == "pilot":
        return load_bfcl_pilot(max_prompts=max_prompts)
    if source == "export":
        return load_bfcl_export(max_prompts=max_prompts)
    if source == "hf":
        per_cat = max_per_category
        if per_cat is None and max_prompts:
            per_cat = max(1, (max_prompts + len(BFCL_HF_CATEGORIES) - 1) // len(BFCL_HF_CATEGORIES))
        return load_bfcl_hf(max_per_category=per_cat or 13, max_total=max_prompts)
    raise ValueError(f"unknown bfcl source {source!r}")


def load_mbpp_pilot(*, max_prompts: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_prompts(
        PILOT_PATHS["mbpp"],
        dataset_family="mbpp",
        default_category="code",
    )
    for row in rows:
        row.setdefault("source", "bundled_pilot")
        if "task_id" not in row:
            row["task_id"] = row["prompt_id"]
    return rows[:max_prompts] if max_prompts else rows


def load_mbpp_hf(*, max_prompts: int = 20, split: str = "train") -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pip install datasets to load MBPP from Hugging Face") from exc

    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        if i >= max_prompts:
            break
        task_id = str(item.get("task_id", i))
        prompt = str(item.get("prompt") or item.get("text") or "")
        if not prompt.strip():
            continue
        rows.append(
            _normalize_row(
                {
                    "prompt_id": f"mbpp_{task_id}",
                    "task_id": task_id,
                    "category": "code",
                    "prompt": prompt,
                    "test_list": item.get("test_list") or item.get("test_list_wo_doc"),
                    "source": "hf",
                    "source_dataset": "google-research-datasets/mbpp",
                },
                dataset_family="mbpp",
                default_category="code",
            ),
        )
    return rows


def load_humaneval_pilot(*, max_prompts: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_prompts(
        PILOT_PATHS["humaneval"],
        dataset_family="humaneval",
        default_category="code",
    )
    return rows[:max_prompts] if max_prompts else rows


def load_humaneval_hf(*, max_prompts: int = 20, split: str = "test") -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pip install datasets to load HumanEval from Hugging Face") from exc

    ds = load_dataset("openai/openai_humaneval", split=split)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        if i >= max_prompts:
            break
        rows.append(
            _normalize_row(
                {
                    "prompt_id": f"he_{item['task_id']}",
                    "category": "code",
                    "prompt": item["prompt"],
                    "source_dataset": "openai/openai_humaneval",
                    "entry_point": item.get("entry_point"),
                },
                dataset_family="humaneval",
                default_category="code",
            ),
        )
    return rows


def load_external_prompts(
    family: str,
    *,
    source: str = "pilot",
    max_prompts: int = 12,
    subsets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    family = family.lower()
    if family == "longbench":
        return load_longbench_prompts(source=source, max_prompts=max_prompts, subsets=subsets)
    if family == "ruler":
        return load_ruler_prompts(max_prompts=max_prompts)
    if family == "bfcl":
        return load_bfcl_prompts(source=source, max_prompts=max_prompts)
    if family == "humaneval":
        if source == "hf":
            return load_humaneval_hf(max_prompts=max_prompts)
        return load_humaneval_pilot(max_prompts=max_prompts)
    if family == "mbpp":
        if source == "hf":
            return load_mbpp_hf(max_prompts=max_prompts)
        return load_mbpp_pilot(max_prompts=max_prompts)
    raise ValueError(f"unknown dataset family {family!r}")


def export_prompts_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return path
