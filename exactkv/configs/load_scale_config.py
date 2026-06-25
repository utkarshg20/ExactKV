"""Scale benchmark configuration loader (Phase H+)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SCALE_CONFIG = Path(__file__).resolve().parent / "scale_7b_8b.yaml"
DEFAULT_SCALE_CONFIG_JSON = Path(__file__).resolve().parent / "scale_7b_8b.json"


def load_scale_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load scale YAML config; falls back to bundled JSON if PyYAML unavailable."""
    cfg_path = Path(path or DEFAULT_SCALE_CONFIG)
    text = cfg_path.read_text()
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"config must be a mapping: {cfg_path}")
        return data
    except ImportError:
        json_path = cfg_path.with_suffix(".json")
        if not json_path.is_file() and DEFAULT_SCALE_CONFIG_JSON.is_file():
            json_path = DEFAULT_SCALE_CONFIG_JSON
        if json_path.is_file():
            return json.loads(json_path.read_text())
        raise ImportError(
            "PyYAML is required to load .yaml configs, or provide scale_7b_8b.json",
        ) from None


def resolve_device(requested: str = "auto") -> str:
    """Resolve torch device from config string."""
    if requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def map_compressors(
    names: list[str],
    compressor_map: dict[str, str] | None = None,
) -> list[str]:
    """Map public compressor aliases to Phase A registry names."""
    mapping = compressor_map or {}
    out: list[str] = []
    for name in names:
        out.append(mapping.get(name, name))
    return out
