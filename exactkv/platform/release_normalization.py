"""Canonical model/compressor ID normalization for public release (Phase J / R1)."""
from __future__ import annotations

from exactkv.benchmarks.leaderboard_platform import short_model_name

# Canonical scale-run model IDs (authoritative).
CANONICAL_MODEL_IDS: dict[str, str] = {
    "meta-llama/llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B",
    "mistralai/mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}

# Public config name → Phase A backend compressor id.
CANONICAL_COMPRESSOR_IDS: dict[str, str] = {
    "int4": "int4_sim",
    "int4_sim": "int4_sim",
    "spectralquant_real": "spectralquant",
    "spectralquant": "spectralquant",
    "shard_real": "shard",
    "shard": "shard",
    "noop": "noop",
    "int8": "int8",
    "k8_v4_sim": "k8_v4_sim",
    "kvquant": "kvquant",
}

# Display names preserve public caveat context (fallback/proxy not hidden).
PUBLIC_COMPRESSOR_DISPLAY: dict[str, str] = {
    "int4_sim": "int4_sim",
    "spectralquant": "spectralquant",
    "shard": "shard",
}


def normalize_model_id(model_id: str) -> str:
    """Return canonical HuggingFace-style model id."""
    if not model_id:
        return model_id
    key = model_id.strip().lower()
    if key in CANONICAL_MODEL_IDS:
        return CANONICAL_MODEL_IDS[key]
    if model_id in CANONICAL_MODEL_IDS.values():
        return model_id
    # Short display aliases → canonical when unambiguous in scale panel.
    if key in ("mistral-7b", "mistral-7b-instruct"):
        return "mistralai/Mistral-7B-Instruct-v0.3"
    if "llama-3.1-8b" in key:
        return "meta-llama/Llama-3.1-8B"
    return model_id


def display_model_name(model_id: str) -> str:
    """Human-readable model label for public tables."""
    return short_model_name(normalize_model_id(model_id))


def normalize_compressor_id(compressor_id: str) -> str:
    """Map public config compressor names to Phase A backend ids."""
    if not compressor_id:
        return compressor_id
    key = compressor_id.strip()
    return CANONICAL_COMPRESSOR_IDS.get(key, CANONICAL_COMPRESSOR_IDS.get(key.lower(), key))


def display_compressor_name(compressor_id: str) -> str:
    """Public leaderboard compressor label (does not hide fallback/proxy status)."""
    norm = normalize_compressor_id(compressor_id)
    return PUBLIC_COMPRESSOR_DISPLAY.get(norm, norm)
