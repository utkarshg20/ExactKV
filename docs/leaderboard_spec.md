# ExactKV Canonical Leaderboard Specification (Phase B)

This document defines the **canonical public ranking system** for KV compression
exactness and divergence behavior. All rankings are derived from Phase A benchmark
outputs — no hidden inference, no runtime commit, no ExactKVGenerator modifications.

## Purpose

Transform ExactKV from a collection of benchmark scripts into a **reproducible
leaderboard platform** that ranks:

- KV compressors (built-in and restricted external adapters)
- Model robustness under compression
- Token-level divergence behavior
- Verifier agreement stability

## Inputs

| Input | Path | Required |
|-------|------|----------|
| Phase A benchmark | `reports/phaseA_benchmark.json` | Yes |
| Exp 116 instability | `reports/experiment_116_instability_regime_analysis.json` | No |

Phase B is **pure aggregation**. It never re-runs models.

## Canonical compressors

| Compressor | Tier | Notes |
|------------|------|-------|
| `noop` | Built-in | Identity baseline |
| `int8` | Built-in | Real INT8 storage |
| `int4_sim` | Built-in | Simulated INT4 |
| `k8_v4_sim` | Built-in | Asymmetric K8/V4 sim |
| `spectralquant` | Restricted | Adapter or `int4_sim` mock |
| `kvquant` | Restricted | Adapter or `int4_sim` mock |
| `shard` | Probe-only | External drafter probe metrics |

## Canonical models

- Qwen/Qwen2.5-0.5B → **Qwen 0.5B**
- Qwen/Qwen2.5-0.5B-Instruct → **Qwen 0.5B-Instruct**
- meta-llama/Llama-3.1-8B → **Llama-3.1-8B**
- mistralai/Mistral-7B-Instruct-v0.3 → **Mistral-7B**

## Normalization layer

For each `(model, compressor)` pair, Phase B reads `per_model_tables` from Phase A
and computes:

| Field | Source |
|-------|--------|
| `acceptance_rate` | `mean_acceptance_rate` |
| `divergence_score` | `divergence_rate` |
| `verifier_agreement` | `mean_verifier_agreement_score` |
| `failure_rate` | `exactkv_failure_rate` |
| `compression_ratio` | `mean_compression_ratio` or estimated fallback |
| `stability_score` | Exp 116 instability (inverted), Phase A instability, or `divergence_stability_score` |

Missing compressors are marked `availability: unavailable`.

## Unified score function

All components normalized to `[0, 1]`:

```
score = 0.35 × acceptance_rate
      + 0.25 × verifier_agreement
      + 0.20 × first_divergence_normalized
      + 0.10 × (1 − exactkv_failure_rate)
      + 0.10 × stability_score
```

**First divergence normalization:** `null` (no divergence) → `1.0`; otherwise
`min(1.0, mean_first_divergence_index / max_new_tokens)`.

## Output schema

### `reports/leaderboard.json`

Top-level keys:

- `leaderboard_id`: `exactkv_leaderboard_platform`
- `entries[]`: ranked rows with schema:

```json
{
  "rank": 1,
  "compressor": "int8",
  "model": "Qwen/Qwen2.5-0.5B",
  "model_short": "Qwen 0.5B",
  "score": 0.8123,
  "acceptance_rate": 0.774,
  "divergence_score": 0.0,
  "verifier_agreement": 0.625,
  "failure_rate": 0.0,
  "stability_score": 0.890,
  "availability": "available"
}
```

- `global_compressor_rankings[]`: mean score per compressor across models
- `per_model_breakdown{}`: entries grouped by short model name
- `insights[]`: 1–5 bullets extracted from computed values only

### `reports/leaderboard.md`

Markdown export with:

1. Global ranked table
2. Global compressor mean score (ASCII bar chart)
3. Per-model breakdown tables
4. Insights section
5. Reproducibility command

## CLI

```bash
# Default: JSON + markdown
python scripts/run_leaderboard.py --all

# Export modes
python scripts/run_leaderboard.py --json
python scripts/run_leaderboard.py --markdown

# Filters
python scripts/run_leaderboard.py --all --filter-model "Llama"
python scripts/run_leaderboard.py --all --filter-compressor int8

# Custom inputs
python scripts/run_leaderboard.py --all \
  --phase-a-input reports/phaseA_benchmark.json \
  --exp116-input reports/experiment_116_instability_regime_analysis.json
```

## Constraints

- **No** ExactKVGenerator modification
- **No** L4 runtime commit
- **No** training or fine-tuning
- **No** hallucinated metrics — insights must cite computed fields
- **No** timing / throughput / speedup fields
- Trace-only philosophy preserved

## Pipeline position

```
Phase A (benchmark) → Phase B (leaderboard) → Phase C (publication)
```

Regenerate leaderboard after every Phase A run:

```bash
python scripts/run_phase_a_scale_benchmark.py --deterministic-mode
python scripts/run_leaderboard.py --all
```

## Module reference

- Core: `exactkv/benchmarks/leaderboard_platform.py`
- Runner: `scripts/run_leaderboard.py`
- Tests: `tests/test_leaderboard_platform.py`
