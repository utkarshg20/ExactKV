# ExactKV Canonical Leaderboard

KV compression ranking by token-level acceptance, divergence, verifier agreement, and stability. Derived from Phase A benchmark outputs only.

**Source:** `phaseA_scale_benchmark`
**Deterministic mode:** False
**Generated:** 2026-06-25T14:57:25.053326+00:00

> No speedup, latency, throughput, or memory savings claims unless directly measured.

## Global Ranked Table

| Rank | Compressor | Model | Score | Acceptance | Divergence | Verifier | Failure | Stability | Availability |
|-----:|------------|-------|------:|-----------:|-----------:|---------:|--------:|----------:|--------------|
| 1 | `noop` | Mistral-7B | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | available |
| 2 | `noop` | Llama-3.1-8B | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | available |
| 3 | `int8` | Llama-3.1-8B | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | available |
| 4 | `noop` | Qwen 0.5B | 0.989 | 1.000 | 0.000 | 1.000 | 0.000 | 0.890 | available |
| 5 | `noop` | Qwen 0.5B-Instruct | 0.989 | 1.000 | 0.000 | 1.000 | 0.000 | 0.890 | available |
| 6 | `int8` | Qwen 0.5B-Instruct | 0.981 | 0.986 | 0.000 | 0.986 | 0.000 | 0.890 | available |
| 7 | `int8` | Qwen 0.5B | 0.848 | 0.995 | 0.167 | 0.995 | 0.000 | 0.890 | available |
| 8 | `k8_v4_sim` | Llama-3.1-8B | 0.848 | 0.961 | 0.250 | 0.961 | 0.000 | 0.750 | available |
| 9 | `int8` | Mistral-7B | 0.833 | 1.000 | 0.167 | 1.000 | 0.000 | 0.833 | available |
| 10 | `k8_v4_sim` | Qwen 0.5B-Instruct | 0.823 | 0.976 | 0.167 | 0.976 | 0.000 | 0.743 | available |
| 11 | `k8_v4_sim` | Mistral-7B | 0.781 | 0.961 | 0.500 | 0.961 | 0.000 | 0.500 | available |
| 12 | `k8_v4_sim` | Qwen 0.5B | 0.755 | 0.895 | 0.417 | 0.895 | 0.000 | 0.782 | available |
| 13 | `int4_sim` | Mistral-7B | 0.697 | 0.839 | 0.500 | 0.839 | 0.000 | 0.500 | available |
| 14 | `spectralquant` | Mistral-7B | 0.697 | 0.839 | 0.500 | 0.839 | 0.000 | 0.500 | mock_fallback |
| 15 | `kvquant` | Mistral-7B | 0.697 | 0.839 | 0.500 | 0.839 | 0.000 | 0.500 | mock_fallback |
| 16 | `int4_sim` | Llama-3.1-8B | 0.665 | 0.817 | 0.500 | 0.817 | 0.000 | 0.500 | available |
| 17 | `spectralquant` | Llama-3.1-8B | 0.665 | 0.817 | 0.500 | 0.817 | 0.000 | 0.500 | mock_fallback |
| 18 | `kvquant` | Llama-3.1-8B | 0.665 | 0.817 | 0.500 | 0.817 | 0.000 | 0.500 | mock_fallback |
| 19 | `int4_sim` | Qwen 0.5B-Instruct | 0.605 | 0.671 | 0.833 | 0.671 | 0.000 | 0.688 | available |
| 20 | `shard` | Mistral-7B | 0.572 | 0.672 | 0.167 | 0.562 | 0.000 | 0.833 | probe_only |
| 21 | `int4_sim` | Qwen 0.5B | 0.563 | 0.617 | 0.500 | 0.617 | 0.000 | 0.804 | available |
| 22 | `shard` | Qwen 0.5B | 0.554 | 0.637 | 0.167 | 0.542 | 0.000 | 0.833 | probe_only |
| 23 | `spectralquant` | Qwen 0.5B-Instruct | 0.553 | 0.671 | 0.833 | 0.671 | 0.000 | 0.167 | mock_fallback |
| 24 | `kvquant` | Qwen 0.5B-Instruct | 0.553 | 0.671 | 0.833 | 0.671 | 0.000 | 0.167 | mock_fallback |
| 25 | `spectralquant` | Qwen 0.5B | 0.532 | 0.617 | 0.500 | 0.617 | 0.000 | 0.500 | mock_fallback |
| 26 | `kvquant` | Qwen 0.5B | 0.532 | 0.617 | 0.500 | 0.617 | 0.000 | 0.500 | mock_fallback |
| 27 | `shard` | Llama-3.1-8B | 0.517 | 0.594 | 0.333 | 0.521 | 0.000 | 0.667 | probe_only |
| 28 | `shard` | Qwen 0.5B-Instruct | 0.472 | 0.532 | 0.417 | 0.458 | 0.000 | 0.583 | probe_only |

## Global Compressor Mean Score

1. `noop` — 0.995  `████████████████████`
2. `int8` — 0.916  `██████████████████░░`
3. `k8_v4_sim` — 0.801  `████████████████░░░░`
4. `int4_sim` — 0.632  `█████████████░░░░░░░`
5. `kvquant` — 0.612  `████████████░░░░░░░░`
6. `spectralquant` — 0.612  `████████████░░░░░░░░`
7. `shard` — 0.529  `███████████░░░░░░░░░`

## Per-Model Breakdown

### Llama-3.1-8B

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|
| 1 | `noop` | 1.000 | 1.000 | 0.000 |
| 2 | `int8` | 1.000 | 1.000 | 0.000 |
| 3 | `k8_v4_sim` | 0.848 | 0.961 | 0.250 |
| 4 | `int4_sim` | 0.665 | 0.817 | 0.500 |
| 5 | `spectralquant` | 0.665 | 0.817 | 0.500 |
| 6 | `kvquant` | 0.665 | 0.817 | 0.500 |
| 7 | `shard` | 0.517 | 0.594 | 0.333 |

### Mistral-7B

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|
| 1 | `noop` | 1.000 | 1.000 | 0.000 |
| 2 | `int8` | 0.833 | 1.000 | 0.167 |
| 3 | `k8_v4_sim` | 0.781 | 0.961 | 0.500 |
| 4 | `int4_sim` | 0.697 | 0.839 | 0.500 |
| 5 | `spectralquant` | 0.697 | 0.839 | 0.500 |
| 6 | `kvquant` | 0.697 | 0.839 | 0.500 |
| 7 | `shard` | 0.572 | 0.672 | 0.167 |

### Qwen 0.5B

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|
| 1 | `noop` | 0.989 | 1.000 | 0.000 |
| 2 | `int8` | 0.848 | 0.995 | 0.167 |
| 3 | `k8_v4_sim` | 0.755 | 0.895 | 0.417 |
| 4 | `int4_sim` | 0.563 | 0.617 | 0.500 |
| 5 | `shard` | 0.554 | 0.637 | 0.167 |
| 6 | `spectralquant` | 0.532 | 0.617 | 0.500 |
| 7 | `kvquant` | 0.532 | 0.617 | 0.500 |

### Qwen 0.5B-Instruct

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|
| 1 | `noop` | 0.989 | 1.000 | 0.000 |
| 2 | `int8` | 0.981 | 0.986 | 0.000 |
| 3 | `k8_v4_sim` | 0.823 | 0.976 | 0.167 |
| 4 | `int4_sim` | 0.605 | 0.671 | 0.833 |
| 5 | `spectralquant` | 0.553 | 0.671 | 0.833 |
| 6 | `kvquant` | 0.553 | 0.671 | 0.833 |
| 7 | `shard` | 0.472 | 0.532 | 0.417 |

## Insights

- `noop` leads the cross-model mean score (0.995 across 4 models).
- `int8` is near Pareto-optimal: mean acceptance 0.995 vs noop 1.000 with zero ExactKV failures across all models.
- `spectralquant` mean acceptance 0.736 vs `int4_sim` 0.736; divergence rates 0.583 vs 0.583.
- `shard` probe-only rows show mean divergence 0.271 and stability 0.729; weakest score on Qwen 0.5B-Instruct (0.472).
- Llama-3.1-8B has the highest mean score (0.766); Qwen 0.5B-Instruct shows the widest compressor spread (0.517).

## Reproducibility

```bash
python scripts/exactkv.py run leaderboard
```
