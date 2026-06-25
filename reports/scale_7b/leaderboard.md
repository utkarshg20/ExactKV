# ExactKV Canonical Leaderboard

KV compression ranking by token-level acceptance, divergence, verifier agreement, and stability. Derived from Phase A benchmark outputs only.

**Source:** `phaseA_scale_benchmark`
**Deterministic mode:** False
**Generated:** n/a

> No speedup, latency, throughput, or memory savings claims unless directly measured.

## Global Ranked Table

| Rank | Compressor | Model | Score | Acceptance | Divergence | Verifier | Failure | Stability | Availability |
|-----:|------------|-------|------:|-----------:|-----------:|---------:|--------:|----------:|--------------|
| 1 | `noop` | Llama-3.1-8B | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | available |
| 2 | `int8` | Llama-3.1-8B | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | available |
| 3 | `noop` | Mistral-7B | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | available |
| 4 | `int8` | Mistral-7B | 0.983 | 1.000 | 0.173 | 1.000 | 0.000 | 0.827 | available |
| 5 | `int4_sim` | Llama-3.1-8B | 0.859 | 0.852 | 0.520 | 0.852 | 0.000 | 0.480 | available |
| 6 | `spectralquant` | Llama-3.1-8B | 0.859 | 0.852 | 0.520 | 0.852 | 0.000 | 0.480 | mock_fallback |
| 7 | `int4_sim` | Mistral-7B | 0.851 | 0.837 | 0.507 | 0.837 | 0.000 | 0.493 | available |
| 8 | `spectralquant` | Mistral-7B | 0.851 | 0.837 | 0.507 | 0.837 | 0.000 | 0.493 | mock_fallback |
| 9 | `shard` | Llama-3.1-8B | 0.732 | 0.632 | 0.227 | 0.532 | 0.000 | 0.773 | probe_only |
| 10 | `shard` | Mistral-7B | 0.727 | 0.623 | 0.247 | 0.533 | 0.000 | 0.753 | probe_only |

## Global Compressor Mean Score

1. `noop` — 1.000  `████████████████████`
2. `int8` — 0.991  `████████████████████`
3. `int4_sim` — 0.855  `█████████████████░░░`
4. `spectralquant` — 0.855  `█████████████████░░░`
5. `shard` — 0.729  `███████████████░░░░░`

## Per-Model Breakdown

### Llama-3.1-8B

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|
| 1 | `noop` | 1.000 | 1.000 | 0.000 |
| 2 | `int8` | 1.000 | 1.000 | 0.000 |
| 3 | `int4_sim` | 0.859 | 0.852 | 0.520 |
| 4 | `spectralquant` | 0.859 | 0.852 | 0.520 |
| 5 | `shard` | 0.732 | 0.632 | 0.227 |

### Mistral-7B

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|
| 1 | `noop` | 1.000 | 1.000 | 0.000 |
| 2 | `int8` | 0.983 | 1.000 | 0.173 |
| 3 | `int4_sim` | 0.851 | 0.837 | 0.507 |
| 4 | `spectralquant` | 0.851 | 0.837 | 0.507 |
| 5 | `shard` | 0.727 | 0.623 | 0.247 |

## Insights

- `noop` leads the cross-model mean score (1.000 across 2 models).
- `int8` is near Pareto-optimal: mean acceptance 1.000 vs noop 1.000 with zero ExactKV failures across all models.
- `spectralquant` mean acceptance 0.844 vs `int4_sim` 0.844; divergence rates 0.513 vs 0.513.
- `shard` probe-only rows show mean divergence 0.237 and stability 0.763; weakest score on Mistral-7B (0.727).
- Llama-3.1-8B has the highest mean score (0.890); Mistral-7B shows the widest compressor spread (0.273).

## Reproducibility

```bash
python scripts/exactkv.py run leaderboard
```
