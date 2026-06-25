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
| 3 | `int4_sim` | Llama-3.1-8B | 0.684 | 0.852 | 0.520 | 0.852 | 0.000 | 0.480 | available |
| 4 | `spectralquant` | Llama-3.1-8B | 0.684 | 0.852 | 0.520 | 0.852 | 0.000 | 0.480 | mock_fallback |
| 5 | `shard` | Llama-3.1-8B | 0.544 | 0.632 | 0.227 | 0.532 | 0.000 | 0.773 | probe_only |

## Global Compressor Mean Score

1. `int8` — 1.000  `████████████████████`
2. `noop` — 1.000  `████████████████████`
3. `int4_sim` — 0.684  `██████████████░░░░░░`
4. `spectralquant` — 0.684  `██████████████░░░░░░`
5. `shard` — 0.544  `███████████░░░░░░░░░`

## Per-Model Breakdown

### Llama-3.1-8B

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|
| 1 | `noop` | 1.000 | 1.000 | 0.000 |
| 2 | `int8` | 1.000 | 1.000 | 0.000 |
| 3 | `int4_sim` | 0.684 | 0.852 | 0.520 |
| 4 | `spectralquant` | 0.684 | 0.852 | 0.520 |
| 5 | `shard` | 0.544 | 0.632 | 0.227 |

### Mistral-7B

| Rank | Compressor | Score | Acceptance | Divergence |
|-----:|------------|------:|-----------:|-----------:|

## Insights

- `int8` leads the cross-model mean score (1.000 across 1 models).
- `int8` is near Pareto-optimal: mean acceptance 1.000 vs noop 1.000 with zero ExactKV failures across all models.
- `spectralquant` mean acceptance 0.852 vs `int4_sim` 0.852; divergence rates 0.520 vs 0.520.
- `shard` probe-only rows show mean divergence 0.227 and stability 0.773; weakest score on Llama-3.1-8B (0.544).
- Llama-3.1-8B has the highest mean score (0.782); Llama-3.1-8B shows the widest compressor spread (0.456).

## Reproducibility

```bash
python scripts/exactkv.py run leaderboard
```
