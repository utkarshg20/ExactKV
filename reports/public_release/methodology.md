# ExactKV Evaluation Methodology

## Divergence (canonical)

All divergence metrics use Phase G `FirstDivergenceAuthority`:
- `canonical_first_divergence_index`
- Types: `token_mismatch`, `length_drift`, `kernel_inconsistency`, `verifier_disagreement`, `none`

## Acceptance

Token-level ExactKV speculative decoding acceptance rate from Phase A cells.

## Leaderboard scoring (locked)

```
score = 0.35 * acceptance_rate
      + 0.25 * verifier_agreement
      + 0.20 * (1 - normalized_first_divergence)
      + 0.10 * (1 - failure_rate)
      + 0.10 * stability_score
```

## Compressors

Built-in: noop, int8, int4_sim, k8_v4_sim
External (adapter/mock): spectralquant (fallback if dependency unavailable), kvquant, shard (probe-first), turboquant
Phase H+ adapters: spectralquant_real (fallback mode when dependency missing), shard_real (probe-first heuristic)

## Reproducibility

All benchmarks are reproducible from disk reports without re-inference when using `--deterministic`.
