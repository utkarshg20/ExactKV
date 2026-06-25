1/9 KV cache compression is everywhere. Token-level drift is not. ExactKV measures when compressed caches start lying — before you ship.

2/9 Structured output case: `int4_sim` on Qwen 0.5B. First divergence at token 1. Acceptance 0.50.

3/9 Worst INT4 cell: acceptance 0.33, divergence at token 1. ExactKV caught it — zero silent failures.

4/9 Cross-model panel: 336 cells, 4 models, 7 compressors. INT8 mean score 0.916.

5/9 int8 is near-optimal baseline: high acceptance, zero divergence rate in aggregated leaderboard, zero ExactKV failures.

6/9 `noop` leads the cross-model mean score (0.995 across 4 models).

7/9 Simulated INT4 + external probes (shard, kvquant mock) show 2–3× higher divergence on small models.

8/9 ExactKV = trace-only verification. No runtime commit. Reproducible benchmark + leaderboard from JSON reports.

9/9 Full paper draft + demo pack in repo. `python scripts/run_leaderboard.py --all`
