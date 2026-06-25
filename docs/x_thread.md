1/9 KV cache compression is everywhere. Token-level drift is not. ExactKV is a compressor-agnostic crash-test and leaderboard for LLM KV-cache compression exactness.

2/9 Structured output case: `int4_sim` on Qwen 0.5B. First divergence at token 1. Acceptance 0.50.

3/9 Worst INT4 cell: acceptance 0.33, divergence at token 1. ExactKV caught it — zero silent failures.

4/9 Cross-model panel: 336 cells, 4 models, 7 compressors. INT8 mean score 0.916.

5/9 int8 is near-optimal baseline: high acceptance, zero divergence rate in aggregated leaderboard, zero ExactKV failures.

6/9 `noop` leads the cross-model mean score (0.995 across 4 models).

7/9 Simulated INT4 + external probes (shard, kvquant mock) show 2–3× higher divergence on small models.

8/9 ExactKV = trace-only verification. Not a production serving system. Does not reproduce VeriCache. Shard = probe-first; SpectralQuant = fallback/proxy when dependency missing.

9/9 Phase H+ scale: 1500 cells on Llama-3.1-8B + Mistral-7B-Instruct-v0.3 (real GPU). Full audit: docs/NOVELTY_AUDIT.md
