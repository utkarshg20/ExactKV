1/9 KV cache compression is everywhere. Token-level drift is not. ExactKV is a compressor-agnostic crash-test and leaderboard for LLM KV-cache compression exactness.

2/9 Structured output case: `int4_sim` on Qwen 0.5B. First divergence at token 1. Acceptance 0.50.

3/9 Worst INT4 cell: acceptance 0.33, divergence at token 1. ExactKV caught it — zero silent failures.

4/9 Phase H+ scale: **1500 real-GPU cells**, Llama-3.1-8B + Mistral-7B-Instruct-v0.3, **exactkv_failures = 0**.

5/9 Llama leaderboard: `noop`/`int8` score 1.0; `int4_sim` 0.684; `shard` (probe-first) 0.544.

6/9 `noop` leads the cross-model mean score (0.995 across 4 models).

7/9 Simulated INT4 + external probes (shard, kvquant mock) show 2–3× higher divergence on small models.

8/9 ExactKV = trace-only verification. Not a production serving system. Does not reproduce VeriCache. Shard = probe-first; SpectralQuant = fallback/proxy when dependency missing.

9/9 Phase H+ scale: 1500 cells on Llama-3.1-8B + Mistral-7B-Instruct-v0.3 (real GPU). Full audit: docs/NOVELTY_AUDIT.md
