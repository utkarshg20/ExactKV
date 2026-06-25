We built ExactKV as an **evaluation framework for KV cache compression robustness** — not another speed benchmark.

Recent cross-model results (28 ranked model×compressor cells):

• INT8 remains the strongest baseline (mean score 0.916)
• Simulated INT4 and external probe adapters show higher token-level divergence
• Verifier agreement drops before ExactKV failures appear — catching drift early matters

The system is fully reproducible from published JSON reports: benchmark → leaderboard → publication artifacts. No hidden inference runs, no serving claims.

If your team compresses KV caches, you need token-level equivalence testing — not just memory ratios.

#MachineLearning #LLM #MLOps #Research #KVCache
