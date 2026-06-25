We built ExactKV as a **compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression** — not another speed benchmark.

Recent cross-model results (28 ranked model×compressor cells on Phase A; Phase H+ adds 1500 real-GPU cells on 7B/8B models):

• INT8 remains the strongest baseline (mean score 0.916)
• Simulated INT4 and external probe adapters (Shard probe-first; SpectralQuant fallback/proxy) show higher token-level divergence
• Verifier agreement drops before ExactKV failures appear — catching drift early matters

The system is fully reproducible from published JSON reports: benchmark → leaderboard → publication artifacts. ExactKV is **not a production serving system** and does **not reproduce VeriCache**. Compression ratios in reports are **stored tensor byte ratios**, not active GPU memory savings claims.

If your team compresses KV caches, you need token-level equivalence testing — not just memory ratios.

#MachineLearning #LLM #MLOps #Research #KVCache
