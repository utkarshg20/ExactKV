We built ExactKV as a **compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression** — not another speed benchmark.

Recent **Phase H+ scale_7b** results (**1500 real-GPU cells**, `exactkv_failures = 0`):

• Llama-3.1-8B: `noop`/`int8` score **1.0**; `int4_sim` **0.684**
• SpectralQuant rows: **fallback/proxy** when dependency unavailable
• Shard rows: **probe-first** heuristic — not full Shard integration

Phase A (336 cells) remains historical cross-model supporting evidence.

The system is fully reproducible from published JSON reports: benchmark → leaderboard → publication artifacts. ExactKV is **not a production serving system** and does **not reproduce VeriCache**. Compression ratios in reports are **stored tensor byte ratios**, not active GPU memory savings claims.

If your team compresses KV caches, you need token-level equivalence testing — not just memory ratios.

#MachineLearning #LLM #MLOps #Research #KVCache
