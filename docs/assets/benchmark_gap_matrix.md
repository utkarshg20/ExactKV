# Benchmark Gap Matrix (compact)

_Artifact companion to [`BENCHMARK_GAP_ANALYSIS.md`](../BENCHMARK_GAP_ANALYSIS.md). ExactKV evidence only — no external benchmark scores cited._

| Layer | LongBench / QA | RULER / needle | Perplexity | Compression reports | ExactKV |
|---|---|---|---|---|---|
| **Measures** | Final answer quality | Retrieval success | Distributional fit | Storage / throughput | Full-KV behavioral equivalence |
| **Can miss** | Token-path drift | Route under compressed KV | Exact greedy identity | Behavioral drift | Task quality; speed; VRAM |

| ExactKV example | Outcome layer | Path / equivalence | exactkv_failures |
|---|---|---|---|
| Exp 037 LongBench-style | Outcome green | Path drifted → repaired | 0 |
| Exp 041 Shard | Not scored | 56.25% draft divergence | 0 |
| Exp 045 SpectralQuant | 12/12 exact final | 11/12 draft divergence | 0 |
| Exp 029 / V10 grids | N/A or acceptance | Full-KV preserved | 0 |

**Message:** Outcome benchmarks and ExactKV answer different questions.
