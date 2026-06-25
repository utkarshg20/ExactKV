# ExactKV Novelty Audit (Phase I)

## 1. Executive summary

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression. It measures token-level drift, first divergence, acceptance rate, verifier agreement, and exactness failures across compressors and models.

- Prior-art systems catalogued: **14**
- Primary sources verified: **7**
- Sources pending: **5**
- Allowed claims: **8**; qualified: **2**; forbidden: **8**

## 2. What ExactKV is

A research-grade, compressor-agnostic evaluation framework and public leaderboard for LLM KV-cache compression exactness.

## 3. What ExactKV is not

- Not a production serving system
- Not a VeriCache reproduction or throughput competitor
- Not proof of end-to-end inference speedups or active GPU memory savings
- Not a verified real SpectralQuant or full Shard integration in the current environment

## 4. Closest prior art

**VeriCache**

## 5. Prior-art matrix

See `reports/novelty_audit_matrix.csv` for the full table.

| System | Category | Evidence | Overlap | Differentiator |
|--------|----------|----------|---------|----------------|
| VeriCache | verifier_mediated_inference_system | verified | Draft/verify/commit semantics and acceptance-oriented evalua | ExactKV is a public exactness benchmark/leaderboard platform |
| KVQuant | kv_compression_method | verified | KV quantization affects token generation; ExactKV can evalua | ExactKV compares compressors on token drift metrics across m |
| KIVI | kv_compression_method | verified | Asymmetric KV quantization is directly relevant to ExactKV c | ExactKV measures cross-compressor exactness, not KIVI throug |
| TurboQuant | kv_compression_method | verified | Same problem domain (KV compression). | ExactKV provides benchmark harness, not TurboQuant algorithm |
| QuantSpec | speculative_decoding_system | source_pending | Acceptance-style metrics may overlap conceptually. | ExactKV focuses on KV compression exactness across compresso |
| SparseSpec | speculative_decoding_system | source_pending | Draft acceptance concepts. | ExactKV is compressor-agnostic KV drift benchmarking. |
| SpecAttn | speculative_decoding_system | source_pending |  |  |
| MagicDec | speculative_decoding_system | source_pending |  |  |
| LMCache | kv_storage_or_offload_system | verified | Both concern KV caches in LLM inference. | ExactKV measures compression exactness, not prefix-cache hit |
| CacheGen | kv_storage_or_offload_system | source_pending |  |  |
| ShardCache (shard-kv) | cache_database_benchmark_system | verified | Name collision risk: 'shard' in ExactKV is a probe adapter,  | ExactKV evaluates LLM KV compression token drift; ShardCache |
| Redis / Valkey cache benchmarks | cache_database_benchmark_system | ambiguous | Both use 'KV' terminology. | ExactKV is transformer KV-cache exactness, not Redis object  |

## 6. VeriCache relationship

VeriCache is the closest conceptual prior art: compressed-KV draft + full-KV verification for lossless inference with serving optimizations. ExactKV must **not** claim to invent this loop or reproduce VeriCache throughput/memory results.

## 7. ShardCache / shard-kv relationship

ShardCache (shard-kv) is primarily a **cache database / LMCache storage benchmark** system. It is adjacent but not equivalent to transformer KV-cache token-drift exactness benchmarking unless primary evidence shows otherwise (none verified here).

## 8. External compressor relationship

KVQuant, KIVI, TurboQuant, and SpectralQuant are **compression methods** or adapter targets. ExactKV compares them in a benchmark harness; it does not subsume their algorithms.

## 9. Storage/offload system relationship

LMCache and CacheGen focus on KV **storage, reuse, and serving**. ExactKV does not replace them; overlap is limited to shared KV terminology.

## 10. Speculative decoding relationship

QuantSpec / SparseSpec / MagicDec / SpecAttn are adjacent speculative-decoding literature. ExactKV must not claim invention of acceptance measurement without qualification.

## 11. ExactKV defensible novelty

- Public, compressor-agnostic **token-level drift** and **first-divergence** leaderboard
- Phase G canonical divergence authority across cells
- Reproducible artifact pipeline (benchmark → leaderboard → public_release)
- Real 7B/8B scale panel with zero ExactKV failures (current evidence)

## 12. Claims allowed

- **ExactKV is a KV-cache compression exactness benchmark.** — allowed
- **ExactKV is a compressor-agnostic token-level drift leaderboard.** — allowed
- **ExactKV measures first divergence across compressors.** — allowed
- **ExactKV reports acceptance rate and accepted span across compressors.** — allowed
- **ExactKV has a real Triton KV compression kernel path.** — allowed_with_qualification
- **ExactKV is a research-grade evaluation framework.** — allowed
- **ExactKV evaluates real 7B/8B models.** — allowed
- **ExactKV includes SpectralQuant fallback/proxy support.** — allowed
- **ExactKV includes Shard probe-first analysis.** — allowed
- **ExactKV is a public benchmark platform.** — allowed_with_qualification

## 13. Claims requiring qualification

- ExactKV has a real Triton KV compression kernel path.: ExactKV includes a CUDA/Triton KV compression kernel microbenchmark path (tested shape/hardware only).
- ExactKV is a public benchmark platform.: ExactKV publishes reproducible benchmark artifacts and a public leaderboard bundle.

## 14. Claims forbidden

- ExactKV is the first system like this.
- ExactKV reproduces VeriCache.
- ExactKV invented compressed-KV verification.
- ExactKV proves end-to-end speedups.
- ExactKV proves active GPU memory savings.
- ExactKV is production ready.
- ExactKV compares real SpectralQuant.
- ExactKV compares real Shard.

## 15. Remaining uncertainty

- QuantSpec, SparseSpec, MagicDec, SpecAttn, CacheGen: **source_pending**
- SpectralQuant real dependency not available in current environment
- Uniqueness vs all exactness benchmarks: **not established** — do not claim 'first'

## 16. Recommended public positioning

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression. It measures token-level drift, first divergence, acceptance rate, verifier agreement, and exactness failures across compressors and models.
