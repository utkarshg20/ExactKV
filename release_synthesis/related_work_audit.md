# ExactKV Related-Work & Prior-Art Audit (Release Synthesis — Part 4)

Prior-art catalogue with verified primary sources where available. This audit
**confirms and does not weaken** the repo's own
[`docs/NOVELTY_AUDIT.md`](../docs/NOVELTY_AUDIT.md). Verified citations are in
[`references.bib`](references.bib). Systems without a confirmed primary source are
marked **source-pending** and carry **no fabricated citation**.

## Positioning

ExactKV is a **compressor-agnostic token-level drift / first-divergence
crash-test and leaderboard** for LLM KV-cache compression, built on
draft/verify/commit (verifier-mediated) semantics. It is an **evaluation
framework**, not a compression algorithm, serving system, or storage layer.

## Prior-art matrix

| System | Category | Source | Overlap with ExactKV | Difference | Can ExactKV compare? | Can claim superiority? | Evidence status |
|--------|----------|--------|----------------------|------------|----------------------|------------------------|-----------------|
| Speculative Decoding (Leviathan et al.) | draft/verify decoding | arXiv:2211.17192 (ICML'23) | Draft-then-verify acceptance loop | Speedup technique for exact sampling; ExactKV uses verify semantics to *measure exactness*, not to accelerate | Conceptual basis | No | verified |
| VeriCache | verifier-mediated compressed-KV inference | [arXiv:2605.17613](https://arxiv.org/abs/2605.17613); local `paper/VeriCache.pdf` | **Closest prior art:** compressed-KV draft + full-KV verify for lossless inference | VeriCache targets serving throughput/memory; ExactKV is a public exactness benchmark | Conceptually adjacent; **not reproduced** | **No** | verified |
| KVQuant | KV quantization method | arXiv:2401.18079 (NeurIPS'24) | Low-bit KV quant affects token generation | A compression *method*; ExactKV benchmarks methods on drift | Yes (as a compressor under test) | No (no same-task head-to-head) | verified |
| KIVI | asymmetric 2-bit KV quant | arXiv:2402.02750 (ICML'24) | Asymmetric K/V quant directly relevant to ExactKV `k8_v4_sim` | Method, not benchmark | Yes (adapter target) | No | verified |
| SnapKV | KV eviction / selection | arXiv:2404.14469 (NeurIPS'24) | KV compression affects exactness | Token-selection method | Yes (factory-only adapter in repo) | No | verified |
| TurboQuant | KV quantization method | repo adapter target (`docs/TURBOQUANT_*`) | Same domain (KV compression) | Method, not benchmark | Probe/adapter only | **No** (no real same-task comparison) | source-pending |
| SpectralQuant | KV quantization method | repo fallback/proxy slot | Compressor slot under test | **Fallback/proxy** (delegates to int4_sim) when dependency missing | Only as disclosed fallback | **No** | source-pending (real dependency unavailable) |
| Shard / ShardCache / shard-kv | cache DB / LMCache storage benchmark | repo probe-only slot | Name overlap ("shard"); KV terminology | Storage/semantic cache benchmark, **not** transformer token-drift exactness | **Probe-first heuristic only** | **No** | verified (as distinct category) |
| LMCache | KV storage/offload layer | arXiv:2510.09665 | Both concern KV caches | Storage/reuse/serving, not exactness | No (different task) | No | verified |
| CacheGen | KV compression + streaming | arXiv:2310.07240 (SIGCOMM'24) | KV bitstream compression | Network/streaming focus | No (different task) | No | verified |
| MagicDec | speculative decoding (long context) | arXiv:2408.11049 (ICLR'25) | Draft + sparse-KV acceptance | Throughput/latency technique | Conceptually adjacent | No | verified |
| PagedAttention / vLLM | KV memory management / serving | arXiv:2309.06180 (SOSP'23) | KV memory in serving | Serving runtime, not exactness benchmark | No (probe-only feasibility in repo) | No | verified |
| QuantSpec | speculative decoding system | — | Acceptance-style metrics | Different focus | Conceptually adjacent | No | source-pending |
| SparseSpec | speculative decoding system | — | Draft acceptance concepts | Different focus | Conceptually adjacent | No | source-pending |
| SpecAttn | speculative attention | — | Acceptance/attention | Different focus | Conceptually adjacent | No | source-pending |

## VeriCache (closest prior art — high overlap)

VeriCache [arXiv:2605.17613](https://arxiv.org/abs/2605.17613) is **closer than a
casual skim suggests**. It explicitly:

- uses **compressed KV to draft** and **full KV to verify/correct**;
- guarantees **identical greedy-decoding output** to full-KV inference;
- frames the same problem: lossy KV looks fine on short/aggregate metrics but
  **diverges more as decoding continues**, with failures in code/tool-calling.

VeriCache's contribution is **not** merely “draft then verify.” It is a **serving/system**
design: compressed KV on GPU, full KV in CPU/storage, load full KV only for
verification, **cross-resource staggering** (HBM-bound draft vs interconnect-bound
verify), reporting serving throughput with identical outputs.

ExactKV **must not** claim novelty for compressed-KV draft + full-KV verify.
VeriCache owns that as a lossless **serving framework**.

| Question | VeriCache | ExactKV |
|----------|-----------|---------|
| Primary goal | Serve faster, same outputs | Measure drift / first divergence |
| System stack | Scheduling, tiering, staggering | Evaluation harness only |
| Reproduced here? | **No** | N/A |

- ExactKV **must not** claim to invent, reproduce, or beat VeriCache.
- External primary source verified: [arXiv:2605.17613](https://arxiv.org/abs/2605.17613).
  Local copy: `paper/VeriCache.pdf`.

**Why ExactKV still matters:** VeriCache uses verification to **serve** without
changing outputs; ExactKV uses verification to **measure where compressors drift**.

## Shard / ShardCache disambiguation

"Shard" in ExactKV is a **probe-first heuristic adapter slot** (`probe_only=true`).
ShardCache / shard-kv in the wild is primarily a **cache-database / LMCache storage
benchmark** — adjacent but not equivalent to transformer KV-cache token-drift
exactness benchmarking. Do not conflate them; do not claim real Shard integration.

## Superiority claims

ExactKV has **no** same-task / same-model / same-metric head-to-head that would
license a "beats X" claim against VeriCache, TurboQuant, Shard, or any other
system. All comparative language is restricted to **descriptive positioning**.

## Uniqueness

Uniqueness vs. all exactness benchmarks is **not established**. Do **not** claim
"first ever," "first and only," or "nothing like this exists."
