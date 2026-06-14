# VeriCache Parity Audit (Phase 11A)

**Status:** Audit and claim firewall only — **no runtime implementation in Phase 11A.**

> ExactKV is **not** a full VeriCache reproduction today.  
> ExactKV currently reproduces **VeriCache-style draft/verify algorithmic semantics** on a Hugging Face correctness harness — **not** the VeriCache serving system, scheduler, or paper-level performance results.

Companion: [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) · [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md) · [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md)

**Attribution:** Draft-then-verify algorithm from Yao et al., *VeriCache: Turning Lossy KV Cache into Lossless LLM Inference*, arXiv:2605.17613, 2026. ExactKV does **not** claim to have invented this algorithm.

---

## 1. Executive summary

| Layer | VeriCache (paper/system) | ExactKV today |
|---|---|---|
| **Algorithm** | Lossy KV drafts; full KV verifies; exact output | **Mostly reproduced** on HF greedy harness |
| **Evaluation** | Correctness + throughput + memory + benchmarks | **Correctness-first**; throughput/memory not reproduced as benefits |
| **Systems** | vLLM, LMCache, remote prefix, staggering | **Missing** — documented no-go or not started |
| **External methods** | Paper compressor ecosystem | **Partial** — built-ins + restricted probes (Shard, SpectralQuant) |

**Shard** and **SpectralQuant** are **restricted external-method probes** under ExactKV verification — **not** VeriCache system parity and **not** paper-level compressor coverage.

---

## 2. Parity table

| VeriCache paper capability | ExactKV current status | Status | Evidence in ExactKV | Missing implementation work | Required tests | Claim allowed today | Claim forbidden today |
|---|---|---|---|---|---|---|---|
| **Compressed-KV drafter** | HF `ExactKVGenerator` drafts from materialized compressed KV | **done** | `exactkv/runtime/exactkv_generator.py`; Exp 012–016 panels | Hot non-materializing draft path; production paged draft blocks | `tests/test_noop_exactkv.py`, compressor exactness suites | Lossy KV proposes draft tokens on tested panels | Compressed KV is always safe without verification |
| **Full-KV verifier** | Sequential default; span opt-in | **done** | `exactkv/verification/engine.py`; Exp 029 span grid | Paper-style parallel batch verify at serving scale; bonus-token acceptance | `tests/test_span_verification*.py`, `tests/test_int8_exactkv.py` | Full-KV verifier is authoritative on tested panels | Verifier optional for production correctness |
| **Exact greedy output preservation** | `exactkv_failures == 0` on published grids | **done** | Exp 012, 015, 016, 029, 033; terminal demos 034b, 037 | Universal proof for all models/compressors/prompts | Panel pytest + `exactkv_failures` invariant tests | On cited panels, final output matches full greedy | Exactness on all prompts without testing |
| **Longest accepted prefix / correction** | Left-to-right accept; first mismatch → correction | **done** | `exactkv/verification/acceptance.py`; Exp 034/034b traces | Bonus-token acceptance (VeriCache extra commit) | Acceptance unit tests; killer demo tests | Drift detected and corrected on cited traces | Every compressor self-corrects without verifier |
| **Compressor abstraction** | `KVCompressor` + factory/registry + adapters | **mostly done** | `exactkv/compressors/`; built-ins; restricted adapters (TurboQuant, KIVI, KVQuant, SnapKV smoke) | Paper-scale compressor matrix; integrated Shard/SpectralQuant defaults | Per-compressor smoke + panel tests | Compressor-agnostic harness on tested compressors | All compressors integrated or ranked equally |
| **Compressed KV memory accounting** | Paper tracks compressed footprint vs full | **partial** | `exactkv/metrics/memory.py`; workspace fields in Exp 004/011 reports | Active compressed-KV residency model (not materialize-then-draft) | `tests/test_workspace_memory_summary.py` | Honest workspace/diagnostic memory accounting on tested path | Active GPU memory savings from compression |
| **Full KV storage separate from compressed KV** | Dual-cache architecture | **partial** | `FullKVState` + `CompressedKVState`; alignment invariant in generator | Dedicated storage manager; tiered residency; non-materializing hot path | Alignment invariant tests; cache state tests | Authoritative full KV separate from draft compressed KV in HF harness | Dual-cache equals VeriCache production memory layout |
| **CPU / host full-KV cache** | Offload full KV to host for verify | **missing** | Exp 017/007 serving harness discusses ownership only | Host-resident full KV pool; sync policy | Host-cache round-trip tests; memory peak tests | Not implemented — correctness harness keeps KV on same device in V1 path | CPU offload reduces GPU memory today |
| **Disk / storage-backed KV cache** | Persistent full/compressed KV tiers | **missing** | None in runtime | Block store, mmap, checkpoint/restore of KV blocks | Storage backend integration tests | Not implemented | Disk-backed KV is production-ready in ExactKV |
| **Remote prefix cache** | Remote drafter + near-storage verify | **partial** | Phase 11H: `LoopbackPrefixCache` + `PrefixRestorePlan` — loopback only | RPC transport, real remote tier, multi-process replay | `tests/test_remote_prefix_cache_semantics.py` | Prefix identity + loopback round-trip on tiny tensors only | Remote prefix caching reproduced or runtime exists |
| **vLLM integration** | Serving engine integration | **missing** | Exp 007/017: **no-go reaffirmed**; sidecar probe only | PagedAttention bridge, block export, scheduler hooks | Serving integration tests (none today) | Sidecar/metadata probe feasibility only (Exp 017) | vLLM integration exists or is production-ready |
| **LMCache integration** | External KV store / tiering | **missing** | Exp 007/017: **no-go reaffirmed** | LMCache-backed full KV restore for verify steps | Tier restore correctness tests | Not implemented | LMCache integration exists |
| **Extended verification** | Parallel / single-pass multi-token verify | **partial** | Span verify (Exp 028–029); bonus token **disabled** | Paper parallel verify at scale; bonus-token acceptance policy | Exp 029 grid; span parity tests (Exp 030b) | Span ≡ sequential exactness on tested grid | Extended verify equals VeriCache throughput path |
| **Cross-resource staggering / scheduling** | Overlap draft, verify, transfer across devices | **missing** | `docs/NON_GOALS.md` explicit non-goal | Scheduler, stream overlap, HBM accounting | Scheduler invariant + timing harness | Not implemented | Cross-resource staggering reproduced |
| **Batching / serving runtime** | Multi-request serving | **missing** | Single-request HF loop only | Request batching, paged batch tables, continuous batching | Multi-request exactness tests | Single-request research harness only | Production serving runtime |
| **Large-model evaluation** | Paper-scale models | **partial** | Qwen 0.5B–3B full panels; Llama-3.1-8B small suite (Exp 033); Shard probe on Llama 8B | Paper model set at paper context lengths | Exp 033, 015, 016 tests | Exactness on cited Qwen/Llama panels only | Paper-scale model coverage complete |
| **Throughput evaluation** | End-to-end tokens/sec after verify | **partial** | Exp 030 diagnostic timing (ExactKV **slower** on panel); Phase 11I methodology contract | Serving-scale harness; staggering | `tests/test_throughput_benchmark_contract.py`; Exp 030 methodology | Panel-bound **diagnostic** timing with exactness gate — **not** benefit | Speedup, throughput, latency, tokens/sec improvement |
| **Quality / correctness evaluation** | Exact + task metrics | **done** (ExactKV focus) | V10 suites; `exactkv_failures`; benchmark-gap docs | Official LongBench/RULER scores (out of scope for ExactKV claim) | Benchmark runner tests; gap analysis tests | Token-level exactness + acceptance on tested panels | Outcome benchmarks replaced by ExactKV |
| **KL / distributional divergence evaluation** | Distributional checks beyond greedy | **missing** | `docs/FUTURE_RESEARCH.md` mentions KL | KL / logprob divergence harness vs full KV | Statistical divergence tests | Not implemented | KL parity with VeriCache paper |
| **Paper-like compressor coverage** | Broad compressor ecosystem in paper | **partial** | Built-ins + restricted factory adapters; Shard/SpectralQuant probes | Integrated paper compressors at paper settings | Per-backend restricted tests | Cited compressor rows with caveats | VeriCache compressor matrix reproduced |
| **Paper-like benchmark coverage** | Paper benchmark suite | **partial** | V10 custom suites; Phase 11J panel **contract**; LongBench-**style** demo only | Locked paper panel run + gates | `tests/test_paper_panel_contract.py` | Panel contract + gate requirements — **not** paper reproduction | Paper benchmarks reproduced as ExactKV results |

---

## 3. What ExactKV already covers (algorithm layer)

- **Draft → verify → commit → align** loop with greedy decoding (`ExactKVGenerator`).
- **Authoritative full KV** vs **compressed draft KV** with per-round alignment asserts.
- **Token-level exactness** gate: `exactkv_output_ids == full_output_ids` when `exactkv_failures == 0`.
- **Acceptance metrics**: accepted prefix length, rejection, correction token (Exp 012–025).
- **HF-first benchmark harness**: `full` / `lossy` / `exactkv` modes (`exactkv/benchmarks/`).
- **Span verification option** with exactness grid (Exp 029) — partial extended-verify path.
- **Correctness-first demos** and leaderboard separating full-panel vs restricted tiers.

---

## 4. What is partial

| Area | Gap |
|---|---|
| **Memory story** | Workspace accounting exists; **active** compressed-KV residency and paper-style HBM savings **not proven** (Exp 031). |
| **Dual-cache systems** | Logical separation yes; **storage manager**, offload tiers, and materialization-free hot path **no**. |
| **Extended verification** | Span verify yes; **bonus-token** and paper **parallel verify at serving scale** **no**. |
| **Compressor / model coverage** | Strong on Qwen built-ins; **restricted** external probes only for Shard/SpectralQuant; not paper matrix. |
| **Serving context** | Harness + sidecar **probe** (Exp 017); **not** integration. vLLM (11F), LMCache (11G) contracts; remote prefix **loopback** (11H) — **not** runtime. |

---

## 5. What is missing (systems layer)

- vLLM integration (documented **no-go**).
- LMCache integration (documented **no-go**).
- Remote prefix caching **runtime** (Phase 11H loopback semantics only).
- Cross-resource staggering / async transfer / CPU host cache tiers.
- Batching and production serving runtime.
- Throughput reproduction harness with VeriCache-comparable methodology.
- KL / distributional divergence evaluation.
- Paper-like benchmark and compressor reproduction panels.

---

## 6. Claim firewall (today)

### Allowed (scoped)

- ExactKV reproduces **VeriCache-style draft/verify semantics** on a **Hugging Face correctness harness** for **tested panels**.
- `exactkv_failures == 0` on cited experiments.
- ExactKV is a **crash-test / evaluation lab** complementary to outcome benchmarks ([`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md)).
- Shard / SpectralQuant have **restricted external-method** results — not VeriCache system parity.

### Forbidden (until Stage 10 gate in roadmap)

- “ExactKV reproduces VeriCache” (full system).
- “ExactKV achieves VeriCache throughput / memory benefits.”
- Speedup, throughput, latency, tokens/sec, active GPU memory savings.
- Production serving, vLLM integration, LMCache integration.
- Paper-level performance claims without measured ExactKV harness results.
- External VeriCache / Shard / SpectralQuant paper numbers as ExactKV results.

---

## 7. Relationship to Shard and SpectralQuant

| Method | VeriCache parity? | ExactKV status |
|---|---|---|
| **Shard** | External compressor/drafter in paper ecosystem | RESTRICTED BACKEND external-drafter probe (Exp 039–041); not integrated; not system parity |
| **SpectralQuant** | External method | RESTRICTED BACKEND materializing adapter (Exp 045); small panel; not system parity |
| **Built-in INT8 / K8V4** | Harness compressors | Full-panel integrated rows on V10 suites |

---

## 8. References

| Doc | Relevance |
|---|---|
| [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) | Staged path toward VeriCache-equivalent **systems** |
| [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md) | vLLM/LMCache no-go |
| [`EXPERIMENT_030_DIAGNOSTIC_TIMING.md`](EXPERIMENT_030_DIAGNOSTIC_TIMING.md) | Throughput honesty |
| [`EXPERIMENT_031_GPU_MEMORY_ISOLATION.md`](EXPERIMENT_031_GPU_MEMORY_ISOLATION.md) | Memory honesty |
| [`NON_GOALS.md`](NON_GOALS.md) | Cross-resource staggering, batching, serving |
| [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) | Public claim firewall |
