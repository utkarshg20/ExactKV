# ExactKV Metric Definitions (Phase J freeze)

Canonical definitions for public-facing metrics. Each entry lists source artifact, public claim status, and limitations.

---

## acceptance_rate

**Definition:** Fraction of drafted tokens accepted by the full-KV verifier across ExactKV rounds in a cell.

**Source:** `reports/scale_7b/raw.json` → `cells[].exactkv.acceptance.acceptance_rate`; aggregated in `reports/scale_7b/leaderboard.json`.

**Public claim status:** Allowed.

**Limitation:** Panel-scoped; greedy decoding only; not throughput or serving acceptance.

---

## first_divergence_index

**Definition:** Index of the first token where lossy compressed-KV greedy output diverges from full-KV reference (Phase G canonical authority when present).

**Source:** Phase G `FirstDivergenceAuthority` in cells; `canonical_first_divergence_index` in unified truth.

**Public claim status:** Allowed.

**Limitation:** Null when no divergence observed in panel; not a universal worst-case bound.

---

## avg_accepted_span

**Definition:** Mean number of tokens accepted per ExactKV verification round (accepted span per round).

**Source:** `cells[].exactkv.acceptance.avg_accepted_per_round` in scale raw.

**Public claim status:** Allowed with panel scope.

**Limitation:** Depends on `draft_len` and prompt panel; not comparable across unlike configs without normalization.

---

## verifier_agreement_score

**Definition:** Agreement between verifier predictions and full-KV reference across measured tokens (leaderboard field: `verifier_agreement`).

**Source:** `reports/scale_7b/leaderboard.json` entries.

**Public claim status:** Allowed.

**Limitation:** Trace-only evaluation layer; not runtime commit path.

---

## exactkv_failure_rate

**Definition:** Fraction of cells where `exactkv_failure=true` (final ExactKV output ≠ full-KV greedy).

**Source:** `reports/scale_7b/scale_summary.json` → `exactkv_failures`; per-cell in raw.

**Public claim status:** Allowed (current scale panel: **0 failures**).

**Limitation:** Hard gate on tested panel only; new compressors/models require re-evaluation.

---

## compression_ratio

**Definition:** Ratio of compressed stored KV bytes to full KV bytes for a cell (`compressed_bytes / full_bytes`).

**Source:** `cells[].memory.compression_ratio` in raw; leaderboard `compression_ratio`.

**Public claim status:** Qualified — **stored tensor byte ratio only**.

**Limitation:** Not active GPU memory unless explicitly measured; simulated compressors may use int8 containers.

---

## stored tensor byte ratio

**Definition:** Byte-level accounting of serialized/stored KV tensor payloads in benchmark cells.

**Source:** `cells[].memory.stored_kv_bytes`, `full_bytes`.

**Public claim status:** Allowed with qualification.

**Limitation:** Does not prove VRAM savings at serving time.

---

## kernel_microbenchmark_speedup

**Definition:** Ratio of torch vs Triton kernel latency for KV compression microbenchmark on a fixed `kv_shape`.

**Source:** `reports/phaseF_kernel_benchmark.json` → `speedups[]`.

**Public claim status:** Qualified — kernel microbenchmark only.

**Limitation:** Not end-to-end inference; tested shape/hardware only; block_sparse uses torch path.

---

## divergence_type

**Definition:** Phase G classification of first divergence: `token_mismatch`, `length_drift`, `kernel_inconsistency`, `verifier_disagreement`, `none`.

**Source:** Phase G unified truth / cell divergence authority.

**Public claim status:** Allowed.

**Limitation:** Taxonomy is ExactKV-specific; not all external systems report identical types.

---

## kernel_consistency

**Definition:** Whether compressed KV kernel path matches reference implementation for tested operations (Phase F / kernel tests).

**Source:** `reports/phaseF_kernel_benchmark.json`, Phase G records.

**Public claim status:** Allowed in kernel scope.

**Limitation:** Microbenchmark scope; block_sparse not Triton-accelerated in current evidence.

---

## fallback/proxy adapter

**Definition:** External compressor slot filled by a stand-in implementation when the real dependency is unavailable (e.g. SpectralQuant `int4_sim` scaling).

**Source:** `spectralquant_available` probe; adapter metadata in cells.

**Public claim status:** Allowed when disclosed as fallback/proxy.

**Limitation:** Must not be called “real SpectralQuant” without dependency evidence.

---

## probe-first adapter

**Definition:** Heuristic analysis adapter that scores/probes without full external integration (e.g. `shard_real`, `probe_only=true`).

**Source:** `exactkv/adapters/shard_real_adapter.py`; leaderboard `probe_only` flags.

**Public claim status:** Allowed when disclosed as probe-first.

**Limitation:** Not a full Shard or ShardCache product integration.
