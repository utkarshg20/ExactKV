# V5 Scope — DRAFT (not approved, not implemented)

> **Status:** Draft proposal only. **V5 is not implemented.** Nothing in this
> document has been built. This is a planning artifact to be reviewed and
> revised before any V5 work begins. It introduces no code, no schema changes,
> and no experiments.

---

## V5 goal

Add **workspace-aware memory accounting** and **prepare for real backend
adapters**, so that ExactKV can describe memory honestly (stored vs working vs
scratch vs total) and lay the groundwork for plugging in real quantisation
backends — all while continuing to make **no speed or production-readiness
claims** until something is actually measured.

---

## 1. Workspace-aware memory schema (proposed)

Today ExactKV reports a small set of byte counts and a `memory_claim_note`. The
proposed V5 memory schema would distinguish where memory actually lives:

| Field | Meaning |
|---|---|
| `stored_kv_bytes` | Bytes of the compressed/stored KV cache as held between steps. |
| `materialized_working_kv_bytes` | Bytes when a compressed cache is dequantized/materialized for use during a step. |
| `metadata_bytes` | Scales, zero-points, indices, and other per-tensor compression metadata. |
| `temporary_workspace_bytes` | Transient scratch buffers allocated during compress/materialize. |
| `active_gpu_kv_bytes` | KV bytes resident on the accelerator at peak (when a device is used). |
| `total_system_kv_bytes` | Aggregate KV-related footprint across the above categories. |
| `supports_real_bytes_claim` | Whether these numbers reflect real storage or a simulation. |

> This schema is **proposed**, not locked. The current report schema is
> unchanged. Adopting it would be a deliberate, separately reviewed step.

---

## 2. Clear distinction between memory categories

A central V5 motivation is that "compressed bytes" is ambiguous. V5 should make
the following explicit and never conflate them:

* **Stored compressed cache** — what persists between decode steps.
* **Materialized working cache** — the (often larger) dequantized form needed to
  actually compute attention.
* **Temporary scratch buffers** — transient allocations during
  compress/materialize.
* **Total memory** — the honest peak footprint, which can exceed naive
  "stored compressed bytes".

The point: a compressor can shrink *stored* bytes while still materializing a
full-precision working cache — so stored-byte savings alone can be misleading.

---

## 3. Better memory honesty for simulated compressors

* Simulated (`_sim`) compressors must keep `supports_real_bytes_claim=False`.
* Under the new schema, `_sim` compressors should report `stored_kv_bytes` as
  the `int8`-container reality, not a hypothetical packed size.
* Reports and Markdown should continue to mark simulated compressors clearly and
  keep average-effective-bit-width labelled as a comparison aid only.

---

## 4. Optional first real backend planning (planning only — not implementation)

V5 may *plan* (design interfaces, document trade-offs) for a first real backend,
without implementing it:

* **KIVI** — asymmetric KV quantization; a natural fit given V4's asymmetric
  findings.
* **kvpress** — KV compression toolkit; useful for adapter-shape exploration.
* **TurboQuant-style adapter** — possible later, lower priority.

Any of these would be introduced behind the existing `KVCompressor` protocol via
an adapter, and only after the capability/memory-honesty story is settled.

---

## 5. Compatibility questions to resolve before building

* **How to expose real backend capabilities** — how should a real backend declare
  bit-widths, real-bytes support, device requirements, and asymmetry through
  `CompressorCapabilities` without special-casing?
* **How to compare simulated vs real compressors honestly** — same acceptance
  harness, but memory numbers are only comparable when `supports_real_bytes_claim`
  matches; reports must prevent apples-to-oranges memory comparisons.
* **How to avoid speed claims until measured** — keep the no-performance-field
  audit in force; any future timing must come with explicit hardware and
  methodology disclaimers and be opt-in, never a default report field.

---

## 6. Potential future experiments (proposed, not scheduled)

* **Experiment 004 — workspace-aware memory accounting.** Re-run an existing
  sweep configuration and report the full memory schema for each compressor,
  illustrating stored vs materialized vs scratch vs total. Acceptance behaviour
  unchanged; this experiment is about memory honesty, not speed.
* **Experiment 005 — first real backend adapter comparison.** Once one real
  backend adapter exists, compare it against the simulated compressors on
  acceptance behaviour and (real) memory, with explicit simulated-vs-real
  labelling.

---

## 7. Explicitly NOT in V5 unless separately approved

* ❌ vLLM integration
* ❌ LMCache integration
* ❌ CUDA / Triton kernels
* ❌ throughput / latency / tokens-per-second benchmarks
* ❌ production-serving claims
* ❌ CPU offload, batching, sampling, parallel verification, bonus-token
  acceptance
* ❌ real bit-packing presented as a default (only behind an explicit, reviewed
  real-backend path)

---

## 8. Guiding principle (unchanged from V1)

ExactKV stays **correctness-first**. V5 adds honesty about memory and a path
toward real backends, but it does not trade away the core guarantee
(`exactkv_output_ids == full_output_ids` under greedy decoding) and does not
introduce performance claims until they are carefully, explicitly measured.
