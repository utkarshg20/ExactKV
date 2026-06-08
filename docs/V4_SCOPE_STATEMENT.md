# ExactKV V4 Scope Statement

> **Status:** Phase 0 (scope documentation). No V4 implementation code exists yet.
> **Theme:** Asymmetric K/V compression — does compressing keys and values at
> different bit-widths improve ExactKV acceptance behaviour?
> **Constraints:** correctness-first, performance-silent, simulated compressors only.

---

## 1. V4 goal

V4 tests a single scientific question:

> **Does compressing keys and values at different bit-widths improve ExactKV
> acceptance behaviour compared with symmetric compression at a comparable
> average bit-width?**

Keys and values play different roles in attention. **Keys** drive attention
routing through the query–key dot product and softmax, so key quantisation error
propagates multiplicatively and can change *which* positions are attended to.
**Values** are aggregated *after* attention weights are formed, so value error is
additive and smoothed by the weighted sum. This asymmetry suggests that
compressing K and V identically is unlikely to be optimal — keys may need higher
precision than values at the same average bit-width.

V4 evaluates this purely with **acceptance behaviour** (acceptance rate, average
accepted length, first-divergence position, lossy-divergence count, rejection
count, and the hard `exactkv_failures == 0` correctness gate). Reconstruction MSE
is explicitly **not** used as a primary metric (see
[`FUTURE_RESEARCH_ASYMMETRIC_KV.md`](FUTURE_RESEARCH_ASYMMETRIC_KV.md) §3).

V4 reuses the entire V1–V3 stack (verification engine, runner, sweeps, reporting,
analysis, markdown renderer) and adds **only** new simulated compressors,
capability metadata, additive report fields, a leaderboard column, and one
documented experiment. No verification, generation, or rendering logic changes.

---

## 2. What V4 adds

### 2.1 `AsymmetricQuantSimCompressor` base
A single parameterized compressor, `AsymmetricQuantSimCompressor(k_bits, v_bits)`,
reusing the existing per-tensor symmetric quantisation math from
`exactkv/compressors/int8.py` and `exactkv/compressors/int4_sim.py`:

- Per-side supported widths: `full` (passthrough, no quantisation), `8` (real
  INT8), `4` (INT4 numeric range, stored in an `int8` container), `2` (INT2
  numeric range, stored in an `int8` container).
- K and V are quantised **independently** with independent scales; a side set to
  `full` is a bit-identical passthrough.
- `materialize_for_draft` dequantises each side at its own width.
- This fits the existing registry contract: named, no-arg registrations bind
  specific `(k_bits, v_bits)` pairs, so `get_compressor(name)` and `run_one`
  work unchanged.

### 2.2 Named asymmetric / ablation compressors

| Name | K width | V width | `is_simulated` | `supports_real_bytes_claim` | Role |
|---|---|---|---|---|---|
| `k8_v4_sim` | INT8 | INT4-sim | yes | no | asymmetric (V more aggressive) |
| `k8_v2_sim` | INT8 | INT2-sim | yes | no | asymmetric (V very aggressive) |
| `k4_v8_sim` | INT4-sim | INT8 | yes | no | asymmetric (K more aggressive) |
| `k_full_v4_sim` | full | INT4-sim | yes | no | V-only compression ablation |
| `k4_v_full_sim` | INT4-sim | full | yes | no | K-only compression ablation |
| `k8_v_full` | INT8 | full | **no** | **yes** | K-only INT8 ablation |
| `k_full_v8` | full | INT8 | **no** | **yes** | V-only INT8 ablation |

The `k*_v*_sim` set includes a simulated sub-INT8 width (INT4-sim or INT2-sim).
The `k8_v_full` and `k_full_v8` ablations use only real INT8 and full precision,
so they carry **no** `_sim` suffix and `is_simulated=False` (see §6).

These compressors let Experiment 003 directly test the paper's hypothesis: the
K-only vs V-only ablations isolate each tensor's contribution to acceptance
degradation.

### 2.3 `CompressorCapabilities` extension (additive, non-breaking)
Three new fields, with backward-compatible defaults so the existing four
compressors stay valid:

| New field | Type | Default | noop | int8 | int4_sim | debug_noise |
|---|---|---|---|---|---|---|
| `key_bit_width` | `int \| None` | `None` (= full) | `None` | `8` | `4` | `None` |
| `value_bit_width` | `int \| None` | `None` | `None` | `8` | `4` | `None` |
| `asymmetric` | `bool` | `False` | `False` | `False` | `False` | `False` |

`is_simulated` and `supports_real_bytes_claim` already exist and are reused. For
asymmetric compressors these become **instance-level** (derived in `__init__`
from the actual widths) rather than class-level, since they vary per
`(k_bits, v_bits)`. `run_one` reads `compressor.capabilities` via `hasattr`, so
this works unchanged.

### 2.4 Report schema additions
`exactkv/benchmarks/reports.py` gains `key_bit_width`, `value_bit_width`, and
`asymmetric` in the JSON per-result enrichment and as CSV columns. These are
additive; the recursive forbidden-field audit (`_assert_no_forbidden_fields`)
is unchanged and still mandatory.

### 2.5 Leaderboard rendering for K/V widths and average effective bit width
`exactkv/reporting/leaderboard.py` gains an optional view showing each
compressor's K and V widths plus a derived **average effective bit width**
(`(k_bits + v_bits) / 2`, treating `full` as the model dtype width). This lets
symmetric and asymmetric policies be compared at a matched average bit width.
This is pure rendering over existing analysis numbers; no new metric is computed
elsewhere and no model is re-run.

### 2.6 Experiment 003 — asymmetric KV sweep
`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`, generated via the existing
`exactkv sweep` + `exactkv report` pipeline over the `core` suite, comparing
symmetric baselines (`int8`, `int4_sim`) against the asymmetric set and the
ablations. Reports acceptance by compressor, lossy-divergence counts, rejection
counts, and ExactKV failures (must be 0).

### 2.7 README V4 results section
A V4 subsection with the asymmetric leaderboard table and a one-line takeaway,
preserving all existing honesty disclaimers.

### 2.8 Release notes for v0.4.0
`docs/RELEASE_NOTES_V0.4.0.md`, following the v0.3.0 structure (what each prior
version proved/added, what V4 adds, supported compressors, out-of-scope items,
known limitations, recommended next direction for V5).

---

## 3. What V4 explicitly does NOT add

- **No real compressor backends** — no TurboQuant, KIVI, kvpress, KVQuant,
  SnapKV. All V4 widths are simulated (`int8`-container), real INT8, or full
  passthrough only.
- **No real INT4 or INT2 bit-packing** — sub-INT8 widths stay in `int8`
  containers; `supports_real_bytes_claim=False` whenever a sub-INT8 width is
  used.
- **No workspace-aware memory schema** — deferred to V5 (see §7).
- **No serving stack** — no vLLM, LMCache, Triton, CUDA kernels, CPU offload.
- **No performance claims** — no timing, latency, throughput, tokens/sec,
  speedup, or runtime metrics or language in any code, report, table, or doc.
- **No generation-logic changes** — no sampling, no batching, no parallel
  verification, no bonus-token acceptance. The draft-verify-commit loop is
  untouched.
- **No production-readiness claims.**
- **No plots/images** — text tables only.
- **No new model-family compatibility guarantee** — still targets
  `Qwen/Qwen2.5-0.5B` on the tested transformers version.

---

## 4. Implementation phases

Each phase is independently shippable and must keep the full existing test suite
green plus pass the no-performance-field audit.

- **Phase 0 — Scope documentation (this file).** Write `V4_SCOPE_STATEMENT.md`,
  update `FUTURE_RESEARCH_ASYMMETRIC_KV.md`, add a README roadmap bullet. No
  implementation code.
- **Phase A — Capability metadata extension.** Add `key_bit_width`,
  `value_bit_width`, `asymmetric` to `CompressorCapabilities` with defaults;
  backfill the four existing compressors. Pure metadata; no behaviour change.
- **Phase B — Asymmetric compressor core.** Implement
  `AsymmetricQuantSimCompressor(k_bits, v_bits)` reusing existing quant/dequant
  helpers, with per-side independent scales and `full` passthrough.
- **Phase C — Named registrations and ablations.** Register the seven named
  compressors; derive instance-level capabilities (`is_simulated`,
  `supports_real_bytes_claim`, widths, `asymmetric`) from the widths.
- **Phase D — Reporting and leaderboard fields.** Add the three fields to
  `reports.py` JSON/CSV; add the K/V-width + average-effective-bit-width column
  to the leaderboard renderer.
- **Phase E — Experiment 003, README, future-research note update, release
  notes.** Run the core-suite sweep, generate `EXPERIMENT_003_…` via the
  `report` CLI, add the README results section, flip the future-research note's
  scope-boundary table to "implemented in V4", write `RELEASE_NOTES_V0.4.0.md`,
  and tag `v0.4.0`.

---

## 5. Tests and gates per phase

**Global gates (every phase)**
- All prior tests remain green (current 542 tests).
- Primary correctness criterion unchanged: `exactkv_output_ids == full_output_ids`.
- No forbidden performance fields anywhere (`tokens_per_second`, `throughput`,
  `latency`, `speedup`, `runtime_seconds`) except in explicit negation prose.

**Phase A — capability metadata**
- The four existing compressors still construct; `asymmetric=False`; widths match
  (`int8`=8/8, `int4_sim`=4/4, `noop`/`debug_noise`=None/None).
- `asdict(capabilities)` flows into the `run_one` result with no forbidden fields.

**Phase B — asymmetric compressor core**
- Per-side width is applied independently: at `k_bits=8, v_bits=4`, K carries
  int8-range error and V is clamped to the INT4 range `[-8, 7]`; a `full` side is
  bit-identical passthrough.
- **ExactKV correctness gate (hard):** for each named compressor,
  `exactkv_output_ids == full_output_ids` across ≥2 prompts × ≥2 draft lengths →
  `exactkv_failures == 0`.
- Acceptance bookkeeping reconciles (`drafted == accepted + rejected`).

**Phase C — named registrations and ablations**
- Every named compressor resolves through the registry via `get_compressor(name)`.
- Every named compressor runs end-to-end with `exactkv_failures == 0`.
- `is_simulated` / `supports_real_bytes_claim` are correct per name — `False` /
  `True` only for `k8_v_full` and `k_full_v8`; `True` / `False` for every
  `*_sim` compressor.

**Phase D — reporting and leaderboard**
- Report schema (JSON enrichment + CSV columns) includes `key_bit_width`,
  `value_bit_width`, and `asymmetric` in every row.
- `_assert_no_forbidden_fields` still passes on the enriched report.
- Leaderboard shows K/V widths and the average effective bit width; symmetric and
  asymmetric rows are directly comparable.
- No forbidden performance fields in the rendered markdown.

**Phase E — experiment and docs**
- `docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md` exists, reports
  `exactkv_failures == 0`, and is honest: it carries the simulation /
  `supports_real_bytes_claim=False` disclaimer for every `*_sim` compressor and
  the VeriCache attribution, and passes the no-performance-word audit.
- README links resolve; release notes are accurate.

---

## 6. Naming decision

The `_sim` suffix is a **simulation honesty marker**, not a stylistic one:

- **`_sim` is used only when a compressor includes a simulated sub-INT8 width**
  (INT4-sim or INT2-sim), i.e. values quantised to a sub-8-bit numeric range but
  stored in `int8` containers with no real bit-packing. For these,
  `is_simulated=True` and `supports_real_bytes_claim=False`.
- **`k8_v_full` and `k_full_v8` do NOT use `_sim`.** They combine only real INT8
  and full precision — there is no simulated sub-INT8 storage — so
  `is_simulated=False` and `supports_real_bytes_claim=True`.

The original draft name `k8_v_full_sim` was **rejected** because the `_sim`
suffix would imply simulation while its capabilities would correctly report
`is_simulated=False`. The name and the capability flag must agree. In all
downstream tooling the **capability flag — not the name — is the source of
truth**, but the name must not contradict it.

---

## 7. Workspace-aware memory decision

The full workspace-aware memory schema (`stored_kv_bytes`,
`materialized_working_kv_bytes`, `metadata_bytes`, `temporary_workspace_bytes`,
`active_gpu_kv_bytes`, `total_system_kv_bytes`) is **deferred to V5.**

Rationale:
1. It is a cross-cutting change to `MemorySummary`, the JSON/CSV report schema,
   the renderer, and all memory tests — orthogonal to V4's acceptance question.
2. On simulated compressors running on CPU, fields such as `active_gpu_kv_bytes`
   and `materialized_working_kv_bytes` would be **synthetic/derived rather than
   measured**, which would be a soft claim ExactKV avoids. They become meaningful
   only alongside a real backend on a real device (V5/V6).
3. V4's thesis is answered entirely with existing acceptance metrics; bundling a
   memory-schema rewrite would dilute a clean, testable release.

Instead, **V4 adds only K/V width metadata and an average effective bit width**
(see §2.3 and §2.5). **V4 does not claim real memory behaviour for simulated
sub-INT8 compressors:** every `*_sim` compressor keeps `is_simulated=True` and
`supports_real_bytes_claim=False`, and its reported byte counts continue to
reflect `int8` container storage, not real packed sub-INT8 storage.

The workspace-aware schema is recorded as the **V5** direction, to be introduced
in one honest place alongside the first real backend and a real device.

---

## 8. Relationship to `docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`

V4 is the direct implementation of the asymmetric-KV future-research note:

- **§5.1 "Asymmetric KV compressor simulator"** → `AsymmetricQuantSimCompressor`
  (Phase B), reusing the `Int4Sim` / `Int8` quantisation logic exactly as the
  note prescribes.
- **§5.2 "K-only / V-only ablations"** → the ablation set `k_full_v4_sim`,
  `k4_v_full_sim`, `k8_v_full`, `k_full_v8` (Phase C).
- **§5.3 "Asymmetric sweep over K bit-width × V bit-width"** and
  **§5.4 "symmetric vs asymmetric at matched average bit-width"** → Experiment
  003 plus the average-effective-bit-width leaderboard column (Phases D–E).
- **§1 / §3 "acceptance, not MSE"** → V4 evaluates purely on acceptance
  behaviour; MSE is explicitly not added.

The note's workspace-aware memory direction (§4) is **not** implemented in V4 and
remains a V5 candidate (see §7). On V4 completion, the note's §6 scope-boundary
table flips the asymmetric-compressor and ablation rows from "Candidate for V4"
to "Implemented in V4 (see EXPERIMENT_003)".

---

## 9. How V4 prepares for real backends (V5/V6)

1. **Width-keyed capabilities are backend-ready.** `key_bit_width` /
   `value_bit_width` / `asymmetric` / `is_simulated` /
   `supports_real_bytes_claim` describe a *policy* independent of *how* it is
   implemented. A V5 real INT4/KIVI backend registers the same fields and
   inherits all V3 rendering and V4 leaderboard logic with no renderer change.
2. **Acceptance harness already validated for asymmetry.** The Experiment-003
   sweep/leaderboard/divergence tooling will characterise where and how often a
   real asymmetric backend diverges, before any performance work.
3. **Schema-stable inputs.** The new fields are additive to the locked report
   schema; real backends slot into the same JSON/CSV.
4. **Clean memory seam for V5.** Because V4 avoids unmeasured memory fields, V5
   can introduce the workspace-aware schema in one honest place, attached to a
   real device/backend, with no retrofitting.
5. **No performance scaffolding to unwind.** V4 adds zero timing fields, so
   V5/V6 can introduce real measurement cleanly.

---

## V4 exit criteria

V4 is complete when all of the following hold:

- [x] `AsymmetricQuantSimCompressor(k_bits, v_bits)` implemented, reusing existing
      quantisation logic, with `full` passthrough and per-side scales.
- [x] All seven named compressors (`k8_v4_sim`, `k8_v2_sim`, `k4_v8_sim`,
      `k_full_v4_sim`, `k4_v_full_sim`, `k8_v_full`, `k_full_v8`) resolve through
      the registry and run end-to-end with `exactkv_failures == 0`.
- [x] `CompressorCapabilities` carries `key_bit_width`, `value_bit_width`,
      `asymmetric`; the four existing compressors backfilled correctly.
- [x] Report JSON/CSV include the three new fields; no-performance-field audit
      passes.
- [x] Leaderboard renders K/V widths and average effective bit width.
- [ ] `docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md` written from a real core-suite
      sweep with `exactkv_failures == 0` and full honesty disclaimers.
- [ ] README V4 results section and `docs/RELEASE_NOTES_V0.4.0.md` written.
- [ ] Full prior test suite remains green.

---

## Citation and novelty note

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is
a compressor-agnostic, Hugging Face-first implementation, a structured benchmark
harness, and a framework for evaluating compressors by acceptance behaviour under
full-KV verification. V4 extends that framework with asymmetric K/V compression
experiments — simulated compressors, K-only/V-only ablations, and acceptance-rate
comparisons — without adding any performance claims.
