# TurboQuant Production-Fidelity Feasibility (V12 Phase 1 / Experiment 021)

**Status:** V12 Phase 1 complete — feasibility research only. **No production adapter implemented.**
**Experiment 022:** **Go with restrictions** (Mode B — restricted external-drafter probe). Mode A **no-go**.
**Date:** 2026-06-11

> This document does **not** claim TurboQuant llama.cpp is integrated into ExactKV.
> Experiment 008 evaluated the **Python `KVCacheCompressor` path only** — not production TurboQuant.
> External TurboQuant+ / REFRACT numbers cited below are **upstream claims**, not ExactKV results.
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec,
> active GPU memory savings, or production serving readiness.

**Prerequisite docs:** [`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md) (V9 Phase A),
[`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md) (V9 Phase B),
[`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md).

**Scratch tools:** [`scripts/research/turboquant_production_phase1_inspect.py`](../scripts/research/turboquant_production_phase1_inspect.py),
[`scripts/research/turboquant_llamacpp_phase1_commands.sh`](../scripts/research/turboquant_llamacpp_phase1_commands.sh).

---

## 1. Purpose

Determine whether TurboQuant's **production-fidelity** path (llama.cpp packed
`turbo2`/`turbo3`/`turbo4` KV cache with WHT rotation) can be evaluated by ExactKV
**without pretending** the existing Python adapter (Experiment 008) is production TurboQuant,
and classify a feasible **Experiment 022** design.

---

## 2. Why Phase 1 is needed

V11 deferred public launch partly because TurboQuant was evaluated only through the
restricted Python NumPy bridge. Production TurboQuant+ lives in **llama.cpp** (and MLX),
with different algorithms, packing, and runtime semantics. Before Exp 022 coding,
ExactKV must document:

- Which upstream path is production-fidelity
- Whether Qwen2.5 models can be converted and run
- Whether HF full-KV verification can remain authoritative
- Whether Mode A (faithful in-loop probe), Mode B (restricted external drafter), or Mode C (no-go) applies

---

## 3. What Experiment 008 did and did not evaluate

### Did evaluate

| Item | Detail |
|---|---|
| Adapter | `TurboQuantPythonAdapter` → upstream `KVCacheCompressor` |
| Model | `Qwen/Qwen2.5-0.5B`, CPU float32 |
| Cells | 272; `exactkv_failures == 0` |
| Accept | `turboquant_python_k3_v3` **0.435** |
| Exactness | HF full-KV verifier unchanged |

### Did not evaluate

| Item | Why it matters |
|---|---|
| llama.cpp TurboQuant runtime | Production packed-bit path |
| GGUF weight format | Exp 008 uses HF float weights |
| WHT fast rotation | Python uses dense Haar `random_rotation_dense` |
| K-path QJL vs production | Python K uses QJL; llama.cpp drops QJL |
| Block-packed `turbo2/3/4` structs | Python stores per-vector numpy indices |
| MLX `TurboKVCache` | Apple Silicon production path |
| REFRACT / vLLM / SGLang scoring | Separate quality frameworks |

---

## 4. Upstream TurboQuant paths inspected

Sources: `vendor/turboquant_plus` (cloned locally), `/tmp/llama-cpp-turboquant`
(cloned `feature/turboquant-kv-cache`), prior V9 Phase A research.

| Path | Location | Production-fidelity? | HF `past_key_values` bridge? |
|---|---|---|---|
| **Python `KVCacheCompressor`** | `turboquant_plus/turboquant/` | **No** — research prototype | ✅ Yes (Exp 008) |
| **llama.cpp `turbo2/3/4`** | `TheTom/llama-cpp-turboquant` | **Yes** — primary cross-platform production path | ❌ No export API |
| **MLX TurboKVCache** | `TheTom/mlx@feature/turboquant-plus` | **Yes** (Apple Silicon) | ❌ Wrong stack |
| **REFRACT** | `turboquant_plus/refract/` | Quality scorer, not compressor | ❌ Compares trajectories externally |
| **vLLM-swift / SGLang** | Upstream forks | Serving runtimes | ❌ Out of ExactKV scope |

**Closest to production TurboQuant:** llama.cpp fork with `-ctk`/`-ctv` turbo types and
WHT kernels in `ggml-turbo-quant.c` / `llama-kv-cache.cpp`.

**Compatible with Qwen/Qwen2.5:** Yes for llama.cpp via `conversion/qwen.py` and documented
Qwen2.5 GGUF usage; Python path validated at `head_dim=64` for 0.5B.

---

## 5. llama.cpp / GGUF findings

### Repository

- **Repo:** <https://github.com/TheTom/llama-cpp-turboquant>
- **Branch inspected:** `feature/turboquant-kv-cache` (default)
- **Turbo cache types:** `turbo2`, `turbo3`, `turbo4` in `llama-bench`, `llama-kv-cache.cpp`, CUDA/HIP FA kernels
- **CLI flags:** `-ctk` / `-ctv` (or `--cache-type-k` / `--cache-type-v`) accept turbo types on this fork
- **Prebuilt binaries:** Available from GitHub Releases (macOS Metal, Windows CUDA) per upstream README

### GGUF conversion

- `convert_hf_to_gguf.py` + `conversion/qwen.py` support Qwen family conversion
- Upstream docs reference `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (e.g. RISC-V build doc)
- **Qwen2.5-0.5B conversion is feasible** with standard HF → GGUF tooling on this fork
- Weights will be **quantized GGUF** (e.g. Q4_K_M), not HF float32 — see §7 alignment risks

### Build/install (this environment)

| Step | Result (2026-06-11) |
|---|---|
| Clone `turboquant_plus` | ✅ `vendor/turboquant_plus` |
| Clone `llama-cpp-turboquant` | ✅ `/tmp/llama-cpp-turboquant` |
| `import turboquant` | ✅ via `PYTHONPATH=vendor/turboquant_plus` |
| `pip install -e vendor/turboquant_plus[dev]` | ✅ `refract` CLI available |
| `refract selftest` | ❌ **Failed** — patched `llama-cli`, `llama-completion`, `llama-tokenize`, `llama-perplexity` missing at default `LLAMA_CPP_BIN_DIR` |
| `cmake` local build | ❌ **Skipped** — `cmake` not installed on research host |
| Prebuilt binary path | Documented; not downloaded in Phase 1 |

**Mitigation for Exp 022:** RunPod or dev machine with cmake **or** use prebuilt
`llama-cpp-turboquant` release tarball; set `LLAMA_CPP_BIN_DIR`.

---

## 6. Qwen2.5 compatibility findings

| Model | head_dim | llama.cpp | TurboQuant config notes |
|---|---:|---|---|
| `Qwen/Qwen2.5-0.5B` | 64 | GGUF feasible; upstream code mentions head_dim padding 64→128 | Use **asymmetric** `-ctk q8_0 -ctv turbo3` or `turbo4`; avoid symmetric turbo on Q4_K_M |
| `Qwen/Qwen2.5-1.5B` | 64 | Same family | Q4_K_M symmetric `turbo3/turbo3` **catastrophic** in upstream matrix (PPL 8641+) |
| `Qwen/Qwen2.5-7B` | 128 | Validated in upstream asymmetric tables | `-ctk q8_0 -ctv turbo4` recommended for Q4_K_M |

**head_dim=64 caveat (upstream):** WHT kernel needs sufficient dimensionality; `head_dim=64`
is at the lower boundary — turbo V compression may be fragile on some models (e.g. GPT-OSS-120B
per getting-started.md). Monitor Qwen2.5-0.5B closely in Exp 022.

**GQA ratio:** llama-kv-cache.cpp notes Qwen2.5's 7:1 Q/KV head ratio makes symmetric turbo3-K
risky vs Mistral 4:1 — reinforces asymmetric configs for Qwen2.5.

---

## 7. Tokenizer and generation-alignment findings

| Concern | Finding |
|---|---|
| Tokenizer source | GGUF embeds tokenizer metadata; HF uses `AutoTokenizer` — **same BPE family** if converted from same HF checkpoint |
| Greedy parity | llama.cpp greedy (`--temp 0`, `--top-k 1`) can match greedy decoding **if** weights and numerics align |
| Weight mismatch | Exp 008 uses **HF float32**; llama.cpp uses **GGUF quant weights** — logits may differ from HF float even with identical tokenizer |
| Chat template | llama.cpp chat templates vs raw prompts — Exp 022 should use **raw token prompts** aligned with ExactKV V10 tokenization (no template drift) |
| Stepwise tokens | `llama-completion` / `llama-cli` can emit discrete tokens; server API exposes per-token completion — **stepwise extraction feasible** via CLI loop or completion API |
| Logits per step | Server supports `n_probs`; perplexity tools record logits — **not required** for ExactKV accept/reject (token ID comparison suffices) |
| Batch vs single-token nondeterminism | llama.cpp docs warn logits may differ between prompt batching and token generation — **document and fix decoding settings** for Exp 022 |

**Alignment risk (high):** Production-fidelity probe compares **GGUF-quantized llama.cpp drafter**
against **HF float32 full verifier**. Token-ID agreement is the ExactKV gate, but divergence may
reflect weight-format differences as much as KV compression — Exp 022 must document this caveat.

---

## 8. KV-cache format findings

| Format | Storage | ExactKV `BackendAdapter`? |
|---|---|---|
| Python `CompressedKVCache` | NumPy indices/norms per vector | ✅ Exp 008 |
| llama.cpp `turbo2/3/4` | Packed blocks, WHT, block_size 32/128 | ❌ Internal to llama runtime |
| MLX `TurboKVCache` | MLX arrays | ❌ |

**Critical:** llama.cpp does **not** expose packed turbo KV blocks for import into HF
`past_key_values`. Production KV compression happens **inside** the llama.cpp graph.

Therefore **Mode A** (faithful `BackendAdapter` using production packed KV) is **not feasible**
without a new tensor export API that does not exist upstream today.

---

## 9. Whether HF full verifier can remain authoritative

**Yes — required and achievable for Exp 022.**

ExactKV architecture:

1. **Authoritative path:** HF `ModelRuntime` + `VerificationEngine` on full-precision KV (unchanged).
2. **Draft path:** External llama.cpp subprocess with turbo KV flags proposes token IDs.
3. **Gate:** `exactkv_output_ids == full_output_ids` where final output follows HF verifier commits.

The verifier **never** reads llama.cpp KV. llama.cpp is an **external drafter**, not a
`BackendAdapter` materialization source.

**Not feasible:** Single-process llama.cpp owning both compressed KV and HF verification without
a dual-runtime rewrite (explicitly out of V12 scope).

---

## 10. Candidate Experiment 022 designs

### Mode A — Faithful production-fidelity probe (in-loop `BackendAdapter`)

| Criterion | Verdict |
|---|---|
| llama.cpp draft tokens + HF verifier | Theoretically possible via subprocess |
| Production packed KV in ExactKV compressor | ❌ **Blocked** |
| Tokenizer/model alignment | ⚠️ Risky (GGUF vs HF float) |
| Stepwise draft capture | ✅ CLI/API supports |
| Exactness gate | ✅ If HF verifier remains authoritative |

**Classification: NO-GO for BackendAdapter integration.** Partial subprocess variant is Mode B.

### Mode B — Restricted external-drafter probe (recommended)

**Design sketch (Phase 2, not implemented in Phase 1):**

1. Convert `Qwen2.5-0.5B-Instruct` → GGUF (Q4_K_M or Q8_0 documented in manifest).
2. Build or download `llama-cpp-turboquant` binaries.
3. For each V10 prompt (small panel ≤25):
   - Tokenize prompt with **HF tokenizer** (same as ExactKV).
   - Run llama.cpp greedy with `-ctk q8_0 -ctv turbo3` (asymmetric, Qwen-safe) to propose draft tokens stepwise **or** in short chunks.
   - Feed proposed tokens through **existing** ExactKV HF verifier loop (experiment harness only — no core generator change).
   - Record accept/reject/correct/exactness vs Exp 008 Python panel.
4. Document weight-format caveat; do not claim Python adapter equals production.

**REFRACT precedent:** Axis A (GTM — Greedy Trajectory Match) compares quantized vs reference
token trajectories on llama.cpp; useful **external** quality reference, not ExactKV acceptance metrics.

### Mode C — No-go for now

Would apply if GGUF conversion blocked, binaries unbuildable, or tokenization irreconcilable.
**Not triggered** — blockers are operational (cmake/binaries), not architectural dead-ends.

---

## 11. Build/install results

Commands run during Phase 1:

```bash
# Python path smoke
PYTHONPATH=vendor/turboquant_plus python3 scripts/research/turboquant_phase_a_inspect.py --with-hf
# → turboquant import ok; head_dim 64/128 roundtrip ok

# Phase 1 inspector
PYTHONPATH=vendor/turboquant_plus python3 scripts/research/turboquant_production_phase1_inspect.py \
  --llama-repo /tmp/llama-cpp-turboquant

# REFRACT
pip install -e vendor/turboquant_plus[dev]
refract selftest
# → FAILED: missing llama-cli, llama-completion, llama-tokenize, llama-perplexity
```

| Component | Status |
|---|---|
| `turboquant` Python import | ✅ |
| `llama-cpp-turboquant` source clone | ✅ |
| llama.cpp binary build | ❌ cmake missing locally |
| REFRACT selftest | ❌ patched binaries missing |
| GGUF conversion run | ⏭️ Not executed (large HF download); path documented |

---

## 12. Blockers and risks

| Blocker / risk | Severity | Mitigation |
|---|---|---|
| No packed KV export from llama.cpp | **Hard** (Mode A) | Use Mode B external drafter |
| Python ≠ production algorithm | **High** (interpretation) | Label Exp 008 as non-production; compare qualitatively in Exp 022 |
| GGUF vs HF float weight mismatch | **High** | Document; prefer Q8_0 GGUF; same prompt tokens |
| Qwen2.5 symmetric turbo configs | **High** | Force asymmetric `-ctk q8_0 -ctv turbo3` |
| head_dim=64 WHT boundary | **Medium** | Monitor 0.5B; fall back to q8_0 V if unstable |
| cmake / binary availability | **Medium** (ops) | Prebuilt releases or RunPod build |
| REFRACT / upstream PPL numbers | **Low** (claims) | Cite as external only |
| Dual-runtime harness complexity | **Medium** | Experiment-layer script only; no core changes |

---

## 13. Recommended go/no-go classification

| Item | Decision |
|---|---|
| **Experiment 021 (this doc)** | ✅ **Complete** |
| **Experiment 022 overall** | ✅ **GO WITH RESTRICTIONS** |
| **Mode A** (`BackendAdapter` + production packed KV) | ❌ **NO-GO** |
| **Mode B** (external llama.cpp drafter + HF verifier) | ✅ **GO** — recommended Exp 022 design |
| **Mode C** (full no-go) | ❌ Not warranted |
| **MLX production path** | ❌ Defer — wrong stack for HF-first ExactKV |
| **REFRACT integration** | Optional external reference only |

---

## 14. What this proves

- Production TurboQuant is **distinct** from Experiment 008's Python adapter (algorithm, packing, runtime).
- llama.cpp turbo path is **real and Qwen2.5-compatible** with documented asymmetric configs.
- HF full-KV verifier **can remain authoritative** via external-drafter experiment harness.
- A **restricted Exp 022** is architecturally feasible without core ExactKV changes.

---

## 15. What this does not prove

- That llama.cpp TurboQuant is integrated into ExactKV (it is **not**).
- That production TurboQuant acceptance matches Exp 008 Python accept (**0.435**).
- That GGUF-quantized llama.cpp greedy matches HF float greedy token-for-token.
- Any speed, memory, throughput, or serving improvement.
- That REFRACT or upstream PPL tables are ExactKV experiment results.

---

## 16. Toolchain prep after Phase 1 (Phase 1b — complete on RunPod)

See [`TURBOQUANT_PRODUCTION_TOOLCHAIN_PREP.md`](TURBOQUANT_PRODUCTION_TOOLCHAIN_PREP.md).

| Step | Result (RunPod 2026-06-11) |
|---|---|
| cmake + ninja build | CPU build **succeeded** (`build-cpu/`); CUDA cmake **failed** |
| Patched binaries | `llama-cli`, `llama-completion`, `llama-server`, `llama-tokenize`, `llama-perplexity` |
| Turbo flags in `--help` | `ctk`, `ctv`, `turbo3`, `q8_0` confirmed |
| REFRACT `selftest` | **PASS** |
| GGUF conversion | **Success** — snapshot path + `--outtype auto` → `qwen2.5-0.5b-auto.gguf` (~949 MiB) |
| Turbo smoke | `llama-completion -ctk q8_0 -ctv turbo3` → `Paris` (text OK) |
| Phase 2 gate | **YES** — Exp 022 may proceed (Mode B harness not written yet) |

---

## 17. Next step for Phase 2 (Experiment 022)

1. **Environment:** RunPod `LLAMA_CPP_BIN_DIR=/workspace/turboquant_prod_prep/llama-cpp-turboquant/build-cpu/bin` (optional CUDA rebuild for speed).
2. **Model:** Use existing `qwen2.5-0.5b-auto.gguf` or reconvert from HF snapshot; record quant type in manifest.
3. **Harness:** Experiment-only external-drafter script (subprocess llama.cpp + HF verifier); **no** `BackendAdapter` production adapter.
4. **Panel:** ≤25 prompts shared with Exp 008 / Exp 020 panels where possible.
5. **Configs:** Compare at minimum:
   - llama.cpp `-ctk q8_0 -ctv turbo3` (production-fidelity asymmetric)
   - Exp 008 `turboquant_python_k3_v3` anchor (same prompts, HF Python path)
6. **Gate:** `exactkv_failures == 0`; document weight-format and head_dim caveats.
7. **No performance claims** in report or docs.

---

## Related

- [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) — Phase 1 / Exp 021–022 plans
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — D2 status
- [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md)
