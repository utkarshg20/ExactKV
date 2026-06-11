# TurboQuant Production Toolchain Prep (V12 Phase 1b)

**Status:** Phase 1b complete on RunPod — toolchain built and smoke-tested. **Experiment 022 not started.**
**Machine:** RunPod `17a0b6d9c4cb` (NVIDIA RTX A5000, Ubuntu 24.04, x86_64)
**Workdir:** `/workspace/turboquant_prod_prep` (outside git; not committed)

> This is **toolchain prep only** — not ExactKV integration, not Experiment 022.
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec,
> active GPU memory savings, or production serving readiness.
> llama.cpp internal perf lines in smoke logs are **upstream diagnostics**, not ExactKV results.

**Prerequisite:** [`TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md`](TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md) (Phase 1 / Exp 021).

**Scripts:** [`turboquant_production_toolchain_prep.sh`](../scripts/research/turboquant_production_toolchain_prep.sh),
[`turboquant_production_toolchain_resume.sh`](../scripts/research/turboquant_production_toolchain_resume.sh),
[`turboquant_production_phase1_inspect.py`](../scripts/research/turboquant_production_phase1_inspect.py).

---

## 1. Purpose

Resolve Phase 1 environment blockers before V12 Phase 2 / Experiment 022:

- cmake / build tools
- patched `llama-cpp-turboquant` binaries
- REFRACT `selftest` with patched binaries
- GGUF conversion path for Qwen2.5-0.5B
- TurboQuant CLI flag smoke (`-ctk q8_0 -ctv turbo3`)

---

## 2. Environment

| Item | Value |
|---|---|
| Host | `17a0b6d9c4cb` |
| OS | Linux 6.8.0-85-generic (Ubuntu 24.04) |
| Python | 3.12.3 |
| GPU | NVIDIA RTX A5000, 24564 MiB |
| cmake | 3.28.3 |
| gcc/g++ | 13.3.0 |
| HF cache | `models--Qwen--Qwen2.5-0.5B` present |

---

## 3. Installed packages

Via `apt-get` (idempotent):

- `cmake`, `build-essential`, `ninja-build`, `git`
- `python3-venv`, `python3-pip`, `pkg-config`

REFRACT venv additionally installed: `torch`, `transformers`, `sentencepiece`, `protobuf` (for GGUF conversion only).

---

## 4. llama.cpp TurboQuant fork / branch / SHA

| Item | Value |
|---|---|
| Repo | `https://github.com/TheTom/llama-cpp-turboquant.git` |
| Branch | `feature/turboquant-kv-cache` |
| SHA | `73eb521daebc85da7c91d37178940b99a5524cf6` |
| `turboquant_plus` SHA | `9eb03410918d2d137b071a48af3a5d1e9c8c3e47` |

---

## 5. Build commands

```bash
cd /workspace/turboquant_prod_prep/llama-cpp-turboquant
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -G Ninja   # failed
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -G Ninja              # succeeded
cmake --build build-cpu -j$(nproc)
```

CUDA cmake failed on this pod (not captured in first manifest); **CPU build succeeded** via `build-cpu/`.

---

## 6. Binary paths found

| Binary | Path |
|---|---|
| `llama-cli` | `/workspace/turboquant_prod_prep/llama-cpp-turboquant/build-cpu/bin/llama-cli` |
| `llama-server` | `.../build-cpu/bin/llama-server` |
| `llama-completion` | `.../build-cpu/bin/llama-completion` |
| `llama-tokenize` | `.../build-cpu/bin/llama-tokenize` |
| `llama-perplexity` | `.../build-cpu/bin/llama-perplexity` |

Set for REFRACT / Phase 2:

```bash
export LLAMA_CPP_BIN_DIR=/workspace/turboquant_prod_prep/llama-cpp-turboquant/build-cpu/bin
```

---

## 7. REFRACT setup result

| Check | Result |
|---|---|
| Install | `pip install -e turboquant_plus[dev]` in `/workspace/turboquant_prod_prep/venv_refract` |
| `refract --help` | OK |
| `refract selftest` | **PASS** (with patched binaries on `PATH` / `LLAMA_CPP_BIN_DIR`) |

**Phase 2 use:** REFRACT is **optional** external quality reference only — not the primary Exp 022 path (Mode B external-drafter + HF verifier).

---

## 8. GGUF / Qwen conversion readiness

| Item | Result |
|---|---|
| `convert_hf_to_gguf.py` | Present in fork |
| `conversion/qwen.py` | Present |
| Valid `--outtype` values | `f32`, `f16`, `bf16`, `q8_0`, `tq1_0`, `tq2_0`, `auto` — **not** `q4_k_m` |
| HF snapshot | `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987` |

**Successful conversion command:**

```bash
source /workspace/turboquant_prod_prep/venv_refract/bin/activate
python3 /workspace/turboquant_prod_prep/llama-cpp-turboquant/convert_hf_to_gguf.py \
  /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --outfile /workspace/turboquant_prod_prep/models/qwen2.5-0.5b-auto.gguf \
  --outtype auto
```

**Output:** `qwen2.5-0.5b-auto.gguf` (~949 MiB). **Not committed** (gitignored `*.gguf`).

**Note:** Passing HuggingFace repo id `Qwen/Qwen2.5-0.5B` directly fails (`not a directory`) — use **snapshot path** or download first.

---

## 9. TurboQuant CLI flag smoke result

**Help smoke:** `-ctk`, `-ctv`, `turbo`, `cache-type-k`, `cache-type-v`, `q8_0`, `turbo3` all found in `llama-cli --help`.

**Inference smoke** (use `llama-completion`, not interactive `llama-cli`):

```bash
LLAMA_CPP_BIN_DIR=.../build-cpu/bin
$LLAMA_CPP_BIN_DIR/llama-completion \
  -m /workspace/turboquant_prod_prep/models/qwen2.5-0.5b-auto.gguf \
  -ctk q8_0 -ctv turbo3 -fa on -ngl 0 \
  --temp 0 --top-k 1 -n 4 -no-cnv \
  --prompt "The capital of France is"
```

**Text output (greedy):** `The capital of France is Paris. The capital` — **success**.

**Token IDs:** Use `llama-tokenize -m <gguf> --stdin` (requires `--stdin` flag). Token extraction for Exp 022 harness is feasible.

**Caveat:** `llama-cli` without careful flags enters interactive/chat mode; prefer **`llama-completion`** for scripted draft probes.

---

## 10. Remaining blockers before Experiment 022

| Blocker | Severity | Notes |
|---|---|---|
| CPU-only build (`build-cpu`) | Medium | CUDA cmake failed; Exp 022 should retry CUDA build or use prebuilt CUDA release for faster probes |
| GGUF vs HF float verifier | High (interpretation) | Document weight-format mismatch in Exp 022 |
| Interactive `llama-cli` pitfall | Low | Use `llama-completion` in harness |
| Token-step harness not written | Expected | Phase 2 deliverable |
| No ExactKV integration | By design | Mode B external subprocess only |

---

## 11. Whether V12 Phase 2 can proceed

**Yes.** Toolchain prep gate passed:

- Patched binaries built and on disk
- TurboQuant flags confirmed
- REFRACT `selftest` passes
- GGUF conversion succeeded
- `llama-completion` smoke with `-ctk q8_0 -ctv turbo3` produced expected text

**Experiment 022** may proceed on RunPod using `LLAMA_CPP_BIN_DIR` and the converted GGUF — still **Mode B** (external drafter + HF verifier), not production `BackendAdapter`.

---

## Reproduce on RunPod

```bash
# Upload prep script from ExactKV repo, then:
bash /workspace/turboquant_prod_prep/run_prep.sh
# Or resume after interrupt:
SKIP_CLONE=1 SKIP_BUILD=1 GGUF_OUTTYPE=auto bash /workspace/turboquant_prod_prep/run_prep.sh
```

Inspector:

```bash
python3 scripts/research/turboquant_production_phase1_inspect.py \
  --workdir /workspace/turboquant_prod_prep --with-refract
```

---

## Related

- [`TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md`](TURBOQUANT_PRODUCTION_FIDELITY_FEASIBILITY.md)
- [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md)
