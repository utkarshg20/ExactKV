# KIVI / KVQuant Integration Research (V9 Phase D1)

**Status:** Phase D1 complete — feasibility research only. **ExactKV does not implement
KIVI or KVQuant.** No adapter code, no compressor registration, no Experiment 009
artifacts.

**Date:** 2026-06-09  
**Scope:** Determine whether KIVI or KVQuant is the better next real-backend adapter
candidate after the TurboQuant Python track (Experiment 008), and produce a go/no-go
feasibility decision for Phase D2 adapter work.  
**Recommendation:** **Implement KIVI adapter next (restricted offline subset first).**
KVQuant faithful integration **requires RunPod validation before adapter decision**
(§23).

> Guardrails inherited from [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md): no
> throughput, latency, tokens/sec, speedup, `runtime_seconds`, `active_gpu_kv_bytes`,
> or production-serving claims as ExactKV results. External KIVI/KVQuant paper and
> README numbers cited below are **upstream claims**, not ExactKV measurements.

---

## 1. Purpose

V9 Phase D1 answers:

> After TurboQuant Python (Experiment 008, `exactkv_failures == 0`), should ExactKV
> pursue **KIVI** or **KVQuant** as the next `BackendAdapter` integration — and if so,
> via what bridge path (offline tensor quant, model rewrite, hooks, CUDA kernels)?

This document records repo inspection, isolated installation attempts, API/cache-format
analysis, Qwen2.5 compatibility, `BackendAdapter` fit, RunPod plans, failure modes,
and a side-by-side feasibility table. It does **not** implement adapters or run
ExactKV experiments.

**Precedent:** TurboQuant Phase A–C ([`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md),
[`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md),
[`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md)) established
the offline numpy/torch bridge pattern: extract post-RoPE HF `past_key_values` → backend
compress → materialize HF cache for draft → verify on full KV only.

---

## 2. Why KIVI / KVQuant after TurboQuant

| Reason | Detail |
|---|---|
| **V9 credibility gap** | Only one restricted real backend (kvpress KnormPress) plus TurboQuant Python evaluated. Simulated `_sim` compressors do not represent KIVI per-channel K / per-token V or KVQuant pre-RoPE NUQ + sparse outliers. |
| **Literature alignment** | [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md) §§2–3: KIVI and KVQuant are the two most cited asymmetric / key-specialized quant families after TurboQuant. ExactKV V4 acceptance evidence aligns with asymmetric K/V — real KIVI/KVQuant backends test whether that pattern holds under **real** quant rules. |
| **TurboQuant lesson** | Experiment 008 proved exactness (`exactkv_failures == 0`) but acceptance was low (**0.435** vs `int8` **0.961**). KIVI targets different error geometry (per-channel K, residual window) and may behave differently — still unknown until integrated. |
| **Deferred register** | D3 (KIVI) and D4 (KVQuant) in [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) block v0.9.0 Phase D exit criteria. |
| **Scope order** | [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) §5 lists KIVI and KVQuant as Phase D backends for optional Experiment 009. |

---

## 3. KIVI summary

**KIVI** (Liu et al., ICML 2024, [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) is a
tuning-free asymmetric KV quantizer:

| Axis | KIVI design |
|---|---|
| **Keys** | Per-channel (along `head_dim`), 2/4-bit, grouped (`group_size`) |
| **Values** | Per-token (along sequence), 2/4-bit |
| **Residual** | Last `residual_length` tokens kept in fp16; streaming quant as cache grows |
| **Inference** | Custom model classes (`LlamaForCausalLM_KIVI`, `MistralForCausalLM_KIVI`) with fused CUDA/Triton attention on **quantized cache tensors** |
| **HF influence** | Upstream README notes HuggingFace Transformers KV-cache quantization docs were inspired by KIVI — **separate code path** from `jy-yuan/KIVI` repo |

**Upstream repo:** [github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)

**External claims (not ExactKV results):** ~2.6× peak memory reduction; near-baseline quality on Llama/Falcon/Mistral families per paper/README.

---

## 4. KVQuant summary

**KVQuant** (Hooper et al., NeurIPS 2024, [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)) is a
calibration-driven sub-4-bit KV methodology:

| Axis | KVQuant design |
|---|---|
| **Keys** | **Per-channel, pre-RoPE** — quantize `k_proj` outputs before rotary embedding |
| **Values** | Per-token `v_proj` outputs |
| **NUQ** | Non-uniform quantization (K-means / NormalFloat signposts) |
| **Dense-and-sparse** | Outlier isolation with configurable sparsity threshold |
| **Pipeline** | Multi-stage: `gradients/` (Fisher) → `quant/` (simquant + pickle quantizers) → `deployment/` (CUDA kernels + forked `transformers`) |
| **Models in scripts** | LLaMA-7B, Mistral, DBRX, LWM long-context — **no Qwen entry script** in main `quant/` tree |

**Upstream repo:** [github.com/SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant)

**External claims (not ExactKV results):** <0.1 perplexity degradation at 3-bit; 1M–10M context serving on A100 systems per paper/README.

---

## 5. Installation results

Inspection date: **2026-06-09**. All installs in **scratch environments** outside ExactKV
default dependencies.

### KIVI (`jy-yuan/KIVI`)

| Step | Environment | Result |
|---|---|---|
| `git clone` → `/tmp/kivi_research` | macOS | ✅ Success |
| `pip install -e .` (main package) | Python 3.13 venv | ❌ `torch==2.4.1` pin not available for 3.13 |
| `pip install -e .` | Python 3.12 venv + `torch==2.4.1` | ❌ `flash-attn` wheel build fails (`ModuleNotFoundError: torch` in isolated build env) |
| `PYTHONPATH=/tmp/kivi_research` import `models.utils_quant` | Python 3.13 + torch (no KIVI pip) | ✅ Success |
| `quant/new_pack` (Triton CUDA pack) | CPU torch | ❌ `ModuleNotFoundError: triton` |
| `cd quant && pip install -e .` (CUDA ext `kivi_gemv`) | Not attempted locally | ⏸ Requires CUDA GPU + nvcc (documented RunPod plan §20) |

### KVQuant (`SqueezeAILab/KVQuant`)

| Step | Environment | Result |
|---|---|---|
| `git clone` → `/tmp/kvquant_research` | macOS | ✅ Success |
| `pip install -e /tmp/kvquant_research/quant` | Python 3.13 isolated venv | ✅ Success (~67s) |
| `import kvquant; from kvquant.simquant_module_quantizer import QuantLinearSim` | Same venv | ✅ Success |
| `pip install flash-attn` | Not attempted on macOS CPU | ⏸ Documented for RunPod |
| `deployment/kvquant/python setup_cuda.py install` | Not attempted locally | ⏸ Requires CUDA GPU |
| `gradients/` Fisher pipeline E2E | Not attempted locally | ⏸ Requires CUDA + HF model + calibration data |

### Scratch script

[`scripts/research/kivi_kvquant_phase_d1_inspect.py`](../scripts/research/kivi_kvquant_phase_d1_inspect.py)
reproduces repo-level checks without touching ExactKV compressors.

---

## 6. Dependency constraints

### KIVI (`pyproject.toml`)

| Dependency | Pin / note | ExactKV conflict risk |
|---|---|---|
| `torch==2.4.1` | Hard pin | ExactKV default env uses newer torch; **isolated `.venv-kivi` required** |
| `transformers==4.43.1` | Hard pin | ExactKV uses transformers 5.x for Qwen2.5; **adapter must not import KIVI model classes into default ExactKV process** |
| `flash-attn` | Required dependency | Build-heavy; CUDA on Linux; fails on macOS CPU |
| `fastchat`, `datasets`, etc. | Eval tooling | Not needed for tensor-bridge adapter |

### KVQuant (`quant/pyproject.toml`)

| Dependency | Note | ExactKV conflict risk |
|---|---|---|
| `torch`, `transformers>=4.28` | Loose pins | Lower pin conflict than KIVI |
| `scikit-learn` | K-means for NUQ calibration | CPU-ok for calibration step |
| `flash-attn` | Required in quant README | GPU build |
| `deployment/transformers` | **Vendored fork** | Must not replace ExactKV's transformers install |

**ExactKV rule (unchanged):** optional extras + lazy import + isolated venv; default
`pip install -e ".[dev]"` and `list_compressors()` unchanged.

---

## 7. Python / CUDA / PyTorch / transformers requirements

| Backend | Python | PyTorch | transformers | CUDA / kernels |
|---|---|---|---|---|
| **KIVI full inference** | 3.10 (upstream conda example) | 2.4.1 | 4.43.1 | **Required:** Triton pack (`quant/new_pack.py`), `cuda_bmm_fA_qB_outer`, `kivi_gemv` ext, flash-attn |
| **KIVI offline simulate** | 3.10+ (tested 3.13) | Any CPU torch | Not required for `utils_quant` only | Not required for `simulate=True` K-channel path |
| **KVQuant simquant** | 3.10 (conda) | CUDA typical | HF `AutoModelForCausalLM` | GPU for layer-wise calibration forward |
| **KVQuant deployment** | 3.9 (deploy README) | half precision | **Forked** `deployment/transformers` | **Required:** `setup_cuda.py`, flash-attn 2.5.5 |

---

## 8. Supported models

### KIVI upstream

| Model family | Upstream support |
|---|---|
| Llama-2 / Llama-3 (GQA) | `models/llama_kivi.py` |
| Mistral | `models/mistral_kivi.py` |
| Falcon | Mentioned in paper; no dedicated file in clone |
| **Qwen / Qwen2.5** | **Not present** — no `qwen_kivi.py`, no README example |

### KVQuant upstream

| Model family | Upstream support |
|---|---|
| LLaMA-7B | `quant/llama_simquant.py`, `deployment/llama.py` |
| Mistral | Mentioned in quant README |
| DBRX | `quant/dbrx_simquant.py` + HF checkpoints |
| LWM (1M context) | `lwm/` directory |
| **Qwen / Qwen2.5** | **No first-class script**; `dbrx/tests` contain Qwen2 MoE **unit tests only** (vendored transformers test tree), not a quant pipeline |

---

## 9. Supported cache formats

### KIVI production cache (`past_key_value` in `LlamaAttention_KIVI`)

Nine logical slots + sequence length (not HF `DynamicCache` / `Cache` API):

| Index | Content |
|---|---|
| 0 | Quantized keys (packed, transposed layout for CUDA matmul) |
| 1 | fp16 residual keys |
| 2–3 | Key scale / min |
| 4 | Quantized values |
| 5 | fp16 residual values |
| 6–7 | Value scale / min |
| 8 | `kv_seq_len` (int) |

Quantization applies to **post-RoPE** `key_states` / `value_states` during forward.

### KIVI offline helpers (`models/utils_quant.py`)

| Function | Role |
|---|---|
| `quantize_by_channel_and_pack_cache` / `dequantize_by_channel_and_unpack_cache` | K: per-channel on `(bsz, heads, seq, head_dim)` |
| `quantize_and_pack` / `dequantize_and_unpack` | V: per-token groups along last dim |
| `simulate=True` | Round-trip without `dequant_cuda` bit packing |

### KVQuant

| Mode | Cache representation |
|---|---|
| **Simquant** | No persistent packed cache — `QuantLinearSim` replaces `k_proj`/`v_proj`; quantizes **projector outputs** each forward; stores quantizer pickles |
| **Deployment** | Custom CUDA-compressed vectors inside forked transformers attention; requires `quantizers.pickle` from calibration |

Neither upstream format is a drop-in HF `past_key_values` tuple of `(K, V)` fp tensors.

---

## 10. API surface discovered

### KIVI

| Surface | Location | Adapter relevance |
|---|---|---|
| `LlamaForCausalLM_KIVI.from_pretrained` | `models/llama_kivi.py` | Full model rewrite — **not** `BackendAdapter` tensor path |
| `LlamaAttention_KIVI.forward` | same | Own cache lifecycle + CUDA kernels |
| `quantize_by_channel_and_pack_cache` | `models/utils_quant.py` | **Offline K compress** on extracted tensors |
| `dequantize_by_channel_and_unpack_cache` | same | **Offline K decompress** → `rebuild_cache` |
| `quantize_and_pack` / `dequantize_and_unpack` | same | **Offline V compress** (CUDA bias in simulate branch — see §21) |
| `triton_quantize_and_pack_along_last_dim` | `quant/new_pack.py` | Production pack — GPU |
| `cuda_bmm_fA_qB_outer` | `quant/matmul.py` | Fused quant attention — GPU |

### KVQuant

| Surface | Location | Adapter relevance |
|---|---|---|
| `llama_simquant.py` | `quant/` | E2E calibration CLI |
| `SimQuant` / `QuantLinearSim` | `kvquant/simquant_module_quantizer.py` | Layer replacement + forward quant |
| `make_quant_sim` | same | Walks module tree; swaps named layers |
| `gradients/run-fisher.py` | `gradients/` | **Prerequisite** Fisher matrices |
| `deployment/llama.py` | `deployment/` | CUDA inference loop |
| `quant_fn_zp`, `quant_fn_nuq_recon` | simquant | Low-level quant primitives |

---

## 11. Whether compressed KV is exposed directly

| Backend | Exposed directly? | Notes |
|---|---|---|
| **KIVI production** | **Yes** — 8-tensor quant state + scales in `past_key_value` | Custom layout; not serializable as standard HF cache |
| **KIVI offline (`utils_quant`)** | **Yes** — quant codes + `(scale, mn)` metadata per compress call | Adapter can store in `backend_data` dict |
| **KVQuant simquant** | **Indirect** — quantizer pickles + replaced modules; no standalone KV blob | Compression is inside forward |
| **KVQuant deployment** | **Yes** — CUDA packed buffers inside forked model | Not extractable without running deployment stack |

For `BackendAdapter`, **KIVI offline helpers** are the cleanest "compressed KV exposed" path (TurboQuant precedent). **KVQuant** compressed state is entangled with model surgery and calibration artifacts.

---

## 12. Whether materialized KV can be produced for draft generation

| Backend | Feasible for ExactKV draft path? |
|---|---|
| **KIVI offline simulate** | **Yes** — dequantize K/V tensors → `rebuild_cache` → HF `past_key_values` → standard `model.forward` for `_get_next_token_id` override (TurboQuant pattern) |
| **KIVI production CUDA** | **Partial** — would require either (a) dequant packed cache to fp16 tensors for materialize, or (b) run `LlamaForCausalLM_KIVI` as draft model — breaks Qwen2.5 without port |
| **KVQuant simquant** | **Only via model rewrite** — must run model with `QuantLinearSim` on `k_proj`/`v_proj`; not a function of extracted `past_key_values` alone |
| **KVQuant deployment** | **Only via forked transformers + CUDA** — same isolation concerns as kvpress but heavier |

---

## 13. Whether full authoritative KV can remain separate for ExactKV verification

| Backend | Compatible with verify isolation? |
|---|---|
| **KIVI offline adapter** | **Yes** — `compress()` clones tensors from `FullKVState`; verify path never calls adapter (unchanged `VerificationEngine`) |
| **KIVI production model** | **Yes if draft uses separate model instance** — but hook/state pollution risk if sharing weights with verify model; prefer offline materialize |
| **KVQuant simquant** | **Risky** — `make_quant_sim` mutates `nn.Module` tree in-place; would require `deepcopy` draft model (kvpress precedent) + restore hooks |
| **KVQuant deployment** | **High risk** — custom forward signatures (`past_key_values_length_inp`); forked transformers |

**Conclusion:** KIVI offline bridge preserves the TurboQuant isolation story. KVQuant
requires draft-model cloning and/or forked transformers — higher blast radius.

---

## 14. Whether hooks, monkeypatching, custom kernels, or model rewrites are used

| Backend | Mechanism |
|---|---|
| **KIVI production** | **Full model rewrite** (`LlamaForCausalLM_KIVI`); **custom CUDA/Triton kernels** in attention; `config.use_flash = True` mandatory |
| **KIVI offline** | **None** — pure tensor functions in `utils_quant.py` |
| **KVQuant simquant** | **Module replacement** (`delattr` + `QuantLinearSim`); calibration **forward hooks** via `Catcher` in `llama_simquant.py` |
| **KVQuant deployment** | **Forked transformers** + **CUDA extensions** + forward hooks for benchmarking |

---

## 15. Whether Qwen/Qwen2.5-0.5B is supported

| Backend | Qwen2.5-0.5B supported upstream? |
|---|---|
| **KIVI repo** | **No** — no Qwen model file or example |
| **KIVI offline on HF cache** | **Plausible** — quant functions are shape-based; `head_dim=64`, GQA `num_kv_heads=2` — no Llama-only assumption in `utils_quant` K-channel simulate (verified MAE round-trip locally) |
| **HF Transformers KIVI-style cache** | **Possible alternate path** (not evaluated in D1) — would be transformers API, not `jy-yuan/KIVI` package |
| **KVQuant** | **No** — no Qwen calibration script; would need new Fisher + simquant port |

**D1 finding:** Qwen2.5-0.5B is **not upstream-supported** by either repo, but **KIVI
offline tensor quant** is architecturally compatible with ExactKV's existing Qwen2.5
HF cache tensors (post-RoPE). KVQuant is **not** compatible without a new calibration
pipeline and pre-RoPE bridge.

---

## 16. Whether Qwen2.5-1.5B / 3B / 7B are likely supported

| Model | KIVI offline bridge | KVQuant |
|---|---|---|
| **1.5B** (`head_dim=128`, 2 KV heads) | **Likely** — same tensor API; residual/group_size tuning needed | **Unknown** — requires new Fisher + quantizer pickle per model |
| **3B** | **Likely** — same | **Unknown** |
| **7B** (`head_dim=128`, 4 KV heads) | **Likely** — GQA shapes supported in KIVI Llama-3 path conceptually | **Unknown** — GPU memory for calibration; no script |

Porting KIVI **production** `LlamaAttention_KIVI` → Qwen2 attention is **engineering work**
(not provided). Offline bridge does not require it for Phase D2 prototype.

---

## 17. Workspace-memory implications

| Path | `stored_kv_bytes` | `materialized_working_kv_bytes` | Honesty notes |
|---|---|---|---|
| **KIVI offline simulate** | Sum of quant codes + scales (+ residual fp16 window if implemented) | Full fp16/fp32 `past_key_values` after dequant | `supports_real_bytes_claim=False` for simulate (int/fp containers, not packed bits) — same honesty pattern as TurboQuant Python |
| **KIVI CUDA packed** | Packed bitstreams + scales | Dequant dense tensors | `supports_real_bytes_claim=True` only after measuring actual packed dtypes |
| **KVQuant simquant** | Quantizer pickle + outlier metadata (dominant) | Full model weights + dense KV during forward | Metadata-heavy; honest accounting requires separating pickle bytes from KV working set |
| **KVQuant deployment** | CUDA compressed buffers | Reconstructed attention operands | Custom kernels; peak workspace not measured in D1 |

ExactKV V5 fields remain **conservative accounting sums**, not measured GPU peaks.

---

## 18. Required BackendAdapter shape

Following [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) and
[`exactkv/compressors/turboquant_adapter.py`](../exactkv/compressors/turboquant_adapter.py):

### KIVI restricted adapter (recommended Phase D2)

```text
class KIVISimulateAdapter(BackendAdapter):
    name = "kivi_sim_k2_v2_r32"  # example — not registered in D1

    _backend_compress(k_tensors, v_tensors, cache_format):
        # Per layer:
        #   K: quantize_by_channel_and_pack_cache(..., simulate=True)
        #   V: quantize_and_pack(..., simulate=True)  # GPU or fixed CPU path
        #   Optional: keep last residual_length tokens in fp16 (KIVI streaming policy)
        # Return backend_data dict with per-layer codes, scales, mn, residual slices

    _backend_materialize(backend_data, cache_format):
        # Dequantize each layer → list[k_hat], list[v_hat]
        # rebuild_cache(k_list, v_list, cache_format)

    _backend_workspace_bytes(...):
        # Count stored quant tensors + metadata; materialized = full dequant KV bytes

    _get_next_token_id(...):  # override — lossy draft
        # materialize + one forward step on draft model
```

Capabilities:

- `is_simulated=False` (real KIVI algorithm via upstream `utils_quant`)
- `supports_real_bytes_claim=False` for simulate path
- `backend_name="kivi"`, `adapter_name="kivi_simulate_python"`, `backend_version` from git SHA or pip pin

Factory: `create_kivi_simulate_adapter(...)` with lazy `PYTHONPATH` or optional `[kivi]` extra.

### KVQuant adapter (if pursued later)

Would require **different shape**:

- `_compresses_via_full_state() -> True` **or** separate draft `AutoModelForCausalLM` with `make_quant_sim` applied
- Calibration artifact (`quantizers.pickle`) loaded per model
- **Cannot** implement faithful pre-RoPE K quant from post-RoPE `extract_kv_tensors` alone without RoPE inversion or projector replay

Not recommended as TurboQuant-style tensor-only adapter.

---

## 19. Required changes, if any, to BackendAdapter

| Change | Needed? | Reason |
|---|---|---|
| New abstract methods | **No** | Existing `_backend_compress` / `_backend_materialize` / `_backend_workspace_bytes` suffice for KIVI offline path |
| `_compresses_via_full_state` | **No** for KIVI offline; **likely Yes** for KVQuant faithful path | KVQuant quantizes at projector outputs during forward |
| Residual-window metadata in `backend_data` | **Optional convention** | Document `residual_length` in adapter; no base-class change |
| Multi-artifact storage (pickle quantizers) | **No base change** | `backend_data` dict can hold pickle bytes; stats accounting in subclass |
| `CompressorCapabilities` | **No** | Existing `backend_name`, `supports_real_bytes_claim` fields sufficient |

**Conclusion:** **No `BackendAdapter` base-class changes** required for KIVI restricted
adapter. KVQuant may force `_compresses_via_full_state` or a documented draft-model
clone pattern (kvpress precedent) without modifying the ABC if clone lifecycle stays in
the subclass.

---

## 20. RunPod GPU plan if needed

Local macOS D1 could not build `flash-attn`, KIVI CUDA extensions, or KVQuant deployment
kernels. The following RunPod plan validates **production-faithful** paths without
blocking D2 KIVI simulate adapter on CPU.

### Pod template

| Parameter | Value |
|---|---|
| Provider | RunPod (or equivalent CUDA Linux host) |
| GPU | 1× RTX 4090 / A100 40GB |
| Image | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` or RunPod PyTorch 2.4 template |
| Python | **3.10** (match upstream conda docs) |
| Disk | ≥ 50 GB (models + clones + venvs) |

### KIVI validation commands (scratch — not ExactKV)

```bash
# Clone outside ExactKV tree
git clone https://github.com/jy-yuan/KIVI.git ~/kivi_research
cd ~/kivi_research
python3.10 -m venv .venv-kivi && source .venv-kivi/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install flash-attn --no-build-isolation
cd quant && pip install -e .

# API smoke (no ExactKV)
python -c "from quant.new_pack import triton_quantize_and_pack_along_last_dim; print('cuda_pack_ok')"
python -c "from models.llama_kivi import LlamaForCausalLM_KIVI; print('kivi_model_ok')"

# Optional: Llama-2-7b KIVI example from upstream README (external model — not Qwen)
```

### KVQuant validation commands (scratch)

```bash
git clone https://github.com/SqueezeAILab/KVQuant.git ~/kvquant_research
cd ~/kvquant_research/gradients && pip install -e . && pip install -r requirements.txt

# Fisher (example — replace MODEL_PATH)
CUDA_VISIBLE_DEVICES=0 python run-fisher.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --output_dir ./fisher_qwen05b --dataset wikitext2 --seqlen 2048 --maxseqlen 2048 --num_examples 16

cd ../quant && pip install -e . && pip install flash-attn --no-build-isolation
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py Qwen/Qwen2.5-0.5B \
  --abits 4 --nsamples 16 --seqlen 2048 --nuq --fisher ../gradients/fisher_qwen05b \
  --quantize --include_sparse --sparsity-threshold 0.99 --quantizer-path quantizers_qwen05b.pickle

cd ../deployment
# Follow deployment/README.md: forked transformers + setup_cuda.py
```

### Qwen2.5 larger models (Phase E alignment)

Repeat Fisher + simquant for `Qwen/Qwen2.5-1.5B` with `core` subset exactness gate
after any adapter exists — **out of D1 scope**.

---

## 21. Failure modes that would reject each backend

### KIVI — would reject adapter if:

| Failure mode | D1 status |
|---|---|
| No API to compress/decompress KV without full model rewrite | **Not triggered** — `utils_quant` provides offline path |
| Post-RoPE mismatch with ExactKV cache | **Not triggered** — KIVI production quantizes post-RoPE |
| Qwen head_dim / GQA incompatibility | **Not triggered** for offline simulate on 0.5B shape |
| `pip install` impossible on Linux CUDA | **Open** — RunPod plan §20; not grounds for rejection per V9 rules |
| Simulate path cannot run V quant on CPU | **Partial** — `quantize_and_pack(..., simulate=True)` hardcodes `device='cuda'` for bit tensor; **workaround:** CPU port of simulate branch or GPU-only V path |
| Acceptance always zero after integration | Would reject **experiment** claims, not feasibility — adapter could still ship with honest low acceptance |
| Exactness failures | **Hard reject** for published cells — same gate as Exp 008 |

### KVQuant — would reject adapter if:

| Failure mode | D1 status |
|---|---|
| Pre-RoPE K quant incompatible with `extract_kv_tensors` | **Triggered** — fundamental tensor-bridge blocker for faithful algorithm |
| No Qwen calibration path | **Triggered** for 0.5B — no upstream script; Fisher+simquant port required |
| Requires forked transformers in ExactKV process | **Triggered** for deployment path — isolation violation risk |
| Cannot materialize HF `past_key_values` for draft without full model | **Triggered** for simquant/deployment |
| Calibration pickle not portable across commits | **Risk** — version pinning required |
| Hook pollution breaks verify model | **Risk** — mitigated by deepcopy draft model only |

**KVQuant is not rejected forever** — but **tensor-only `BackendAdapter` (TurboQuant
pattern) is a no-go** without architectural compromise (e.g., post-RoPE approximate
KVQuant, which would **not** be faithful KVQuant).

---

## 22. Side-by-side feasibility table

| Criterion | KIVI | KVQuant |
|---|---|---|
| **pip install (quant pkg only)** | ❌ Heavy pins + flash-attn | ✅ `quant/` installs cleanly |
| **Import without GPU** | ✅ `utils_quant` via PYTHONPATH | ✅ `kvquant.simquant_module_quantizer` |
| **HF `past_key_values` bridge** | ✅ Offline dequant → `rebuild_cache` | ❌ Pre-RoPE projector quant |
| **Compressed KV as data** | ✅ Per-layer codes + scales | ⚠️ Pickle quantizers + module state |
| **Verify path isolation** | ✅ Tensor-only adapter | ⚠️ Model rewrite / hooks |
| **Qwen2.5-0.5B upstream** | ❌ No model file | ❌ No script |
| **Qwen2.5 offline feasibility** | ✅ K simulate round-trip OK | ❌ Needs new calibration + forward replay |
| **Custom CUDA in ExactKV** | ❌ Stays in upstream venv | ❌ Stays in upstream venv |
| **BackendAdapter ABC changes** | **None** | Likely `_compresses_via_full_state` pattern |
| **RunPod needed for D2 prototype** | **Optional** (CPU simulate) | **Required** for faithful path |
| **RunPod needed for production-faithful** | **Yes** (packed bits + Triton) | **Yes** (Fisher + CUDA deploy) |
| **Experiment 009 readiness (D2)** | **High** (restricted subset) | **Low** (blocked on architecture) |
| **Alignment with Exp 008 pattern** | **High** | **Low** |

---

## 23. Recommendation

### Decision

| Option | Verdict |
|---|---|
| Implement **KIVI** adapter next | ✅ **Yes** — primary recommendation |
| Implement **KVQuant** adapter next | ❌ **Not next** — architectural mismatch |
| Implement **restricted subset only** | ✅ **Yes** — Phase D2 should start with **KIVI offline simulate** (`utils_quant`, post-RoPE tensors), not production `LlamaForCausalLM_KIVI` |
| **Requires RunPod before decision** | ⚠️ **KVQuant only** — faithful KVQuant needs RunPod Fisher/simquant/deployment validation on Qwen2.5 before any adapter commitment |
| **No-go for both** | ❌ **No** — KIVI restricted path is feasible |

### Summary statement

**Proceed to Phase D2: KIVI restricted Python adapter** using upstream
`models/utils_quant.py` on ExactKV-extracted post-RoPE K/V tensors, following the
TurboQuant isolation pattern (`lazy import`, optional extra, not in default registry,
`supports_real_bytes_claim=False` until packed CUDA path validated on RunPod).

**Defer KVQuant adapter** until RunPod confirms a Qwen2.5 Fisher + simquant pipeline
and ExactKV chooses either (a) draft-model clone with `QuantLinearSim` (heavy), or
(b) documents **KVQuant-style simquant on post-RoPE tensors** as a non-faithful
research variant (explicitly labelled, not "KVQuant adapter").

### Phase D1 exit criteria met

- [x] Feasibility doc (this file)
- [x] Installation/import evidence
- [x] Qwen2.5 compatibility analysis
- [x] `BackendAdapter` shape + ABC change assessment
- [x] RunPod plan
- [x] Recommendation for Phase D2

**Commit D1 before adapter implementation:** Yes — this research doc and register
updates should be committed as a **docs-only Phase D1 milestone** before any
`exactkv/compressors/kivi_*.py` code lands (separate Phase D2 approval).

---

## Related documents

| Document | Relevance |
|---|---|
| [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) | Phase D1/D2 scope |
| [`TURBOQUANT_INTEGRATION_RESEARCH.md`](TURBOQUANT_INTEGRATION_RESEARCH.md) | Precedent offline bridge |
| [`TURBOQUANT_ADAPTER_PROTOTYPE.md`](TURBOQUANT_ADAPTER_PROTOTYPE.md) | Adapter pattern |
| [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) | Last real-backend eval |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Contract |
| [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) | D3/D4 status |
| [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md) | Survey |

## Attribution

**KIVI:** Liu et al., ICML 2024, [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) — external claims only.

**KVQuant:** Hooper et al., NeurIPS 2024, [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) — external claims only.

ExactKV does not reproduce or claim external-backend performance or accuracy results.
