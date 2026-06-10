# KVQuant RunPod Validation (V9 Phase D4)

**Status:** Phase D4 complete — **static validation + Qwen2.5 module walk + RunPod playbook**.
**Live RunPod GPU pipeline not executed** in the development environment (no CUDA GPU,
no RunPod CLI). GPU confirmation steps are documented in
[`scripts/research/kvquant_runpod_commands.sh`](../scripts/research/kvquant_runpod_commands.sh).

**Date:** 2026-06-09  
**Recommendation:** **Option A — faithful adapter feasible (provisional)** → Phase D5
draft-model-clone replay adapter. **Not** tensor-only post-RoPE bridge.

> This phase does **not** implement a KVQuant adapter. This phase does **not** run
> ExactKV Experiment 010. ExactKV does **not** claim KVQuant results yet. External
> KVQuant paper results are **not** ExactKV results. No throughput, latency, speedup,
> runtime, tokens/sec, active GPU memory, or production-serving claim is made. If a
> forward pass works on RunPod, that is a **feasibility result only**, not an ExactKV
> acceptance result.

---

## 1. Purpose

V9 Phase D4 answers whether KVQuant can support a **faithful** ExactKV adapter path
for `Qwen/Qwen2.5-0.5B` after:

- Experiment 008 (TurboQuant Python, accept **0.435**)
- Experiment 009 (KIVI offline k2/v2, accept **0.012**)
- Phase D1 feasibility (KVQuant deferred pending RunPod)

Phase D4 must decide **go / restricted / no-go** for Phase D5 adapter prototype and
document the exact adapter shape — without implementing adapter code or Experiment 010.

---

## 2. RunPod environment

### Target pod (illustrative)

| Parameter | Preferred | Acceptable first attempt |
|---|---|---|
| Provider | RunPod | Any CUDA Linux host |
| GPU | A100 40GB | L40S 48GB, **A40** |
| Image | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` | RunPod PyTorch 2.4+ template |
| Python | 3.10 | 3.9–3.11 |
| Disk | ≥ 30 GB | Model + clone + scratch artifacts |

### Dev environment (this validation session)

| Item | Value |
|---|---|
| Host | macOS, **no `nvidia-smi`** |
| RunPod CLI | **Not installed** |
| GPU pipeline | **Not executed** |
| Static / CPU checks | Executed locally |

### Reproduce GPU validation on RunPod

```bash
# On pod after cloning ExactKV + KVQuant
bash scripts/research/kvquant_runpod_commands.sh 2>&1 | tee /tmp/kvquant_d4_run.log
```

Scratch output (gitignored): `/tmp/kvquant_d4/quantizers_qwen05b.pickle`

---

## 3. KVQuant repo and commit

| Item | Value |
|---|---|
| Upstream | [github.com/SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant) |
| Clone path (scratch) | `/tmp/kvquant_research` |
| Commit (inspect) | `57a238357f0f` |
| Subprojects used | `quant/` (simquant), `gradients/` (Fisher — optional), `deployment/` (CUDA — out of D4 scope) |

---

## 4. Install result

### Local (CPU macOS, 2026-06-09)

| Step | Result |
|---|---|
| `git clone` KVQuant | ✅ (from D1) |
| `pip install -e quant/` | ✅ (`/tmp/kvquant_venv_test`, Python 3.13) |
| `import kvquant` | ✅ (requires `scikit-learn` in venv) |
| `pip install flash-attn` | ⏸ Not on macOS CPU |
| `deployment/setup_cuda.py` | ⏸ Not attempted (CUDA required) |
| `gradients/` pip install | ⏸ Not required for first-pass calibration (`fisher=None`) |

### RunPod (documented, not executed here)

See [`kvquant_runpod_commands.sh`](../scripts/research/kvquant_runpod_commands.sh):

1. Clone KVQuant
2. `python -m venv .venv-kvquant && pip install -e quant/`
3. Optional `pip install flash-attn` (patch `llama_simquant.py` if build fails)
4. Record `torch`, `transformers`, CUDA via `nvidia-smi`

---

## 5. Import result

| Symbol | Path | Local result |
|---|---|---|
| `kvquant` package | `quant/kvquant/` | ✅ |
| `QuantLinearSim` | `kvquant/simquant_module_quantizer.py` | ✅ |
| `SimQuant` | same | ✅ |
| `make_quant_sim` | same | ✅ |
| `find_layers` | `kvquant/modelutils.py` | ✅ |
| Fisher `run-fisher.py` | `gradients/run-fisher.py` | Present; **Llama-specific hooks** |

Scratch inspector: [`scripts/research/kvquant_phase_d4_inspect.py`](../scripts/research/kvquant_phase_d4_inspect.py)

---

## 6. Qwen2.5 module compatibility

**Local CPU walk (2026-06-09):**

| Check | Result |
|---|---|
| `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")` | ✅ |
| `config.model_type` | `qwen2` |
| `kvquant.parse_model()` classification | **`llama`** (fallback for non-opt/dbrx) |
| `model.model.layers` | ✅ 24 layers |
| GQA | `num_key_value_heads=2`, `head_dim=64` |
| `find_layers(layer0)` contains `self_attn.k_proj` | ✅ |
| `find_layers(layer0)` contains `self_attn.v_proj` | ✅ |
| Expected quantizer keys | `model.layers.{i}.self_attn.k_proj`, `...v_proj` |

**Conclusion:** Qwen2.5 uses the same linear module naming and `model.model.layers`
layout as Llama-class models in KVQuant's `model_parse.py`. **No Qwen-specific script**
exists, but the **llama_simquant.py** path is **mechanically compatible** after
flash-attn patch.

### Known patches for Qwen on RunPod

| Issue | Mitigation |
|---|---|
| `use_flash_attention_2=True` in `get_model` | Patch to `attn_implementation='sdpa'` (script included) |
| Fisher `k_proj.act.grad` | **Skip Fisher** on first pass (`fisher=None`); optional later |
| Fisher `model.model.set_devices()` | Use stock HF Qwen + calibration without gradients repo |
| `DEV = cuda:0` | Run on GPU pod only |

---

## 7. Fisher / calibration result

| Task | Dev environment | RunPod (documented) |
|---|---|---|
| Fisher on Qwen2.5 | **Not attempted** — Llama-patched `run-fisher.py` | Optional Phase D5+; not blocking |
| `llama_simquant.py --quantize` without Fisher | **Not attempted** (no CUDA) | **Primary D4 GPU gate** |
| Dataset | — | `wikitext2` (built-in `datautils.get_loaders`) |
| Tiny config | — | `--nsamples 4 --seqlen 128` |

**Expected calibration command (RunPod):**

```bash
cd KVQuant/quant
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py Qwen/Qwen2.5-0.5B \
  --abits 4 --nsamples 4 --seqlen 128 --maxseqlen 128 \
  --dataset wikitext2 --quantize \
  --quantizer-path /tmp/kvquant_d4/quantizers_qwen05b.pickle
```

**Feasibility assessment (static):** Calibration hooks `k_proj`/`v_proj` via
`register_forward_hook` + `SimQuant.add_batch` on layer outputs — **pre-RoPE**
projector outputs. Structure matches Qwen2 attention. **High confidence** pending
GPU confirmation.

---

## 8. Quantizer artifact result

| Item | Dev | RunPod (expected) |
|---|---|---|
| `quantizers.pickle` produced | ❌ (no GPU run) | ✅ if calibration succeeds |
| Key format | — | `model.layers.{i}.self_attn.{k,v}_proj` |
| Value per key | — | Tuple of thresholds / centroids / optional NUQ LUT |
| Adapter loadable later | — | ✅ via `pickle.load` + `make_quant_sim` |

Artifact path (gitignored): `/tmp/kvquant_d4/quantizers_qwen05b.pickle`

---

## 9. QuantLinearSim / forward-pass result

| Step | Dev | RunPod (documented) |
|---|---|---|
| `make_quant_sim` replaces `k_proj`/`v_proj` | Static code review ✅ | Execute on pod |
| `QuantLinearSim.forward` | Requires CUDA (`.cuda()` hardcoded) | GPU pod |
| One forward with `use_cache=True` | Not run | Script step 5 in `kvquant_runpod_commands.sh` |
| `past_key_values` returned | Expected post-RoPE HF cache | Feasibility check only |

**Pre-RoPE semantics:** Quantization applies to **linear projector outputs** before
rotary embedding inside attention. ExactKV's `extract_kv_tensors` sees **post-RoPE**
`past_key_values`. A tensor-only offline bridge (KIVI/TurboQuant pattern) would be
**non-faithful**. Faithful path must **replay draft forward** through `QuantLinearSim`
modules.

---

## 10. Draft model isolation assessment

| Question | Finding |
|---|---|
| Does KVQuant mutate modules in place? | **Yes** — `make_quant_sim` uses `delattr` + `setattr` to swap `nn.Linear` → `QuantLinearSim` |
| Do calibration hooks persist? | **No** — `register_forward_hook` removed after `llama_calibration` layer loop |
| Can verify model stay separate? | **Yes** — if ExactKV keeps **two model instances**: `draft_model` (quantized) and `runtime.model` (clean) |
| Draft clone needed? | **Yes** — mirror [`kvpress_knorm`](../exactkv/compressors/kvpress_knorm.py) `isolate_compression_model=True` |
| Forked transformers required for simquant? | **No** — `quant/` uses stock HF; **deployment/** fork is separate CUDA path (out of D5 scope) |

---

## 11. Whether full authoritative KV can remain separate

**Yes**, under the same invariant as kvpress and TurboQuant:

- `VerificationEngine` uses `FullKVState` / `runtime.model` only — never the compressor.
- Draft path uses **cloned** `draft_model` with `QuantLinearSim` applied.
- Verify model must have **zero** `QuantLinearSim` modules (assertion in RunPod script).
- Compress path replays from full state through draft model; verify never calls compressor.

---

## 12. Whether KVQuant can fit BackendAdapter

| Adapter pattern | Fits? | Notes |
|---|---|---|
| Tensor-only `_backend_compress` (TurboQuant/KIVI) | **No** | Pre-RoPE quant vs post-RoPE HF cache |
| `_compresses_via_full_state()` replay (kvpress) | **Yes** | Primary faithful path |
| `make_quant_sim` + quantizers.pickle | **Yes** | Per-model calibration artifact |
| `supports_real_bytes_claim` | **False** for simquant | No packed CUDA deployment format in D5 |
| Registry | **No** | Factory-only, isolated venv |

---

## 13. Required adapter shape for Phase D5

**Provisional `KVQuantSimAdapter` (not implemented in D4):**

```text
class KVQuantSimAdapter(BackendAdapter):
    _compresses_via_full_state() -> True

    __init__(runtime, quantizer_path, *, abits=4, isolate_draft_model=True):
        load quantizers.pickle
        draft_model = deepcopy(runtime.model) if isolate else runtime.model
        make_quant_sim(draft_model, k_proj quantizers, perchannel=True)
        make_quant_sim(draft_model, v_proj quantizers, perchannel=False, dynamicquantization=True)

    _backend_compress_from_full_state(state):
        replay prefill tokens through draft_model with use_cache=True
        store past_key_values + quantizer metadata in backend_data

    _backend_materialize(backend_data):
        return stored/rebuilt HF past_key_values for draft forward

    _get_next_token_id:
        override for lossy draft (materialize + partial forward)

    capabilities:
        backend_name="kvquant"
        adapter_name="KVQuantSimAdapter"
        supports_real_bytes_claim=False  # simquant path
        notes: faithful simquant replay; not deployment CUDA; not post-RoPE approx
```

**Not in D5 scope:** `deployment/` CUDA kernels, forked transformers, Fisher/NUQ until
baseline simquant path works.

---

## 14. Failure modes observed

| Failure mode | Observed in D4? | Severity |
|---|---|---|
| Qwen not in upstream scripts | Static only | **Mitigated** — llama path works |
| `use_flash_attention_2=True` load failure | Static | **Patch** to sdpa |
| Fisher incompatible with Qwen | Static | **Bypass** — `fisher=None` |
| No CUDA for calibration/forward | Dev env | **RunPod required** |
| In-place `make_quant_sim` pollutes verify model | Static | **Mitigated** — draft clone |
| Post-RoPE tensor bridge | D1 + D4 | **Blocker** for non-faithful shortcut |
| `gradients/` `set_devices()` on Qwen | Static | **Blocker** for Fisher only |
| Deployment fork replaces transformers | Static | **Avoid** in D5; simquant only |

---

## 15. Go / no-go recommendation

### Classification: **A — Faithful adapter feasible (provisional)**

| Option | Verdict |
|---|---|
| **A. Faithful adapter** (draft clone + `_compresses_via_full_state` + simquant replay) | ✅ **Recommend Phase D5** after RunPod GPU script succeeds |
| **B. Restricted non-faithful** (post-RoPE tensor quant) | ❌ **Do not** label as KVQuant; not recommended |
| **C. No-go** | ❌ Qwen mechanical compatibility is sufficient to reject C |

### Decision gates before D5 adapter code

1. **Operator runs** `kvquant_runpod_commands.sh` on A100/L40S/A40 and confirms:
   - `quantizers_qwen05b.pickle` created
   - draft forward + `use_cache=True` succeeds
   - verify model has no `QuantLinearSim` modules
2. If GPU calibration fails after sdpa patch → downgrade to **C** and document blocker.
3. Experiment 010 remains **separate approval** after D5 prototype smoke gate.

### Phase D5 should **not** use:

- `deployment/` CUDA path
- Post-RoPE `extract_kv_tensors` quant
- Default registry
- Fisher/NUQ until baseline uniform simquant works

---

## 16. What this does not prove

- ExactKV acceptance or exactness with KVQuant (Experiment 010 not run).
- Upstream KVQuant paper perplexity or long-context claims.
- Deployment CUDA kernel compatibility.
- That simquant acceptance will exceed KIVI (0.012) or TurboQuant (0.435) — unknown until D5/D10.
- Production readiness, throughput, latency, or GPU peak memory.

---

## 17. No-performance-claim note

This document and scratch scripts record **feasibility and adapter-shape decisions**
only. No timing, throughput, tokens/sec, speedup, `runtime_seconds`, or
`active_gpu_kv_bytes` fields are produced. External KVQuant metrics cited in
[`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md) remain
**upstream claims**, not ExactKV results.

---

## Related documents

| Document | Relevance |
|---|---|
| [`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md) | Phase D1 baseline |
| [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) | KIVI accept **0.012** |
| [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) | TurboQuant accept **0.435** |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Adapter contract |
| [`KVPRESS_KNORM_VALIDATION.md`](KVPRESS_KNORM_VALIDATION.md) | Draft/verify isolation precedent |

## Attribution

**KVQuant:** Hooper et al., NeurIPS 2024, [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) —
external claims only.
