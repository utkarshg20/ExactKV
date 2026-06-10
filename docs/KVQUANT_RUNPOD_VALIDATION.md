# KVQuant RunPod Validation (V9 Phase D4)

**Status:** Phase D4 complete — **D4a static research + D4b live RunPod GPU validation**.
**D4b executed 2026-06-10** on RunPod L40S (46 GB) via proxy SSH.

**Date:** 2026-06-10
**Recommendation:** **Option A — faithful adapter go** (with documented D5 patches) →
Phase D5 `KVQuantSimAdapter` draft-clone replay. **Not** tensor-only post-RoPE bridge.

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

### D4b live pod (2026-06-10)

| Item | Value |
|---|---|
| Provider | RunPod proxy SSH (`ssh.runpod.io`) |
| Pod hostname | `116ee198b217` |
| GPU | **NVIDIA L40S**, 46068 MiB |
| Driver / CUDA (nvidia-smi) | 550.127.05 / **12.4** |
| Python | 3.12.3 (`/usr/local/bin/python`) |
| torch | 2.8.0+cu128 (system; venv uses `--system-site-packages`) |
| transformers | **4.44.2** (pinned; 5.x breaks calibration) |
| ExactKV SHA (reference) | `b19c2107f6a640cf995d71b13865deab62809422` |
| Scratch workdir | `/workspace/kvquant_d4` |

### D4a dev environment (static only)

| Item | Value |
|---|---|
| Host | macOS, no local CUDA |
| Checks | Import walk, Qwen module naming, adapter-shape research |

### Reproduce D4b on RunPod

```bash
# Proxy SSH (no SCP); pipe scripts via stdin — see scripts/research/
bash scripts/research/kvquant_runpod_d4b_execute.sh   # or stepwise SSH commands
python scripts/research/kvquant_runpod_synthetic_calib.py  # if wikitext2 fails
python scripts/research/kvquant_runpod_forward_check.py
```

Artifact path (outside git): `/workspace/kvquant_d4/quantizers_qwen05b.pickle` (151745 bytes, 48 keys)

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

### D4a local (CPU macOS)

| Step | Result |
|---|---|
| `git clone` KVQuant | ✅ (from D1) |
| `pip install -e quant/` | ✅ |
| `import kvquant` | ✅ (requires `scikit-learn`) |

### D4b RunPod (2026-06-10)

| Step | Result |
|---|---|
| Clone KVQuant | ✅ `/workspace/kvquant_d4/KVQuant` |
| venv `--system-site-packages` | ✅ **Required** — plain venv reinstalled torch cu130 → `cuda=False` |
| `pip install -e quant/ --no-deps` | ✅ + explicit deps |
| `pip install flash-attn` | ❌ failed (not needed; upstream uses `sdpa`) |
| `deployment/setup_cuda.py` | ⏸ Not used (simquant only) |
| `gradients/` | ⏸ Not used (`fisher=None`) |

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

| Task | D4a static | D4b RunPod |
|---|---|---|
| Fisher on Qwen2.5 | Skipped (Llama hooks) | Skipped (same) |
| `llama_simquant.py --dataset wikitext2` | Not run | **FAILED** — `HfUriError` (datasets/huggingface_hub URI) |
| Synthetic calibration (`synthetic_calib.py`) | N/A | **OK** — 4 samples, seqlen 128 |
| transformers pin | N/A | **Required** `transformers==4.44.2` (5.x → `position_embeddings` None) |

**D4b calibration command (workaround used):**

```bash
cd /workspace/kvquant_d4/KVQuant/quant
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python /workspace/kvquant_d4/synthetic_calib.py
```

Calibration hooks `k_proj`/`v_proj` projector outputs (pre-RoPE). **Confirmed on GPU.**

---

## 8. Quantizer artifact result

| Item | D4a | D4b RunPod |
|---|---|---|
| `quantizers_qwen05b.pickle` produced | ❌ | ✅ |
| Path | — | `/workspace/kvquant_d4/quantizers_qwen05b.pickle` |
| Size | — | **151745 bytes** |
| Key count | — | **48** (24×k_proj + 24×v_proj) |
| Key format | — | `model.layers.{i}.self_attn.{k,v}_proj` |
| Adapter loadable | — | ✅ `pickle.load` + patched `make_quant_sim` |

Keep outside git; document path for D5 prototype calibration.

---

## 9. QuantLinearSim / forward-pass result

| Step | D4a | D4b RunPod |
|---|---|---|
| `make_quant_sim` on k_proj/v_proj | Code review ✅ | ✅ after **bias patch** |
| `QuantLinearSim.forward` | CUDA required | ✅ on L40S |
| One forward `use_cache=True` | Not run | ✅ logits `(1, 5, 151936)`, `past_key_values` (24 layers) |
| Qwen bias handling | Noted statically | **Patch required:** pass `tmp.bias` not `tmp.bias is not None`; `if bias is not None:` in `QuantLinearSim` |

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

| Failure mode | Observed | Severity / fix |
|---|---|---|
| venv reinstalls torch → CUDA broken | D4b | **Fix:** `--system-site-packages` venv |
| wikitext2 `HfUriError` | D4b | **Workaround:** synthetic calibration |
| transformers 5.x breaks Qwen2 calib | D4b | **Fix:** pin `transformers==4.44.2` |
| Qwen k_proj/v_proj have bias | D4b | **Patch:** `make_quant_sim` bias args |
| Fisher incompatible with Qwen | D4a/b | Bypass — `fisher=None` |
| In-place `make_quant_sim` | D4b | Mitigated — `deepcopy` draft |
| Post-RoPE tensor bridge | D1 | Blocker for non-faithful shortcut |
| deployment/ CUDA fork | D4b | Not used; avoid in D5 |

---

## 15. Go / no-go recommendation

### Classification: **A — Faithful adapter go** (D4b GPU confirmed)

| Option | Verdict |
|---|---|
| **A. Faithful adapter** (draft clone + `_compresses_via_full_state` + simquant replay) | ✅ **Proceed to Phase D5** |
| **B. Restricted non-faithful** (post-RoPE tensor quant) | ❌ Do not label as KVQuant |
| **C. No-go** | ❌ Rejected — GPU pipeline succeeded with documented patches |

### D4b gates passed

1. ✅ `quantizers_qwen05b.pickle` created (48 keys)
2. ✅ draft forward + `use_cache=True` succeeds
3. ✅ verify model has no `QuantLinearSim` modules (`verify_model_clean=True`)
4. ✅ `deepcopy` draft isolation confirmed

### D5 prerequisites from D4b

1. Pin `transformers~=4.44` in isolated KVQuant venv
2. Ship Qwen **bias patch** for `make_quant_sim` / `QuantLinearSim`
3. Calibration: per-model pickle; synthetic or fixed dataset loader acceptable for prototype
4. Experiment 010 remains **separate approval** after D5 smoke gate

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
