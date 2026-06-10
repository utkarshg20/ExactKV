# ExactKV Experiment Index

One-line reference for published experiments **001–011**. All report
`exactkv_failures == 0` unless noted. JSON/CSV artifacts are **gitignored**
under `reports/`.

> ExactKV experiments measure **exactness, acceptance, divergence, and memory
> honesty** — not throughput, latency, speedup, or production serving behaviour.

---

| # | Name | Document | Purpose | Cells | Headline result | Failures | Key caveat |
|---|---|---|---|---:|---|---:|---|
| 001 | Smoke sweep | [`EXPERIMENT_001_SMOKE_SWEEP.md`](EXPERIMENT_001_SMOKE_SWEEP.md) | First multi-compressor sweep | 36 | `int8` accept **0.931** | 0 | `int4_sim` simulated; 6 smoke prompts only |
| 002 | Core sweep | [`EXPERIMENT_002_CORE_SWEEP.md`](EXPERIMENT_002_CORE_SWEEP.md) | Core suite baseline | 204 | `int8` accept **0.951** | 0 | 3 compressors × 2 draft lengths |
| 003 | Asymmetric K/V | [`EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md) | K vs V fragility under compression | 612 | Keys far more fragile than values | 0 | `_sim` uses int8 containers |
| 004 | Workspace memory | [`EXPERIMENT_004_WORKSPACE_MEMORY.md`](EXPERIMENT_004_WORKSPACE_MEMORY.md) | V5 workspace accounting validation | 340 | Five-field memory schema honest | 0 | `total_kv_footprint_bytes` is accounting sum |
| 005 | KVPress Knorm | [`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | Restricted real-backend adapter (kvpress) | 272 | Pruned cache + exactness preserved | 0 | Isolated `[kvpress]` env; not default registry |
| 006A | Proxy divergence | [`EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md) | Reanalyze 003–005 divergence fields | — (analysis) | Aggressive compressors diverge early | N/A | **No new runs**; proxy only, no attention weights |
| 006 | Layer-aware V | [`EXPERIMENT_006_LAYER_AWARE_V.md`](EXPERIMENT_006_LAYER_AWARE_V.md) | Layer-aware boundary V sweep | 374 | `k8_v4_boundary_v8_sim` +0.013 vs `k8_v4_sim` | 0 | Simulated policies; int8 containers |
| 006C | Boundary depth | [`EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md`](EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) | Boundary depth N=1/2/4 ablation | 170 | `k8_v4_boundary4_v8_sim` accept **0.954** | 0 | Not real packed-bit; not attention-gated |
| 007 | Serving harness | [`EXPERIMENT_007_SERVING_CONTEXT.md`](EXPERIMENT_007_SERVING_CONTEXT.md) | Serving-context compatibility (Mode B) | 238 | All harness invariants pass; 907 commit rounds | 0 | **Not** vLLM/LMCache; identity phys/logical panel |
| 008 | TurboQuant Python | [`EXPERIMENT_008_TURBOQUANT_PYTHON.md`](EXPERIMENT_008_TURBOQUANT_PYTHON.md) | Restricted TurboQuant adapter vs baselines | 272 | `turboquant_python_k3_v3` accept **0.435**; exactness preserved | 0 | Python path only; not llama.cpp/MLX; `supports_real_bytes_claim=False` |
| 009 | KIVI offline | [`EXPERIMENT_009_KIVI_OFFLINE.md`](EXPERIMENT_009_KIVI_OFFLINE.md) | Restricted KIVI offline adapter vs baselines | 272 | `kivi_offline_k2_v2` accept **0.012**; exactness preserved | 0 | Offline simulate only; not CUDA/Triton; `supports_real_bytes_claim=False` |
| 010 | KVQuant simquant | [`EXPERIMENT_010_KVQUANT_SIM.md`](EXPERIMENT_010_KVQUANT_SIM.md) | Restricted KVQuant simquant adapter vs baselines | 272 | `kvquant_sim_qwen05b` accept **0.792**; exactness preserved | 0 | Simquant only; not deployment CUDA; `supports_real_bytes_claim=False`; RunPod GPU |
| 011 | Larger-model validation | [`EXPERIMENT_011_LARGER_MODEL_VALIDATION.md`](EXPERIMENT_011_LARGER_MODEL_VALIDATION.md) | Qwen2.5-1.5B RunPod exactness + acceptance stability | 238 | `int8` accept **0.980**; boundary4 **0.954** > k8_v4_sim **0.945** | 0 | Not a performance benchmark; cuda fp16; 3B stretch not run |

---

## Reproduction commands

| Experiment | Command |
|---|---|
| 001–004, 006, 006C | `python3 -m exactkv sweep` (see each report) |
| 005 | `.venv-kvpress/bin/python scripts/run_experiment_005_kvpress_knorm.py` |
| 007 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_007_serving_context.py` |
| 008 | `PYTHONPATH=vendor/turboquant_plus .venv-turboquant/bin/python scripts/run_experiment_008_turboquant_python.py` |
| 009 | `PYTHONPATH=/tmp/kivi_research .venv-turboquant/bin/python scripts/run_experiment_009_kivi_offline.py` |
| 010 | `EXACTKV_KVQUANT_QUANTIZERS=/path/to/quantizers.pickle python scripts/run_experiment_010_kvquant_sim.py` (KVQuant venv, CUDA) |
| 011 | `python scripts/run_experiment_011_larger_model_validation.py` (RunPod CUDA, fp16) |

---

## Related

- [`RELEASE_NOTES_V0.9.0.md`](RELEASE_NOTES_V0.9.0.md) — V9 changelog (latest)
- [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md) — current project status
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — V10+ tracker
