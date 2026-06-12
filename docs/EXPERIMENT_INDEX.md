# ExactKV Experiment Index

One-line reference for published experiments **001–020**. All report
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
| 012 | V10 suite expansion | [`EXPERIMENT_012_EVAL_SUITE_EXPANSION.md`](EXPERIMENT_012_EVAL_SUITE_EXPANSION.md) | V10 eval suites + per-category leaderboards | 896 | boundary4 **0.923** > k8_v4_sim **0.914**; `int8` **0.957** | 0 | 128 prompts × 7 suites; not a universal benchmark; cpu float32 |
| 013 | Sensitivity forensics | [`EXPERIMENT_013_SENSITIVITY_FORENSICS.md`](EXPERIMENT_013_SENSITIVITY_FORENSICS.md) | draft_len × max_new_tokens grid + divergence forensics | 2160 | boundary4 **0.932** > k8_v4_sim **0.923**; `int8` **0.970** | 0 | core_v2 + stress subset; 3×3 grid; RunPod A5000 fp16 |
| 014 | Real-backend spot-checks | [`EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md`](EXPERIMENT_014_REAL_BACKEND_SPOTCHECKS.md) | V10 harder-category subset + factory-only real backends | 280 | KVQuant **0.634** > TurboQuant **0.309** > KIVI **0.019**; boundary4 **0.876** > k8_v4_sim **0.864** | 0 | 40 prompts × 7 compressors; cross-panel merge; not comprehensive |
| 015 | 1.5B V10 suite validation | [`EXPERIMENT_015_QWEN15B_V10_SUITES.md`](EXPERIMENT_015_QWEN15B_V10_SUITES.md) | Qwen2.5-1.5B transfer check on full V10 suites | 896 | `int8` **0.978**; boundary4 **0.951** > k8_v4_sim **0.942**; margin **+0.009** (matches Exp 012) | 0 | Larger-model validation; RunPod A5000 fp16; not a performance benchmark |
| 016 | 3B V10 suite validation | [`EXPERIMENT_016_QWEN3B_V10_SUITES.md`](EXPERIMENT_016_QWEN3B_V10_SUITES.md) | Qwen2.5-3B built-in stretch on full V10 suites | 896 | `int8` **0.991**; boundary4 **0.952** > k8_v4_sim **0.951**; margin **+0.001** | 0 | 3B scale validation; RunPod A5000 fp16; not a performance benchmark |
| 017 | Serving sidecar/probe refresh | [`EXPERIMENT_017_SERVING_SIDECAR_PROBE.md`](EXPERIMENT_017_SERVING_SIDECAR_PROBE.md) | V11 serving feasibility refresh + metadata-only sidecar probe | 32 | Sidecar probe **pass**; direct vLLM/LMCache **no-go reaffirmed**; `exactkv_failures == 0` | 0 | Not production serving; harness/sidecar only |
| 018 | GPU memory methodology pilot | [`EXPERIMENT_018_GPU_MEMORY_PILOT.md`](EXPERIMENT_018_GPU_MEMORY_PILOT.md) | PyTorch CUDA allocation pilot vs V5 accounting | 100 | `pilot_success`; 0.5B+1.5B × 10 prompts × 5 compressors; `exactkv_failures == 0` | 0 | Isolated pilot artifact; not a performance benchmark; no `active_gpu_kv_bytes` |
| 019 | Divergence autopsy | [`EXPERIMENT_019_DIVERGENCE_AUTOPSY.md`](EXPERIMENT_019_DIVERGENCE_AUTOPSY.md) | Mechanistic divergence forensics + repair hypotheses | 400 | 0.5B+1.5B × 25 prompts × 4 compressors × draft_len {4,8}; `int8` accept **0.967**; attention **deferred** | 0 | Forensics only; repair policies hypothesis-only; RunPod A5000 fp16 |
| 020 | Repair policy pilot | [`EXPERIMENT_020_REPAIR_POLICY_PILOT.md`](EXPERIMENT_020_REPAIR_POLICY_PILOT.md) | Autopsy-guided policy selection pilot vs baselines | 300 | `fallback_int8` accept **0.979**; `category_adaptive` **0.973**; `exactkv_failures == 0` | 0 | Experiment-layer only; not production policy; RunPod A5000 fp16 |
| 022 | TurboQuant llama.cpp probe | [`EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md`](EXPERIMENT_022_TURBOQUANT_LLAMACPP_PROBE.md) | External llama.cpp/GGUF drafter vs HF verifier (Mode B) | 10 | External-probe accept **0.486**; token-safe **6/10**; Mode B **go with restrictions** | 0 | Not BackendAdapter; not llama.cpp integration; RunPod CPU |
| 023 | KVQuant 1.5B validation | [`EXPERIMENT_023_KVQUANT_LARGER_MODEL.md`](EXPERIMENT_023_KVQUANT_LARGER_MODEL.md) | KVQuant simquant on Qwen2.5-1.5B hard V10 panel | 200 | `kvquant_sim_qwen15b` accept **0.609**; `int8` **0.962**; `exactkv_failures == 0` | 0 | Simquant only; not deployment CUDA; RunPod A5000 fp16 |
| 024 | KIVI CUDA/Triton feasibility | [`EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md`](EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md) | Packed-path CUDA/Triton audit vs Exp 009 offline simulate | — (feasibility) | **`B_restricted_go`**; Triton pack OK; `dequant_cuda` missing; no Qwen KIVI model | N/A | Feasibility only; not acceptance panel; not production serving; RunPod A5000 |
| 025 | Full-suite repair-policy validation | [`EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md`](EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md) | Exp 020 policies on full 128-prompt V10 suites | 768 | `int8_all` **0.957**; `category_adaptive` **0.948** > boundary4 **0.923**; pilot gains **shrank** vs Exp 020 | 0 | Experiment-layer only; not production policy; CPU float32 0.5B |
| 026 | Attention logging feasibility | [`EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md`](EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md) | True attention weights: sdpa no-go; eager prefill **restricted_go** | — (feasibility) | Qwen eager prefill OK (24 layers, 14 heads); default/decode-step blocked | N/A | Feasibility only; diagnostic; not production; CPU float32 |
| 027 | Performance/memory truth boundary | [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md) | Claim-boundary review after V1–V12; not a benchmark | — (review) | Speed and active GPU memory savings **forbidden**; V13 Practicality Proof recommended | N/A | No new timing/GPU measurements; see [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md) |

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
| 012 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_012_eval_suite_expansion.py` |
| 013 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_013_sensitivity_forensics.py --device cuda --dtype float16 --include-stress-subset` |
| 014 | `python3 scripts/run_experiment_014_real_backend_spotchecks.py --merge-only --merge-from reports/exp014_panel_*.json` (after per-panel runs; see report) |
| 015 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_015_qwen15b_v10_suites.py --device cuda --dtype float16` (RunPod GPU) |
| 016 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_016_qwen3b_v10_suites.py --device cuda --dtype float16` (RunPod GPU) |
| 017 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_017_serving_sidecar_probe.py` |
| 018 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_018_gpu_memory_pilot.py --device cuda --dtype float16 --include-15b` (RunPod GPU) |
| 019 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_019_divergence_autopsy.py --device cuda --dtype float16 --include-15b` (RunPod GPU) |
| 020 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_020_repair_policy_pilot.py --device cuda --dtype float16 --include-15b` (RunPod GPU) |
| 022 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_022_turboquant_llamacpp_probe.py` (RunPod; `LLAMA_CPP_BIN_DIR` + GGUF) |
| 023 | `EXACTKV_KVQUANT_QUANTIZERS=/workspace/kvquant_d4/quantizers_qwen15b.pickle python scripts/run_experiment_023_kvquant_larger_model.py` (KVQuant venv, RunPod CUDA) |
| 024 | `bash scripts/research/kivi_cuda_triton_exp024_runpod_execute.sh` (RunPod CUDA; isolated KIVI venv at `/workspace/kivi_exp024`) |
| 025 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_025_full_suite_repair_policy.py --device cuda --dtype float16` (optional `--include-15b`) |
| 026 | `TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_026_attention_logging_feasibility.py` |
| 027 | Review only — [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md); optional inspect: `python3 scripts/research/performance_memory_boundary_inspect.py` |

---

## Related

- [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md) — V10 changelog (latest)
- [`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md) — current project status
- [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) — v1.0.0 launch-gate decision
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — V10+ tracker
