# Release Evidence Status (Gate R0)

**Status:** PASS

## Evidence-complete

- Phase A 336-cell multi-model benchmark (`reports/phaseA_benchmark.json`)
- Phase D Llama runtime probe (`reports/phaseD_runtime_probe.json`)
- Phase F CUDA/Triton kernel microbenchmark (`reports/phaseF_kernel_benchmark.json`)
- Phase G unified truth engine (`reports/phaseG_unified_truth.json`)
- Phase H platform layer (`reports/benchmark.json`)
- Phase H+ real 7B/8B scale benchmark (`reports/scale_7b/raw.json`)

## Qualified / disclosure required

- SpectralQuant: `int4_sim_scaling_fallback`
- Shard: `probe_first_heuristic` (probe-only)
- Phase F speedups are **kernel microbenchmarks only** (not end-to-end inference)
- Compression ratios are stored tensor byte ratios unless active GPU memory was measured
- ExactKV is a research-grade evaluation framework, not a production serving runtime

## Current headline evidence

- Scale cells: 1500
- Models: meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-Instruct-v0.3
- ExactKV failures: 0
- Deterministic mode: False
- Phase F INT8 kernel speedup: 1.63x
- Phase F INT4 kernel speedup: 1.54x
- block_sparse execution backend: torch

## Validation checks

| Check | Pass | Detail |
|-------|------|--------|
| scale_raw_exists | yes |  |
| scale_cell_count | yes | got 1500, expected 1500 |
| scale_model_Llama-3.1-8B | yes | present=True |
| scale_model_Mistral-7B-Instruct-v0.3 | yes | present=True |
| scale_no_blocked_models | yes | blocked={} |
| scale_zero_failures | yes | exactkv_failures=0 |
| scale_not_deterministic | yes | deterministic_mode=False |
| scale_summary_not_deterministic | yes |  |
| public_leaderboard_exists | yes | /Users/utkarshgupta/Documents/ExactKV/reports/public_release/leaderboard_final.j |
| public_manifest_exists | yes | /Users/utkarshgupta/Documents/ExactKV/reports/public_release/release_manifest.js |
| manifest_references_scale_7b | yes |  |
| public_leaderboard_covers_raw_models | yes | ok |
| public_mistral_numeric_rows | yes | 750 raw Mistral cells |
| phase_f_exists | yes |  |
| phase_f_cuda | yes | True |
| phase_f_triton | yes | True |
| phase_f_device_cuda | yes | device='cuda' |
| phase_f_int8_speedup | yes | {'mode': 'int8', 'torch_latency_ms': 0.115, 'triton_latency_ms': 0.0706, 'speedu |
| phase_f_int4_speedup | yes | {'mode': 'int4', 'torch_latency_ms': 0.3319, 'triton_latency_ms': 0.2162, 'speed |
| phase_f_block_sparse_not_triton_speedup | yes | execution_backend='torch' |
| phase_g_exists | yes |  |
| spectralquant_fallback_disclosed | yes | spectralquant_available=False — public docs must not claim real SpectralQuant |
| shard_probe_disclosed | yes | shard_real is probe-first; no real Shard backend wired |
| public_claim_safety | yes | clean |

## Warnings

- Scale run used sequential model execution (volume constraint).

## Remaining release blockers

- Full `pytest` must pass
- Novelty audit must be complete before public claims are finalized
- Token/secret scan must pass before publishing
- Final README/public posts must use claim-safe language
