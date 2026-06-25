# ExactKV Demo Cards

Generated from on-disk release and historical demo artifacts only.

## Best stable release case (noop baseline)

- **demo_id:** `release_best_stable_noop`
- **source:** `reports/scale_7b/raw.json`
- **model:** `meta-llama/Llama-3.1-8B`
- **compressor:** `noop`
- **prompt:** `p00_p0_capital_france` (capital_france)
- **first_divergence_index:** None
- **acceptance_rate:** 1.0
- **avg_accepted_span:** 4.0
- **verifier_agreement:** None
- **exactkv_failure_count:** 0
- **decoded output:** decoded text unavailable in artifact.
- **what changed:** No compression; full-KV greedy reference with 100% draft acceptance.
- **why it matters:** Establishes the exactness baseline on the 1500-cell public panel.
- **public-safe claim:** noop achieves acceptance_rate=1.0 with exactkv_failures=0 on tested cells.
- **caveat:** Panel-scoped greedy decoding; not serving throughput.

## Strong INT8 release case

- **demo_id:** `release_strong_int8`
- **source:** `reports/scale_7b/raw.json`
- **model:** `meta-llama/Llama-3.1-8B`
- **compressor:** `int8`
- **prompt:** `p00_p0_capital_france` (capital_france)
- **first_divergence_index:** None
- **acceptance_rate:** 1.0
- **avg_accepted_span:** 4.0
- **verifier_agreement:** None
- **exactkv_failure_count:** 0
- **decoded output:** decoded text unavailable in artifact.
- **what changed:** INT8 stored KV compression with verifier-backed draft acceptance.
- **why it matters:** Shows int8 can remain exactness-compatible on many prompts in the public panel.
- **public-safe claim:** int8 achieves acceptance_rate=1.0 with exactkv_failures=0 on cited Llama cells.
- **caveat:** Compression ratio is stored tensor byte ratio; not active GPU memory savings.

## INT4 drift / compression-break case

- **demo_id:** `release_int4_drift_case`
- **source:** `reports/scale_7b/raw.json`
- **model:** `meta-llama/Llama-3.1-8B`
- **compressor:** `int4_sim`
- **prompt:** `p01_p1_simple_math` (simple_math)
- **first_divergence_index:** None
- **acceptance_rate:** 0.5
- **avg_accepted_span:** 1.5
- **verifier_agreement:** None
- **exactkv_failure_count:** 0
- **decoded output:** decoded text unavailable in artifact.
- **what changed:** Lower acceptance (0.5) under int4_sim on p01_p1_simple_math.
- **why it matters:** Demonstrates when compressed KV paths diverge before full exactness recovery.
- **public-safe claim:** int4_sim shows reduced acceptance_rate=0.5 while exactkv_failures remain 0.
- **caveat:** Simulated int4 container; illustrates drift metrics, not deployment speedups.

## Shard probe-first case

- **demo_id:** `release_shard_probe_first`
- **source:** `reports/scale_7b/raw.json`
- **model:** `meta-llama/Llama-3.1-8B`
- **compressor:** `shard`
- **prompt:** `p01_p1_simple_math` (simple_math)
- **first_divergence_index:** None
- **acceptance_rate:** 0.25
- **avg_accepted_span:** None
- **verifier_agreement:** None
- **exactkv_failure_count:** 0
- **decoded output:** decoded text unavailable in artifact.
- **what changed:** Probe-first shard heuristic with acceptance_rate=0.25.
- **why it matters:** Shows bounded probe analysis for shard-style compression hypotheses.
- **public-safe claim:** Shard slot is probe-first heuristic analysis, not full Shard product integration.
- **caveat:** probe_only=True; not real Shard / ShardCache integration.

## SpectralQuant fallback/proxy case

- **demo_id:** `release_spectralquant_fallback`
- **source:** `reports/scale_7b/raw.json`
- **model:** `meta-llama/Llama-3.1-8B`
- **compressor:** `spectralquant`
- **prompt:** `p01_p1_simple_math` (simple_math)
- **first_divergence_index:** None
- **acceptance_rate:** 0.5
- **avg_accepted_span:** 1.5
- **verifier_agreement:** None
- **exactkv_failure_count:** 0
- **decoded output:** decoded text unavailable in artifact.
- **what changed:** Fallback/proxy SpectralQuant slot (backend_tier=MOCK).
- **why it matters:** Documents adapter honesty when the real dependency is unavailable.
- **public-safe claim:** SpectralQuant results use fallback/proxy in the current environment.
- **caveat:** spectralquant_available=False; do not claim real SpectralQuant integration.

## Historical structured-output JSON drift demo

- **demo_id:** `historical_structured_output_json`
- **source:** `reports/phaseA_benchmark.json`
- **model:** `Qwen/Qwen2.5-0.5B`
- **compressor:** `int4_sim`
- **prompt:** `p2_json_tool` (structured_output_drift)
- **first_divergence_index:** 1
- **acceptance_rate:** 0.5
- **avg_accepted_span:** None
- **verifier_agreement:** 0.5
- **exactkv_failure_count:** 0
- **decoded output:** full_reference:  "New York",; compressed_draft:  "state": {"
- **what changed:** Pharmacy-style intent-flip prompt is not in the Phase A panel; p2_json_tool is the closest in-panel structured-output drift case.
- **why it matters:** Illustrates token-level drift and verifier-mediated exactness on a historical demo panel.
- **public-safe claim:** Illustrative exactness evidence from pre-release demo artifacts; not the 1500-cell public headline.
- **caveat:** Historical/internal panel; not throughput or serving evidence.

## Historical V-series / crash-test style int4 drift

- **demo_id:** `historical_terminal_crash_style_int4`
- **source:** `reports/phaseA_benchmark.json`
- **model:** `Qwen/Qwen2.5-0.5B`
- **compressor:** `int4_sim`
- **prompt:** `p0_capital_france` (worst_case_compression)
- **first_divergence_index:** 1
- **acceptance_rate:** 0.3333333333333333
- **avg_accepted_span:** None
- **verifier_agreement:** 0.3333333333333333
- **exactkv_failure_count:** 0
- **decoded output:** full_reference:  Paris. It is; compressed_draft:  Paris, the capital
- **what changed:** Lowest acceptance int4_sim cell in Phase A benchmark.
- **why it matters:** Illustrates token-level drift and verifier-mediated exactness on a historical demo panel.
- **public-safe claim:** Illustrative exactness evidence from pre-release demo artifacts; not the 1500-cell public headline.
- **caveat:** Historical/internal panel; not throughput or serving evidence.
