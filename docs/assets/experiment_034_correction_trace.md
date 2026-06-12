# Experiment 034 — Correction Trace

**Prompt:** `tj_002` (tool_json)
**Compressor:** `int4_sim` | **draft_len:** 4
**Verification:** sequential

> **Everyone is racing to shrink KV caches. ExactKV tells you when they start lying.**

### Output comparison

| Mode | Output |
| --- | --- |
| Full KV | ` "metric"}} To complete this tool call JSON, you would need to define a function that takes in the necessary parameters and returns the weather data for the specified city` |
| Lossy compressed KV draft | ` "}}}\n\n{"name": "get_weather", "arguments": {"city": "Paris", "units": "metric"}}}\n\n{"name": "get_weather` |
| ExactKV | ` "metric"}} To complete this tool call JSON, you would need to define a function that takes in the necessary parameters and returns the weather data for the specified city` |

### Correction at a glance

| Field | Value |
| --- | --- |
| First divergence | token index 1 |
| Draft token rejected | '}}' (id 3417) |
| Full-KV correction | 'metric' (id 15903) |
| ExactKV failures | 0 |
| Final output match | true |

## Highlight round

| Field | Value |
|---|---|
| Round | 0 |
| Draft tokens | `[330, 3417, 630, 4913]` |
| Verifier tokens | `[330, 15903]` |
| Accepted prefix | `[330]` |
| First rejected | `3417` → '}}' |
| Correction | `15903` → 'metric' |

## Outputs

### Full greedy
```
 "metric"}} To complete this tool call JSON, you would need to define a function that takes in the necessary parameters and returns the weather data for the specified city
```

### Lossy greedy (diverges)
```
 "}}}

{"name": "get_weather", "arguments": {"city": "Paris", "units": "metric"}}}

{"name": "get_weather
```

### ExactKV (exact)
```
 "metric"}} To complete this tool call JSON, you would need to define a function that takes in the necessary parameters and returns the weather data for the specified city
```

## Explanation

Everyone is racing to shrink KV caches. ExactKV tells you when they start lying. With compressor `int4_sim`, the lossy compressed-KV draft proposed '}}' (id 3417) at round 0. The full-KV verifier predicted 'metric' (id 15903) instead. ExactKV rejected the draft token and committed the verifier correction; the wrong draft was never written to the authoritative KV cache. Final ExactKV output matches full greedy (True). Without verification, lossy greedy output could carry malformed structured tokens into the continuation.
