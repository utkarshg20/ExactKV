# ExactKV — LinkedIn Launch Post (Phase K Final)

**When does compressed KV start lying?**

I'm sharing ExactKV — a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression. It measures token-level drift: when a lossy compressed-KV path first diverges from full-precision greedy decoding, how much of each draft prefix a full-KV verifier accepts, and whether verifier-backed execution preserves exactness on tested panels.

**The project did not begin at "Phase A."** ExactKV grew through a long verifier-first research arc — the **V1–V21 version arc** (V1–V13 scope statements; V14–V21 safety/runtime ladder via Phase 14–21 docs), 120+ experiment documents, trace correctness suites, structured-output demos, a safety ladder with L3/L4 no-commit boundaries, shadow observer runtime probes, and explicit no-go investigations for vLLM, LMCache, and unqualified memory/timing claims. Release archaeology catalogued 1,176 historical artifacts. The formal A–J release pipeline then scaled, packaged, validated, and published that earlier system.

**Public release evidence (authoritative):**

- **1,500-cell (1500-cell)** real-GPU benchmark (`reports/scale_7b/raw.json`)
- Models: **Llama-3.1-8B** and **Mistral-7B-Instruct-v0.3**
- **`exactkv_failures = 0`** on the public panel
- Compressors: noop, int8, int4_sim, spectralquant (fallback/proxy), shard (probe-first)
- Public leaderboard with numeric Llama and Mistral rows

**Positioning:** Research-grade evaluation infrastructure — **not a production serving system**. VeriCache is the closest conceptual prior art for compressed-KV draft plus full-KV verification; **ExactKV does not reproduce VeriCache** serving throughput.

**Qualified technical notes:**

- Phase F INT8 (~1.63×) and INT4 (~1.54×) results are **kernel microbenchmarks** on a fixed KV shape — not end-to-end inference speedups
- Compression ratios reported are **stored tensor byte ratios**, not active GPU memory savings
- SpectralQuant runs as **fallback/proxy** in the current environment; Shard is **probe-first** heuristic analysis
- Scale benchmark used sequential model execution due to infrastructure volume constraints

**Reproduce:** `python3 scripts/exactkv_repro.py --release-check`

**Read more:**

- Technical report: `docs/EXACTKV_TECHNICAL_REPORT.md`
- Project lineage: `docs/PROJECT_LINEAGE.md`
- Claim boundaries: `docs/CLAIM_BOUNDARIES.md`

If you work on inference systems, KV-cache compression, or evaluation methodology — I'd welcome feedback on the leaderboard framing and claim boundaries.

#LLM #Inference #Systems #MachineLearning #Research
