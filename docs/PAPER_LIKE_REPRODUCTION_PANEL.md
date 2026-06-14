# Paper-Like Reproduction Panel (Phase 11J)

**Status:** Panel contract only — **not a reproduction result.**

> This is a **panel contract**, not a reproduction result.  
> **ExactKV has not reproduced VeriCache throughput results.**  
> **ExactKV has not reproduced VeriCache memory benefits.**  
> **ExactKV has not reproduced VeriCache production serving.**  
> **External paper numbers are not ExactKV results.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.

Companion: [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md) · [`THROUGHPUT_BENCHMARK_HARNESS.md`](THROUGHPUT_BENCHMARK_HARNESS.md) · [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md) · `exactkv/benchmarks/paper_panel_contract.py`

---

## 1. Why a paper-like panel is required for VeriCache parity

VeriCache reports results on a **fixed model × compressor × benchmark × metrics** panel including throughput and memory. ExactKV V10–V13 built **correctness-first** panels with different goals. Stage 9 defines the **contract** for what would need to match before any paper-like VeriCache reproduction claim — without running that panel today.

---

## 2. What is included in the panel contract

| Dimension | Types |
|---|---|
| **Models** | `PaperPanelModelSpec` — small HF sanity + larger Llama/Qwen + paper-scale placeholder |
| **Compressors** | Built-ins, simulated rows, paper set placeholder, restricted Shard/SpectralQuant |
| **Workloads** | V10 suites (implemented), paper long-context / remote-prefix / quality placeholders |
| **Metrics** | Exactness, acceptance, drift, throughput, memory accounting |
| **Hardware** | GPU/dtype metadata requirements |
| **Runtime** | HF harness vs vLLM/LMCache/serving (not implemented) |
| **Gates** | Exactness, throughput, memory, runtime |

Factory: `build_default_paper_like_panel()` → `CONTRACT_ONLY`, `claim_eligible=False`.

---

## 3. What is not implemented yet

| Area | Status |
|---|---|
| Paper model/context-length matrix | **Not implemented** |
| Paper compressor ecosystem at paper settings | **Not implemented** |
| Official paper benchmark workloads | **Not implemented** |
| Remote-prefix paper workload runtime | **Not implemented** (11H loopback only) |
| Throughput claim-ready panel | **Not implemented** (Exp 030 diagnostic only) |
| Active memory benefit panel | **Not implemented** (Exp 031 diagnostic only) |
| vLLM / LMCache / serving runtime | **Not implemented** |

---

## 4. How ExactKV results differ from VeriCache paper results today

| Topic | VeriCache paper | ExactKV today |
|---|---|---|
| Primary question | Throughput + memory + exactness on paper panel | **Exactness** on custom crash-test panels |
| Throughput | Reported benefit path | Exp 030: ExactKV **slower** than full greedy (diagnostic) |
| Memory | HBM savings narrative | Exp 031: no active VRAM savings at tested scale |
| Workloads | Paper benchmarks | V10 custom suites; LongBench-**style** demo only |
| Runtime | vLLM + LMCache | HF harness only |

See [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md) — complementary questions, not competitive replacement.

---

## 5. Why paper numbers cannot be reused as ExactKV results

Yao et al. (*VeriCache*, arXiv:2605.17613) throughput and memory figures describe **their system and panel**. ExactKV `paper_numbers_as_exactkv_results` must remain **False**. Citing external paper numbers as ExactKV results is **forbidden**.

---

## 6. Restricted Shard / SpectralQuant rows

| Row | Why not paper-equivalent |
|---|---|
| **Shard external-drafter** | Mode B restricted probe (Exp 039–041); external drafter; not integrated default compressor |
| **SpectralQuant experimental** | Materializing factory-only adapter (Exp 044–045); no active memory savings; small restricted panel |

These may appear on ExactKV leaderboard as **RESTRICTED BACKEND** — not as VeriCache paper compressor reproduction.

---

## 7. Claim gates

| Gate | Requirement for `CLAIM_ELIGIBLE` |
|---|---|
| **Exactness** | `exactness_gate_passed` on locked panel (`exactkv_failures == 0`) |
| **Throughput** | Phase 11I methodology + `throughput_gate_passed` (not diagnostic-only) |
| **Memory** | Active measurement + `memory_gate_passed` (not diagnostic-only) |
| **Runtime/backend** | `runtime_gate_passed` — paper backend requirements met |
| **Workloads** | Required workloads `implemented=True`; non-paper workloads carry caveats |
| **Hardware** | `hardware_requirements.implemented=True` |

Default panel: all gates **not passed**; status **`CONTRACT_ONLY`**.

---

## 8. How Stage 10 parity RC would use this

Stage 10 release candidate checklist would require:

1. Panel status advances to `RUN_COMPLETE_UNVERIFIED` only after a locked panel run
2. Independent review verifies gates and claim boundaries
3. `CLAIM_ELIGIBLE` only if all dimensions implemented and gates pass
4. Still **not** automatic production/speed claim — explicit wording in parity audit

---

## 9. JSON schema (panel header)

```json
{
  "status": "CONTRACT_ONLY",
  "exactness_gate_required": true,
  "exactness_gate_passed": false,
  "throughput_gate_passed": false,
  "memory_gate_passed": false,
  "paper_numbers_as_exactkv_results": false,
  "claim_eligible": false,
  "claim_note": "..."
}
```

---

## 10. Claims boundary

| Allowed | Forbidden |
|---|---|
| Panel contract metadata exists | VeriCache reproduction complete |
| Cited ExactKV panel exactness with panel name | VeriCache throughput reproduced |
| Restricted rows with caveats documented | VeriCache memory benefits reproduced |
| Planning dimensions listed | External paper numbers as ExactKV results |
| Gate requirements documented | Production serving readiness |
| | Speed / latency / throughput improvement |
