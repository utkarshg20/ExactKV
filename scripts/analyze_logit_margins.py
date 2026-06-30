#!/usr/bin/env python3
"""Divergence Autopsy: Logit Margins at First Flip (v2.7 forensic analysis).

Reads `top_k_logits_at_divergence` data stored by --store-top-k-logits in ExactKV
panels and computes:

  - Full-KV vs lossy top-1 logit margin at the first divergence token
  - KL divergence between full and lossy top-k distributions
  - Near-tie flip rate (margin < threshold)
  - Semantic shift classification (benign / structural / catastrophic)
  - Per-compressor, per-subset summary table

Usage:
  python3 scripts/analyze_logit_margins.py \
      --input reports/external_panels/hf_longbench_v26_merged_raw.json \
      --output reports/external_panels/logit_autopsy_v26.json \
      --print-top 10

Output JSON keys (per cell with logit data):
  full_top1_logit, full_top1_margin, lossy_top1_logit, lossy_top1_margin,
  full_top1_token, lossy_top1_token, kl_full_to_lossy, top5_overlap,
  shift_class (near_tie / moderate / large), logit_delta
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


# ── Wilson CI helper ────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── Logit extraction ─────────────────────────────────────────────────────────
def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) in nats, clipped to avoid log(0)."""
    eps = 1e-10
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def _top5_overlap(full_tokens: list[str], lossy_tokens: list[str]) -> float:
    """Fraction of top-5 full-KV tokens that appear in top-5 lossy tokens."""
    if not full_tokens or not lossy_tokens:
        return 0.0
    return len(set(full_tokens) & set(lossy_tokens)) / max(len(full_tokens), 1)


NEAR_TIE_THRESHOLD = 1.0   # logit units; margin < this = near-tie
LARGE_SHIFT_THRESHOLD = 5.0  # logit units; margin > this = large shift


def classify_shift(full_margin: float, logit_delta: float) -> str:
    """Classify the shift as near_tie / moderate / large."""
    if full_margin < NEAR_TIE_THRESHOLD:
        return "near_tie"
    if abs(logit_delta) > LARGE_SHIFT_THRESHOLD:
        return "large"
    return "moderate"


def extract_logit_metrics(cell: dict[str, Any]) -> dict[str, Any] | None:
    """Extract logit-level metrics from a divergent cell.

    The cell must contain `top_k_logits_at_divergence` with keys:
      full_top_tokens, full_top_logits, lossy_top_tokens, lossy_top_logits
    """
    logit_data = cell.get("top_k_logits_at_divergence")
    if not logit_data:
        return None

    full_tokens  = logit_data.get("full_top_tokens",  [])
    full_logits  = logit_data.get("full_top_logits",  [])
    lossy_tokens = logit_data.get("lossy_top_tokens", [])
    lossy_logits = logit_data.get("lossy_top_logits", [])

    if not full_logits or not lossy_logits:
        return None

    full_probs  = _softmax(full_logits)
    lossy_probs = _softmax(lossy_logits)

    full_top1_logit  = full_logits[0]
    lossy_top1_logit = lossy_logits[0]
    full_margin      = full_logits[0] - full_logits[1]  if len(full_logits)  > 1 else 0.0
    lossy_margin     = lossy_logits[0] - lossy_logits[1] if len(lossy_logits) > 1 else 0.0
    logit_delta      = full_top1_logit - lossy_top1_logit

    # Align distributions to same top-k token set for KL computation
    # Use full distribution as p, lossy distribution as q (aligned by position)
    kl = _kl_divergence(full_probs, lossy_probs)
    overlap = _top5_overlap(full_tokens, lossy_tokens)

    return {
        "full_top1_token":   full_tokens[0]  if full_tokens  else "",
        "lossy_top1_token":  lossy_tokens[0] if lossy_tokens else "",
        "full_top1_logit":   round(full_top1_logit,  4),
        "lossy_top1_logit":  round(lossy_top1_logit, 4),
        "full_top1_margin":  round(full_margin,  4),
        "lossy_top1_margin": round(lossy_margin, 4),
        "logit_delta":       round(logit_delta,  4),
        "kl_full_to_lossy":  round(kl,      4),
        "top5_overlap":      round(overlap,  4),
        "shift_class":       classify_shift(full_margin, logit_delta),
        "full_top5":  list(zip(full_tokens,  [round(x, 3) for x in full_logits]))[:5],
        "lossy_top5": list(zip(lossy_tokens, [round(x, 3) for x in lossy_logits]))[:5],
    }


# ── Aggregate summaries ──────────────────────────────────────────────────────
def aggregate_metrics(annotated: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary stats over all cells that have logit data."""
    with_logits = [c for c in annotated if c.get("_logit")]
    n = len(with_logits)
    if n == 0:
        return {"n_with_logits": 0}

    near_tie  = sum(1 for c in with_logits if c["_logit"]["shift_class"] == "near_tie")
    moderate  = sum(1 for c in with_logits if c["_logit"]["shift_class"] == "moderate")
    large     = sum(1 for c in with_logits if c["_logit"]["shift_class"] == "large")

    margins   = [c["_logit"]["full_top1_margin"]  for c in with_logits]
    deltas    = [c["_logit"]["logit_delta"]        for c in with_logits]
    kls       = [c["_logit"]["kl_full_to_lossy"]  for c in with_logits]
    overlaps  = [c["_logit"]["top5_overlap"]       for c in with_logits]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    nt_lo, nt_hi = wilson_ci(near_tie, n)

    return {
        "n_with_logits":          n,
        "near_tie_count":         near_tie,
        "near_tie_rate":          round(near_tie / n, 4),
        "near_tie_ci95":          [round(nt_lo, 4), round(nt_hi, 4)],
        "moderate_count":         moderate,
        "large_shift_count":      large,
        "mean_full_margin":       round(mean(margins), 4),
        "mean_logit_delta":       round(mean(deltas),  4),
        "mean_kl":                round(mean(kls),     4),
        "mean_top5_overlap":      round(mean(overlaps), 4),
        "min_full_margin":        round(min(margins),  4),
        "max_logit_delta":        round(max(abs(d) for d in deltas), 4),
    }


def breakdown_by_compressor(annotated: list[dict]) -> dict:
    by_comp: dict[str, list[dict]] = {}
    for c in annotated:
        if c.get("_logit"):
            comp = c.get("compressor_name", "unknown")
            by_comp.setdefault(comp, []).append(c)
    return {comp: aggregate_metrics(cells) for comp, cells in sorted(by_comp.items())}


def breakdown_by_subset(annotated: list[dict]) -> dict:
    by_sub: dict[str, list[dict]] = {}
    for c in annotated:
        if c.get("_logit"):
            sub = c.get("category", "unknown")
            by_sub.setdefault(sub, []).append(c)
    return {sub: aggregate_metrics(cells) for sub, cells in sorted(by_sub.items())}


# ── Top-N interesting cells ──────────────────────────────────────────────────
def top_interesting_cells(annotated: list[dict], n: int = 10) -> list[dict]:
    """Return the N most forensically interesting divergent cells.

    Interesting = small full-KV margin (near-tie flip) or large KL shift.
    """
    scored = []
    for c in annotated:
        lm = c.get("_logit")
        if not lm:
            continue
        # Score: low margin (near-tie) OR high KL
        interestingness = lm["kl_full_to_lossy"] - lm["full_top1_margin"] * 0.5
        scored.append((interestingness, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:n]]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Logit margin autopsy for ExactKV divergent cells")
    parser.add_argument("--input",     required=True, type=Path, help="Raw JSON panel (merged or single-model)")
    parser.add_argument("--output",    default="",    help="Output JSON path (default: <input>_logit_autopsy.json)")
    parser.add_argument("--print-top", type=int, default=10, help="Print top-N interesting cells to stdout")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    if not cells:
        print(f"[WARN] No 'cells' key in {args.input}", file=sys.stderr)
        return 1

    # Annotate each divergent cell with logit metrics
    annotated = []
    n_with_logits = 0
    for cell in cells:
        c = dict(cell)
        if cell.get("diverged") or cell.get("first_divergence_index") is not None:
            metrics = extract_logit_metrics(cell)
            if metrics:
                c["_logit"] = metrics
                n_with_logits += 1
        annotated.append(c)

    total_divergent = sum(1 for c in annotated if c.get("diverged"))
    print(f"Cells: {len(annotated)}  |  Divergent: {total_divergent}  |  With logit data: {n_with_logits}")

    if n_with_logits == 0:
        print("[WARN] No top-k logit data found. Was --store-top-k-logits used in the panel run?")

    agg      = aggregate_metrics(annotated)
    by_comp  = breakdown_by_compressor(annotated)
    by_sub   = breakdown_by_subset(annotated)
    top_n    = top_interesting_cells(annotated, args.print_top)

    result = {
        "source":      str(args.input),
        "total_cells": len(annotated),
        "total_divergent": total_divergent,
        "aggregate":   agg,
        "by_compressor": by_comp,
        "by_subset":   by_sub,
        "top_interesting_cells": [
            {k: v for k, v in c.items() if k not in ("prompt", "full_output", "lossy_output")}
            for c in top_n
        ],
    }

    out_path = Path(args.output) if args.output else args.input.with_suffix("").with_name(
        args.input.stem + "_logit_autopsy.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Print summary
    print()
    print("=== Aggregate logit-margin summary ===")
    if agg.get("n_with_logits", 0) > 0:
        print(f"  Near-tie flips (<{NEAR_TIE_THRESHOLD} logit margin): "
              f"{agg['near_tie_count']}/{agg['n_with_logits']} "
              f"({agg['near_tie_rate']:.1%})  "
              f"CI95=[{agg['near_tie_ci95'][0]:.1%}, {agg['near_tie_ci95'][1]:.1%}]")
        print(f"  Mean full-KV top-1 margin:  {agg['mean_full_margin']:.3f} logit units")
        print(f"  Mean KL(full||lossy):        {agg['mean_kl']:.4f} nats")
        print(f"  Mean top-5 token overlap:    {agg['mean_top5_overlap']:.1%}")
        print()

        print("=== By compressor ===")
        for comp, stats in by_comp.items():
            if stats.get("n_with_logits", 0) == 0:
                continue
            print(f"  {comp:<16}  near_tie={stats['near_tie_rate']:.1%}  "
                  f"mean_margin={stats['mean_full_margin']:.2f}  "
                  f"mean_kl={stats['mean_kl']:.3f}")

        if args.print_top > 0 and top_n:
            print()
            print(f"=== Top-{args.print_top} most interesting divergent cells ===")
            for i, c in enumerate(top_n, 1):
                lm = c.get("_logit", {})
                prompt_id = c.get("prompt_id", "?")
                comp      = c.get("compressor_name", "?")
                subset    = c.get("category", "?")
                model     = c.get("model_id", "?").split("/")[-1]
                print(f"  {i:2d}. [{lm.get('shift_class','?')}]  "
                      f"{model}/{comp}/{subset}  "
                      f"full_tok={lm.get('full_top1_token','?')!r}  "
                      f"lossy_tok={lm.get('lossy_top1_token','?')!r}  "
                      f"margin={lm.get('full_top1_margin',0):.2f}  "
                      f"kl={lm.get('kl_full_to_lossy',0):.3f}")
    else:
        print("  No logit data to summarize.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
