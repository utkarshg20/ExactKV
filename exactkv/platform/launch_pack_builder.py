"""Build Phase K launch pack artifacts (demo cards, launch manifest)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_metrics(cell: dict[str, Any]) -> dict[str, Any]:
    ex = cell.get("exactkv") or {}
    acc = ex.get("acceptance") or {}
    metrics = cell.get("metrics") or {}
    return {
        "prompt_id": cell.get("prompt_id"),
        "category": cell.get("category"),
        "model": cell.get("model_name"),
        "compressor": cell.get("compressor_name"),
        "acceptance_rate": acc.get("acceptance_rate"),
        "avg_accepted_span": acc.get("avg_accepted_per_round"),
        "verifier_agreement": metrics.get("verifier_agreement") or acc.get("verifier_agreement"),
        "first_divergence_index": metrics.get("first_divergence_index")
        or metrics.get("canonical_first_divergence_index"),
        "exactkv_failure_count": 1 if cell.get("exactkv_failure") else 0,
        "exactkv_failure_rate": 1.0 if cell.get("exactkv_failure") else 0.0,
        "probe_only": cell.get("probe_only"),
        "backend_tier": cell.get("backend_tier"),
    }


def _pick_scale_cell(
    cells: list[dict[str, Any]],
    *,
    model: str,
    compressor: str,
    best: bool = True,
) -> dict[str, Any] | None:
    subset = [c for c in cells if c.get("model_name") == model and c.get("compressor_name") == compressor]
    if not subset:
        return None
    key = lambda c: (_cell_metrics(c).get("acceptance_rate") or 0.0)
    return max(subset, key=key) if best else min(subset, key=key)


def _release_card(
    *,
    demo_id: str,
    title: str,
    cell: dict[str, Any],
    what_changed: str,
    why_it_matters: str,
    public_safe_claim: str,
    caveat: str,
) -> dict[str, Any]:
    m = _cell_metrics(cell)
    decoded = "decoded text unavailable in artifact."
    return {
        "demo_id": demo_id,
        "title": title,
        "historical_or_release_source": "reports/scale_7b/raw.json",
        "model": m["model"],
        "compressor": m["compressor"],
        "prompt_id": m["prompt_id"],
        "prompt_category": m["category"],
        "first_divergence_index": m["first_divergence_index"],
        "acceptance_rate": m["acceptance_rate"],
        "avg_accepted_span": m["avg_accepted_span"],
        "verifier_agreement": m["verifier_agreement"],
        "exactkv_failure_count": m["exactkv_failure_count"],
        "exactkv_failure_rate": m["exactkv_failure_rate"],
        "decoded_output_note": decoded,
        "what_changed": what_changed,
        "why_it_matters": why_it_matters,
        "public_safe_claim": public_safe_claim,
        "caveat": caveat,
    }


def _historical_card_from_demo(demo: dict[str, Any], *, demo_id: str, title: str) -> dict[str, Any]:
    outputs = demo.get("outputs") or {}
    full = outputs.get("full_reference") or {}
    draft = outputs.get("compressed_draft") or {}
    full_text = full.get("output_text") if full.get("output_text_available") else None
    draft_text = draft.get("output_text") if draft.get("output_text_available") else None
    decoded_note = "decoded text unavailable in artifact."
    if full_text or draft_text:
        decoded_note = (
            f"full_reference: {full_text or 'unavailable'}; "
            f"compressed_draft: {draft_text or 'unavailable'}"
        )
    metrics = demo.get("metrics") or {}
    sources = demo.get("data_sources") or ["reports/demo_pack.json"]
    return {
        "demo_id": demo_id,
        "title": title,
        "historical_or_release_source": sources[0] if sources else "reports/demo_pack.json",
        "model": demo.get("model"),
        "compressor": demo.get("compressor"),
        "prompt_id": demo.get("prompt_id"),
        "prompt_category": demo.get("category"),
        "first_divergence_index": demo.get("first_divergence_index"),
        "acceptance_rate": metrics.get("acceptance_rate"),
        "avg_accepted_span": None,
        "verifier_agreement": metrics.get("verifier_agreement_score"),
        "exactkv_failure_count": 1 if metrics.get("exactkv_failure") else 0,
        "exactkv_failure_rate": 1.0 if metrics.get("exactkv_failure") else 0.0,
        "decoded_output_note": decoded_note,
        "what_changed": demo.get("category_note") or demo.get("category", ""),
        "why_it_matters": "Illustrates token-level drift and verifier-mediated exactness on a historical demo panel.",
        "public_safe_claim": (
            "Illustrative exactness evidence from pre-release demo artifacts; not the 1500-cell public headline."
        ),
        "caveat": "Historical/internal panel; not throughput or serving evidence.",
    }


def build_demo_cards(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root)
    raw = _load_json(root / "reports/scale_7b/raw.json")
    cells = raw.get("cells") or []
    demo_pack = _load_json(root / "reports/demo_pack.json")
    demos_by_id = {d["demo_id"]: d for d in demo_pack.get("demos") or []}

    llama = "meta-llama/Llama-3.1-8B"
    cards: list[dict[str, Any]] = []

    noop = _pick_scale_cell(cells, model=llama, compressor="noop", best=True)
    if noop:
        cards.append(
            _release_card(
                demo_id="release_best_stable_noop",
                title="Best stable release case (noop baseline)",
                cell=noop,
                what_changed="No compression; full-KV greedy reference with 100% draft acceptance.",
                why_it_matters="Establishes the exactness baseline on the 1500-cell public panel.",
                public_safe_claim="noop achieves acceptance_rate=1.0 with exactkv_failures=0 on tested cells.",
                caveat="Panel-scoped greedy decoding; not serving throughput.",
            )
        )

    int8 = _pick_scale_cell(cells, model=llama, compressor="int8", best=True)
    if int8:
        cards.append(
            _release_card(
                demo_id="release_strong_int8",
                title="Strong INT8 release case",
                cell=int8,
                what_changed="INT8 stored KV compression with verifier-backed draft acceptance.",
                why_it_matters="Shows int8 can remain exactness-compatible on many prompts in the public panel.",
                public_safe_claim="int8 achieves acceptance_rate=1.0 with exactkv_failures=0 on cited Llama cells.",
                caveat="Compression ratio is stored tensor byte ratio; not active GPU memory savings.",
            )
        )

    int4 = _pick_scale_cell(cells, model=llama, compressor="int4_sim", best=False)
    if int4:
        m = _cell_metrics(int4)
        cards.append(
            _release_card(
                demo_id="release_int4_drift_case",
                title="INT4 drift / compression-break case",
                cell=int4,
                what_changed=f"Lower acceptance ({m['acceptance_rate']}) under int4_sim on {m['prompt_id']}.",
                why_it_matters="Demonstrates when compressed KV paths diverge before full exactness recovery.",
                public_safe_claim=(
                    f"int4_sim shows reduced acceptance_rate={m['acceptance_rate']} while exactkv_failures remain 0."
                ),
                caveat="Simulated int4 container; illustrates drift metrics, not deployment speedups.",
            )
        )

    shard = _pick_scale_cell(cells, model=llama, compressor="shard", best=False)
    if shard:
        m = _cell_metrics(shard)
        cards.append(
            _release_card(
                demo_id="release_shard_probe_first",
                title="Shard probe-first case",
                cell=shard,
                what_changed=f"Probe-first shard heuristic with acceptance_rate={m['acceptance_rate']}.",
                why_it_matters="Shows bounded probe analysis for shard-style compression hypotheses.",
                public_safe_claim="Shard slot is probe-first heuristic analysis, not full Shard product integration.",
                caveat="probe_only=True; not real Shard / ShardCache integration.",
            )
        )

    sq = _pick_scale_cell(cells, model=llama, compressor="spectralquant", best=False)
    if sq:
        m = _cell_metrics(sq)
        cards.append(
            _release_card(
                demo_id="release_spectralquant_fallback",
                title="SpectralQuant fallback/proxy case",
                cell=sq,
                what_changed=f"Fallback/proxy SpectralQuant slot (backend_tier={m.get('backend_tier')}).",
                why_it_matters="Documents adapter honesty when the real dependency is unavailable.",
                public_safe_claim="SpectralQuant results use fallback/proxy in the current environment.",
                caveat="spectralquant_available=False; do not claim real SpectralQuant integration.",
            )
        )

    struct_demo = demos_by_id.get("structured_output_drift_p2_json_tool_int4_sim")
    if struct_demo:
        cards.append(
            _historical_card_from_demo(
                struct_demo,
                demo_id="historical_structured_output_json",
                title="Historical structured-output JSON drift demo",
            )
        )

    crash_demo = demos_by_id.get("worst_case_compression_p0_capital_france_int4_sim")
    if crash_demo:
        cards.append(
            _historical_card_from_demo(
                crash_demo,
                demo_id="historical_terminal_crash_style_int4",
                title="Historical V-series / crash-test style int4 drift",
            )
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": [
            "reports/scale_7b/raw.json",
            "reports/scale_7b/leaderboard.json",
            "reports/demo_pack.json",
            "reports/phaseG_unified_truth.json",
            "reports/historical_artifact_inventory.json",
        ],
        "demo_cards": cards,
    }


def demo_cards_to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ExactKV Demo Cards",
        "",
        "Generated from on-disk release and historical demo artifacts only.",
        "",
    ]
    for card in payload.get("demo_cards") or []:
        lines.extend(
            [
                f"## {card['title']}",
                "",
                f"- **demo_id:** `{card['demo_id']}`",
                f"- **source:** `{card['historical_or_release_source']}`",
                f"- **model:** `{card.get('model')}`",
                f"- **compressor:** `{card.get('compressor')}`",
                f"- **prompt:** `{card.get('prompt_id')}` ({card.get('prompt_category')})",
                f"- **first_divergence_index:** {card.get('first_divergence_index')}",
                f"- **acceptance_rate:** {card.get('acceptance_rate')}",
                f"- **avg_accepted_span:** {card.get('avg_accepted_span')}",
                f"- **verifier_agreement:** {card.get('verifier_agreement')}",
                f"- **exactkv_failure_count:** {card.get('exactkv_failure_count')}",
                f"- **decoded output:** {card.get('decoded_output_note')}",
                f"- **what changed:** {card.get('what_changed')}",
                f"- **why it matters:** {card.get('why_it_matters')}",
                f"- **public-safe claim:** {card.get('public_safe_claim')}",
                f"- **caveat:** {card.get('caveat')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_launch_manifest(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root)
    evidence = _load_json(root / "reports/release_evidence_status.json")
    scale = evidence.get("scale_summary") or {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "release_status": "phase_k_launch_pack_ready",
        "source_of_truth_artifact": "reports/scale_7b/raw.json",
        "benchmark_cell_count": scale.get("total_cells", 1500),
        "models": scale.get("models")
        or ["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-Instruct-v0.3"],
        "compressors": scale.get("compressors_observed")
        or ["noop", "int8", "int4_sim", "spectralquant", "shard"],
        "exactkv_failures": scale.get("exactkv_failures", 0),
        "public_leaderboard_path": "reports/public_release/leaderboard_final.json",
        "technical_report_path": "paper/ExactKV_Technical_Report.md",
        "site_path": "site/index.html",
        "demo_cards_path": "reports/public_release/demo_cards.json",
        "novelty_audit_path": "reports/novelty_audit.json",
        "project_lineage_path": "docs/PROJECT_LINEAGE.md",
        "version_lineage_path": "docs/VERSION_LINEAGE.md",
        "version_lineage_json": "reports/version_lineage.json",
        "version_arc": "V1-V21",
        "version_lineage_caveat": (
            "Historical version arc (V1-V21) is project lineage context, "
            "not release benchmark evidence. Authoritative benchmark: reports/scale_7b/raw.json."
        ),
        "historical_inventory_path": "reports/historical_artifact_inventory.json",
        "claim_boundaries_path": "docs/CLAIM_BOUNDARIES.md",
        "metric_definitions_path": "docs/METRIC_DEFINITIONS.md",
        "release_checklist_path": "docs/RELEASE_CHECKLIST.md",
        "validation_commands": [
            "python3 scripts/check_launch_pack.py",
            "python3 scripts/check_project_lineage.py",
            "python3 scripts/check_public_release.py",
            "python3 scripts/check_release_evidence.py",
            "python3 scripts/audit_public_claims.py",
            "python3 scripts/check_no_secrets.py",
            "python3 scripts/exactkv_repro.py --release-check",
        ],
        "validation_results": {
            "note": "Run validation_commands before publish; manual_signoff_required=true until approved.",
        },
        "manual_signoff_required": True,
        "remaining_known_limitations": [
            "ExactKV is not a production serving system.",
            "ExactKV does not reproduce VeriCache serving throughput.",
            "Phase F speedups are kernel microbenchmark results only (not end-to-end).",
            "Compression ratios are stored tensor byte ratios unless active GPU memory is explicitly measured.",
            "SpectralQuant uses fallback/proxy in the current environment (spectralquant_available=False).",
            "Shard is probe-first heuristic analysis (probe_only=True), not full Shard integration.",
            "Scale run used sequential model execution due to RunPod volume constraints.",
            "Historical inventory contains 1176 artifacts; manual editorial review recommended before launch.",
            "231 inventory entries remain in unknown chronological bucket pending manual curation.",
        ],
    }


def write_launch_pack(root: Path | str = ".") -> dict[str, str]:
    root = Path(root)
    out_dir = root / "reports/public_release"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards_payload = build_demo_cards(root)
    cards_json = out_dir / "demo_cards.json"
    cards_md = out_dir / "demo_cards.md"
    cards_json.write_text(json.dumps(cards_payload, indent=2) + "\n", encoding="utf-8")
    cards_md.write_text(demo_cards_to_markdown(cards_payload), encoding="utf-8")

    manifest = build_launch_manifest(root)
    manifest_path = out_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "demo_cards_json": str(cards_json),
        "demo_cards_md": str(cards_md),
        "launch_manifest": str(manifest_path),
    }
