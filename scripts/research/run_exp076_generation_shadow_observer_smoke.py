#!/usr/bin/env python3
"""Experiment 076: external L1 generation-shadow observer smoke (Phase 16K)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.generation_shadow_observer import (  # noqa: E402
    DEFAULT_EXP076_REPORT,
    DEFAULT_MODEL_ID,
    GenerationShadowObserverConfig,
    build_exp076_report,
    run_generation_shadow_observer,
    validate_exp076_report,
)
from exactkv.attention.generation_shadow_review import PROPOSED_SHADOW_CLI_FLAG  # noqa: E402
from exactkv.attention.hf_single_layer_probe import long_context_prompts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 076 generation-shadow observer smoke")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP076_REPORT)
    parser.add_argument(PROPOSED_SHADOW_CLI_FLAG, action="store_true", dest="shadow_observer")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument(
        "--shadow-mode",
        choices=["prompt_prefix_only", "prompt_plus_generated_tokens"],
        default="prompt_prefix_only",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    parser.add_argument("--no-allow-shadow-fail", action="store_false", dest="allow_shadow_fail")
    args = parser.parse_args()

    if not args.shadow_observer:
        report = build_exp076_report(
            run_generation_shadow_observer(
                prompts=[],
                config=GenerationShadowObserverConfig(shadow_observer_enabled=False),
            )
        )
        report["status"] = "skipped"
        report["blockers"] = [f"{PROPOSED_SHADOW_CLI_FLAG} not set"]
    else:
        config = GenerationShadowObserverConfig(
            shadow_observer_enabled=True,
            shadow_mode=args.shadow_mode,
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            skip_generation=args.skip_generation,
            allow_shadow_fail=args.allow_shadow_fail,
            local_files_only=args.local_files_only,
        )
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id, local_files_only=args.local_files_only,
            )
            prompts = long_context_prompts(
                tokenizer, (32, 64), max_prompts=args.max_prompts,
            )
            prompt_panel = [(p[0], p[1]) for p in prompts]
        except Exception as exc:  # noqa: BLE001
            prompt_panel = [
                (f"fallback_{i}", f"ExactKV shadow observer fallback prompt {i}")
                for i in range(args.max_prompts)
            ]
            config_blockers = [f"prompt provider failed: {type(exc).__name__}: {exc}"]
        else:
            config_blockers = []

        result = run_generation_shadow_observer(prompt_panel, config=config)
        if config_blockers:
            result.blockers.extend(config_blockers)
        report = build_exp076_report(result)

    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    errors = validate_exp076_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 076: {report['status']} observer={report['generation_shadow_observer_enabled']} "
        f"gen_ok={report['generation_successful_prompts']} "
        f"shadow_ok={report['shadow_successful_prompts']}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("diagnostic_complete", "diagnostic_partial", "skipped", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
