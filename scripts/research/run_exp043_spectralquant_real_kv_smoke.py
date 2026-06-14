#!/usr/bin/env python3
"""Experiment 043: SpectralQuant real model K/V tensor smoke (Phase 10F).

Captures real past_key_values from a small HF model prefill, runs SpectralQuant
compress/decompress on real K/V tensors after minimal calibration.

NOT ExactKV generation. No speed/memory/serving claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.external.spectralquant_real_kv import (  # noqa: E402
    DEFAULT_MODEL,
    CalibrationConfig,
    default_calibration_prompts,
    default_smoke_prompt,
    run_real_kv_tensor_smoke,
)

DEFAULT_JSON = _ROOT / "reports" / "experiment_043_spectralquant_real_kv_smoke.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 043 SpectralQuant real KV tensor smoke")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--calib-prompts", nargs="*", default=None)
    args = parser.parse_args()

    report = run_real_kv_tensor_smoke(
        model_name=args.model,
        prompt=args.prompt or default_smoke_prompt(),
        calibration_prompts=args.calib_prompts or default_calibration_prompts(),
        calibration_config=CalibrationConfig(),
        device=args.device,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Exp 043: {report['status']} label={report['label']} recommendation={report['recommendation']}")
    if report.get("summary"):
        s = report["summary"]
        print(
            f"  key_max_err={s.get('key_max_abs_error')} "
            f"value_max_err={s.get('value_max_abs_error')} "
            f"calibration_required={s.get('calibration_required')}"
        )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
