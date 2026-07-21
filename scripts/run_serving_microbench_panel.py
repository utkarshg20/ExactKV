#!/usr/bin/env python3
"""CLI for the ExactKV HF multi-request serving microbench."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.serving_microbench_panel import (  # noqa: E402
    DEFAULT_COMPRESSORS,
    DEFAULT_CONTEXT_BUCKETS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODELS,
    DEFAULT_N_REQUESTS,
    DEFAULT_OUTPUT_DIR,
    run_serving_microbench_panel,
    write_serving_microbench_outputs,
)


def _model_tag(model: str) -> str:
    tag = model.split("/")[-1]
    return tag.replace(".", "_").replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--compressors", default=",".join(DEFAULT_COMPRESSORS))
    parser.add_argument(
        "--context-buckets",
        default=",".join(str(x) for x in DEFAULT_CONTEXT_BUCKETS),
    )
    parser.add_argument(
        "--max-new-tokens",
        default=",".join(str(x) for x in DEFAULT_MAX_NEW_TOKENS),
    )
    parser.add_argument(
        "--n-requests",
        default=",".join(str(x) for x in DEFAULT_N_REQUESTS),
        help="Serial request counts per load shape (e.g. 1,4)",
    )
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--checkpoint-json", type=Path, default=None)
    parser.add_argument("--resume-json", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--deterministic-mode", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    compressors = [c.strip() for c in args.compressors.split(",") if c.strip()]
    buckets = [int(x) for x in args.context_buckets.split(",") if x.strip()]
    mnts = [int(x) for x in args.max_new_tokens.split(",") if x.strip()]
    nreqs = [int(x) for x in args.n_requests.split(",") if x.strip()]

    resume_cells = []
    if args.resume_json and args.resume_json.exists():
        prev = json.loads(args.resume_json.read_text(encoding="utf-8"))
        resume_cells = list(prev.get("cells") or [])

    out = args.output_json
    if out is None:
        if len(models) == 1:
            out = args.out_dir / f"{_model_tag(models[0])}_raw.json"
        else:
            out = args.out_dir / "combined_raw.json"
    ckpt = args.checkpoint_json or out

    def _progress(msg: str) -> None:
        print(msg, flush=True)
        status = ckpt.with_suffix(".status.txt")
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(msg + "\n", encoding="utf-8")

    def _checkpoint(partial: dict) -> None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps(partial, indent=2, default=str) + "\n", encoding="utf-8")
        # Also stamp a rotating backup so a mid-write crash can't wipe the last good file.
        bak = ckpt.with_suffix(f".bak.{partial.get('n_cells', 0)}.json")
        bak.write_text(ckpt.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"checkpoint n_cells={partial.get('n_cells')} -> {ckpt}", flush=True)

    report = run_serving_microbench_panel(
        device=args.device,
        dtype=args.dtype,
        models=models,
        compressors=compressors,
        context_buckets=buckets,
        max_new_tokens_list=mnts,
        n_requests_list=nreqs,
        draft_len=args.draft_len,
        smoke=args.smoke,
        deterministic_mode=args.deterministic_mode,
        resume_cells=resume_cells,
        progress_callback=_progress,
        checkpoint_callback=_checkpoint,
    )
    md = out.with_name(out.stem.replace("_raw", "_summary") + ".md")
    if md == out:
        md = out.with_suffix(".md")
    write_serving_microbench_outputs(report, json_path=out, markdown_path=md)
    # checkpoint same path
    if ckpt != out:
        write_serving_microbench_outputs(report, json_path=ckpt)
    print(f"Wrote {out} ({report['n_cells']} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
