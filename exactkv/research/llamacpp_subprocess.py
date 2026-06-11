"""Subprocess helpers for llama.cpp TurboQuant external-drafter probes (Exp 022).

Not integrated into ExactKV runtime. No performance metrics.
"""
from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from typing import Sequence

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PERF_LINE_RE = re.compile(
    r"^(?:\d+\.\d+\.\d+\.\d+ I |.*perf_print:|.*tokens per second)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubprocessResult:
    """Captured subprocess output (no timing fields)."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def strip_llama_output(text: str) -> str:
    """Remove ANSI escapes and common llama.cpp log/perf lines from captured text."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).strip()
        if not line:
            continue
        if _PERF_LINE_RE.search(line):
            continue
        if line.startswith("warning:"):
            continue
        if "llama_model_loader:" in line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_llama_token_ids(stdout: str) -> list[int]:
    """Parse ``llama-tokenize --ids`` output (Python list format)."""
    cleaned = strip_llama_output(stdout)
    for line in reversed(cleaned.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            parsed = ast.literal_eval(line)
            if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                return list(parsed)
    raise ValueError(f"Could not parse llama token IDs from output: {stdout!r}")


def extract_continuation_text(full_text: str, prompt: str) -> str:
    """Return generated continuation when stdout echoes the raw prompt prefix."""
    full = full_text.strip()
    if full.startswith(prompt):
        return full[len(prompt):]
    return full


def tokenizer_ids_match(
    hf_ids: Sequence[int],
    llama_ids: Sequence[int],
) -> bool:
    return list(hf_ids) == list(llama_ids)


def run_subprocess(
    argv: list[str],
    *,
    input_text: str | None = None,
    timeout_seconds: float | None = 120.0,
) -> SubprocessResult:
    try:
        proc = subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return SubprocessResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return SubprocessResult(
            returncode=-1,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + "\nTIMEOUT",
            timed_out=True,
        )


def run_llama_tokenize(
    llama_tokenize_bin: str,
    gguf_model: str,
    text: str,
    *,
    timeout_seconds: float | None = 60.0,
    use_stdin: bool = False,
) -> list[int]:
    argv = [
        llama_tokenize_bin,
        "-m",
        gguf_model,
        "--log-disable",
        "--ids",
    ]
    if use_stdin:
        argv.append("--stdin")
        result = run_subprocess(argv, input_text=text, timeout_seconds=timeout_seconds)
    else:
        argv.extend(["-p", text])
        result = run_subprocess(argv, timeout_seconds=timeout_seconds)
    if result.timed_out:
        raise TimeoutError(f"llama-tokenize timed out after {timeout_seconds}s")
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed (rc={result.returncode}): {result.stderr}"
        )
    return parse_llama_token_ids(result.stdout)


def run_llama_completion(
    llama_completion_bin: str,
    gguf_model: str,
    prompt: str,
    *,
    max_new_tokens: int,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "turbo3",
    n_gpu_layers: int = 0,
    timeout_seconds: float | None = 300.0,
) -> SubprocessResult:
    """Run non-interactive ``llama-completion`` (preferred over ``llama-cli``)."""
    argv = [
        llama_completion_bin,
        "-m",
        gguf_model,
        "-ctk",
        cache_type_k,
        "-ctv",
        cache_type_v,
        "-fa",
        "on",
        "-ngl",
        str(n_gpu_layers),
        "--temp",
        "0",
        "--top-k",
        "1",
        "-n",
        str(max_new_tokens),
        "-no-cnv",
        "--prompt",
        prompt,
    ]
    result = run_subprocess(argv, timeout_seconds=timeout_seconds)
    if result.timed_out:
        raise TimeoutError(
            f"llama-completion timed out after {timeout_seconds}s; "
            "check interactive-mode hang or increase --timeout-seconds"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-completion failed (rc={result.returncode}): {result.stderr}"
        )
    return result


def resolve_completion_binary(llama_cli_bin: str) -> str:
    """Prefer ``llama-completion`` sibling when ``llama-cli`` path is given."""
    from pathlib import Path

    p = Path(llama_cli_bin)
    completion = p.parent / "llama-completion"
    if completion.is_file():
        return str(completion)
    return str(p)
