#!/usr/bin/env bash
# V12 Phase 1b — TurboQuant production-fidelity toolchain prep (NOT Experiment 022).
# Idempotent where practical. Does not modify ExactKV runtime or commit weights/GGUF.
set -euo pipefail

if [[ -d /workspace ]] && [[ "$(id -u)" -eq 0 || -w /workspace ]]; then
  WORKDIR="${TQ_PREP_WORKDIR:-/workspace/turboquant_prod_prep}"
else
  WORKDIR="${TQ_PREP_WORKDIR:-/tmp/turboquant_prod_prep}"
fi

mkdir -p "$WORKDIR"
LOG="$WORKDIR/prep.log"
MANIFEST="$WORKDIR/prep_manifest.txt"
FLAG_HELP="$WORKDIR/llama_cli_help.txt"
REFRACT_LOG="$WORKDIR/refract_selftest.log"
SMOKE_LOG="$WORKDIR/llama_cli_smoke.log"
CONVERT_LOG="$WORKDIR/gguf_convert.log"

LLAMA_REPO_URL="${LLAMA_REPO_URL:-https://github.com/TheTom/llama-cpp-turboquant.git}"
LLAMA_BRANCH="${LLAMA_BRANCH:-feature/turboquant-kv-cache}"
TQ_PLUS_URL="${TQ_PLUS_URL:-https://github.com/TheTom/turboquant_plus.git}"
LLAMA_DIR="$WORKDIR/llama-cpp-turboquant"
TQ_PLUS_DIR="$WORKDIR/turboquant_plus"
VENV_DIR="$WORKDIR/venv_refract"
MODELS_DIR="$WORKDIR/models"
GGUF_OUT="${GGUF_OUT:-$MODELS_DIR/qwen2.5-0.5b-auto.gguf}"
GGUF_OUTTYPE="${GGUF_OUTTYPE:-auto}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-0.5B}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_CLONE="${SKIP_CLONE:-0}"

exec > >(tee -a "$LOG") 2>&1

echo "=== TurboQuant production toolchain prep ==="
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "workdir: $WORKDIR"

{
  echo "# TurboQuant toolchain prep manifest"
  echo "generated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "workdir: $WORKDIR"
  echo "hostname: $(hostname)"
  echo "uname: $(uname -a)"
  echo "python: $(python3 --version 2>&1)"
  echo "git: $(git --version 2>&1)"
  echo "cmake: $(cmake --version 2>/dev/null | head -1 || echo MISSING)"
  echo "gcc: $(gcc --version 2>/dev/null | head -1 || echo MISSING)"
  echo "g++: $(g++ --version 2>/dev/null | head -1 || echo MISSING)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia_smi: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  else
    echo "nvidia_smi: absent"
  fi
} > "$MANIFEST"

install_deps() {
  echo "=== Dependency install ==="
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq cmake build-essential ninja-build git python3-venv python3-pip \
      pkg-config 2>&1 | tail -20
    echo "apt_install: success" >> "$MANIFEST"
  else
    echo "apt_install: skipped — manual install: cmake build-essential ninja-build git python3-venv python3-pip" >> "$MANIFEST"
    echo "WARN: apt-get not available; continuing with existing tools"
  fi
}

clone_repos() {
  echo "=== Clone upstream repos ==="
  if [[ "$SKIP_CLONE" == "1" ]]; then
    echo "skip_clone: repos assumed present" >> "$MANIFEST"
    return 0
  fi
  if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    git clone --depth 1 --branch "$LLAMA_BRANCH" "$LLAMA_REPO_URL" "$LLAMA_DIR"
  else
    git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_BRANCH" 2>/dev/null || true
    git -C "$LLAMA_DIR" checkout "$LLAMA_BRANCH" 2>/dev/null || true
    git -C "$LLAMA_DIR" pull --ff-only 2>/dev/null || true
  fi
  if [[ ! -d "$TQ_PLUS_DIR/.git" ]]; then
    git clone --depth 1 "$TQ_PLUS_URL" "$TQ_PLUS_DIR"
  fi
  LLAMA_SHA=$(git -C "$LLAMA_DIR" rev-parse HEAD)
  TQ_SHA=$(git -C "$TQ_PLUS_DIR" rev-parse HEAD)
  echo "llama_repo: $LLAMA_REPO_URL" >> "$MANIFEST"
  echo "llama_branch: $LLAMA_BRANCH" >> "$MANIFEST"
  echo "llama_sha: $LLAMA_SHA" >> "$MANIFEST"
  echo "turboquant_plus_sha: $TQ_SHA" >> "$MANIFEST"
}

build_llama() {
  echo "=== Build llama-cpp-turboquant ==="
  if [[ "$SKIP_BUILD" == "1" ]]; then
    for d in "$LLAMA_DIR/build/bin" "$LLAMA_DIR/build-cpu/bin" "$LLAMA_DIR/build-make/bin"; do
      if [[ -x "$d/llama-cli" ]]; then
        export LLAMA_BUILD_DIR="$(dirname "$d")"
        echo "skip_build: using $d/llama-cli" >> "$MANIFEST"
        echo "llama_build_dir: $LLAMA_BUILD_DIR" >> "$MANIFEST"
        return 0
      fi
    done
  fi
  cd "$LLAMA_DIR"
  BUILD_DIR="build"
  CMAKE_ARGS=(-B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -G Ninja)
  if command -v nvidia-smi >/dev/null 2>&1; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
    echo "build_target: CUDA" >> "$MANIFEST"
  else
    echo "build_target: CPU" >> "$MANIFEST"
  fi
  if ! cmake "${CMAKE_ARGS[@]}" 2>&1; then
    echo "WARN: CUDA cmake failed; retry CPU-only"
    BUILD_DIR="build-cpu"
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -G Ninja
    echo "build_target: CPU (fallback)" >> "$MANIFEST"
  fi
  if ! cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"; then
    echo "WARN: Ninja build failed; retry Unix Makefiles"
    BUILD_DIR="build-make"
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"
    echo "build_generator: Unix Makefiles (fallback)" >> "$MANIFEST"
  else
    echo "build_generator: Ninja" >> "$MANIFEST"
  fi
  export LLAMA_BUILD_DIR="$LLAMA_DIR/$BUILD_DIR"
  echo "llama_build_dir: $LLAMA_BUILD_DIR" >> "$MANIFEST"
  cd "$WORKDIR"
}

locate_binaries() {
  echo "=== Locate binaries ==="
  BIN_DIR=""
  for d in "$LLAMA_BUILD_DIR/bin" "$LLAMA_DIR/build/bin" "$LLAMA_DIR/build-cpu/bin" "$LLAMA_DIR/build-make/bin"; do
    if [[ -x "$d/llama-cli" ]]; then
      BIN_DIR="$d"
      break
    fi
  done
  if [[ -z "$BIN_DIR" ]]; then
    echo "llama_cli: MISSING" >> "$MANIFEST"
    echo "ERROR: llama-cli not found after build"
    return 1
  fi
  echo "llama_bin_dir: $BIN_DIR" >> "$MANIFEST"
  export LLAMA_CPP_BIN_DIR="$BIN_DIR"
  export PATH="$BIN_DIR:$PATH"
  for b in llama-cli llama-server llama-completion llama-tokenize llama-perplexity; do
    if [[ -x "$BIN_DIR/$b" ]]; then
      echo "binary_found: $BIN_DIR/$b" >> "$MANIFEST"
    else
      echo "binary_missing: $b" >> "$MANIFEST"
    fi
  done
  "$BIN_DIR/llama-cli" --help > "$FLAG_HELP" 2>&1 || true
  for flag in ctk ctv turbo cache-type-k cache-type-v q8_0 turbo3; do
    if grep -qi "$flag" "$FLAG_HELP"; then
      echo "help_flag_found: $flag" >> "$MANIFEST"
    else
      echo "help_flag_missing: $flag" >> "$MANIFEST"
    fi
  done
}

setup_refract() {
  echo "=== REFRACT setup (optional tooling check) ==="
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install -q --upgrade pip
  pip install -q -e "$TQ_PLUS_DIR[dev]" || pip install -q refract-llm
  export LLAMA_CPP_BIN_DIR="${LLAMA_CPP_BIN_DIR:-}"
  refract --help > "$WORKDIR/refract_help.txt" 2>&1 || true
  echo "refract_help: $WORKDIR/refract_help.txt" >> "$MANIFEST"
  if refract selftest > "$REFRACT_LOG" 2>&1; then
    echo "refract_selftest: PASS" >> "$MANIFEST"
  else
    echo "refract_selftest: FAIL (see $REFRACT_LOG)" >> "$MANIFEST"
    tail -15 "$REFRACT_LOG" || true
  fi
  deactivate || true
}

gguf_readiness() {
  echo "=== GGUF conversion readiness ==="
  CONVERTER="$LLAMA_DIR/convert_hf_to_gguf.py"
  if [[ -f "$CONVERTER" ]] && [[ -f "$LLAMA_DIR/conversion/qwen.py" ]]; then
    echo "gguf_converter: present" >> "$MANIFEST"
    echo "qwen_conversion_module: present" >> "$MANIFEST"
  else
    echo "gguf_converter: missing" >> "$MANIFEST"
    return 0
  fi
  mkdir -p "$MODELS_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install -q torch transformers sentencepiece protobuf 2>&1 | tail -5 || true
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  HF_SNAP=""
  for d in "$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B/snapshots"/*; do
    if [[ -f "$d/config.json" ]]; then HF_SNAP="$d"; break; fi
  done
  if [[ -n "$HF_SNAP" ]] || [[ "${FORCE_GGUF_CONVERT:-0}" == "1" ]]; then
    echo "hf_cache_qwen05b: present" >> "$MANIFEST"
    [[ -n "$HF_SNAP" ]] && echo "hf_snapshot: $HF_SNAP" >> "$MANIFEST"
    CONVERT_SRC="${HF_SNAP:-$HF_MODEL}"
    if [[ ! -f "$GGUF_OUT" ]]; then
      echo "Attempting GGUF conversion to $GGUF_OUT (outtype=$GGUF_OUTTYPE)"
      if python3 "$CONVERTER" "$CONVERT_SRC" --outfile "$GGUF_OUT" --outtype "$GGUF_OUTTYPE" > "$CONVERT_LOG" 2>&1; then
        echo "gguf_convert: success" >> "$MANIFEST"
        echo "gguf_path: $GGUF_OUT" >> "$MANIFEST"
        ls -lh "$GGUF_OUT" >> "$MANIFEST"
      else
        echo "gguf_convert: FAIL" >> "$MANIFEST"
        tail -20 "$CONVERT_LOG" >> "$MANIFEST" || true
      fi
    else
      echo "gguf_convert: skipped (exists)" >> "$MANIFEST"
      echo "gguf_path: $GGUF_OUT" >> "$MANIFEST"
    fi
  else
    echo "hf_cache_qwen05b: absent — defer conversion to Phase 2" >> "$MANIFEST"
    echo "gguf_convert_command: python3 $CONVERTER $HF_MODEL --outfile $GGUF_OUT --outtype $GGUF_OUTTYPE" >> "$MANIFEST"
  fi
  deactivate || true
}

cli_smoke() {
  echo "=== TurboQuant CLI smoke ==="
  BIN_DIR="${LLAMA_CPP_BIN_DIR:-}"
  if [[ -z "$BIN_DIR" ]] || [[ ! -x "$BIN_DIR/llama-cli" ]]; then
    echo "cli_smoke: skipped (no binary)" >> "$MANIFEST"
    return 0
  fi
  if [[ ! -f "$GGUF_OUT" ]]; then
    echo "cli_smoke: help-only (no GGUF)" >> "$MANIFEST"
    return 0
  fi
  PROMPT="The capital of France is"
  NGL=0
  set +e
  if [[ -x "$BIN_DIR/llama-completion" ]]; then
    timeout 300 "$BIN_DIR/llama-completion" \
      -m "$GGUF_OUT" \
      -ctk q8_0 -ctv turbo3 \
      -fa on -ngl "$NGL" \
      --temp 0 --top-k 1 \
      -n 4 \
      -no-cnv \
      --prompt "$PROMPT" \
      > "$SMOKE_LOG" 2>&1
    SMOKE_RC=$?
  else
    timeout 300 "$BIN_DIR/llama-cli" \
      -m "$GGUF_OUT" \
      -ctk q8_0 -ctv turbo3 \
      -fa on -ngl "$NGL" \
      --temp 0 --top-k 1 \
      -n 4 \
      -no-cnv \
      --prompt "$PROMPT" \
      > "$SMOKE_LOG" 2>&1
    SMOKE_RC=$?
  fi
  set -e
  echo "cli_smoke_exit: $SMOKE_RC" >> "$MANIFEST"
  tail -30 "$SMOKE_LOG" >> "$MANIFEST" || true
  if grep -qi "Paris" "$SMOKE_LOG"; then
    echo "cli_smoke_text_ok: yes" >> "$MANIFEST"
  fi
  if [[ -x "$BIN_DIR/llama-tokenize" ]]; then
    echo "$PROMPT" | "$BIN_DIR/llama-tokenize" -m "$GGUF_OUT" --stdin --show-count 2>/dev/null \
      > "$WORKDIR/tokenize_smoke.txt" 2>&1 || true
    echo "tokenize_smoke: $WORKDIR/tokenize_smoke.txt" >> "$MANIFEST"
  fi
}

phase2_gate() {
  echo "=== Phase 2 gate ==="
  CAN=1
  grep -q "llama_cli: MISSING" "$MANIFEST" 2>/dev/null && CAN=0
  grep -q "help_flag_found: turbo3" "$MANIFEST" 2>/dev/null || CAN=0
  if [[ $CAN -eq 1 ]]; then
    echo "phase2_proceed: YES (toolchain ready; Exp 022 not started)" >> "$MANIFEST"
  else
    echo "phase2_proceed: NO — see blockers in manifest" >> "$MANIFEST"
  fi
}

install_deps
clone_repos
build_llama
locate_binaries
setup_refract
gguf_readiness
cli_smoke
phase2_gate

echo "=== Done ==="
echo "manifest: $MANIFEST"
echo "log: $LOG"
cat "$MANIFEST"
