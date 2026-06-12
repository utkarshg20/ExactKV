#!/usr/bin/env bash
# V12 Phase 1 — TurboQuant llama.cpp research commands (NOT ExactKV integration).
# Run in an isolated environment. Does not modify ExactKV runtime.
set -euo pipefail

TQ_PLUS_REPO="${TQ_PLUS_REPO:-/tmp/turboquant_plus_research}"
LLAMA_TQ_REPO="${LLAMA_TQ_REPO:-/tmp/llama-cpp-turboquant}"
MODEL_HF="${MODEL_HF:-Qwen/Qwen2.5-0.5B-Instruct}"
GGUF_OUT="${GGUF_OUT:-/tmp/qwen25-05b-tq-research.gguf}"

echo "=== Clone upstream (skip if present) ==="
if [[ ! -d "$TQ_PLUS_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/TheTom/turboquant_plus.git "$TQ_PLUS_REPO"
fi
if [[ ! -d "$LLAMA_TQ_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/TheTom/llama-cpp-turboquant.git "$LLAMA_TQ_REPO"
fi
cd "$LLAMA_TQ_REPO"
git checkout feature/turboquant-kv-cache 2>/dev/null || true

echo "=== Python turboquant import smoke ==="
python3 -m venv "$TQ_PLUS_REPO/.venv_phase1" || true
# shellcheck disable=SC1091
source "$TQ_PLUS_REPO/.venv_phase1/bin/activate"
pip install -q numpy scipy
PYTHONPATH="$TQ_PLUS_REPO" python3 -c "from turboquant import KVCacheCompressor; print('turboquant ok')"

echo "=== REFRACT install smoke (optional quality reference) ==="
pip install -q -e "$TQ_PLUS_REPO[dev]" || pip install -q refract-llm
refract selftest || echo "REFRACT selftest failed — need patched llama-cpp-turboquant binaries"

echo "=== Build llama-cpp-turboquant (requires cmake) ==="
if command -v cmake >/dev/null 2>&1; then
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
else
  echo "SKIP: cmake not found. Use prebuilt release from:"
  echo "  https://github.com/TheTom/llama-cpp-turboquant/releases"
fi

echo "=== HF -> GGUF conversion (requires torch, transformers; large download) ==="
echo "Example (not run automatically):"
echo "  pip install torch transformers sentencepiece"
echo "  python3 convert_hf_to_gguf.py $MODEL_HF --outfile $GGUF_OUT --outtype q4_k_m"

echo "=== llama.cpp TurboQuant inference smoke (after GGUF + build) ==="
echo "Example (not run automatically):"
echo "  ./build/bin/llama-cli -m $GGUF_OUT -ctk q8_0 -ctv turbo3 -fa on -ngl 99 \\"
echo "    --temp 0 --top-k 1 -n 16 -no-cnv --prompt 'Hello'"

echo "=== ExactKV Phase 1 inspector ==="
echo "  PYTHONPATH=$TQ_PLUS_REPO python3 scripts/research/turboquant_production_phase1_inspect.py \\"
echo "    --llama-repo $LLAMA_TQ_REPO --with-refract"
