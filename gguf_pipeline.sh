#!/bin/bash
# GGUF conversion + quantization for KodaLite-1.3B.
#
# Requires:
#   - cmake (apt-get install cmake)
#   - llama.cpp checked out and buildable
# Inputs:
#   - HF Transformers format model at $KODA_ROOT/hf_llama (or hf_llama_eos)
# Outputs:
#   - $KODA_ROOT/gguf/{kodalite-f16,kodalite-Q8_0,kodalite-Q4_K_M}.gguf
#
# Override defaults via env: KODA_ROOT, LLAMA_CPP_DIR, HF_SRC_DIR, GGUF_OUT_DIR.

set -e

KODA_ROOT=${KODA_ROOT:-./work}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$KODA_ROOT/llama.cpp}
HF_SRC_DIR=${HF_SRC_DIR:-$KODA_ROOT/hf_llama_eos}
GGUF_OUT_DIR=${GGUF_OUT_DIR:-$KODA_ROOT/gguf_eos}

mkdir -p "$GGUF_OUT_DIR"

echo "=== [1/3] Build llama-quantize ==="
cd "$LLAMA_CPP_DIR"
if [ ! -f build/bin/llama-quantize ]; then
  cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF 2>&1 | tail -5
  cmake --build build --target llama-quantize -j 4 2>&1 | tail -5
fi

echo "=== [2/3] Convert HF -> GGUF F16 ==="
python convert_hf_to_gguf.py "$HF_SRC_DIR" \
  --outfile "$GGUF_OUT_DIR/kodalite-f16.gguf" --outtype f16 2>&1 | tail -10

echo "=== [3/3] Quantize Q8_0 + Q4_K_M ==="
./build/bin/llama-quantize "$GGUF_OUT_DIR/kodalite-f16.gguf"   "$GGUF_OUT_DIR/kodalite-Q8_0.gguf"   Q8_0   2>&1 | tail -3
./build/bin/llama-quantize "$GGUF_OUT_DIR/kodalite-f16.gguf"   "$GGUF_OUT_DIR/kodalite-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -3
ls -la "$GGUF_OUT_DIR"

echo "=== DONE ==="
echo "To upload: hf upload \${HF_REPO_GGUF:-YoAbriel/KodaLite-1.3B-GGUF} $GGUF_OUT_DIR"
