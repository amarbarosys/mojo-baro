#!/usr/bin/env bash
# Build tools/mtmd-embed.c against the llama.cpp checkout's libmtmd/libllama (no rebuild of llama.cpp).
set -euo pipefail
cd "$(dirname "$0")/.."
L=${LLAMA_CPP:-$HOME/Projects/imports/llama.cpp}
mkdir -p .work/vision
gcc -O2 -Wall -Wno-deprecated-declarations -I "$L/include" -I "$L/ggml/include" -I "$L/tools/mtmd" tools/mtmd-embed.c \
  -L "$L/build/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin" -o .work/vision/mtmd-embed
echo "built .work/vision/mtmd-embed"
