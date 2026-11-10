#!/usr/bin/env bash
# Build the soft-thinking proposer backend: libsoft_shim.so (C glue to libllama) and the Mojo driver.
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${LLAMA_DIR:-$HOME/llama.cpp}
mkdir -p .work/soft-think
cc -O2 -shared -fPIC tools/soft-think/soft_shim.c -I "$L/include" -I "$L/ggml/include" -L "$L/build/bin" \
  -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin" -o .work/soft-think/libsoft_shim.so \
  || { echo "FAIL build: soft_shim.c"; exit 1; }
./.venv/bin/mojo build tools/soft-think/soft-think.mojo -I . -o .work/soft-think/soft-think > .work/soft-think/build.log 2>&1 \
  || { grep -m3 error: .work/soft-think/build.log; echo "FAIL build: soft-think.mojo, see .work/soft-think/build.log"; exit 1; }
echo "built .work/soft-think/{libsoft_shim.so,soft-think}"
