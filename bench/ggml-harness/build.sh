#!/usr/bin/env bash
# builds bench/ggml-harness/op_bench against ~/llama.cpp-master (headers + build/bin libs)
set -euo pipefail
cd "$(dirname "$0")"
L=${LLAMA_ROOT:-$HOME/llama.cpp-master}
gcc -O2 -o op_bench op_bench.c -I"$L/ggml/include" -L"$L/build/bin" -lggml -lggml-base -lggml-hip -lm -Wl,-rpath,"$L/build/bin"
git -C "$L" rev-parse --short HEAD > op_bench.llama-commit
echo "built: $(pwd)/op_bench against llama.cpp $(cat op_bench.llama-commit)"
