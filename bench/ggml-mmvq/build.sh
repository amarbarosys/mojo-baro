#!/usr/bin/env bash
# builds bench/ggml-mmvq/mmvq_shapes against ~/llama.cpp-master (headers + build/bin libs)
set -euo pipefail
cd "$(dirname "$0")"
L=$HOME/llama.cpp-master
gcc -O2 -o mmvq_shapes mmvq_shapes.c -I"$L/ggml/include" -L"$L/build/bin" -lggml -lggml-base -lggml-hip -Wl,-rpath,"$L/build/bin"
echo built: $(pwd)/mmvq_shapes
