#!/usr/bin/env bash
# mojo build with the lab's kernels and mojo-baro's (this repo, two levels up) on the include path; opens the GPU
set -euo pipefail
cd "$(dirname "$0")/.."
B=${BARO:-$(cd ../.. && pwd)}
[ -d "$B/kernels" ] || { echo "FAIL mojo.sh: no mojo-baro at $B (set BARO)"; exit 1; }
exec gpu-wait run --timeout 380 -- .venv/bin/mojo build -I kernels/gemm -I kernels/gemv -I kernels/llama -I "$B/kernels" -I "$B" -I "$B/serve" -I "$B/tools" "$@"
