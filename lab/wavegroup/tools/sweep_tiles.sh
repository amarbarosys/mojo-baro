#!/usr/bin/env bash
# tile-config sweep for the GEMM: usage tools/sweep_tiles.sh M N K TB; one line per config, device TFLOPS
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1 N=$2 K=$3 TB=$4
for c in "2 2 2 2" "2 4 2 2" "4 2 2 2" "2 2 2 4" "2 2 4 2" "4 2 2 4" "2 4 2 4" "4 4 2 2" "2 2 4 4"; do
  set -- $c; b=.work/sweep/g_${M}_${N}_${K}_${TB}_$1$2$3$4; mkdir -p .work/sweep
  if ! gpu-wait run --timeout 280 -- .venv/bin/mojo build -I kernels/gemm -D WMMA_C_F16=1 -D WMMA_TB=$TB -D TG_M=$M -D TG_N=$N -D TG_K=$K \
      -D TG_WM=$1 -D TG_WN=$2 -D TG_TM=$3 -D TG_TN=$4 -o $b kernels/gemm/run.mojo >$b.log 2>&1; then echo "cfg $c BUILD_FAIL"; continue; fi
  r=$(gpu-wait run --timeout 180 -- $b 2>&1 | tail -1 | grep -oE 'dev_tflops [0-9.]+|max_rel_err [0-9.e-]+|FAIL.*' | tr '\n' ' ') || true
  echo "shape $M $N $K tb$TB cfg $c $r"
done
