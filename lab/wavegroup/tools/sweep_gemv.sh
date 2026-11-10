#!/usr/bin/env bash
# split-K sweep for the k-quant GEMV: one line per (shape, S); S=0 is mojo-baro's amar_gemv_kq
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .work/m2
fail=0
for shape in "512 2048 12" "2048 2048 12" "3072 2048 12" "8192 2048 12" "2048 8192 12" "128256 2048 14"; do
  set -- $shape
  for S in 0 2 4 8; do
    o=.work/m2/gs_$1_$2_$3_$S
    tools/mojo.sh -D GV_N=$1 -D GV_K=$2 -D GV_GT=$3 -D GV_S=$S -o $o kernels/gemv/bench.mojo >$o.log 2>&1 || { echo "shape $shape s $S BUILD_FAIL"; fail=1; continue; }
    r=$(gpu-wait run --timeout 180 -- $o 2>&1 | tail -1) || { echo "shape $shape s $S RUN_FAIL"; fail=1; continue; }
    echo "$r"
  done
done
[ $fail = 0 ] || { echo "FAIL sweep: a run failed"; exit 1; }
echo SWEEP_DONE
