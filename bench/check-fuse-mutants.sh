#!/usr/bin/env bash
# The fused q4x kernels' gate (kernels/test_q4x_hostops.mojo, fused vs unfused bit-identity) must catch a planted bug in
# each fused kernel: Q4X_FUSE_MUTANT 1 hcdown inject, 2 hcup cur, 3 gnorm_ffn combine, each a ~5e-7 relative nudge of one
# output. Mutant 0 (the shipped build) must pass. One gpu-wait job.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.work/fuse-mutants; mkdir -p "$OUT"
fail() { echo "FAIL $1: $2"; exit 1; }
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  for m in 0 1 2 3; do
    timeout 900 ./.venv/bin/mojo build -D Q4X_FUSE_MUTANT=$m kernels/test_q4x_hostops.mojo -I kernels -o "$OUT/t$m" \
      --target-accelerator "${BARO_TARGET_ACCELERATOR:-gfx1100}" > "$OUT/build$m.log" 2>&1 || fail build "mutant $m, $OUT/build$m.log"
  done
  exec gpu-wait run --timeout 600 -- bash "$0"
fi
"$OUT/t0" > "$OUT/run0.log" 2>&1 || fail shipped "mutant 0 failed its own gate, $OUT/run0.log"
caught=0
for m in 1 2 3; do
  if "$OUT/t$m" > "$OUT/run$m.log" 2>&1; then echo "MISSED mutant $m"; else caught=$((caught + 1)); echo "caught mutant $m: $(grep -m1 -o 'FAIL [^:]*' "$OUT/run$m.log")"; fi
done
[ $caught -eq 3 ] || fail mutants "caught $caught/3"
echo "PASS fuse-mutants: shipped build passes, 3/3 planted fused-kernel bugs caught"
