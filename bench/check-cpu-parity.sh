#!/usr/bin/env bash
# bench/cpu-parity-protocol.md: cpu/ops.mojo q4_row_dot is bitwise equal to amar_matmul_skinny_q4rowb on real pack
# rows; a planted mutant (32-lane tree order swapped) must fail. Fail word and arm line read on every run.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.work/cpu-parity; mkdir -p "$OUT/mut"
MOJO=${MOJO:-./.venv/bin/mojo}
PACK=${PACK:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.work/engine-pack-q4}
export MODULAR_CRASH_REPORTING_ENABLED=false
fail() { echo "FAIL $1: $2"; exit 1; }
[ -f "$PACK/pack.bin" ] || fail pack "$PACK/pack.bin missing"
sed 's/fold\[32\](p, True)/fold[32](p, False)/' cpu/ops.mojo >"$OUT/mut/ops.mojo"
cmp -s cpu/ops.mojo "$OUT/mut/ops.mojo" && fail mutant "sed changed nothing"
timeout 300 "$MOJO" build -I kernels -I cpu kernels/test_cpu_dot.mojo -o "$OUT/dot" >"$OUT/build.log" 2>&1 || fail build "$OUT/build.log"
timeout 300 "$MOJO" build -I kernels -I "$OUT/mut" kernels/test_cpu_dot.mojo -o "$OUT/dot-mut" >"$OUT/build-mut.log" 2>&1 || fail build "$OUT/build-mut.log"
PACKABS=$(readlink -f "$PACK")
gpu-wait run --timeout 240 -- sh -c "$OUT/dot $PACKABS >$OUT/run.txt 2>&1" || true
gpu-wait run --timeout 240 -- sh -c "$OUT/dot-mut $PACKABS >$OUT/run-mut.txt 2>&1" || true
for f in run run-mut; do
  grep -q '^arm gpu=amar_matmul_skinny_q4rowb\[UNROLL=2,MR=1\] cpu=cpu/ops.mojo:q4_row_dot' "$OUT/$f.txt" || fail arm "$f: no arm line (job died before the GPU?)"
  [ "$(grep -c 'mismatches [0-9]* of 4096' "$OUT/$f.txt")" -eq 3 ] || fail incomplete "$f: expected 3 tensor lines"
done
grep -q '^PASS test_cpu_dot' "$OUT/run.txt" || fail bitwise "$(grep 'mismatches' "$OUT/run.txt" | tr '\n' ' ')"
[ "$(grep -c 'mismatches 0 of 4096' "$OUT/run.txt")" -eq 3 ] || fail bitwise "pass word without 3 zero counts"
grep -q '^PASS' "$OUT/run-mut.txt" && fail mutant "mutant passed"
bad=$(awk '/mismatches [0-9]+ of/{s+=$(NF-2)} END{print s+0}' "$OUT/run-mut.txt")
[ "$bad" -gt 0 ] || fail mutant "mutant run shows 0 mismatches"
echo "PASS cpu-parity: q4 dot cpu == gfx1100 on 12288 real-row dots, mutant caught ($bad mismatches)"
