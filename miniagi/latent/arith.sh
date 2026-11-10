#!/usr/bin/env bash
# Arithmetic with a program checker (arithgen.mojo). Families: 0 flat +/-, 1 flat with *, 2 one paren pair.
# usage: gpu-wait run --timeout 150 (pilot) or 420 (folds) -- bash miniagi/latent/arith.sh pilot|folds
# pilot: seed 110, trained on all 3 families, nocot arm (validity test) and cot arm (same-budget ceiling).
# folds: cot arm, hold out one family (train on the other two), seeds 110-112, test on all three; family folds = out0 out1 out2.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=.work/spec/arith; O=.work/spec/arith-$1; ST=${ST:-3000}
K="lr=1e-3 warmup=50 decay=1 n=5 S=40 every=1000 evaln=512 neval=2048 test=0,1,2"
run() { # name arm seed fams
  local d=$O/$1; mkdir -p $d
  $R $2 $ST 256 $3 - fams=$4 $K > $d/log 2>&1 || { echo "FAIL $1: run"; exit 1; }
  grep -q ' np ' $d/log || { echo "FAIL $1: readback"; exit 1; }
  for f in 0 1 2; do grep "RESULT .* fam $f " $d/log | tail -1 | awk -v n=$1 '{print n, $2, $3, $4, "trained", $6, "acc", $8, "chain", $10, "noend", $12, "ms", $20}'; done
}
case $1 in
  pilot) run nocot-all nocot 110 0,1,2; run cot-all cot 110 0,1,2;;
  folds) for s in 110 111 112; do run out0-s$s cot $s 1,2; run out1-s$s cot $s 0,2; run out2-s$s cot $s 0,1; done;;
  *) echo "FAIL arg: $1"; exit 1;;
esac
echo DONE
