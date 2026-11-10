#!/usr/bin/env bash
# Adapt vs scratch on held-out arithmetic shapes. Base: 3000 updates on two families. Then only the held-out family,
# 1500 updates, same lr, every=50, arms adapt (init=base) and scratch. Folds out0 out1 out2, seeds 110-112.
# Frozen before the run: adapt reaches 90% on the held-out family in at most half the scratch updates on 2 of 3 folds
# (median of seeds); a run that never hits counts as 1500. Held-out start 11-21%. About 8 GPU minutes.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=.work/spec/arith; O=.work/spec/arith-adapt
K="lr=1e-3 warmup=50 decay=1 n=5 S=40 evaln=512 neval=2048 test=0,1,2"
for s in 110 111 112; do for f in 0 1 2; do
  t=$(echo 0 1 2 | tr ' ' '\n' | grep -v $f | paste -sd,); b=$O/out$f-s$s/base; mkdir -p $b
  $R cot 3000 256 $s $b fams=$t every=1000 $K > $b/log 2>&1 || { echo "FAIL base $f $s"; exit 1; }
  [ -s $b/params.f32 ] || { echo "FAIL params $f $s"; exit 1; }
  for arm in adapt scratch; do d=$O/out$f-s$s/$arm; mkdir -p $d; I=""; [ $arm = adapt ] && I=init=$b/params.f32
    $R cot 1500 256 $s - fams=$f every=50 $K $I > $d/log 2>&1 || { echo "FAIL $arm $f $s"; exit 1; }
    echo "$s out$f $arm $(grep "RESULT .* fam $f " $d/log | tail -1 | cut -c1-90)"
  done
done; done
echo DONE
