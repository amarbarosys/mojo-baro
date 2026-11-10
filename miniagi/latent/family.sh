#!/usr/bin/env bash
# Family hold-out. Rules 1-4 = add family, 5-10 = xor family (hop_rule in run.mojo). Rule 0 (plain f) is in neither.
# usage: gpu-wait run --timeout 420 -- bash miniagi/latent/family.sh add|xor   (24000 updates x 7 ms = 168 s, margin 2.5x for 18 launches + evals)
# add: base trains on add rules 1-4, held-out family = xor 5-10. xor: base trains on xor 5-10, held-out = add 1-4.
# Per held-out rule: zero-update floor, adapt (init=base) vs scratch, same lr/updates/seeds, hit_step at 90%.
set -euo pipefail
cd "$(dirname "$0")/../.."
fam=$1; O=.work/spec/fam-$fam; R=.work/spec/run; A="short=1 nobot=1 hit=soft temp=0.5"
case $fam in add) tr=1,2,3,4; te="5 8";; xor) tr=5,6,7,8,9,10; te="1 3";; *) echo "FAIL arg: $fam"; exit 1;; esac
BU=${BU:-4000}; AU=${AU:-1000}
K="lr=1e-3 warmup=50 decay=1 horizon=$AU every=25 evaln=1024 target=0.9 stop=1"
hit() { grep RESULT "$1" | tail -1 | grep -o 'hit_step [-0-9]*' | cut -d' ' -f2; }
for s in 110 111 112; do
  b=$O/base/s$s; mkdir -p $b
  env LAT_RULES=$tr $R text $BU 256 $s $b 4096 lr=1e-3 warmup=50 decay=1 horizon=$BU $A > $b/log 2>&1 || { echo "FAIL base $s: train"; exit 1; }
  grep -q ' np ' $b/log || { echo "FAIL base $s: readback"; exit 1; }
  echo "base $s $(grep RESULT $b/log | tail -1 | cut -d' ' -f4)"
  for r in $te; do
    env LAT_RULES=$r $R text 0 256 $s - 4096 init=$b/params.f32 $A > $b/floor$r 2>&1 || { echo "FAIL floor $s $r"; exit 1; }
    for arm in adapt scratch; do
      d=$O/$arm/r$r-s$s; mkdir -p $d; I=""; [ $arm = adapt ] && I="init=$b/params.f32"
      env LAT_RULES=$r $R text $AU 256 $s - 4096 $I $K $A > $d/log 2>&1 || { echo "FAIL $arm $s $r: train"; exit 1; }
      echo "$s r$r $arm hit_step $(hit $d/log) acc $(grep RESULT $d/log | tail -1 | cut -d' ' -f4) floor $(grep RESULT $b/floor$r | tail -1 | cut -d' ' -f4)"
    done
  done
done
echo DONE
