#!/usr/bin/env bash
# Latent parity gate. `dcheck.sh prep` (CPU): oracle fixtures (DCHECK_CASES, default small; cert = certified shape) and
# test_latent binaries per width into .work/miniagi. `dcheck.sh` (GPU, under gpu-wait): run them.
# Optional run identity: put a pre-change runner at .work/miniagi/run_base and the new one at run_new.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=.work/miniagi
PY=".work/oracle-venv/bin/python -P"
DIMS=${DCHECK_DIMS:-256 512}
CASES=${DCHECK_CASES:-small}
if [ "${1:-}" = prep ]; then
  mkdir -p $W
  for d in $DIMS; do
    for c in $CASES; do
      LAT_D=$d LAT_H=8 ORACLE_CASE=$c $PY tools/miniagi_latent_oracle.py $W/case-$c-d$d > /dev/null || { echo "FAIL prep oracle $c d$d"; exit 1; }
    done
    ./.venv/bin/mojo build miniagi/latent/test_latent.mojo -D LAT_D=$d -D LAT_H=8 -I miniagi/latent -I miniagi/train -I kernels -o $W/test_latent_d$d \
      || { echo "FAIL prep build test_latent d$d"; exit 1; }
  done
  echo "prep done: $W/case-{$(echo $CASES | tr ' ' ,)}-d{$(echo $DIMS | tr ' ' ,)}, $W/test_latent_d*"
  exit 0
fi
fail=()
for d in $DIMS; do
  for c in $CASES; do
    k=$c-d$d
    [ -x $W/test_latent_d$d ] && [ -f $W/case-$k/grad.f32 ] || { fail+=("$k"); echo "FAIL $k: run 'dcheck.sh prep' first"; continue; }
    rc=0; $W/test_latent_d$d $W/case-$k $W/dump-$k > $W/dcheck-$k.log 2>&1 || rc=$?
    if [ -f $W/dump-$k.bf16 ]; then
      rc=0; grep -q -E "^FAIL.*(exact|row_weights)" $W/dcheck-$k.log && rc=1
      LAT_D=$d LAT_H=8 ORACLE_CASE=$c $PY tools/miniagi_bf16_spread.py $W/case-$k $W/dump-$k.bf16 16 > $W/dcheck-spread-$k.txt 2>&1 || rc=1
      [ "$rc" = 0 ] && echo "$k: exact PASS, bf16 inside torch draw spread ($W/dcheck-spread-$k.txt)"
    fi
    if [ "$rc" != 0 ]; then fail+=("$k"); echo "FAIL $k: $W/dcheck-$k.log"; fi
    grep -E "row_weights|loss" $W/dcheck-$k.log
  done
done
if [ -x $W/run_base ] && [ -x $W/run_new ]; then
  for arm in base new; do $W/run_$arm text 60 256 101 - 512 short=1 nobot=1 lr=3e-3 warmup=50 > $W/dcheck-run-$arm.log 2>&1; done
  if cmp -s <(grep -v -i -E "ms|wall|s/step|^params" $W/dcheck-run-base.log) <(grep -v -i -E "ms|wall|s/step|^params" $W/dcheck-run-new.log); then echo "run d256 identical to base"; else fail+=(run-identity); echo "FAIL run-identity: diff $W/dcheck-run-{base,new}.log"; fi
else
  echo "NOT EXERCISED run-identity: no $W/run_base + $W/run_new"
fi
[ ${#fail[@]} -eq 0 ] && echo PASS || { echo "FAIL ${#fail[@]} ${fail[*]}"; exit 1; }
