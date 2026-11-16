#!/usr/bin/env bash
# usage: bench/mega-first-req.sh ENGINE PACK OUT [N] : NOT-RESIDENT repro on the plain megakernel's first request.
# N fresh resident engines per arm, one request each (p01, n 64), fail word recorded. Arms: default env, and
# HSA_NO_SCRATCH_RECLAIM=1 (ROCr keeps scratch assigned to the queue instead of use-once). Run inside one gpu-wait job.
set -euo pipefail
cd "$(dirname "$0")/.."
e=$(realpath "$1"); pack=$(realpath "$2"); out=$(realpath -m "$3"); n=${4:-4}; mkdir -p "$out"
printf '{"id":1,"prompt":[%s],"n":64,"spec":false}\n' "$(tr -s ' \n' ',,' < bench/mtp-prompts/p01-water.tokens | sed 's/,$//')" > "$out/req.jsonl"
for arm in default noreclaim; do
  ext=(); [ $arm = noreclaim ] && ext=(HSA_NO_SCRATCH_RECLAIM=1)
  for i in $(seq "$n"); do
    env BARO_PACK="$pack" BARO_SERVE=1 BARO_TMAX=16384 BARO_MEGA=1 BARO_SPEC=0 "${ext[@]}" "$e" < "$out/req.jsonl" > "$out/$arm-$i.log" 2>&1 || true
    printf '%s %s %s\n' "$arm" "$i" "$(grep -m1 -oE 'mega fail word: [0-9]+' "$out/$arm-$i.log" || echo 'no fail word line')"
  done
done | tee "$out/summary.txt"
