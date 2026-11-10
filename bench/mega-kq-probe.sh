#!/usr/bin/env bash
# usage: bench/mega-kq-probe.sh ENGINE KQ_PACK LOG [MEGA=1] [PROMPT=p01-water]
# One-prompt megakernel K-quant probe under gpu-wait: tok/s_gen and the per-phase profile (BARO_PROFILE=5),
# with the read-backs the gate demands. Iteration only; the claim is bench/mega-kq-check.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 kq=$2 log=$3 mg=${4:-1} p=${5:-p01-water}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 300 -- "$0" "$@"
fi
want=False; [ "$mg" = 1 ] && want=True
env BARO_MEGA="$mg" BARO_PACK="$kq" BARO_SPEC=0 BARO_TMAX=16384 BARO_PROFILE=5 BARO_PROMPT="bench/mtp-prompts/$p.tokens" "$eng" > "$log" 2>&1 || { echo "FAIL probe: engine exit"; tail -3 "$log"; exit 1; }
grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL probe: TMAX read-back"; exit 1; }
grep -q 'pack kq trunk: True' "$log" || { echo "FAIL probe: not a kq trunk"; exit 1; }
grep -q "BARO_MEGA: $want" "$log" || { echo "FAIL probe: BARO_MEGA read-back, want $want"; exit 1; }
grep -q '^GENERATED' "$log" || { echo "FAIL probe: no GENERATED line"; exit 1; }
echo "eng=$(sha256sum "$eng" | cut -c1-16) mega=$mg prompt=$p"
grep -E 'tok/s_gen|mega profile|mega phases|mega fail word' "$log"
echo "ids md5 $(grep '^GENERATED' "$log" | md5sum | cut -c1-12)"
