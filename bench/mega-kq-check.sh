#!/usr/bin/env bash
# usage: bench/mega-kq-check.sh ENGINE KQ_PACK OUTDIR
# Megakernel K-quant dot: on an engine-pack --kq trunk the m=1 megakernel (BARO_MEGA=1) must generate the
# same ids as the launch path (BARO_MEGA=0), same engine, all 20 mtp-prompts (tools/mega-gate.sh identity
# stage, for the kq pack). One gpu-wait job.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 kq=$2 out=$3
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 2400 -- "$0" "$@"
fi
mkdir -p "$out"
echo "eng=$(sha256sum "$eng" | cut -c1-16) pack=$kq load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
run() { # prompt log mega
  local tf=$1 log=$2 mg=$3 want=False
  [ "$mg" = 1 ] && want=True
  env BARO_MEGA="$mg" BARO_PACK="$kq" BARO_SPEC=0 BARO_TMAX=16384 BARO_PROMPT="$tf" "$eng" > "$log" 2>&1 || { echo "FAIL run $log: engine exit"; tail -3 "$log"; exit 1; }
  grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL run $log: TMAX read-back"; exit 1; }
  grep -q 'pack kq trunk: True' "$log" || { echo "FAIL run $log: not a kq trunk"; exit 1; }
  grep -q "BARO_MEGA: $want" "$log" || { echo "FAIL run $log: BARO_MEGA read-back, want $want"; exit 1; }
  grep -q '^GENERATED' "$log" || { echo "FAIL run $log: no GENERATED line"; exit 1; }
}
same=0 n=0
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens)
  run "$tf" "$out/$p.mega0.log" 0
  run "$tf" "$out/$p.mega1.log" 1
  n=$((n + 1))
  if [ "$(grep '^GENERATED' "$out/$p.mega0.log")" = "$(grep '^GENERATED' "$out/$p.mega1.log")" ]; then same=$((same + 1)); else echo "DIFF $p"; fi
  echo "$p launch $(grep -m1 -o 'tok/s_gen[^0-9]*[0-9.]*' "$out/$p.mega0.log" || echo '?') mega $(grep -m1 -o 'tok/s_gen[^0-9]*[0-9.]*' "$out/$p.mega1.log" || echo '?')"
done
echo "M: generated ids equal $same/$n"
[ "$same" = "$n" ] || { echo "FAIL M: $same/$n"; exit 1; }
echo "PASS M"
