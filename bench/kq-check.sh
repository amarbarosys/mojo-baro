#!/usr/bin/env bash
# usage: bench/kq-check.sh OLD_ENGINE NEW_ENGINE Q8_PACK Q4_PACK KQ_PACK OUTDIR
# Run 6 check R (bench/llama-stint-protocol.md): the per-weight kind routing leaves q8 / q4 packs unchanged (generated ids
# equal, old vs new engine, launch path and default path, 5 prompts), then one kq-pack smoke run. One gpu-wait job.
set -euo pipefail
cd "$(dirname "$0")/.."
old=$1 new=$2 q8=$3 q4=$4 kq=$5 out=$6
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 1500 -- "$0" "$@"
fi
mkdir -p "$out"
echo "old=$(sha256sum "$old" | cut -c1-16) new=$(sha256sum "$new" | cut -c1-16) load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
run() { # engine pack prompt log [env...]
  local e=$1 p=$2 tf=$3 log=$4; shift 4
  env "$@" BARO_PACK="$p" BARO_SPEC=0 BARO_TMAX=16384 BARO_PROMPT="$tf" "$e" > "$log" 2>&1 || { echo "FAIL run $log: engine exit"; tail -3 "$log"; exit 1; }
  grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL run $log: TMAX read-back"; exit 1; }
  grep -q '^GENERATED' "$log" || { echo "FAIL run $log: no GENERATED line"; exit 1; }
}
same=0 n=0
for pk in q8 q4; do
  pack=$q8; [ $pk = q4 ] && pack=$q4
  for mode in mega0 default; do
    ev=(BARO_MEGA=0); [ $mode = default ] && ev=(BARO_NOOP=1)
    for tf in $(ls bench/mtp-prompts/p*.tokens | head -5); do
      p=$(basename "$tf" .tokens)
      run "$old" "$pack" "$tf" "$out/$pk.$mode.$p.old.log" "${ev[@]}"
      run "$new" "$pack" "$tf" "$out/$pk.$mode.$p.new.log" "${ev[@]}"
      n=$((n + 1))
      if [ "$(grep '^GENERATED' "$out/$pk.$mode.$p.old.log")" = "$(grep '^GENERATED' "$out/$pk.$mode.$p.new.log")" ]; then same=$((same + 1)); else echo "DIFF $pk $mode $p"; fi
    done
  done
done
echo "R: generated ids equal $same/$n"
run "$new" "$kq" bench/mtp-prompts/p01-water.tokens "$out/kq.smoke.log"
grep -E 'pack kq trunk|BARO_MEGA:|tok/s_gen' "$out/kq.smoke.log" | tr '\n' ' '; echo
[ "$same" = "$n" ] || { echo "FAIL R: $same/$n"; exit 1; }
echo "PASS R"
