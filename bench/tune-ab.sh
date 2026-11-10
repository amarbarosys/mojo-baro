#!/usr/bin/env bash
# usage: bench/tune-ab.sh ENGINE PACKDIR OUTDIR [lengths...]   (inside gpu-wait)
# Same binary, two arms: A = BARO_TUNE=0 (registry rule), B = BARO_TUNE=1 (PACKDIR/tune.txt).
# Per length: arms alternate A B A B A B, one process per run; read-back per run = the TUNE line
# (A must say rows 0, B rows > 0), prefill rows, prompt tokens, mega fail word. Identity:
# GENERATED ids equal across all six runs. Prints median prefill_s per arm and A/B.
set -euo pipefail
ENG=${1:?engine}; PACK=${2:?packdir}; OUT=${3:?outdir}; shift 3
LENS=${*:-0128 0256 0512 1024 8192}
mkdir -p "$OUT"; S=$OUT/summary.txt; : > "$S"
echo "engine_sha=$(sha256sum "$ENG" | cut -c1-16) pack=$PACK tune_sha=$(sha256sum "$PACK/tune.txt" | cut -c1-16) load=$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$S"
[ -f bench/prefill-prompts/p0256.tokens ] || { tr -s ' \n' '\n\n' < bench/prefill-prompts/p0512.tokens | grep . | head -256 | paste -sd' ' > bench/prefill-prompts/p0256.tokens; }
fails=""
for n in $LENS; do
  p=bench/prefill-prompts/p$n.tokens
  for i in $(seq 1 ${REPS:-3}); do
    for arm in A B; do
      t=$([ $arm = A ] && echo 0 || echo 1)
      log=$OUT/$n-$arm-$i.log
      env BARO_PACK="$PACK" BARO_TMAX=9216 BARO_TUNE=$t BARO_PROMPT="$p" "$ENG" > "$log" 2>&1 || { echo "FAIL run $n $arm $i: exit $? $log" | tee -a "$S"; fails="$fails $n$arm$i"; continue; }
      rows=$(grep -oE '^TUNE rows [0-9]+' "$log" | awk '{print $3}')
      { [ $arm = A ] && [ "$rows" = 0 ]; } || { [ $arm = B ] && [ "${rows:-0}" -gt 0 ]; } || { echo "FAIL readback $n $arm $i: TUNE rows '$rows'" | tee -a "$S"; fails="$fails rb$n$arm$i"; }
      grep -q '^GENERATED' "$log" || { echo "FAIL $n $arm $i: no GENERATED" | tee -a "$S"; fails="$fails gen$n$arm$i"; }
    done
  done
  ref=$(grep -h '^GENERATED' "$OUT/$n-A-1.log" || true)
  for f in "$OUT/$n"-*.log; do [ "$(grep -h '^GENERATED' "$f")" = "$ref" ] || { echo "FAIL identity $n: $f differs from A-1" | tee -a "$S"; fails="$fails id$n"; }; done
  med() { for f in "$OUT/$n-$1"-*.log; do grep -oE 'prefill_s: [0-9.e-]+' "$f" | awk '{print $2}'; done | sort -g | awk "{a[NR]=\$1} END{print a[int((NR+1)/2)]}"; }
  a=$(med A); b=$(med B)
  pairs=0; bwins=0
  for f in "$OUT/$n"-A-*.log; do i=${f##*-A-}; i=${i%.log}; pa=$(grep -oE 'prefill_s: [0-9.e-]+' "$f" | awk '{print $2}'); pb=$(grep -oE 'prefill_s: [0-9.e-]+' "$OUT/$n-B-$i.log" | awk '{print $2}'); [ -n "$pa" ] && [ -n "$pb" ] || continue; pairs=$((pairs+1)); awk -v a="$pa" -v b="$pb" 'BEGIN{exit !(b<a)}' && bwins=$((bwins+1)); done
  echo "len $n pairs $pairs B_faster $bwins" | tee -a "$S"
  echo "len $n prefill_rows $(grep -ohE 'prefill rows: [0-9]+' "$OUT/$n-B-1.log" | head -1 | awk '{print $3}') A_med $a B_med $b A/B $(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.3f", a/b}') fail_word $(grep -ohE 'mega fail word: [0-9]+' "$OUT/$n-B-1.log" | head -1)" | tee -a "$S"
done
[ -z "$fails" ] || { echo "FAIL $(echo $fails | wc -w):$fails" | tee -a "$S"; exit 1; }
echo "PASS tune-ab identity + read-back all lengths" | tee -a "$S"
