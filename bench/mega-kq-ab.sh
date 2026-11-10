#!/usr/bin/env bash
# usage: bench/mega-kq-ab.sh OUTDIR KQ_PACK NAME=ENGINE[:LIBDIR] NAME=ENGINE[:LIBDIR] ...
# Same-job megakernel A/B on a kq pack (bench/mega-kq-check.sh read-backs, bench/mega-g-ab.sh interleave): every
# arm runs BARO_MEGA=1 on all 20 mtp-prompts, arms interleaved per prompt, fail word read on every run, ids of every
# arm compared with the first arm. LIBDIR = modular/lib of an arm built on another toolchain (LD_LIBRARY_PATH, as
# bench/engine-ab.sh OLD_LIB). Prints the 20-prompt median tok/s_gen per arm. One gpu-wait job.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1 kq=$2; shift 2
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 3000 -- "$0" "$out" "$kq" "$@"
fi
[ $# -ge 2 ] || { echo "FAIL args: need two arms"; exit 1; }
mkdir -p "$out"; : > "$out/toks.txt"; : > "$out/arm.txt"
names=() engs=() libs=()
for a in "$@"; do
  n=${a%%=*} r=${a#*=}; e=${r%%:*} l=
  [ "$e" != "$r" ] && l=${r#*:}
  [ -x "$e" ] || { echo "FAIL args: $e not executable"; exit 1; }
  names+=("$n") engs+=("$e") libs+=("$l")
  echo "arm $n eng=$(sha256sum "$e" | cut -c1-16) lib=${l:-runpath}" | tee -a "$out/arm.txt"
done
echo "pack=$kq load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$out/arm.txt"
bad=0
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens) line=$(basename "$tf" .tokens)
  for i in "${!names[@]}"; do
    log="$out/$p.${names[$i]}.log"
    env LD_LIBRARY_PATH="${libs[$i]}" BARO_MEGA=1 BARO_PACK="$kq" BARO_SPEC=0 BARO_TMAX=16384 BARO_PROMPT="$tf" \
      "${engs[$i]}" > "$log" 2>&1 || { echo "FAIL run $log: engine exit"; tail -3 "$log"; exit 1; }
    grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL run $log: TMAX read-back"; exit 1; }
    grep -q 'pack kq trunk: True' "$log" || { echo "FAIL run $log: not a kq trunk"; exit 1; }
    grep -q 'BARO_MEGA: True' "$log" || { echo "FAIL run $log: BARO_MEGA read-back"; exit 1; }
    grep -q 'mega fail word: 0' "$log" || { echo "FAIL run $log: fail word"; exit 1; }
    grep -q '^GENERATED' "$log" || { echo "FAIL run $log: no GENERATED line"; exit 1; }
    t=$(grep -m1 -oE 'tok/s_gen: [0-9.]+' "$log" | cut -d' ' -f2)
    echo "$p ${names[$i]} $t" >> "$out/toks.txt"
    line="$line ${names[$i]} $t"
    if [ "$i" -gt 0 ] && [ "$(grep '^GENERATED' "$log")" != "$(grep '^GENERATED' "$out/$p.${names[0]}.log")" ]; then
      echo "DIFF $p ${names[$i]} vs ${names[0]}"; bad=$((bad + 1))
    fi
  done
  echo "$line"
done
python3 - "$out/toks.txt" "${names[@]}" <<'PY' | tee "$out/medians.txt"
import sys, statistics as st
rows = [l.split() for l in open(sys.argv[1])]
for n in sys.argv[2:]:
    v = [float(r[2]) for r in rows if r[1] == n]
    print(f"median {n} {st.median(v):.2f} (min {min(v):.2f} max {max(v):.2f}, n {len(v)})")
PY
[ "$bad" = 0 ] || { echo "FAIL ids: $bad runs differ from ${names[0]}"; exit 1; }
echo "PASS ids: every arm equal to ${names[0]} on every prompt"
