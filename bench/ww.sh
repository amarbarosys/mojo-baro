#!/usr/bin/env bash
# One warm SQTT capture of one kernel, then the wavewait json, classes and top stalls.
# usage: bench/ww.sh KERNEL_REGEX ENGINE PACK OUT [K=V ...]   (wraps wavewait tools/att-capture.sh, its own gpu-wait job)
# Writes OUT/{arm.txt,engine.log,ww.json,classes.txt,top.txt}. Fails when the engine or the trace fails. TMAX defaults to 16384.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 4 ] || { echo "FAIL usage: bench/ww.sh KERNEL_REGEX ENGINE PACK OUT [K=V ...]"; exit 1; }
re=$1; eng=$(realpath "$2"); pack=$(realpath "$3"); out=$(realpath -m "$4"); shift 4
lib=$(ls -d "$PWD"/.venv/lib/python*/site-packages/modular/lib); root=$(tools/wavewait --root)
rm -rf "$out"; mkdir -p "$out"
it=${WW_ITER:-12}; prompt=${WW_PROMPT:-$PWD/bench/mtp-prompts/p01-water.tokens}
echo "regex $re engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) pack $pack knobs BARO_TMAX=16384 $* wavewait $(sed -n 's/^PIN=\([^;]*\).*/\1/p' tools/wavewait) iter $it" > "$out/arm.txt"
ATT_SLIM=1 ATT_KERNEL="$re" ATT_EXTRA="--kernel-iteration-range [$it-$it]" ATT_TIMEOUT=${WW_TIMEOUT:-300} \
  "$root/tools/att-capture.sh" "$out/att" -- env LD_LIBRARY_PATH="$lib" BARO_TMAX=16384 BARO_PROFILE=5 "$@" BARO_PACK="$pack" BARO_PROMPT="$prompt" "$eng" \
  > "$out/engine.log" 2>&1 || { echo "FAIL capture: see $out/engine.log"; exit 1; }
tools/wavewait app "$out/att" > "$out/app.txt" 2>&1 || { echo "FAIL app: see $out/app.txt"; exit 1; }
d=$(awk -v r="$re" '$1 ~ r {print $NF; exit}' "$out/app.txt")
[ -n "$d" ] || { echo "FAIL app: no dispatch for $re"; exit 1; }
tools/wavewait app "$out/att" --dispatch "$d" --json > "$out/ww.json" || { echo "FAIL json"; exit 1; }
tools/wavewait app "$out/att" --dispatch "$d" --by-class > "$out/classes.txt"
tools/wavewait app "$out/att" --dispatch "$d" --causes --top 10 > "$out/top.txt"
grep -E '^(spill|barrier|vmem load) ' "$out/classes.txt"; echo "PASS ww: $out"
