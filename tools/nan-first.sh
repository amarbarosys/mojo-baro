#!/usr/bin/env bash
# First non-finite stage per run: runs ENGINE (built from this checkout) with
# BARO_NANPROBE=1 RUNS times under gpu-wait and prints, per run, the first
# (layer, stage, row) whose non-finite count > 0 plus the stage just before it
# for that row. A position that moves between runs means a race or stray write,
# not math. Refuses an engine older than any serve/ or kernels/ source.
# usage: tools/nan-first.sh ENGINE PACKDIR PROMPT_TOKENS [RUNS] [EXTRA_ENV...]
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
eng=$(realpath "${1:?usage: nan-first.sh ENGINE PACKDIR PROMPT_TOKENS [RUNS] [ENV=V...]}")
pack=$(realpath "${2:?PACKDIR}"); prompt=$(realpath "${3:?PROMPT_TOKENS}"); runs=${4:-1}; shift $(( $# < 4 ? $# : 4 ))
[ -x "$eng" ] || { echo "FAIL nan-first: no engine $eng"; exit 1; }
newer=$(find "$here/serve" "$here/kernels" -name '*.mojo' -newer "$eng" | head -3)
[ -z "$newer" ] || { echo "FAIL nan-first: $eng is older than $newer (rebuild first)"; exit 1; }
out=$here/.work/nan-first; mkdir -p "$out"
for i in $(seq 1 "$runs"); do
  log=$out/run-$i.txt
  gpu-wait run --priority 90 --timeout 600 -- env BARO_NANPROBE=1 BARO_PACK="$pack" BARO_PROMPT="$prompt" "$@" "$eng" > "$log" 2>&1 \
    || echo "note: run $i exited non-zero (see $log)"
  grep -q '^NANTRACE' "$log" || { echo "FAIL nan-first: run $i printed no NANTRACE (probe not compiled in?)"; exit 1; }
  awk -v run="$i" '$1=="NANTRACE" {
      key=$4; if ($6>0 && !done) { print "run " run ": first bad layer " $2 " stage " $3 " (m " $4 ", row " $5 ", " $6 " values), previous for row: " prev[$5]; done=1 }
      prev[$5]="layer " $2 " " $3 " m " $4 }
    END { if (!done) print "run " run ": all stages finite" }' "$log"
done
