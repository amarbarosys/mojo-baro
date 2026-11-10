#!/usr/bin/env bash
# usage: tools/tune-family.sh GGUF PACK PROFILEDIR OUTDIR
# Tune spark's split-K knobs for one model (bench/family-gen-protocol.md). Stage 1: base + each knob moved alone, one
# binary per arm (comptime -D), built in parallel, kq-spilling builds dropped (isa-receipt). Stage 2: the per-knob winners
# combined and raced against base. One gpu-wait job: arms interleaved per prompt, BARO_GEN 64, 2 rounds over the first NP
# mtp-prompts. A knob wins on the median per-prompt ratio to base with ids equal on every prompt. Read-back: the
# "spark knobs:" line of every log against the arm's defines. Output OUTDIR/flags (empty = base), OUTDIR/summary.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
gguf=$1 pack=$2 prof=$3 out=$4; NP=${NP:-6}
fail() { echo "FAIL $1: $2"; exit 1; }
build() { # name defs...
  local n=$1; shift
  ./.venv/bin/mojo build serve/spark.mojo "$@" -I "$prof" -I . -I kernels -I serve -I tools -o "$out/bin.$n" > "$out/build.$n.log" 2>&1 \
    || { echo "FAIL build $n: $(grep -m1 error: "$out/build.$n.log" | cut -c1-160)"; return 1; }
  ./tools/isa-receipt.py "$out/bin.$n" > "$out/isa.$n.txt" 2>&1 || { echo "FAIL isa $n"; return 1; }
  if grep -E "kq.*(vgpr|sgpr)_spill_count=[1-9]" "$out/isa.$n.txt" > /dev/null; then echo "DROP $n: spills"; rm "$out/bin.$n"; fi
}
want() { # defs... -> the knobs line the binary must print
  local q=4 o=1 d=1 h=1 x
  for x in "$@"; do case $x in -DBARO_SP_QKV_S=*) q=${x#*=};; -DBARO_SP_O_S=*) o=${x#*=};; -DBARO_SP_DOWN_S=*) d=${x#*=};; -DBARO_SP_HEAD_S=*) h=${x#*=};; esac; done
  echo "spark knobs: QKV_S $q O_S $o DOWN_S $d HEAD_S $h"
}
race() { # stage name...
  local st=$1 r f n log ts pf gen; shift
  for r in 1 2; do for f in $(ls "$out"/p*.tokens | head -"$NP"); do for n in "$@"; do
    log=$out/$st.$n.$(basename "$f" .tokens).r$r.log
    env BARO_PACK="$pack" BARO_PROMPT="$f" BARO_GEN=64 BARO_SPEC=0 "$out/bin.$n" > "$log" 2>&1 || fail run "$n $f: $(tail -1 "$log")"
    grep -qx "$(want $(grep "^$n " "$out/arms.txt" | cut -d' ' -f2-))" "$log" || fail readback "$n: knobs line differs from its defines"
    ts=$(grep -oP 'tok/s_gen: \K[0-9.]+' "$log") || fail run "$n: no tok/s_gen"
    pf=$(grep -m1 -oP 'prefill_s: \K[0-9.e-]+' "$log") || fail run "$n: no prefill_s"
    gen=$(grep -oP '^generated: \K.*' "$log") || fail run "$n: no generated ids"
    echo "$st $n $(basename "$f" .tokens) $r $ts $pf $(echo "$gen" | md5sum | cut -c1-8)" >> "$out/runs.txt"
  done; done; done
}
score() { # stage name -> "median_ratio ids_same/total"
  awk -v s="$1" -v n="$2" '$1 == s { k = $3 " " $4; if ($2 == "base") { b[k] = $5; bi[k] = $7 } if ($2 == n) { c[k] = $5; ci[k] = $7 } }
    END { m = 0; same = 0; for (k in c) { r[++m] = c[k] / b[k]; same += (ci[k] == bi[k]) } asort(r)
          printf "%.4f %d/%d\n", (m % 2 ? r[(m + 1) / 2] : (r[m / 2] + r[m / 2 + 1]) / 2), same, m }' "$out/runs.txt"
}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  mkdir -p "$out"
  printf '%s\n' "base" "q2 -DBARO_SP_QKV_S=2" "q8 -DBARO_SP_QKV_S=8" "o2 -DBARO_SP_O_S=2" "o4 -DBARO_SP_O_S=4" \
    "d2 -DBARO_SP_DOWN_S=2" "d4 -DBARO_SP_DOWN_S=4" "h2 -DBARO_SP_HEAD_S=2" > "$out/arms.txt"
  pids=()
  while read -r n defs; do build "$n" $defs & pids+=($!); done < "$out/arms.txt"
  for p in "${pids[@]}"; do wait "$p" || fail build "see $out/build.*.log"; done
  { [ -x .work/baro-tokenize ] && ! ldd .work/baro-tokenize | grep -q "not found"; } \
    || ./.venv/bin/mojo build tools/baro-tokenize.mojo -I . -I serve -o .work/baro-tokenize > "$out/build-tok.log" 2>&1 || fail build "baro-tokenize"
  for f in bench/mtp-prompts/p*.txt; do ./.work/baro-tokenize encode "$f" "$gguf" | tr -s '\n' ' ' > "$out/$(basename "$f" .txt).tokens"; done
  exec gpu-wait run --vram 22 --timeout 2400 -- "$0" "$@"
fi
: > "$out/runs.txt"
echo "gguf=$gguf pack=$pack tmax=$(grep -oP '^comptime TMAX = \K[0-9]+' "$prof/profile.mojo") load=$(cut -d' ' -f1-3 /proc/loadavg)" > "$out/arm.txt"
for b in "$out"/bin.*; do echo "$(basename "$b") sha=$(sha256sum "$b" | cut -c1-16)" >> "$out/arm.txt"; done
names=$(ls "$out"/bin.* | sed 's/.*bin\.//' | tr '\n' ' ')
env BARO_PACK="$pack" BARO_PROMPT="$(ls "$out"/p*.tokens | head -1)" BARO_GEN=8 BARO_SPEC=0 "$out/bin.base" > "$out/probe.log" 2>&1 || fail run "probe: $(tail -1 "$out/probe.log")"
if grep -q "kquant 0 " "$out/probe.log"; then
  : > "$out/flags"; echo "SKIP tune: pack has no K-quant matrices (the knobs act on kq GEMVs only), flags empty" | tee -a "$out/summary.txt"; exit 0
fi
race s1 $names
defs=""
for k in q o d h; do
  best="" bv=1.0
  for n in $names; do
    [ "${n:0:1}" = "$k" ] || continue
    read -r v ids < <(score s1 "$n"); echo "s1 $n ratio $v ids $ids" | tee -a "$out/summary.txt"
    [ "${ids%/*}" = "${ids#*/}" ] && awk -v a="$v" -v b="$bv" 'BEGIN { exit !(a > b) }' && { best=$n; bv=$v; }
  done
  [ -n "$best" ] && defs="$defs $(grep "^$best " "$out/arms.txt" | cut -d' ' -f2-)"
done
echo "$defs" | xargs > "$out/flags"
if [ -n "$(cat "$out/flags")" ]; then
  echo "tuned $(cat "$out/flags")" >> "$out/arms.txt"
  ( build tuned $(cat "$out/flags") ) || fail build tuned
  race s2 base tuned
  read -r v ids < <(score s2 tuned)
  echo "s2 tuned [$(cat "$out/flags")] ratio $v ids $ids" | tee -a "$out/summary.txt"
  awk -v a="$v" 'BEGIN { exit !(a > 1.0) }' || { echo "s2: combination not faster, flags cleared" | tee -a "$out/summary.txt"; : > "$out/flags"; }
else
  echo "s2: no knob beat base, flags empty" | tee -a "$out/summary.txt"
fi
echo "PASS tune: flags [$(cat "$out/flags")]" | tee -a "$out/summary.txt"
