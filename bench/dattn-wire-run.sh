#!/usr/bin/env bash
# Wiring stint of bench/dattn-wire-protocol.md, ONE gpu-wait job, fail-closed at each gate.
#   base = .work/engine-base (prebuilt from main's engine sources; sha recorded)
#   new  = .work/engine, rebuilt from this tree by tools/mega-gate.sh inside the job
# 1. mega-gate on new: build, kernel, tests, mega == launch identity on every pack, q8 / q4 vs
#    reference tokens (default path below the split threshold must stay bit-identical)
# 2. split-path identity on new: BARO_ATT_SPLIT=1, BARO_MEGA=1 vs 0, p0512 and p8192, GENERATED equal
# 3. short context: bench/ab-prompts.sh base vs new on the 20 prompts (identity + median, P4)
# 4. long context: p8192 and p32768, REPS alternating runs per arm (base, new), plus one exact-attention
#    run on base (split threshold above T) as the correctness reference; GENERATED, tok/s_gen, fail word
# usage: bench/dattn-wire-run.sh [OUT_DIR] [REPS]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-.work/dattn-wire/$(date +%F-%H%M%S)}"); REPS=${2:-3}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 16 --timeout 7200 -- "$PWD/bench/dattn-wire-run.sh" "$OUT" "$REPS"
fi
mkdir -p "$OUT"; S="$OUT/SUMMARY.txt"; : > "$S"
say() { echo "$*" | tee -a "$S"; }
die() { say "ABORT: $*"; exit 1; }
{ date -Is; git rev-parse --short HEAD; git status --short | head -5
  awk '{printf "power cap %d W\n", $1/1e6}' /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1; } > "$OUT/gpu-state.txt" 2>&1
[ -x .work/engine-base ] || die "no .work/engine-base"
tools/mega-gate.sh "$OUT/mega-gate" > "$OUT/mega-gate.log" 2>&1
cat "$OUT/mega-gate/SUMMARY.txt" >> "$S"
grep -q "^FAIL" "$OUT/mega-gate/SUMMARY.txt" && die "mega-gate"
grep -q "ALL PASS" "$OUT/mega-gate/SUMMARY.txt" || die "mega-gate did not finish"
say "engines: base $(sha256sum .work/engine-base | cut -c1-16) new $(sha256sum .work/engine | cut -c1-16)"
run() {  # label engine env... -> OUT/label.log ; prints GENERATED-hash tok/s fail-word
  local lab=$1 eng=$2; shift 2
  env "$@" "$eng" > "$OUT/$lab.log" 2>&1
  local fw; fw=$(grep -oE 'mega fail word: [0-9]+' "$OUT/$lab.log" | tail -1 | grep -oE '[0-9]+$')
  printf "%s tok/s_gen %s fail %s att_split %s TMAX %s gen %s\n" "$lab" \
    "$(grep -oE 'tok/s_gen: [0-9.]+' "$OUT/$lab.log" | tail -1 | cut -d' ' -f2)" "${fw:-none}" \
    "$(grep -oE 'att split: [0-9]+' "$OUT/$lab.log" | cut -d' ' -f3)" "$(grep -oE 'TMAX: [0-9]+' "$OUT/$lab.log" | cut -d' ' -f2)" \
    "$(grep -m1 '^GENERATED' "$OUT/$lab.log" | sha256sum | cut -c1-12)" | tee -a "$S"
}
gen() { grep -m1 '^GENERATED' "$OUT/$1.log" | cut -d: -f2-; }
agree() {  # common prefix length of two GENERATED lines, in tokens
  python3 - "$(gen "$1")" "$(gen "$2")" <<'PY'
import sys
a, b = sys.argv[1].split(), sys.argv[2].split()
n = 0
while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
print(f"{n}/{max(len(a), len(b))}")
PY
}
PK=.work/engine-pack-q4
for p in p0512 p8192; do
  run "splitid.$p.mega" .work/engine BARO_PACK=$PK BARO_PROMPT=bench/prefill-prompts/$p.tokens BARO_TMAX=9216 BARO_ATT_SPLIT=1 BARO_MEGA=1
  run "splitid.$p.launch" .work/engine BARO_PACK=$PK BARO_PROMPT=bench/prefill-prompts/$p.tokens BARO_TMAX=9216 BARO_ATT_SPLIT=1 BARO_MEGA=0
  a=$(agree "splitid.$p.mega" "splitid.$p.launch"); say "split identity $p mega vs launch: $a"
  case $a in 0/*|"") die "split identity $p produced no tokens";; esac
  [ "${a%/*}" = "${a#*/}" ] || die "split identity $p mega != launch ($a)"
done
AB_ENGINE_B=.work/engine bench/ab-prompts.sh .work/engine-base "$OUT/ab20" "BARO_PACK=$PK" "BARO_PACK=$PK" base dattn > "$OUT/ab20.log" 2>&1
say "20-prompt: $(tail -n1 "$OUT/ab20.log")"; head -1 "$OUT/ab20/arm.txt" >> "$S"
for p in p8192 p32768; do
  E="BARO_PACK=$PK BARO_PROMPT=bench/prefill-prompts/$p.tokens BARO_TMAX=33792 BARO_PREFILL_C=1024"
  for r in $(seq 1 "$REPS"); do
    if [ $((r % 2)) -eq 1 ]; then o="base new"; else o="new base"; fi
    for a in $o; do
      eng=.work/engine; [ "$a" = base ] && eng=.work/engine-base
      run "long.$p.$a.r$r" "$eng" $E
    done
  done
  run "long.$p.exact" .work/engine-base $E BARO_ATT_SPLIT_T=1000000
  say "agreement $p: base-vs-new $(agree long.$p.base.r1 long.$p.new.r1)  base-vs-exact $(agree long.$p.base.r1 long.$p.exact)  new-vs-exact $(agree long.$p.new.r1 long.$p.exact)"
done
python3 - "$S" <<'PY' | tee -a "$S"
import re, sys, statistics as st
rows = {}
for l in open(sys.argv[1]):
    m = re.match(r"long\.(p\d+)\.(base|new)\.r\d+ tok/s_gen ([0-9.]+)", l)
    if m: rows.setdefault((m[1], m[2]), []).append(float(m[3]))
for p in ("p8192", "p32768"):
    b, n = rows.get((p, "base"), []), rows.get((p, "new"), [])
    if b and n:
        sp = lambda x: 100 * (max(x) - min(x)) / st.median(x)
        print(f"long {p}: base median {st.median(b):.2f} (spread {sp(b):.1f} %)  new median {st.median(n):.2f} (spread {sp(n):.1f} %)  ratio {st.median(n)/st.median(b):.3f}")
PY
say "DONE"
