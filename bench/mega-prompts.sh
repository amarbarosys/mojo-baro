#!/usr/bin/env bash
# usage: bench/mega-prompts.sh ENGINE OUTDIR
# Megakernel stage-2 receipt (PROTOCOL-RULES P4): for every bench/mtp-prompts/*.tokens
# run arm A (BARO_MEGA=0, launch path) and arm M (BARO_MEGA=1) back to back, same
# stint. Gate: M GENERATED == A GENERATED per prompt. Prints per-arm median tok/s_gen
# and spread; reads the power cap / voltage offset back first (P1 receipt).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2
mkdir -p "$out"
echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1)" | tee "$out/arm.txt"
echo "prompt n_prompt A_tok_s M_tok_s identity" > "$out/results.txt"
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens); n=$(wc -w < "$tf")
  BARO_PROMPT="$tf" BARO_MEGA=0 "$eng" > "$out/$p.A.log" 2>&1
  BARO_PROMPT="$tf" BARO_MEGA=1 "$eng" > "$out/$p.M.log" 2>&1
  ga=$(grep '^GENERATED' "$out/$p.A.log"); gm=$(grep '^GENERATED' "$out/$p.M.log")
  ta=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.A.log" | cut -d' ' -f2)
  tm=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.M.log" | cut -d' ' -f2)
  [ "$ga" = "$gm" ] && id=PASS || id=FAIL
  echo "$p $n $ta $tm $id" >> "$out/results.txt"
done
column -t "$out/results.txt"
python3 - "$out/results.txt" <<'PY'
import sys, statistics as st
rows=[l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
a=[float(r[2]) for r in rows]; m=[float(r[3]) for r in rows]
fails=[r[0] for r in rows if r[4]!="PASS"]
def spread(x): return (max(x)-min(x))/st.median(x)*100
print(f"A median {st.median(a):.2f} tok/s_gen spread {spread(a):.1f}%  |  M median {st.median(m):.2f} spread {spread(m):.1f}%  |  ratio {st.median(m)/st.median(a):.3f}  |  identity fails: {fails or 'none'}")
PY
