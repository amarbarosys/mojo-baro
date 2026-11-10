#!/usr/bin/env bash
# usage: bench/ab-prompts.sh ENGINE OUTDIR "ENV_A" "ENV_B" [label_A label_B]
#   e.g. bench/ab-prompts.sh ./.work/engine .work/ab "BARO_MEGA=0" "BARO_MEGA=1"
#        bench/ab-prompts.sh ./.work/engine .work/ab "BARO_SPEC=1 BARO_MEGA_WIN=0" "BARO_SPEC=1 BARO_MEGA_WIN=1"
# PROTOCOL-RULES P4 receipt for any two arms: every bench/mtp-prompts/*.tokens, A then B
# back to back (same stint), identity = B GENERATED == A GENERATED, per-arm median and
# spread, power cap / voltage offset read back first (P1). Wrap in bench/clock-probe.sh
# for the clock receipt. AB_ENGINE_B=path runs arm B on a second binary (same env
# strings), for whole-kernel A/B of two builds on the real pack.
# Kill line (gpu-lane-budget, 2026-09-24): after AB_KILL_AT prompts (default 5) the running
# B/A median ratio under AB_KILL_RATIO (default 0.95) or any identity FAIL stops the run,
# exit 3, results so far kept. A killed run is not a 20-prompt receipt. AB_KILL_AT=0 disables.
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2; envA=$3; envB=$4; la=${5:-A}; lb=${6:-B}; engB=${AB_ENGINE_B:-$eng}
mkdir -p "$out"
ha=$(sha256sum "$eng" | cut -c1-16); hb=$(sha256sum "$engB" | cut -c1-16)
if [ "$ha" = "$hb" ] && [ "$envA" = "$envB" ]; then
  echo "REFUSED: engA and engB are the same binary ($ha) with identical env -- this is champion vs itself (ledger 2026-09-08). Set AB_ENGINE_B or differ the env strings." >&2; exit 2
fi
echo "engA=$eng engB=$engB shaA=$ha shaB=$hb" | tee "$out/arm.txt"
echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1) armA='$envA' armB='$envB'" | tee -a "$out/arm.txt"
echo "prompt n_prompt ${la}_tok_s ${lb}_tok_s identity ${la}_gen ${lb}_gen ${la}_fail ${lb}_fail" > "$out/results.txt"
for tf in ${AB_PROMPTS:-bench/mtp-prompts/p*.tokens}; do   # AB_PROMPTS="bench/mtp-prompts/p0[1-3].tokens" for a QUICK run
  p=$(basename "$tf" .tokens); n=$(wc -w < "$tf")
  env BARO_PROMPT="$tf" $envA "$eng" > "$out/$p.$la.log" 2>&1
  env BARO_PROMPT="$tf" $envB "$engB" > "$out/$p.$lb.log" 2>&1
  ga=$(grep '^GENERATED' "$out/$p.$la.log"); gb=$(grep '^GENERATED' "$out/$p.$lb.log")
  ta=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.$la.log" | cut -d' ' -f2); tb=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.$lb.log" | cut -d' ' -f2)
  [ -n "$gb" ] && [ "$ga" = "$gb" ] && id=PASS || id=FAIL
  ga_=$(grep -oE 'mega barrier gen: [0-9]+' "$out/$p.$la.log" | grep -oE '[0-9]+$'); gb_=$(grep -oE 'mega barrier gen: [0-9]+' "$out/$p.$lb.log" | grep -oE '[0-9]+$')
  fa_=$(grep -oE 'mega fail word: [0-9]+' "$out/$p.$la.log" | grep -oE '[0-9]+$'); fb_=$(grep -oE 'mega fail word: [0-9]+' "$out/$p.$lb.log" | grep -oE '[0-9]+$')
  echo "$p $n ${ta:-nan} ${tb:-nan} $id ${ga_:--} ${gb_:--} ${fa_:--} ${fb_:--}" >> "$out/results.txt"
  kill_at=${AB_KILL_AT:-5}; done_n=$(( $(wc -l < "$out/results.txt") - 1 ))
  if [ "$kill_at" -gt 0 ]; then
    if [ "$id" != PASS ]; then echo "KILLED after $done_n prompts: identity FAIL on $p (the rest cannot be a receipt)" | tee -a "$out/results.txt"; exit 3; fi
    if [ "$done_n" -ge "$kill_at" ]; then
      ratio=$(python3 -c "import sys,statistics as st; r=[l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]; a=[float(x[2]) for x in r]; b=[float(x[3]) for x in r]; print(f'{st.median(b)/st.median(a):.3f}')" "$out/results.txt")
      if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$ratio" "${AB_KILL_RATIO:-0.95}"; then
        echo "KILLED after $done_n prompts: B/A median ratio $ratio < ${AB_KILL_RATIO:-0.95} (kill line, gpu-lane-budget)" | tee -a "$out/results.txt"; exit 3
      fi
    fi
  fi
done
column -t "$out/results.txt"
python3 - "$out/results.txt" "$la" "$lb" <<'PY'
import sys, statistics as st
rows=[l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
a=[float(r[2]) for r in rows]; b=[float(r[3]) for r in rows]
fails=[r[0] for r in rows if r[4]!="PASS"]
sp=lambda x:(max(x)-min(x))/st.median(x)*100
print(f"{sys.argv[2]} median {st.median(a):.2f} tok/s_gen spread {sp(a):.1f}%  |  {sys.argv[3]} median {st.median(b):.2f} spread {sp(b):.1f}%  |  ratio {st.median(b)/st.median(a):.3f}  |  identity fails: {fails or 'none'}")
PY
