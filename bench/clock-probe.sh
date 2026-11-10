#!/usr/bin/env bash
# usage: [SKIP_S=<seconds>] bench/clock-probe.sh <cmd...>
# Runs <cmd> while sampling GPU 0 every ~0.2 s, then prints the busy-window (GPU use
# >= 90%) receipt: sclk min/median/max, power range, peak junction. Samples land in
# .work/clock-probe-<ts>.log. The sustained clock is an arm-defining parameter
# (PROTOCOL-RULES P1): peak FLOP/s scales with it, and WMMA load holds a lower
# clock than the fp32 kernels do at the same package power.
#
# SKIP_S (default 3.5) discards samples from the first N seconds. Warm-up runs
# the same kernel at full tilt, so its samples are "busy" too: with a 10 s
# warm-up around a 0.28 s measurement, 97% of busy samples described the
# warm-up, and warm-up clocks high off a cool card -- so sclk_med read high and
# FLOP/clk/CU (gflops / CU / sclk) consequently read low. Every Round 6/7 clock
# receipt carries that bias. SKIP_S must exceed the bench's own WARMUP_SECONDS;
# the skipped-sample count is printed so a wrong value is visible rather than
# silently inert (P1).
set -u; cd "$(dirname "$0")/.."
SKIP_S=${SKIP_S:-3.5}
CAP_W=$(awk '{printf "%d", $1/1e6}' /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
mkdir -p .work; SMI=.work/clock-probe-$(date +%Y%m%d-%H%M%S).log; : > "$SMI"
( while :; do
    l=$(rocm-smi -d 0 --showuse --showclocks --showpower --showtemp 2>/dev/null \
        | grep -E "GPU use|sclk clock level|Graphics Package Power|Sensor junction" \
        | sed -E 's/^GPU\[0\][[:space:]]*: //' | tr '\n' '|')
    echo "$(date +%s.%N)|$l" >> "$SMI"; sleep 0.2
  done ) & S=$!
"$@"; rc=$?
kill $S 2>/dev/null; wait $S 2>/dev/null
awk -F'|' -v f="$SMI" -v skip="$SKIP_S" -v cap="${CAP_W:-0}" '
  NR == 1 { t0 = $1 }
  { if ($1 - t0 < skip) { skipped++; next } }
  { u=s=p=t=""
    for (i=2; i<=NF; i++) {
      if ($i ~ /GPU use/)        { sub(/.*: /, "", $i); u=$i+0 }
      if ($i ~ /sclk/)           { match($i, /[0-9]+Mhz/); s=substr($i, RSTART, RLENGTH-3)+0 }
      if ($i ~ /Package Power/)  { sub(/.*: /, "", $i); p=$i+0 }
      if ($i ~ /junction/)       { sub(/.*: /, "", $i); t=$i+0 } }
    n++
    if (u >= 90 && s > 0) { b++; sc[b]=s; if (p>pmax) pmax=p; if (pmin=="" || p<pmin) pmin=p; if (t>tmax) tmax=t; if (cap>0 && p>cap) over++ } }
  END {
    if (!b) { printf "clock-probe: %d samples, none busy (GPU use >= 90%%) after skipping %d in first %ss -- %s\n", n, skipped+0, skip, f; exit }
    asort(sc)
    printf "clock-probe: busy %d/%d samples (skipped %d in first %ss)  sclk min/med/max %d/%d/%d MHz  power %.0f-%.0f W (cap %d W, %d busy samples above it: rocm-smi package power is instantaneous, the cap is a moving average)  junction max %.0f C  -- %s\n", b, n, skipped+0, skip, sc[1], sc[int((b+1)/2)], sc[b], pmin, pmax, cap, over+0, tmax, f }' "$SMI"
exit $rc
