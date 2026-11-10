#!/usr/bin/env bash
# usage: bench/race-fp16.sh <rounds> <label=binary>...
# Interleaves the given bench binaries round-robin for <rounds> rounds (same thermal
# window for every arm), prints every gflops sample and the per-arm median.
# PROBE=1: wrap every sample in bench/clock-probe.sh (byte-for-byte, same script,
# same output), read back its sclk_med and print FLOP/clk/CU alongside gflops.
# SPREAD_MAX (default 0.02): exit non-zero if any arm's (max-min)/median exceeds
# it. A 0.28 s timed window on the card that also drives the displays gave 29%
# spread and a median dragged 10% low by compositor dropouts, which is how the
# Round 7 gap decomposition came out wrong; the spread was always printed and
# never gated. Contention only ever removes throughput, so the distribution is
# one-sided and the median is not robust to it -- gate, do not eyeball.
set -eu; cd "$(dirname "$0")/.."
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
rounds=$1; shift
probe=${PROBE:-0}
spread_max=${SPREAD_MAX:-0.02}
NUM_CU=${NUM_CU:-$(bench/gpu-info.py | python3 -c 'import json,sys; print(json.load(sys.stdin)["compute_units"])')}
[ -n "$NUM_CU" ] || { echo "race-fp16.sh: could not read compute_units from bench/gpu-info.py" >&2; exit 1; }
echo "num_cu $NUM_CU (read from GPU)  spread_max $spread_max"
declare -A samples
for ((r=1; r<=rounds; r++)); do
  for arm in "$@"; do
    label=${arm%%=*}; bin=${arm#*=}
    if [ "$probe" = "1" ]; then
      out=$(bench/clock-probe.sh "$bin")
    else
      out=$($bin)
    fi
    g=$(grep -oE '"gflops": [0-9.]+' <<<"$out" | grep -oE '[0-9.]+$'); ok=$(grep -oE '"correct": [a-z]+' <<<"$out")
    if [ "$probe" = "1" ]; then
      sclk_med=$(grep -oE 'sclk min/med/max [0-9]+/[0-9]+/[0-9]+' <<<"$out" | awk '{split($NF,a,"/"); print a[2]}')
      fpc=$(awk -v g="$g" -v s="$sclk_med" -v cu="$NUM_CU" 'BEGIN{ if (s>0) printf "%.1f", (g*1.0e9)/(cu*s*1.0e6); else print "NA" }')
      printf "round %d  %-10s %8.0f  sclk_med %sMHz  flop/clk/cu %s  %s\n" "$r" "$label" "$g" "$sclk_med" "$fpc" "$ok"
      grep -E "^clock-probe:" <<<"$out"
    else
      printf "round %d  %-10s %8.0f  %s\n" "$r" "$label" "$g" "$ok"
    fi
    samples[$label]+="$g "
  done
done
echo "--- medians over $rounds rounds"
rc=0
for arm in "$@"; do
  label=${arm%%=*}
  read -r med lo hi sp < <(tr ' ' '\n' <<<"${samples[$label]}" | grep . | sort -n \
    | awk '{a[NR]=$1} END{m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "%.0f %.0f %.0f %.4f\n", m, a[1], a[NR], (m>0)?(a[NR]-a[1])/m:9}')
  verdict=OK; awk -v s="$sp" -v x="$spread_max" 'BEGIN{exit !(s>x)}' && { verdict="SPREAD"; rc=1; }
  printf "%-10s median %8s  (min %s max %s)  spread %.2f%%  %s\n" "$label" "$med" "$lo" "$hi" \
    "$(awk -v s="$sp" 'BEGIN{print s*100}')" "$verdict"
done
[ "$rc" -eq 0 ] || echo "race-fp16.sh: spread exceeds SPREAD_MAX=$spread_max -- the instrument is too noisy to compare arms (PROTOCOL-RULES P6)" >&2
exit $rc
