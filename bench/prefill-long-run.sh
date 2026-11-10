#!/usr/bin/env bash
# Prefill-long candidate run: 3 timed runs + 1 BARO_PROFILE=1 split per length,
# greedy parity of GENERATED (64 tokens) against the champion build's stage-0
# log at the same length. Every run through gpu-wait at priority 20.
# usage: bench/prefill-long-run.sh <engine> <tag> [lengths...]
set -u
cd "$(dirname "$0")/.."
ENG=${1:?engine}; TAG=${2:?tag}; shift 2
LENS=${*:-8192 16384 32768}
B=.work/logs/stage0
L=.work/logs/$TAG; mkdir -p "$L"; S=$L/summary.txt
E="env BARO_PACK=.work/engine-pack-q4 BARO_TMAX=32896"
echo "run $TAG $(date -Is) engine_sha=$(sha256sum "$ENG" | cut -c1-16) power_cap_W=$(awk '{printf "%d", $1/1e6}' /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)" >> "$S"
rcpt() { grep -hoE 'prompt tokens: [0-9]+|pack q4 trunk: [A-Za-z]+|BARO_MEGA: [A-Za-z]+|BARO_SPEC: [A-Za-z]+|TMAX: [0-9]+|prefill rows: [0-9]+|prefill_s: [0-9.]+|tok/s_gen: [0-9.]+|mega fail word: [0-9]+' "$1" | tr '\n' ' '; }
parity() {
  local a b; a=$(grep '^GENERATED' "$1"); b=$(grep '^GENERATED' "$2")
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "parity PASS 64/64"; return; fi
  python3 - "$1" "$2" <<'EOF'
import sys
g = [open(p).read().split('GENERATED:')[-1].split('\n')[0].split() for p in sys.argv[1:3]]
n = next((i for i, (x, y) in enumerate(zip(*g)) if x != y), min(map(len, g)))
print(f"parity FAIL first divergence at generated token {n} of {min(map(len, g))}")
EOF
}
for n in $LENS; do
  p=bench/prefill-prompts/p$(printf '%04d' "$n").tokens
  for i in 1 2 3; do
    gpu-wait list > "$L/q-$n-$i.txt" 2>&1
    gpu-wait run --priority 20 --vram 12 -- $E BARO_PROMPT="$p" "$ENG" > "$L/run-$n-$i.log" 2>&1
    echo "$TAG $n run$i exit $? $(rcpt "$L/run-$n-$i.log") $(parity "$L/run-$n-$i.log" "$B/base-$n-1.log")" >> "$S"
  done
  gpu-wait list > "$L/q-prof-$n.txt" 2>&1
  gpu-wait run --priority 20 --vram 12 -- $E BARO_PROFILE=1 BARO_PROMPT="$p" "$ENG" > "$L/prof-$n.log" 2>&1
  echo "$TAG prof $n exit $? $(grep -h 'prefill split' "$L/prof-$n.log")" >> "$S"
done
echo "RUN DONE $TAG $(date -Is)" >> "$S"
