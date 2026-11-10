#!/usr/bin/env bash
# Build the stamped C/D2 engines from bench/carryover-stamp.py's output and run
# them behind the GPU waiting room, one job per run so each gets its own
# pre-launch queue/thermal read. Promoted from .work/carryover/run.sh
# (2026-09-15 carry-over probe, exchange/2026-09-15-carryover-probe.md).
#
# Phase R: reproduce the unstamped C/D2 tok/s (3 rounds, alternating) --
# needs pre-built .work/jsplit-review/engine-{C,D2} (bench/ssm-occupancy
# tool). Phase S: the stamped engines this tool builds (5 rounds, alternating).
#
# usage: bench/carryover-run.sh [R|S|smoke|all]
set -u
cd "$(dirname "$0")/.."
GW=${GPU_WAIT_BIN:-gpu-wait}
R=.work/carryover/runs; mkdir -p "$R"
P=bench/mtp-prompts/p09-explain-gpu.tokens
HW=/sys/class/drm/card1/device/hwmon/hwmon0

build_stamped() {
  for arm in C D2; do
    local bin=".work/carryover/engine-${arm}s"
    ./.venv/bin/mojo build ".work/carryover/src-$arm/serve/engine.mojo" -I ".work/carryover/src-$arm" -I ".work/carryover/src-$arm/kernels" \
      -o "$bin" > ".work/carryover/build-engine-${arm}s.log" 2>&1 \
      || { echo "FAIL build $arm: $(grep -m1 'error:' ".work/carryover/build-engine-${arm}s.log")"; exit 1; }
  done
}

run_one() {
  local tag=$1 bin=$2
  local q; q=$($GW list 2>&1 | head -1); [ "$q" = "(no jobs)" ] || echo "WARN queue not empty before $tag: $q"
  local g; g=$($GW gpu 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('\n')[0]); print('temp_c=%s power_w=%s busy=%s hold=%s' % (d['temp_c'], d['power_w'], d['busy_percent'], d['hold']['active']))" 2>/dev/null)
  local cap; cap=$(cat "$HW/power1_cap" 2>/dev/null || echo "?")
  local log="$R/$tag.log"
  $GW run --vram 20 -- env BARO_SPEC=1 BARO_SPEC_K=2 BARO_PROMPT=$P "$bin" > "$log" 2>&1
  local rc=$?
  echo "$tag exit=$rc bin_sha=$(sha256sum "$bin" | cut -c1-12) pre[$g cap_uw=$cap] tok_s_gen=$(grep -oE 'tok/s_gen: [0-9.]+' "$log" | cut -d' ' -f2) decode_s=$(grep -oE 'decode_s: [0-9.]+' "$log" | cut -d' ' -f2) gen_sha=$(grep '^GENERATED' "$log" | sha256sum | cut -c1-12) fail=$(grep -oE 'mega fail word: [0-9]+' "$log" | awk '{print $4}' | sort -u | tr '\n' ,) readback=[$(grep -oE '^BARO_SPEC: [A-Za-z]+|^spec k: [0-9]+|^BARO_MEGA: [A-Za-z]+|^pack q4 trunk: [A-Za-z]+|^STAMPS [0-9]+' "$log" | tr '\n' ';')]"
}

phase=${1:-all}
if [ "$phase" = R ] || [ "$phase" = all ]; then
  for i in 1 2 3; do for arm in C D2; do run_one R-$arm-$i .work/jsplit-review/engine-$arm; done; done
  echo R-DONE
fi
if [ "$phase" = S ] || [ "$phase" = all ] || [ "$phase" = smoke ]; then
  build_stamped
fi
if [ "$phase" = S ] || [ "$phase" = all ]; then
  for i in 1 2 3 4 5; do for arm in Cs D2s; do run_one S-$arm-$i .work/carryover/engine-$arm; done; done
  echo S-DONE
fi
if [ "$phase" = smoke ]; then
  run_one SMOKE-Cs .work/carryover/engine-Cs
fi
