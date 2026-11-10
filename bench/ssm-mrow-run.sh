#!/usr/bin/env bash
# usage: bench/ssm-mrow-run.sh OUTDIR [PROMPT] [REPEATS]
#
# Row-scaling receipt for bench/ssm-mrow-protocol.md round 2: the dense q4
# engine at m = 1, 2, 4, 8 rows per window (BARO_SPEC_K = 0, 1, 3, 7), under
# the three profile modes (1 sub-block totals, 2 SSM stages, 4 FFN stages),
# REPEATS runs each, plus one rocprofv3 kernel trace per m. Every arm runs
# the launch path (BARO_MEGA=0): the megakernel takes m = 1 by default and
# bypasses the stage timers, which voided the first m = 1 arm. Rebuilds the
# engine and prints its sha in the same command as the runs (P1). Every
# stdout goes to OUTDIR/<mode>-m<m>-r<i>.log; bench/ssm-mrow-summarize.py
# turns them into the per-window table. Run under gpu-wait.
set -uo pipefail
cd "$(dirname "$0")/.."
out=$1; prompt=${2:-bench/mtp-prompts/p14-history.tokens}; reps=${3:-3}
mkdir -p "$out"
./.venv/bin/mojo build serve/engine.mojo -I kernels -I . -o .work/engine-mrow > "$out/build.log" 2>&1
echo "build exit: $?" | tee "$out/arm.txt"
sha256sum .work/engine-mrow | tee -a "$out/arm.txt"
echo "prompt=$prompt reps=$reps pack=.work/engine-pack-q4 git=$(git rev-parse --short HEAD)" | tee -a "$out/arm.txt"
echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap)" | tee -a "$out/arm.txt"
for mode in 2 4 1; do
  for k in 0 1 3 7; do
    m=$((k + 1))
    for i in $(seq 1 "$reps"); do
      if [ "$k" = 0 ]; then envm="BARO_SPEC=0"; else envm="BARO_SPEC=1 BARO_SPEC_K=$k"; fi
      env BARO_PACK=.work/engine-pack-q4 BARO_PROMPT="$prompt" BARO_PROFILE=$mode BARO_MEGA=0 $envm \
        .work/engine-mrow > "$out/p$mode-m$m-r$i.log" 2>&1
      echo "p$mode m$m r$i exit $?"
    done
  done
done
if command -v rocprofv3 >/dev/null; then
  for k in 0 1 3 7; do
    m=$((k + 1)); d="$out/trace-m$m"; rm -rf "$d"; mkdir -p "$d"
    if [ "$k" = 0 ]; then envm="BARO_SPEC=0"; else envm="BARO_SPEC=1 BARO_SPEC_K=$k"; fi
    env BARO_PACK=.work/engine-pack-q4 BARO_PROMPT="$prompt" BARO_MEGA=0 $envm \
      rocprofv3 --kernel-trace -f csv -d "$d" -o trace -- .work/engine-mrow > "$d/stdout.txt" 2> "$d/stderr.txt"
    echo "trace m$m exit $?"
  done
fi
echo "done $out"
