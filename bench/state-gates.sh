#!/usr/bin/env bash
# usage: bench/state-gates.sh ENGINE PACK OUTDIR [PORT]
#
# The state-layer gates in one GPU job, in order: checkpoint-api (28 checks,
# BAROST02 default), ckpt-share (S1, 3 items), ckpt-s4 (S4, 3 items + the
# 28-check gate in both formats). Each gate gets its own OUTDIR/<gate> and its
# own port; each kills its servers on exit. Outside gpu-wait this script
# queues itself. Exit 0 only when every gate printed its PASS line; a gate
# that fails stops the run there (its receipt names the failure).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$(realpath "$1"); pack=$(realpath "$2"); out=$(realpath -m "$3"); port=${4:-8120}
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$eng" ] || fail setup "no engine at $eng"
[ -d "$pack" ] || fail setup "no pack at $pack"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  mkdir -p "$out"
  # gpu-wait carries no environment: knobs the gates read go in by name.
  exec gpu-wait run --priority "${STATE_GATES_PRIORITY:-20}" --vram 22 --timeout "${STATE_GATES_TIMEOUT:-420}" -- env ${CKPT_S4_FORMATS:+CKPT_S4_FORMATS="$CKPT_S4_FORMATS"} "$0" "$@"
fi
mkdir -p "$out"
export LC_ALL=C
echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | head -c 80)' engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) pack=$pack port=$port job=${GPU_WAITING_ROOM_JOB:-}" | tee "$out/arm.txt"
n=0
for g in checkpoint-api ckpt-share ckpt-s4; do
  n=$((n + 1)); t0=${EPOCHREALTIME%.*}
  if bench/$g.sh "$eng" "$pack" "$out/$g" $((port + n)) > "$out/$g.log" 2>&1; then
    echo "PASS $g: $(( ${EPOCHREALTIME%.*} - t0 )) s, log $out/$g.log" | tee -a "$out/arm.txt"
  else
    echo "FAIL $g: rc=$? after $(( ${EPOCHREALTIME%.*} - t0 )) s, log $out/$g.log" | tee -a "$out/arm.txt"
    tail -5 "$out/$g.log"; exit 1
  fi
done
echo "PASS state-gates: checkpoint-api, ckpt-share, ckpt-s4" | tee -a "$out/arm.txt"
