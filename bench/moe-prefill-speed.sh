#!/usr/bin/env bash
# MOEPF gate 2 (bench/moe-prefill-protocol.md): prefill tok/s, replay (BARO_PREFILL=0) vs
# batched (BARO_PREFILL=1), same binary, at prompt lengths 1024/8192/32768.
#   bench/moe-prefill-speed.sh ENGINE PACK OUT [resident|tier]
# env: LENS="1024 8192 32768" (space-separated prompt lengths, files bench/prefill-prompts/pN.tokens);
#      REPS=3 (repeats per arm per length); REPS_REPLAY_LONG=1 (replay repeats at the longest length
#      only, ~8 min/run); QUICK=1 forces LENS="1024" REPS=1.
# One engine process per (length, arm, rep); every process's first request is a warmup
# (p0128.tokens, discarded) so the timed request never shares a process with anything but a
# warmup of different content -- see bench/moe-prefill-speed.py header for why that, plus
# BARO_CKPT=0, keeps "cached" at 0 (p32768.tokens has p8192.tokens as an exact prefix).
# PASS = every batched run's first 16 tokens equal that length's first replay run's 16 tokens,
# tmax read back from every engine's ready line matches what was computed and passed, cached==0
# and prefill_rows>0(batched)/==0(replay) on every timed request, no fail word in any log.
# Receipts: OUT/arm.txt, OUT/<mode>-<len>-<arm>-repN.log, OUT/summary.tsv, OUT/medians.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; mode=${4:-resident}
lens_in=${LENS:-"1024 8192 32768"}; reps=${REPS:-3}; reps_long=${REPS_REPLAY_LONG:-1}; quick=${QUICK:-0}
if [ "$quick" != 0 ]; then lens_in="1024"; reps=1; fi
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --timeout 5400 -- env LENS="$lens_in" REPS="$reps" REPS_REPLAY_LONG="$reps_long" QUICK="$quick" "$0" "$@"
fi
mkdir -p "$out"
bench/preflight.sh --check || { echo "FAIL preflight: tree changed since the last passing bench/preflight.sh"; exit 1; }
case "$mode" in
  resident|tier) ;;
  *) echo "FAIL args: mode '$mode' is not resident|tier"; exit 1 ;;
esac

read -r -a lens <<< "$lens_in"
maxlen=0
for L in "${lens[@]}"; do [ "$L" -gt "$maxlen" ] && maxlen=$L; done
tmax=$((maxlen + 16 + 1024))

for L in "${lens[@]}"; do
  f="bench/prefill-prompts/p${L}.tokens"
  [ -f "$f" ] || { echo "FAIL args: no prompt file $f for length $L"; exit 1; }
done
[ -f bench/prefill-prompts/p0128.tokens ] || { echo "FAIL args: warmup file bench/prefill-prompts/p0128.tokens missing"; exit 1; }

{ echo "gate=moe-prefill-speed mode=$mode lens='$lens_in' reps=$reps reps_replay_long=$reps_long quick=$quick tmax_expected=$tmax"
  echo "eng=$eng sha=$(sha256sum "$eng" | cut -c1-16)"
  echo "pack=$pack index_sha=$(sha256sum "$pack/index.txt" | cut -c1-16) pack_bytes=$(stat -c %s "$pack/pack.bin")"
  echo "pcie=$(cat /sys/bus/pci/devices/0000:00:01.1/current_link_speed)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --short | tr '\n' ';')'"; } | tee "$out/arm.txt"

python3 bench/moe-prefill-speed.py \
  --engine "$eng" --pack "$pack" --out "$out" --mode "$mode" --tmax "$tmax" \
  --lens "$lens_in" --reps "$reps" --reps-replay-long "$reps_long" \
  --warmup bench/prefill-prompts/p0128.tokens --prompts-dir bench/prefill-prompts \
  --arm-file "$out/arm.txt" \
  || { echo "FAIL driver: bench/moe-prefill-speed.py exited non-zero, see $out/*.log"; exit 1; }

echo "PASS moe-prefill-speed: see $out/summary.tsv and $out/medians.tsv"
