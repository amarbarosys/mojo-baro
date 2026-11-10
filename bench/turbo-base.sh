#!/usr/bin/env bash
# usage: bench/turbo-base.sh ENGINE PACK SPEC OUTDIR   (SPEC = 0|1)
# The 1-min load gate is 12 here (other lanes' gpu-room jobs are CPU-heavy and their load lingers after they end;
# the gpu room serializes them against this job): read load_max per arm in arm.txt, above 4 the arm is suspect.
# Clean decode baseline for one pack: launch (BARO_MEGA=0) vs megakernel (BARO_MEGA=1), resident arms over all 20
# mtp-prompts (bench/ab-resident.sh), spec per request as SPEC. One gpu-wait job. Read-backs added on top of
# ab-resident's: pack kind (kq trunk line matches the pack's index), TMAX 16384, spec per done line (a no-spec
# arm with a drafted count is VOID, a spec arm without one is VOID). Prints both medians and the acceptance sum.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 pack=$2 spec=$3 out=$4
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 20 --timed --timeout 1200 --label turbo-base -- "$0" "$@"
fi
[ "$spec" = 0 ] || [ "$spec" = 1 ] || { echo "FAIL args: SPEC must be 0 or 1"; exit 1; }
rs=false; [ "$spec" = 1 ] && rs=true
kq=False; grep -q ' q4_k ' "$pack/index.txt" && kq=True
mkdir -p "$out"
env BARO_PACK="$pack" BARO_TMAX=16384 SEQ=0 REQ_SPEC=$rs LOAD_START_MAX="${LOAD_START_MAX:-12}" PROMPTS='bench/mtp-prompts/p[0-9][0-9]-*.tokens' \
  bench/ab-resident.sh "$eng" "$out" "BARO_MEGA=0 BARO_SPEC=$spec" "BARO_MEGA=1 BARO_SPEC=$spec" launch mega \
  | tee "$out/ab.out" || { echo "FAIL ab-resident: see $out/ab.out"; exit 1; }
for a in launch mega; do
  log=$out/$a.log
  grep -q "pack kq trunk: $kq" "$log" || { echo "FAIL readback $a: pack kq trunk != $kq"; exit 1; }
  grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL readback $a: TMAX"; exit 1; }
  nd=$(grep -c '"done": *true' "$log" || true); ns=$(grep -c '"done": *true.*"drafted"' "$log" || true)
  if [ "$spec" = 1 ]; then [ "$ns" = "$nd" ] || { echo "FAIL readback $a: spec arm, $ns of $nd done lines carry drafted: VOID"; exit 1; }
  else [ "$ns" = 0 ] || { echo "FAIL readback $a: no-spec arm with $ns drafted lines: VOID"; exit 1; }; fi
  acc=$(grep -oE '"drafted":[0-9]+,"accepted":[0-9]+' "$log" | awk -F'[:,]' '{d+=$2; a+=$4} END {printf "drafted %d accepted %d", d, a}')
  echo "$a kq=$kq spec=$spec done=$nd $acc" | tee -a "$out/arm.txt"
done
echo "PASS turbo-base $out"
