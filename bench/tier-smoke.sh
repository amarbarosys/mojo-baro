#!/usr/bin/env bash
# usage: bench/tier-smoke.sh ENGINE OUT REQ.jsonl [WANT_MD5]      (bench/moe-tier-cpu-protocol.md, Round 5)
# Resident engine, one process per arm, REQ over stdin: A1 = BARO_TIER=64 pinned (bell + byte slots),
# B1 = A1 + BARO_TIER_CPU=1. Prints each arm's md5 of its GENERATED lines, the mode line, cpu_misses and
# cpu_wait_s (P1 read-back). ARMS="A1 B1" picks arms, EXTRA="K=V ..." adds env to both (BARO_TIER_STAMP=1). With WANT_MD5 the A1 md5 must equal it (CPU off must not move a token).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=${1:?engine}; out=$(realpath -m "${2:?out dir}"); req=${3:?req.jsonl}; want=${4:-}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --vram 20 --timeout 900 -- env ARMS="${ARMS:-A1 B1}" EXTRA="${EXTRA:-}" "$0" "$eng" "$out" "$req" $want
fi
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$eng" ] && [ -s "$req" ] || fail setup "engine or request missing"
mkdir -p "$out"
T=${TIER_PACK:-$HOME/.cache/baro/qwen35moe-f2628b18234dbbb4/tier-e5c7fc0d}
echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) pack $T load $(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
for arm in ${ARMS:-A1 B1}; do
  e="BARO_TIER=64 BARO_TIER_PINNED=1"; [ $arm = B1 ] && e="$e BARO_TIER_CPU=1"; e="$e ${EXTRA:-}"
  env BARO_PACK=$T BARO_SERVE=1 BARO_TMAX=8192 $e timeout 600 "$eng" < "$req" > "$out/$arm.log" 2>&1 || fail "$arm" "engine exit, see $out/$arm.log"
  grep -qE 'mega fail word: [1-9]' "$out/$arm.log" && fail "$arm" "fail word set"
  md5=$(grep ^GENERATED "$out/$arm.log" | md5sum | cut -c1-12)
  echo "$arm env='$e' ids_md5 $md5 n $(grep -c ^GENERATED "$out/$arm.log") tok/s $(grep -oE 'tok/s_gen: [0-9.]+' "$out/$arm.log" | awk '{print $2}' | tr '\n' ' ')" | tee -a "$out/arm.txt"
  grep -oE 'mode [a-z+]+( cpu_max [0-9]+)?  bell [A-Za-z]+' "$out/$arm.log" | head -1 | sed "s/^/$arm /" | tee -a "$out/arm.txt"
  grep -oE 'cpu_misses [0-9]+ +cpu_s [0-9.]+( +cpu_wait_s [0-9.]+)?' "$out/$arm.log" | tail -1 | sed "s/^/$arm /" | tee -a "$out/arm.txt"
  [ $arm = A1 ] && [ -n "$want" ] && [ "$md5" != "$want" ] && fail "A1 identity" "ids md5 $md5, want $want"
done
echo "PASS smoke"
