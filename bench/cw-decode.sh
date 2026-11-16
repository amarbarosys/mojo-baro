#!/usr/bin/env bash
# usage: bench/cw-decode.sh ENGINE OUT [K=V ...] : corewait --pid on a resident engine during decode only,
# plus gpu_busy_percent sampled every 50 ms in the same window. Run inside one gpu-wait job.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$(realpath "$1"); out=$(realpath -m "$2"); shift 2; mkdir -p "$out"
pack=$(realpath ../nerkyor-vs-forge/.work/nerkyor/pack-q4)
busy=$(ls /sys/class/drm/card*/device/gpu_busy_percent | head -1)
req=$(printf '{"id":1,"prompt":[%s],"n":1500,"spec":false}' "$(tr -s ' \n' ',,' < bench/mtp-prompts/p02-python-fib.tokens | sed 's/,$//')")
fifo=$out/in; rm -f "$fifo"; mkfifo "$fifo"
env BARO_PACK="$pack" BARO_SERVE=1 BARO_TMAX=16384 BARO_MEGA=0 "$@" "$eng" < "$fifo" > "$out/engine.log" 2>&1 &
epid=$!
exec 9> "$fifo"
for _ in $(seq 120); do grep -q '"ready":true' "$out/engine.log" && break; sleep 1; done
grep -q '"ready":true' "$out/engine.log" || { echo "FAIL ready: see $out/engine.log"; kill $epid; exit 1; }
echo "$req" >&9
sleep 8   # prefill done, decode running
( for _ in $(seq 200); do cat "$busy"; sleep 0.05; done > "$out/busy.txt" ) &
bpid=$!
corewait record desktop "$out/cw" --pid $epid -- sleep 10 > "$out/cw.log" 2>&1 || { echo "FAIL corewait: see $out/cw.log"; kill $epid; exit 1; }
wait $bpid || true
exec 9>&-; kill $epid 2>/dev/null || true; wait $epid 2>/dev/null || true
awk '{ s += $1; n++ } END { printf "gpu_busy_percent mean %.1f over %d samples\n", s / n, n }' "$out/busy.txt"
echo "PASS cw-decode $out"
