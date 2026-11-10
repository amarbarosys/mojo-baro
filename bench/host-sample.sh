#!/usr/bin/env bash
# usage: bench/host-sample.sh OUT.csv [TOP_EVERY_S]   (runs until killed; start it in the background of a GPU job)
# Per second: host CPU busy % and iowait % (from /proc/stat deltas), runnable tasks, load average,
# MemAvailable. Every TOP_EVERY_S seconds (default 2) the three busiest processes over a 0.5 s top
# window go to OUT.top. Written for the int8 speed A/B (2026-10-12): an arm ran at half speed with
# identical engine work and nothing on record said what the host was doing (bench/int8-moe-protocol.md).
# GPU side: pair it with ~/iTools/gpu-telemetry/gpu-live-log/gpu-live-log.sh 1 OUT.gpu.csv.
set -euo pipefail
out=${1:?usage: host-sample.sh OUT.csv [TOP_EVERY_S]}; every=${2:-2}
echo "ts,busy_pct,iowait_pct,procs_running,load1,mem_avail_gb" > "$out"
read -r _ u n s i w q sq st _ < /proc/stat; prev_t=$((u + n + s + i + w + q + sq + st)); prev_i=$i; prev_w=$w; k=0
while :; do
  sleep 1
  read -r _ u n s i w q sq st _ < /proc/stat; t=$((u + n + s + i + w + q + sq + st)); dt=$((t - prev_t))
  [ "$dt" -gt 0 ] || continue
  busy=$(( 100 * (dt - (i - prev_i) - (w - prev_w)) / dt )); iow=$(( 100 * (w - prev_w) / dt ))
  prev_t=$t; prev_i=$i; prev_w=$w
  run=$(awk '/^procs_running/{print $2}' /proc/stat); load=$(cut -d' ' -f1 /proc/loadavg)
  mem=$(awk '/^MemAvailable/{printf "%.1f", $2 / 1048576}' /proc/meminfo)
  echo "$(date +%H:%M:%S),$busy,$iow,$run,$load,$mem" >> "$out"
  k=$((k + 1))
  if [ $((k % every)) -eq 0 ]; then
    { echo "== $(date +%H:%M:%S)"; top -b -n 2 -d 0.5 -w 160 -o %CPU | awk '/^ *PID/{f++} f==2' | sed -n 2,4p; } >> "${out%.csv}.top"
  fi
done
