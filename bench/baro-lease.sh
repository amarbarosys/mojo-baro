#!/usr/bin/env bash
# Resident lease for bench/regescore-blender-ab.sh: one baro-serve that consecutive A/B
# runs share, so each run skips the ~40 s pack load and ~12 s warm.
#
#   bench/baro-lease.sh start  DIR   # queue the server, return once its /health answers
#   bench/baro-lease.sh status DIR   # /health + the arm identity
#   bench/baro-lease.sh stop   DIR   # SIGTERM the lease's process group; the lease job ends with it
#   gpu-wait run --shared --vram 0.5 --priority 90 --timeout 300 -- env LEASE=DIR/lease.json MEGA=1 ... \
#     bench/regescore-blender-ab.sh baro OUT
#
# Declared VRAM must fit the daemon's budget (vram_total - 512 MiB = 22.02 GiB here): the lease
# declares its measured 21.3 GiB (run 9: 23230 MiB used beside a 1364 MiB desktop), a run 0.5;
# 22 + 1 would never co-admit. A run queues at priority 90: the daemon admits only the head of
# the queue, so any waiting job that cannot fit beside the lease would leave the card held idle.
#
# start queues `gpu-wait run --shared --vram LEASE_VRAM (21.3) --timeout LEASE_TIMEOUT (1800)`
# running the A/B script in serve-only mode (LEASE_SERVE=1): same fit check, launch, P1
# read-back, spill watch and fail word as a normal run, then it writes DIR/lease.json and
# serves until stop. The A/B knobs (ENGINE MEGA MEM_PCT WARM PF_CHUNK TIER_CAP PERSONA
# PERSONA_SHA RESIDENT_NEED_MB FIT_MARGIN_MB, FAKE_OUT for the stub) are passed into the job,
# because gpu-wait run drops the environment; LEASE_KIND=baro-tier leases the tier arm.
# DRY=1 runs the job without gpu-wait (bench/stub-engine.py dry run). job.log ends "lease job exit N".
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cmd=${1:?usage: baro-lease.sh start|status|stop DIR}
dir=$(realpath -m "${2:?usage: baro-lease.sh start|status|stop DIR}")
lease=$dir/lease.json
live() { [[ -f $lease ]] && kill -0 "$(jq -r .pid "$lease")" 2>/dev/null; }
case $cmd in
start)
  ! live || { echo "FAIL lease: $lease is live, stop it first"; exit 1; }
  mkdir -p "$dir"; rm -f "$lease" "$dir/STOPPED" "$dir/SPILL"
  knobs=(LEASE_SERVE=1)
  for k in ENGINE MEGA MEM_PCT WARM PF_CHUNK TIER_CAP PERSONA PERSONA_SHA RESIDENT_NEED_MB FIT_MARGIN_MB FAKE_OUT; do
    [[ -z ${!k:-} ]] || knobs+=("$k=${!k}")
  done
  q=(gpu-wait run --shared --vram "${LEASE_VRAM:-21.3}" --timeout "${LEASE_TIMEOUT:-1800}" --)
  [[ -z ${DRY:-} ]] || q=(setsid)   # own group, as gpu-wait gives a job, so stop leaves this wrapper alive
  setsid nohup bash -c '"$@"; echo "lease job exit $?"' lease "${q[@]}" env "${knobs[@]}" \
    "$here/bench/regescore-blender-ab.sh" "${LEASE_KIND:-baro}" "$dir" > "$dir/job.log" 2>&1 < /dev/null &
  echo $! > "$dir/client.pid"
  echo "lease queued, client $!, log $dir/job.log"
  until [[ -f $lease ]]; do
    kill -0 "$(cat "$dir/client.pid")" 2>/dev/null || { echo "FAIL lease: job ended before serving: $(tail -1 "$dir/job.log")"; exit 1; }
    sleep 1
  done
  echo "lease ready: $(jq -c '{job, started_at, port, pid}' "$lease")"
  ;;
status)
  [[ -f $lease ]] || { echo "FAIL lease: no $lease"; exit 1; }
  live && curl -sf "localhost:$(jq -r .port "$lease")/health" > /dev/null \
    || { echo "FAIL lease: server $(jq -r .pid "$lease") not answering /health"; exit 1; }
  jq . "$lease"
  echo "OK lease live"
  ;;
stop)
  [[ -f $lease ]] || { echo "FAIL lease: no $lease"; exit 1; }
  pgid=$(jq -r .pgid "$lease")
  kill -TERM -- "-$pgid" 2>/dev/null || echo "lease group $pgid already gone"
  for _ in $(seq 60); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -KILL -- "-$pgid"; echo "FAIL stop: group $pgid outlived SIGTERM by 60 s, killed"; exit 1
  fi
  for _ in $(seq 30); do kill -0 "$(cat "$dir/client.pid")" 2>/dev/null || break; sleep 1; done
  echo "lease stopped: $(grep -E '^(PASS|FAIL)' "$dir/job.log" | tail -1)"
  [[ $(tail -1 "$dir/job.log") == "lease job exit 0" ]] || { echo "FAIL stop: $(tail -1 "$dir/job.log"), see $dir/job.log"; exit 1; }
  ;;
*)
  echo "FAIL usage: baro-lease.sh start|status|stop DIR"; exit 2
  ;;
esac
