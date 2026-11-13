#!/usr/bin/env bash
set -euo pipefail
pidfile=.work/gui-front.pid
[[ -s "$pidfile" ]] || { echo 'FAIL stop: engine is not owned by this face'; exit 1; }
pid=$(<"$pidfile")
[[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || { rm -f "$pidfile"; echo 'FAIL stop: stale engine pid'; exit 1; }
cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
[[ ( "$cmd" == *tools/baro* && "$cmd" == *serve* ) || "$cmd" == */baro-serve\ * ]] || { rm -f "$pidfile"; echo 'FAIL stop: pid is not tools/baro serve or baro-serve'; exit 1; }
kill -TERM -- "-$pid"
for _ in $(seq 30); do
  if ! kill -0 -- "-$pid" 2>/dev/null; then rm -f "$pidfile"; exit 0; fi
  sleep 1
done
echo 'FAIL stop: engine process group did not exit'
exit 1
