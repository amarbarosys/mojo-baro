#!/usr/bin/env bash
# usage: face/apply_gate.sh MODEL.gguf [OUTDIR]     (default .work/apply-gate)
# F5 step 6: Apply restarts a REAL engine. One gpu-wait job: start `tools/baro serve MODEL` at BARO_TMAX=16384 the way
# the face owns it (setsid + .work/gui-front.pid), read /health limits, then run face/apply.mojo's apply_action with
# restart_cmd(BARO_TMAX=32768) and face/stop-engine.sh, and read back: /health tmax 32768, a new pid, the old process
# group gone, and the engine log line "for TMAX 32768" (state load). Falsifier: APPLY_GATE_EXPECT=16384 must exit non-zero.
# Needs the serve cache built (tools/baro serve MODEL --no-serve, CPU). Exit non-zero on any miss: FAIL <step>: <reason>.
set -euo pipefail
fail() { echo "FAIL $1: $2"; exit 1; }
model=$(readlink -f "${1:?usage: face/apply_gate.sh MODEL.gguf [OUTDIR]}"); out=${2:-.work/apply-gate}
port=${APPLY_GATE_PORT:-18431}; t0=16384; t1=32768; expect=${APPLY_GATE_EXPECT:-$t1}
mkdir -p "$out"
if [ -z "${APPLY_GATE_IN_JOB:-}" ]; then
  tools/baro serve "$model" --no-serve > "$out/cache.log" 2>&1 || fail cache "serve cache not built, see $out/cache.log"
  grep -q "engine=hit pack=hit" "$out/cache.log" || fail cache "cache miss: build it on CPU first (tools/baro serve MODEL --no-serve)"
  ./.venv/bin/mojo build -I . -I face face/apply_gate.mojo -o .work/apply_gate > "$out/build.log" 2>&1 || fail build "see $out/build.log"
  exec gpu-wait run --vram 16 --timeout 900 -- env APPLY_GATE_IN_JOB=1 APPLY_GATE_EXPECT="$expect" bash -c "cd '$PWD' && face/apply_gate.sh '$model' '$out' > '$out/job.log' 2>&1; rc=\$?; cat '$out/job.log'; exit \$rc"
fi
health() { curl -s -m 2 "http://127.0.0.1:$port/health" || true; }
wait_up() { for _ in $(seq 180); do curl -sf -m 2 -o /dev/null "http://127.0.0.1:$port/health" && return 0; sleep 1; done; return 1; }
rm -f .work/gui-front.pid
setsid env BARO_TMAX=$t0 tools/baro serve "$model" --port $port > .work/gui-front.log 2>&1 & echo $! > .work/gui-front.pid
pid0=$(<.work/gui-front.pid)
trap 'p=$(cat .work/gui-front.pid 2>/dev/null || true); [ -n "$p" ] && kill -TERM -- "-$p" 2>/dev/null || true' EXIT
wait_up || fail start "front never answered /health (see .work/gui-front.log)"
h0=$(health); echo "before: pid $pid0 health $h0"
grep -q "\"tmax\":$t0" <<< "$h0" || fail before "/health tmax is not $t0: $h0"
cp .work/gui-front.log "$out/front-before.log"
.work/apply_gate "$model" $port $t1 | tee "$out/apply.txt" || fail apply "apply_action did not reach a healthy front"
pid1=$(<.work/gui-front.pid)
h1=$(health); echo "after: pid $pid1 health $h1"
[ "$pid1" != "$pid0" ] || fail pid "pid file still names the old front $pid0"
kill -0 -- "-$pid0" 2>/dev/null && fail old "old process group $pid0 still alive"
grep -q "\"tmax\":$expect" <<< "$h1" || fail after "/health tmax is not $expect: $h1"
for _ in $(seq 30); do grep -q "for TMAX $expect$" .work/gui-front.log && break; sleep 1; done
grep -q "for TMAX $expect$" .work/gui-front.log || fail log "engine log has no 'for TMAX $expect'"
cp .work/gui-front.log "$out/front-after.log"
echo "PASS apply gate: restart pid $pid0 -> $pid1, /health tmax $t0 -> $expect, engine log for TMAX $expect ($out)"
