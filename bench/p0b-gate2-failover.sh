#!/usr/bin/env bash
# P0b gate 2 (docs/PLATFORM-PLAN.md): kill one engine mid-run: requests in
# flight on it fail loudly, new ones route to the other; receipt in the
# catalog.
#
# Same two-engine rig as gate 1 (bench/p0b-gate1-placement.sh):
# BARO_TMAX=4096, MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10 --
# see that script's header for the VRAM numbers (lane FORK,
# docs/P1-FORK-TARGET.md) and why 32k is out of reach on this box.
#
# Known limitation, honest: the workload catalog logs a request "done" as
# soon as the upstream engine's response HEADERS arrive and streaming to the
# downstream client begins (serve/src/bin/router.rs::proxy), not when the
# stream finishes. A request killed mid-STREAM (after headers, during body
# transfer) already has a "done" row by the time it breaks -- the catalog
# does not retroactively mark it failed. This gate proves the client-visible
# half honestly instead: the downstream client's own transfer breaks loudly
# (curl reports a transfer error), never hangs silently, and a genuinely
# NEW request after the kill is both served correctly and correctly logged
# against the survivor.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/p0b-gate2}
engine=${BARO_ENGINE:-.work/engine}
pack=${BARO_PACK:-.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-.work/team-A/sonnet/target/release/baro-serve}
router_bin=${ROUTER_BIN:-.work/team-A/sonnet/target/release/router}
mkdir -p "$out"
exec > >(tee "$out/gate2.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
for f in "$engine" "$serve" "$router_bin" "$pack/pack.bin"; do [ -e "$f" ] || fail setup "missing $f"; done
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"

echo "engine sha256=$(sha256sum "$engine" | cut -d' ' -f1)"

pids=()
cleanup() {
  for p in "${pids[@]}"; do pkill -TERM -P "$p" 2>/dev/null || true; kill -TERM "$p" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && return 0
    sleep 0.5
  done
  for p in "${pids[@]}"; do pkill -KILL -P "$p" 2>/dev/null || true; kill -KILL "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

# NAME; sets $LAST_URL and $LAST_PID. Never call inside a command
# substitution -- see bench/p0b-gate1-placement.sh's header for why.
start_engine() {
  local name="$1"
  env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10 BARO_TMAX=4096 BARO_SPEC=0 BARO_CKPT_DIR="$out/ckpts-$name" \
    "$serve" --engine "$engine" --pack "$pack" --port 0 > "$out/$name.stdout" 2> "$out/$name.stderr" &
  LAST_PID=$!
  pids+=("$LAST_PID")
  for _ in $(seq 1 600); do
    grep -q '^listening on' "$out/$name.stdout" && break
    kill -0 "$LAST_PID" 2>/dev/null || fail "start-$name" "engine exited: $(tail -3 "$out/$name.stderr" | tr '\n' ' ')"
    sleep 0.5
  done
  LAST_URL=$(grep -m1 -oE 'http://[0-9.:]+' "$out/$name.stdout") || fail "start-$name" "no listening line in 300 s"
}

start_engine a; url_a="$LAST_URL"; pid_a="$LAST_PID"
start_engine b; url_b="$LAST_URL"; pid_b="$LAST_PID"
port_a=$(python3 -c "import sys,urllib.parse; print(urllib.parse.urlsplit(sys.argv[1]).port)" "$url_a")
port_b=$(python3 -c "import sys,urllib.parse; print(urllib.parse.urlsplit(sys.argv[1]).port)" "$url_b")
echo "engine a=$url_a (pid $pid_a) engine b=$url_b (pid $pid_b)"

router_port=18320
BARO_ROUTER_ENGINES="a=http://127.0.0.1:${port_a},b=http://127.0.0.1:${port_b}" \
  "$router_bin" --host 127.0.0.1 --port "$router_port" --node-id 00000000-0000-4000-8000-0000000000c2 \
  > "$out/router.stdout" 2> "$out/router.stderr" &
router_pid=$!
pids+=("$router_pid")
for _ in $(seq 1 60); do
  grep -q '^listening on' "$out/router.stdout" && break
  kill -0 "$router_pid" 2>/dev/null || fail start-router "router exited: $(tail -5 "$out/router.stderr" | tr '\n' ' ')"
  sleep 0.5
done
router_url="http://127.0.0.1:${router_port}"

sleep 12  # probe_loop: 2 passes at a 5 s interval to mark both healthy
curl -fsS "$router_url/v1/node-info" > "$out/node-info-before.json" || fail health "router node-info failed"
python3 - "$out/node-info-before.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
for eid in ("a", "b"):
    e = next((x for x in info["engines"] if x["id"] == eid), None)
    assert e is not None and e["healthy"] is True, (eid, e)
print("node-info OK: both engines healthy before the kill")
PY

# ---- an in-flight request on "a" (both engines idle, tied pending -> the
# lexicographically-first id "a" wins choose()'s tie-break, matching
# router.rs::EngineRegistry::choose's own (pending, !preferred, id) key)

long_body='{"prompt":"Once upon a time, in a land far away,","max_tokens":300,"temperature":0.0,"stream":true}'
curl -sS -N -X POST "$router_url/v1/completions" -H 'content-type: application/json' -d "$long_body" \
  > "$out/inflight.sse" 2> "$out/inflight.curl.stderr" &
inflight_pid=$!
pids+=("$inflight_pid")

# Let it connect and receive real SSE frames before killing its engine.
for _ in $(seq 1 40); do
  [ -s "$out/inflight.sse" ] && [ "$(grep -c '^data: ' "$out/inflight.sse" 2>/dev/null || echo 0)" -ge 1 ] && break
  kill -0 "$inflight_pid" 2>/dev/null || fail inflight "curl exited before any SSE frame arrived"
  sleep 0.25
done
frames_before_kill=$(grep -c '^data: ' "$out/inflight.sse" || true)
[ "$frames_before_kill" -ge 1 ] || fail inflight "no SSE frames arrived before the kill; nothing was genuinely in flight"
echo "in-flight request has $frames_before_kill SSE frame(s) before the kill"

# ---- kill engine a outright (SIGKILL: a real crash, not a graceful exit)

kill -KILL "$pid_a"
for _ in $(seq 1 40); do kill -0 "$pid_a" 2>/dev/null || break; sleep 0.25; done
kill -0 "$pid_a" 2>/dev/null && fail kill "engine a did not die"
echo "engine a killed (pid $pid_a)"

# The in-flight curl must end (loudly: nonzero exit or a body shorter than
# a real 300-token completion would produce), never hang. `wait` alone
# would trip `set -e` on the very nonzero exit this line exists to observe;
# the `|| inflight_rc=$?` guards it instead of letting bash abort here.
inflight_rc=0
wait "$inflight_pid" || inflight_rc=$?
frames_after=$(grep -c '^data: ' "$out/inflight.sse" || true)
done_reached=$(grep -c '^data: \[DONE\]' "$out/inflight.sse" || true)
echo "in-flight curl exit=$inflight_rc frames_total=$frames_after done_reached=$done_reached"
if [ "$inflight_rc" -eq 0 ] && [ "$done_reached" -ge 1 ]; then
  fail inflight "the killed request completed cleanly -- it should have failed loudly, not quietly finished"
fi
echo "in-flight request failed loudly as expected (exit=$inflight_rc, no terminal DONE reached)"

# ---- a NEW request after the kill must route to the survivor, b --------

new_body='{"prompt":"Say the word survivor.","max_tokens":8,"temperature":0.0}'
curl -fsS -X POST "$router_url/v1/completions" -H 'content-type: application/json' -d "$new_body" > "$out/after-kill.json" \
  || fail failover "new request after the kill failed"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('after-kill text:', repr(d['choices'][0]['text']))" "$out/after-kill.json"

curl -fsS "$router_url/v1/workloads" > "$out/workloads.json" || fail failover "GET /v1/workloads failed"
python3 - "$out/workloads.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
completions = [r for r in rows if r["path"] == "/v1/completions" and r["state"] == "done"]
assert completions, rows
last = completions[-1]
assert last["engine"] == "b", (last, "expected the new request to route to the survivor, b")
print(f"catalog OK: new post-kill request landed on engine={last['engine']} placement={last['placement']}")
PY

# HealthHysteresis needs 3 consecutive failed probes at probe_loop's 5 s
# interval to flip healthy from true to false (router.rs), so this
# assertion specifically needs real time to pass, unlike the routing checks
# above (which are correct immediately via the retry-on-connect-failure
# path in proxy(), not dependent on the hysteresis at all).
sleep 16
curl -fsS "$router_url/v1/node-info" > "$out/node-info-after.json" || fail failover "router node-info failed"
python3 - "$out/node-info-after.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
a = next(x for x in info["engines"] if x["id"] == "a")
b = next(x for x in info["engines"] if x["id"] == "b")
assert a["healthy"] is False, a
assert b["healthy"] is True, b
print("node-info OK: a unhealthy, b still healthy")
PY

echo "PASS p0b-gate2-failover"
