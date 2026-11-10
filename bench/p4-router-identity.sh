#!/usr/bin/env bash
# P4's real identity gate: the 20 bench/mtp-prompts/p*.tokens prompts, sent as token ids,
# max_tokens 64, temperature 0, spec false, THROUGH THE ROUTER against two real baro-serve
# engine processes (both on the XTX at BARO_TMAX=4096), each response's token ids compared
# by `cmp`, never on text, against the same prompt sent directly to engine a (the solo
# baseline). This replaces the earlier claim that team A's gate 1 (a placement gate with a
# 5-prompt x 4-repeat, ~70-token text identity check) stood in for this: it does not, it is
# about 35x fewer compared tokens (mojo-baro lane P4B follow-up, 2026-09-18).
#
# Placement must spread: the gate FAILS if either engine served zero of the 20 (this gate's
# reverse arm, see the reverse-arm-gates skill -- an identity pass with one engine doing all
# the work proves nothing about the split).
#
# The launch half (start_engine, router bring-up, health/ready wait) is FORKED from
# bench/p0b-gate1-placement.sh, not sourced: that script is linear (its own calls to
# start_engine, then pair-dispatch/placement/identity, run immediately after the function
# definitions, and its EXIT trap kills both engines the moment IT finishes), so sourcing it
# would either re-run its own weaker identity check first or require restructuring team A's
# file, which is out of scope here. The payload/tokens/request functions are lifted from
# .work/p4/run-two-engines.sh in the main mojo-baro checkout (team B), unchanged.
#
# usage: p4-router-identity.sh OUT_DIR
#   p4-router-identity.sh --selftest    CPU-only test of the token-id comparator, no engines
#   P4_CPU_PREFLIGHT=1                  verify paths/hashes/tools only, no GPU
#   GATE_DRYRUN=1                       (gate-dryrun harness) stop before the first engine launch
# env: BARO_ENGINE BARO_PACK BARO_SERVE_BIN ROUTER_BIN (default to team A's built paths under
#      this repo's .work/; override when running from a worktree that lacks them, as this one
#      does -- point them at another worktree's already-built, sha256-verified binaries)
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "FAIL $1: $2"; exit 1; }

cmp_tokens() { # solo_tokens_file router_tokens_file -> prints verdict, returns 0 PASS / 1 FAIL
  if cmp -s "$1" "$2"; then
    echo "PASS identity: $1 == $2"
  else
    echo "FAIL identity: $1 != $2"
    return 1
  fi
}

if [ "${1:-}" = --selftest ]; then
  mkdir -p .work; t=$(mktemp -d .work/p4ri.XXXXXX)
  printf '27336 85895 506 220 16\n' > "$t/solo.tokens"
  printf '27336 85895 506 220 16\n' > "$t/router-same.tokens"
  printf '27336 85895 506 999 16\n' > "$t/router-diff.tokens"
  cmp_tokens "$t/solo.tokens" "$t/router-same.tokens" || fail selftest "identical token files must PASS"
  if cmp_tokens "$t/solo.tokens" "$t/router-diff.tokens" 2>/dev/null; then
    fail selftest "a one-id difference must FAIL, but the comparator returned PASS"
  fi
  echo "PASS p4-router-identity --selftest: identical=PASS, one-id-different=FAIL (exit 1), CPU only"
  rm -rf "$t"
  exit 0
fi

out=${1:?usage: p4-router-identity.sh OUT_DIR}
engine=${BARO_ENGINE:-.work/team-A/sonnet/target/release/engine}
pack=${BARO_PACK:-.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-.work/team-A/sonnet/target/release/baro-serve}
router_bin=${ROUTER_BIN:-.work/team-A/sonnet/target/release/router}
mkdir -p "$out"
exec > >(tee "$out/gate.log") 2>&1

ARM="$out/arm.txt"
write_arm() {
  {
    echo "engine=$engine sha256=$(sha256sum "$engine" 2>/dev/null | cut -d' ' -f1 || echo missing)"
    echo "serve=$serve sha256=$(sha256sum "$serve" 2>/dev/null | cut -d' ' -f1 || echo missing)"
    echo "router=$router_bin sha256=$(sha256sum "$router_bin" 2>/dev/null | cut -d' ' -f1 || echo missing)"
    echo "pack=$pack sha256=$(sha256sum "$pack/pack.bin" 2>/dev/null | cut -d' ' -f1 || echo missing)"
    echo "prompts=bench/mtp-prompts/p*.tokens count=20 max_tokens=64 temperature=0 spec=false"
    echo "tmax_expected=4096 mem_mgr_pct_expected=10"
  } > "$ARM"
}
write_arm

if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL p4-router-identity: would start two baro-serve engines and a router next (GATE_DRYRUN stop, no GPU touched)"
  exit 77
fi

if [ "${P4_CPU_PREFLIGHT:-0}" = 1 ]; then
  for f in "$engine" "$serve" "$router_bin" "$pack/pack.bin"; do [ -e "$f" ] || fail preflight "missing $f"; done
  [ "$(sha256sum "$pack/pack.bin" | cut -d' ' -f1)" = 715d4c347b86787922198190be0fff3ed2fede708cfd07629f812c97d59c7698 ] \
    || fail preflight "engine-pack-q4 index hash mismatch"
  command -v curl >/dev/null || fail preflight "curl is required"
  command -v python3 >/dev/null || fail preflight "python3 is required"
  command -v rocm-smi >/dev/null || fail preflight "rocm-smi is required"
  n=$(ls bench/mtp-prompts/p*.tokens | wc -l)
  [ "$n" -eq 20 ] || fail preflight "expected 20 prompt files under bench/mtp-prompts/, found $n"
  echo "PASS p4-router-identity CPU preflight: binaries present, pack hash matches, tools present, 20 prompt files"
  exit 0
fi

[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || fail admission "run under gpu-wait"

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

# start_engine: forked verbatim (params renamed) from bench/p0b-gate1-placement.sh. Sets
# $LAST_URL/$LAST_PID in THIS shell; never call inside a command substitution (that mistake
# hung a GPU job for 9 minutes in front of a priority-90 job, ebb406d, lane-fork, 2026-09-17).
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
for n in a b; do
  grep -m1 -o 'limits Limits { tmax: [0-9]*' "$out/$n.stderr" || fail "readback-$n" "no engine limits line, BARO_TMAX unconfirmed"
done

router_port=18311
BARO_ROUTER_ENGINES="a=http://127.0.0.1:${port_a},b=http://127.0.0.1:${port_b}" \
  "$router_bin" --host 127.0.0.1 --port "$router_port" --node-id 00000000-0000-4000-8000-0000000000d2 \
  > "$out/router.stdout" 2> "$out/router.stderr" &
pids+=($!)
router_pid=${pids[-1]}
for _ in $(seq 1 60); do
  grep -q '^listening on' "$out/router.stdout" && break
  kill -0 "$router_pid" 2>/dev/null || fail start-router "router exited: $(tail -5 "$out/router.stderr" | tr '\n' ' ')"
  sleep 0.5
done
router_url="http://127.0.0.1:${router_port}"

# probe_loop needs 2 passes at a 5 s interval to mark both engines healthy.
sleep 12
curl -fsS "$router_url/v1/node-info" > "$out/node-info.json" || fail health "router node-info failed"
python3 - "$out/node-info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
for eid in ("a", "b"):
    e = next((x for x in info["engines"] if x["id"] == eid), None)
    assert e is not None and e["healthy"] is True, (eid, e)
print("node-info OK: both engines healthy")
PY

# ---- device read-back: real files on disk, both engine child PIDs, memory-manager cap ------

engine_child_pid() { # baro-serve pid -> pid of its "engine" child, or empty
  local parent="$1" c comm exe
  for c in $(pgrep -P "$parent" 2>/dev/null || true); do
    comm=$(cat "/proc/$c/comm" 2>/dev/null || true)
    exe=$(basename "$(readlink "/proc/$c/exe" 2>/dev/null || true)")
    if [ "$comm" = "engine" ] || [ "$exe" = "engine" ] || [[ "$comm" == *engine* ]]; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}
engine_pid_a=$(engine_child_pid "$pid_a") || fail readback "no engine child of baro-serve pid $pid_a"
engine_pid_b=$(engine_child_pid "$pid_b") || fail readback "no engine child of baro-serve pid $pid_b"
ROCM_SMI=$(command -v rocm-smi 2>/dev/null || echo "/opt/rocm/bin/rocm-smi")
"$ROCM_SMI" --showpids > "$out/rocm-smi-showpids.log" 2>&1
grep -qw "$engine_pid_a" "$out/rocm-smi-showpids.log" || fail readback "rocm-smi did not list engine a's PID $engine_pid_a"
grep -qw "$engine_pid_b" "$out/rocm-smi-showpids.log" || fail readback "rocm-smi did not list engine b's PID $engine_pid_b"
{
  echo "engine_a_pid=$engine_pid_a"
  echo "engine_b_pid=$engine_pid_b"
  for eid_pid in "a:$engine_pid_a" "b:$engine_pid_b"; do
    n=${eid_pid%%:*}; p=${eid_pid##*:}
    val=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=' || echo "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=unset")
    echo "engine_${n}_environ_readback.$val"
    tmax_val=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^BARO_TMAX=' || echo "BARO_TMAX=unset")
    echo "engine_${n}_environ_readback.$tmax_val"
  done
} > "$out/device-readback.txt"
cat "$out/device-readback.txt"

# ---- the real identity gate: 20 prompts, token ids, through the router vs solo on a --------

payload() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps({"prompt": [int(x) for x in open(sys.argv[1]).read().split()],
                  "max_tokens": 64, "temperature": 0, "spec": False}))
PY
}
tokens() {
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ".join(str(x) for x in d["choices"][0]["tokens"]))
PY
}
request() {
  local url=$1 prompt=$2 response=$3
  curl -sf --max-time 900 "$url/v1/completions" \
    -H 'content-type: application/json' --data-binary "$(payload "$prompt")" > "$response"
  tokens "$response" > "${response%.json}.tokens"
}

# Solo baselines direct to engine a: sequential is fine, correctness only, no router involved.
for prompt in bench/mtp-prompts/p*.tokens; do
  name=$(basename "$prompt" .tokens)
  request "$url_a" "$prompt" "$out/$name.solo.json"
done

# Router requests IN PARALLEL: choose()'s tie-break is (pending, !preferred, engine id), so
# with every request issued sequentially both engines sit at pending=0 at every decision and
# the id tie-break ("a" < "b") picks engine a for all 20, every time -- not a router bug, a
# gate bug (round 1 caught this live: a=20 b=0). Real concurrent load, like team A's own
# pair-dispatch --mode parallel, is what actually produces differing pending counts.
router_pids=()
for prompt in bench/mtp-prompts/p*.tokens; do
  name=$(basename "$prompt" .tokens)
  request "$router_url" "$prompt" "$out/$name.router.json" &
  router_pids+=($!)
done
for p in "${router_pids[@]}"; do
  wait "$p" || fail identity "a parallel router request (pid $p) failed"
done

echo "prompt,match" > "$out/identity.csv"
i=0
for prompt in bench/mtp-prompts/p*.tokens; do
  name=$(basename "$prompt" .tokens)
  if cmp_tokens "$out/$name.solo.tokens" "$out/$name.router.tokens"; then
    echo "$name,PASS" >> "$out/identity.csv"
  else
    echo "$name,FAIL" >> "$out/identity.csv"
    fail identity "$name mismatch, see $out/$name.solo.tokens and $out/$name.router.tokens"
  fi
  i=$((i + 1))
done
[ "$i" -eq 20 ] || fail identity "expected 20 prompts, ran $i"

# ---- placement must spread: this gate's reverse arm -----------------------------------------

curl -fsS "$router_url/v1/workloads" > "$out/workloads.json" || fail placement "GET /v1/workloads failed"
python3 - "$out/workloads.json" 20 > "$out/placement-summary.txt" <<'PY'
import collections, json, sys
rows = json.load(open(sys.argv[1]))
want = int(sys.argv[2])
chat = [r for r in rows if r["path"] == "/v1/completions" and r["state"] == "done"]
assert len(chat) >= want, (len(chat), want, "fewer completed /v1/completions rows than prompts sent")
counts = collections.Counter(r["engine"] for r in chat[-want:])
assert set(counts) <= {"a", "b"}, counts
a, b = counts.get("a", 0), counts.get("b", 0)
assert a + b == want, (a, b, want)
assert a > 0 and b > 0, (a, b, "one engine served zero of the 20: reverse arm FAILED, placement gate is dead")
print(f"placement OK: a={a} b={b}, both engines served at least one of the {want} requests")
PY
cat "$out/placement-summary.txt"

echo "PASS p4-router-identity: 20/20 token-id identity through the router vs solo-a, placement spread confirmed" | tee "$out/SUMMARY.txt"
