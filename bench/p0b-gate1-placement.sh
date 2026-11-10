#!/usr/bin/env bash
# P0b gate 1 (docs/PLATFORM-PLAN.md): pair-dispatch --count 20 --mode parallel
# against the router with two real engines: 20/20 complete, placement
# follows the rank rule (catalog shows pending balanced within one), every
# response identical to its single-engine run at T=0.
#
# Two engines rig: lane FORK measured two live 9B dense q4 baro-serve
# processes fitting the XTX only at BARO_TMAX=4096,
# MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10 (10.7 GB each,
# docs/P1-FORK-TARGET.md on lane-fork; at BARO_TMAX=33024 one engine alone
# needs ~19 GB). This gate runs under that same TMAX=4096 ceiling -- it
# proves placement and identity, not 32k-context behavior with two engines,
# which nothing on this box can do yet.
#
# pair-dispatch (Personal-AI-Router's inference-dispatcher) deliberately
# never logs prompt or response TEXT, only a digest+length
# (proxy-inference-routing.mdc) -- it proves the 20/20-complete, real-PAIR-
# client half of this gate. The identity half (byte-for-byte text) is a
# separate direct comparison this script makes itself, the same split
# bench/p0a-contract-gate.sh already uses for gate 1/2 there.
#
# Servers are direct children of THIS script, pids tracked in the main
# shell (never inside a command substitution -- that exact mistake hung a
# GPU job for 9 minutes in front of a priority-90 job, ebb406d on
# lane-fork, ledger gpuwaitingroom.md 2026-09-17).
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/p0b-gate1}
engine=${BARO_ENGINE:-.work/engine}
pack=${BARO_PACK:-.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-.work/team-A/sonnet/target/release/baro-serve}
router_bin=${ROUTER_BIN:-.work/team-A/sonnet/target/release/router}
count=${COUNT:-20}
mkdir -p "$out"
exec > >(tee "$out/gate1.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
for f in "$engine" "$serve" "$router_bin" "$pack/pack.bin"; do [ -e "$f" ] || fail setup "missing $f"; done
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"

echo "engine sha256=$(sha256sum "$engine" | cut -d' ' -f1)"
echo "serve sha256=$(sha256sum "$serve" | cut -d' ' -f1) router sha256=$(sha256sum "$router_bin" | cut -d' ' -f1)"

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

# NAME; sets $LAST_URL and $LAST_PID in THIS shell. Never call inside a
# command substitution: that runs the whole function (pids+=() included) in
# a subshell, so the backgrounding and the pid tracking never reach the
# script's own EXIT trap -- exactly the mistake that hung a GPU job for
# 9 minutes in front of a priority-90 job (ebb406d, lane-fork,
# gpuwaitingroom.md 2026-09-17). Call as `start_engine a; url_a="$LAST_URL"`,
# not `url_a=$(start_engine a)`.
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

start_engine a; url_a="$LAST_URL"
start_engine b; url_b="$LAST_URL"
port_a=$(python3 -c "import sys,urllib.parse; print(urllib.parse.urlsplit(sys.argv[1]).port)" "$url_a")
port_b=$(python3 -c "import sys,urllib.parse; print(urllib.parse.urlsplit(sys.argv[1]).port)" "$url_b")
echo "engine a=$url_a engine b=$url_b"
for n in a b; do
  grep -m1 -o 'limits Limits { tmax: [0-9]*' "$out/$n.stderr" || fail "readback-$n" "no engine limits line, BARO_TMAX unconfirmed"
done

router_port=18310
BARO_ROUTER_ENGINES="a=http://127.0.0.1:${port_a},b=http://127.0.0.1:${port_b}" \
  "$router_bin" --host 127.0.0.1 --port "$router_port" --node-id 00000000-0000-4000-8000-0000000000c1 \
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

model=$(python3 -c "import json,urllib.request; print(json.load(urllib.request.urlopen('$router_url/api/tags'))['models'][0]['name'])")
echo "model=$model"

# ---- 20/20 through pair-dispatch, a real PAIR-shaped client -----------------

prompts=(
  "Reply with exactly the word amber."
  "What is two plus two? Reply with one digit."
  "Name the first month of the year."
  "Reply with exactly the word router."
  "Give one short word for a cold color."
)
pair_args=(--backend ollama --port "$router_port" --model "$model" --count "$count" --mode parallel
  --seed 0 --temperature 0 --max-tokens 8 --result-log "$out/pair-results.jsonl")
for ((i = 0; i < count; i++)); do
  pair_args+=(--prompt "${prompts[$((i % ${#prompts[@]}))]}")
done
rm -f "$out/pair-results.jsonl"
~/iTools/bin/pair-dispatch "${pair_args[@]}" > "$out/pair-dispatch.log" 2>&1 || fail pair-dispatch "see $out/pair-dispatch.log"
python3 - "$out/pair-results.jsonl" "$count" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
want = int(sys.argv[2])
assert len(rows) >= want, (len(rows), want)
tail = rows[-want:]
assert all(r.get("ok") is True for r in tail), tail
print(f"pair-dispatch OK: {want}/{want} ok through the router")
PY

# ---- placement: catalog shows pending balanced within one, all plain rank --

curl -fsS "$router_url/v1/workloads" > "$out/workloads.json" || fail placement "GET /v1/workloads failed"
python3 - "$out/workloads.json" "$count" <<'PY'
import collections, json, sys
rows = json.load(open(sys.argv[1]))
want = int(sys.argv[2])
chat = [r for r in rows if r["path"] == "/api/generate" and r["state"] == "done"]
assert len(chat) >= want, (len(chat), want, "pair-dispatch's ollama backend speaks /api/generate (measured live 2026-09-17, not /api/chat)")
counts = collections.Counter(r["engine"] for r in chat[-want:])
assert set(counts) <= {"a", "b"}, counts
a, b = counts.get("a", 0), counts.get("b", 0)
assert abs(a - b) <= 1, (a, b, "expected balanced within one")
assert all(r["placement"] == "rank" for r in chat[-want:]), [r for r in chat[-want:] if r["placement"] != "rank"]
print(f"placement OK: a={a} b={b} (balanced within one), every row placement=rank")
PY

# ---- identity: router responses match a single-engine (a) baseline at T=0 --

direct_url="$url_a"
python3 - "$router_url" "$direct_url" "$out" <<'PY'
import json, sys, urllib.request

router_url, direct_url, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
    "Reply with exactly the word amber.",
    "What is two plus two? Reply with one digit.",
    "Name the first month of the year.",
    "Reply with exactly the word router.",
    "Give one short word for a cold color.",
]

def completion(base, prompt):
    body = {"prompt": prompt, "max_tokens": 8, "temperature": 0.0}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["choices"][0]["text"]

rows = []
mismatches = []
for prompt in prompts:
    baseline = completion(direct_url, prompt)
    # Four repeats: with two engines balanced by rank, this exercises both
    # physical engines for the same prompt, not just whichever answers first.
    for i in range(4):
        proxied = completion(router_url, prompt)
        rows.append({"prompt": prompt, "repeat": i, "baseline": baseline, "proxied": proxied, "match": baseline == proxied})
        if baseline != proxied:
            mismatches.append((prompt, i, baseline, proxied))
json.dump(rows, open(f"{out}/identity.json", "w"), indent=2)
assert not mismatches, mismatches
print(f"identity OK: {len(rows)}/{len(rows)} proxied responses match the single-engine (a) baseline at T=0")
PY

echo "PASS p0b-gate1-placement"
