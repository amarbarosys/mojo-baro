#!/usr/bin/env bash
# usage: tools/test_pool.sh [OUTDIR]      (needs the GPU; re-execs itself under gpu-wait)
# Gate for the C4 engine pool (bench/chat-protocol.md): BARO_POOL=2 loads two
# packs (~2x BARO_PACK's VRAM) and routes two concurrent requests one per
# engine (P-J2), both still matching ref-tokens-64.txt; records the aggregate
# tokens/s at pool size 1 vs 2 (P-J3, not gated -- decode is memory-bound, so
# no target is frozen). Separate from tools/test_server.sh (default pool of
# one): this test's VRAM and runtime cost doubles what that one pays, so it
# is not folded into the standing single-engine gate.
# Pack: BARO_PACK (default .work/engine-pack-q4); ref = $pack/ref-tokens-64.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --priority 60 --timeout 1800 --vram 16 -- "$0" "$@"
fi
out=${1:-.work/pool-test}; mkdir -p "$out"; : > "$out/SUMMARY.txt"
pack=${BARO_PACK:-.work/engine-pack-q4}; ref=$pack/ref-tokens-64.txt
ok() { echo "PASS $1: $2" | tee -a "$out/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$out/SUMMARY.txt"; exit 1; }
[ -f "$ref" ] || die setup "no $ref"
prompt_json=$(python3 -c "import sys; print([int(x) for x in open('$pack/prompt-tokens.txt').read().split()])")

if [ ! -x .work/engine ] || [ serve/engine.mojo -nt .work/engine ] || [ serve/registry.mojo -nt .work/engine ]; then
  ./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/engine > "$out/build-engine.log" 2>&1 || die build "$(grep -m1 error: "$out/build-engine.log" | cut -c1-160)"
fi
(cd serve && cargo build --release) > "$out/cargo-build.log" 2>&1 || die build "$(grep -m1 error "$out/cargo-build.log")"
ok build "engine + baro-serve"

start_server() {
  local poolsize=$1 dir=$2
  mkdir -p "$dir"
  BARO_POOL=$poolsize BARO_PACK=$pack ./serve/target/release/baro-serve --engine .work/engine --pack "$pack" --port 0 \
    > "$dir/server.stdout" 2> "$dir/server.stderr" &
  local srv=$!
  for _ in $(seq 1 600); do
    grep -q '^listening on' "$dir/server.stdout" && break
    kill -0 "$srv" 2>/dev/null || { echo "FAIL server exited: $(tail -3 "$dir/server.stderr")" >&2; return 1; }
    sleep 0.5
  done
  echo "$srv"
}

# --- pool size 2: two concurrent requests, one per engine (P-J2) -----------------
p2="$out/pool2"
srv2=$(start_server 2 "$p2") || die pool2-start "$(tail -5 "$p2/server.stderr")"
trap 'kill -9 '"$srv2"' 2>/dev/null; pkill -9 -P '"$srv2"' 2>/dev/null' EXIT
url2=$(grep -m1 -oE 'http://[0-9.:]+' "$p2/server.stdout") || die pool2-start "no listening line"
grep -q 'engine pool ready: 2 engine' "$p2/server.stderr" || die pool2-start "did not report 2 engines: $(cat "$p2/server.stderr")"
ok pool2-start "$url2, 2 engines up"

for i in 1 2; do
  curl -sf "$url2/v1/completions" -H 'content-type: application/json' \
    -d "{\"prompt\": $prompt_json, \"max_tokens\": 64, \"spec\": false}" > "$p2/req$i.json" &
done
sleep 0.3
curl -sf "$url2/health" > "$p2/health-busy.json" || die pool2-routing "curl /health failed"
wait
python3 -c "
import json
h = json.load(open('$p2/health-busy.json'))
pool = h.get('pool')
assert pool == [1, 1], f'expected both engines busy [1,1] while both requests were in flight, got {pool}'
print('pool depths during concurrent requests:', pool)
" > "$p2/pool.check" 2>&1 || die pool2-routing "$(cat "$p2/pool.check")"
ok pool2-routing "$(cat "$p2/pool.check")"

for i in 1 2; do
  python3 -c "import json; d=json.load(open('$p2/req$i.json')); print('GENERATED:', ' '.join(map(str, d['choices'][0]['tokens'])), '')" > "$p2/req$i.gen" || die pool2-identity "request $i: $(cat "$p2/req$i.json")"
  tools/check-tokens.sh "$ref" "$p2/req$i.gen" > "$p2/req$i.check" 2>&1 || die pool2-identity "request $i: $(cat "$p2/req$i.check")"
done
ok pool2-identity "both engines independently match ref-tokens-64.txt"

kill -INT "$srv2"; wait "$srv2"; rc2=$?
[ "$rc2" = 0 ] || die pool2-shutdown "server exit $rc2"
trap - EXIT
ok pool2-shutdown "2-engine pool: server exit 0"

# --- aggregate throughput, pool 1 vs pool 2 (P-J3, recorded not gated) -----------
bench_pool() {
  local poolsize=$1 dir=$2 nreq=$3
  srv=$(start_server "$poolsize" "$dir") || { echo "FAIL bench pool=$poolsize start"; return 1; }
  local url=$(grep -m1 -oE 'http://[0-9.:]+' "$dir/server.stdout")
  local t0=$(date +%s.%N)
  local pids=()
  for _ in $(seq 1 "$nreq"); do
    curl -sf "$url/v1/completions" -H 'content-type: application/json' \
      -d "{\"prompt\": $prompt_json, \"max_tokens\": 64, \"spec\": false}" > /dev/null &
    pids+=($!)
  done
  wait "${pids[@]}"
  local t1=$(date +%s.%N)
  kill -INT "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
  python3 -c "print(f'{$nreq * 64 / ($t1 - $t0):.1f}')"
}
agg1=$(bench_pool 1 "$out/bench1" 4)
agg2=$(bench_pool 2 "$out/bench2" 4)
ok pool-throughput "4 concurrent requests x 64 tokens: pool=1 aggregate ${agg1} tok/s, pool=2 aggregate ${agg2} tok/s (recorded, no target frozen)"

echo "ALL PASS" | tee -a "$out/SUMMARY.txt"
