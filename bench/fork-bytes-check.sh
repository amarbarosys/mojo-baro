#!/usr/bin/env bash
# usage: gpu-wait run --vram 23 --timeout 900 -- env HOME=$HOME PATH=$PATH bench/fork-bytes-check.sh [OUT]
#
# Did node B keep the bytes node A exported? The ids cannot say: gate 2's identity half
# (bench/p1-fork-protocol.md, amendment 1) let 2 of 5 K/V-swapped states through with the right 32
# ids. This compares BYTES instead. Post hoc, not part of that freeze, and it judges the import, not
# the gate.
#   plain arm    A exports P to a file; A forks P to B; B re-exports P; B's file must equal A's
#                (salt, tokens, conv, SSM, and K/V below pos), bit for bit
#   swapped arm  the same through the forwarder's swapkv port, on OTHER prompts (B must be cold for
#                them): B's re-export must hold A's V where K belongs and A's K where V belongs.
#                This is the arm that discriminates. A node that ignored the import and re-prefilled
#                would reproduce A's own K and V, so plain equality alone cannot prove the import
#                was used; a swapped re-export can only come from the imported bytes.
# f32 only (int8 is lossy by design, there is nothing bit-exact to compare). Loopback, no shaping:
# transport integrity is already proven per import by node B's own sha256 of the received body.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/fork/bytes-check}
engine=${BARO_ENGINE:-.work/fork/engine}
pack=${BARO_PACK:-$HOME/Projects/mojo/mojo-baro/.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
py=./.venv/bin/python
bport=${BPORT:-18473}; fbase=${FBASE:-19480}
plain=(p01-water p03-story p12-rust p17-summarize)
swapped=(p02-python-fib p04-list-planets p05-math)
mkdir -p "$out"; out=$(realpath "$out"); rm -f "$out"/*.state   # absolute once: the nodes write the export paths, and "$PWD/$out" on an already absolute OUT put 816 MB under a stray tree
exec > >(tee "$out/check.log") 2>&1
fail() { echo "FAIL $1: $2 (log $out/check.log)"; exit 1; }
for f in "$engine" "$serve" "$pack/pack.bin" "$pack/identity.json"; do [ -e "$f" ] || fail setup "missing $f"; done
ln -sfn "$(readlink -f "$pack")" "$out/pack-b"

pids=()
cleanup() {
  for p in "${pids[@]}"; do pkill -TERM -P "$p" 2>/dev/null || true; kill -TERM "$p" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && return 0; sleep 0.5
  done
  for p in "${pids[@]}"; do pkill -KILL -P "$p" 2>/dev/null || true; kill -KILL "$p" 2>/dev/null || true; done
}
trap cleanup EXIT
start() {  # NAME PACKPATH ARGS...; sets $url. Never call in a command substitution.
  local name=$1 pk=$2; shift 2
  env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10 BARO_TMAX=4096 BARO_SPEC=0 BARO_STATE_INT8=0 BARO_CKPT_DIR="$out/ckpts-$name" \
    "$serve" --engine "$engine" --pack "$pk" "$@" > "$out/$name.stdout" 2> "$out/$name.stderr" &
  pids+=($!)
  for _ in $(seq 1 600); do
    grep -q '^listening on' "$out/$name.stdout" && break
    kill -0 "${pids[-1]}" 2>/dev/null || fail "start-$name" "server exited: $(tail -2 "$out/$name.stderr" | tr '\n' ' ')"
    sleep 0.5
  done
  url=$(grep -m1 -oE 'http://[0-9.:]+' "$out/$name.stdout") || fail "start-$name" "no listening line in 300 s"
  local lim; lim=$(grep -o 'limits Limits { tmax: [0-9]*' "$out/$name.stderr" | tail -1 || true)
  [ "$lim" = "limits Limits { tmax: 4096" ] || fail "readback-$name" "engine limits line says '$lim'"
}
python3 tools/fork-link-forwarder.py 127.0.0.1 "$fbase" 127.0.0.1 "$bport" > "$out/forwarder.log" 2>&1 &
pids+=($!)
start a "$pack" --port 0; ua=$url
start b "$out/pack-b" --port "$bport"; ub=$url
echo "arm: engine_sha=$(sha256sum "$engine" | cut -c1-16) serve_sha=$(sha256sum "$serve" | cut -c1-16) commit=$(git rev-parse --short HEAD) a=$ua b=$ub forwarder=127.0.0.1:$fbase"

post() { curl -sS --fail-with-body -X POST "$1" -H 'content-type: application/json' -d "$2"; }
bad=()
run() {  # ARM TARGET FLAG PROMPT...
  local arm=$1 target=$2 flag=$3; shift 3
  for p in "$@"; do
    ids=$(tr -s ' \n' ',' < "bench/mtp-prompts/$p.tokens" | sed 's/^,//; s/,$//')
    post "$ua/v1/state/export" "{\"tokens\":[$ids],\"path\":\"$out/A-$p.state\"}" > "$out/A-$p.json" || fail "$arm" "$p export on A: $(cat "$out/A-$p.json")"
    post "$ua/v1/fork" "{\"prompt\":[$ids],\"target\":\"$target\",\"branches\":[{\"max_tokens\":32,\"temperature\":0,\"spec\":false}]}" > "$out/fork-$p.json" || fail "$arm" "$p fork: $(head -c 200 "$out/fork-$p.json")"
    cached=$($py -c "import json;d=json.load(open('$out/fork-$p.json'));print(d['branches'][0]['timings'].get('cached'), d['target']['pos'])")
    post "$ub/v1/state/export" "{\"tokens\":[$ids],\"path\":\"$out/B-$p.state\"}" > "$out/B-$p.json" || fail "$arm" "$p re-export on B: $(cat "$out/B-$p.json")"
    rc=0; res=$($py tools/state-bytes-diff.py "$out/A-$p.state" "$out/B-$p.state" $flag 2>&1) || rc=$?
    echo "$arm $p: B.cached pos = $cached | $res"
    [ "$rc" = 0 ] || bad+=("$arm:$p")
  done
}
run plain "${ub#http://}" "" "${plain[@]}"
run swapped "127.0.0.1:$((fbase + 2))" "--swapped" "${swapped[@]}"
echo "forwarder: $(grep -c 'swapkv: rewrote' "$out/forwarder.log" || true) states rewritten in flight (want ${#swapped[@]})"
[ "$(grep -c 'swapkv: rewrote' "$out/forwarder.log" || true)" = "${#swapped[@]}" ] || bad+=("forwarder-count")
n=$(( ${#plain[@]} + ${#swapped[@]} ))
[ "${#bad[@]}" = 0 ] || { echo "FAIL fork-bytes-check ${#bad[@]}/$n ${bad[*]}"; exit 1; }
echo "PASS fork-bytes-check $n/$n: node B's re-export is bit-identical to what node A sent (${#plain[@]} plain, ${#swapped[@]} swapped)"
