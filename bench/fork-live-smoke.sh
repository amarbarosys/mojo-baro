#!/usr/bin/env bash
# usage: gpu-wait run --vram 23 --timeout 900 -- env HOME=$HOME PATH=$PATH QUICK=3 bench/fork-live-smoke.sh [OUT]
#
# First live contact for POST /v1/fork with "target" (serve/src/fork_target.rs): two 9B nodes of
# ours alive on the XTX at once (BARO_TMAX=4096, MAX memory cap 10, the operating point
# bench/fork-cap-probe.sh measured at 23.5 GB), loopback, no link shaping. NOT gate 2: no veth, no
# link rates, no 32k, no frozen predictions. It answers one question: does a fork exported on node
# A and imported on node B answer from B with the ids a single node gives?
#   per prompt: single-node ids on A  vs  A -> B fork ids
#   reuse check: B never saw the prompt, so its chain holds nothing for it; the branch must report
#     cached >= pos or B re-prefilled and the ids would match having tested nothing
#   portability: B serves the same pack through a DIFFERENT path (a symlink), the case P1's
#     identity.json salt exists for
# The whole script is one gpu-wait job; servers are direct children, reaped child-first.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/fork/live-smoke}
quick=${QUICK:-3}
engine=${BARO_ENGINE:-.work/fork/engine}
pack=${BARO_PACK:-$HOME/Projects/mojo/mojo-baro/.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
mkdir -p "$out"
exec > >(tee "$out/smoke.log") 2>&1
fail() { echo "FAIL $1: $2 (log $out/smoke.log)"; exit 1; }
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
start() {  # NAME PACKPATH; sets $url. Never call in a command substitution (pids would be lost).
  env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10 BARO_TMAX=4096 BARO_SPEC=0 BARO_CKPT_DIR="$out/ckpts-$1" \
    "$serve" --engine "$engine" --pack "$2" --port 0 > "$out/$1.stdout" 2> "$out/$1.stderr" &
  pids+=($!)
  for _ in $(seq 1 600); do
    grep -q '^listening on' "$out/$1.stdout" && break
    kill -0 "${pids[-1]}" 2>/dev/null || fail "start-$1" "server exited: $(tail -2 "$out/$1.stderr" | tr '\n' ' ')"
    sleep 0.5
  done
  url=$(grep -m1 -oE 'http://[0-9.:]+' "$out/$1.stdout") || fail "start-$1" "no listening line in 300 s"
}
start a "$pack"; ua=$url
start b "$out/pack-b"; ub=$url
echo "arm: engine_sha=$(sha256sum "$engine" | cut -c1-16) serve_sha=$(sha256sum "$serve" | cut -c1-16) commit=$(git rev-parse --short HEAD) a=$ua pack_a=$pack b=$ub pack_b=$out/pack-b"
for n in a b; do u=$ua; [ $n = b ] && u=$ub; st=$(curl -fsS "$u/v1/state" 2>&1) || fail "readback-$n" "GET $u/v1/state: $st"; echo "readback $n: $(echo "$st" | python3 -c "import json,sys;d=json.load(sys.stdin);print('portable',d['portable'],'kv',d['kv'],'resident_states',len(d['states']))") $(grep -m1 -o 'limits Limits { tmax: [0-9]*' "$out/$n.stderr" || fail "readback-$n" "no engine limits line, BARO_TMAX unconfirmed") $(grep -m1 -o 'identity Identity { pack: "[0-9a-f]\{16\}' "$out/$n.stderr" || true)"; done

python3 - "$out" "$ua" "$ub" "$quick" <<'PYEOF'
import glob, json, sys, time, urllib.request, urllib.error
out, ua, ub, quick = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def call(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
branch = {"max_tokens": 32, "temperature": 0, "spec": False}
bad = []
files = sorted(glob.glob("bench/mtp-prompts/p*.tokens"))[:quick]
for f in files:
    name = f.split("/")[-1][:-7]
    ids = [int(x) for x in open(f).read().split()]
    s1, ref = call(ua, "/v1/fork", {"prompt": ids, "branches": [branch]})
    t0 = time.monotonic()
    s2, fk = call(ua, "/v1/fork", {"prompt": ids, "target": ub.replace("http://", ""), "branches": [branch]})
    wall = time.monotonic() - t0
    json.dump({"ref": ref, "fork": fk}, open(f"{out}/{name}.json", "w"))
    if s1 != 200 or s2 != 200:
        print(f"{name}: HTTP ref {s1} fork {s2}: {json.dumps(fk)[:300]}"); bad.append(name); continue
    a, b = ref["branches"][0]["tokens"], fk["branches"][0]["tokens"]
    tg, tm = fk.get("target", {}), fk["branches"][0].get("timings", {})
    pos, cached = tg.get("pos"), tm.get("cached")
    same = a == b and len(b) > 0
    reused = pos == len(ids) - 1 and isinstance(cached, int) and cached >= pos
    print(f"{name}: |P|={len(ids)} pos={pos} B.cached={cached} ids_equal={same} ({len(b)} tokens) format={tg.get('format')} bytes={tg.get('state_bytes')} export_s={tg.get('export_s', 0):.3f} import_s={tg.get('import_s', 0):.3f} answer_s={tg.get('answer_s', 0):.3f} wall_s={wall:.3f} runtime_differs={tg.get('import', {}).get('runtime_differs')}")
    if not reused:
        print(f"  VOID {name}: node B did not restore the imported state (pos {pos}, cached {cached})")
    if not same:
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"  MISS {name}: first divergence at {k}")
    if not (same and reused):
        bad.append(name)
if bad:
    print(f"FAIL fork-live-smoke {len(bad)}/{len(files)} {' '.join(bad)}"); sys.exit(1)
print(f"PASS fork-live-smoke {len(files)}/{len(files)}: forked A -> B, B restored the imported state, ids equal single-node ids")
PYEOF
