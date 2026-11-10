#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/a3-wire}
pack=${BARO_PACK:-.work/engine-pack-q4}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  bench/preflight.sh --check
  exec gpu-wait run --priority 70 --timeout 1800 --vram 24 -- env A3_WIRE_JOB=1 "$0" "$@"
fi

mkdir -p "$out"
rm -f "$out"/single.json "$out"/concurrent-*.json "$out"/health-busy.json
: > "$out/receipt.md"
fail() { echo "FAIL $1: $2" | tee -a "$out/receipt.md" >&2; exit 1; }

engine_sha=DRYRUN
if [ "${GATE_DRYRUN:-0}" != 1 ]; then
  [ -x .work/engine ] || fail setup "missing .work/engine"
  engine_sha=$(sha256sum .work/engine | cut -d' ' -f1)
fi
cat > "$out/arm.txt" <<EOF
commit=$(git rev-parse HEAD)
dirty=$(git status --short)
engine=.work/engine sha256=$engine_sha
pack=$pack index_sha256=$(sha256sum "$pack/index.txt" | cut -d' ' -f1)
requests=2
n=128
spec=false
temperature=0
EOF
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL GPU step: dry-run reached server startup" | tee -a "$out/receipt.md" >&2
  exit 1
fi
[ -x serve/target/release/baro-serve ] || fail setup "missing baro-serve"
[ -f "$pack/prompt-tokens.txt" ] || fail setup "missing prompt"

prompt_json=$(python3 - "$pack/prompt-tokens.txt" <<'PY'
import json, sys
print(json.dumps([int(x) for x in open(sys.argv[1]).read().split()]))
PY
)
printf '%s\n' "{\"id\":1,\"prompt\":$prompt_json,\"n\":128,\"spec\":false,\"temperature\":0}" > "$out/ref-request.json"
refkey=$(~/iTools/bin/refcache key "$engine_sha" "$(sha256sum "$pack/index.txt" | cut -d' ' -f1)" "A3a n128 T0 spec0" "@$out/ref-request.json")
export REFCACHE_ROOT=.work/refcache/a3

BARO_PACK="$pack" BARO_SPEC=0 ./serve/target/release/baro-serve --engine .work/engine --pack "$pack" --port 0 \
  > "$out/server.stdout" 2> "$out/server.stderr" &
srv=$!
cleanup() {
  if kill -0 "$srv" 2>/dev/null; then
    kill -INT "$srv" 2>/dev/null || true
    sleep 1
    kill -TERM "$srv" 2>/dev/null || true
  fi
}
trap cleanup EXIT
for _ in $(seq 1 600); do
  grep -q '^listening on' "$out/server.stdout" && break
  kill -0 "$srv" 2>/dev/null || fail start "server exited: $(tail -5 "$out/server.stderr")"
  sleep 0.5
done
grep -q '^listening on' "$out/server.stdout" || fail start "no listening line"
for _ in $(seq 1 600); do
  grep -q 'engine pool ready:' "$out/server.stderr" && break
  kill -0 "$srv" 2>/dev/null || fail ready "server exited: $(tail -8 "$out/server.stderr")"
  sleep 0.5
done
grep -q 'engine pool ready:' "$out/server.stderr" || fail ready "no engine ready read-back"
url=$(grep -m1 -oE 'http://[0-9.:]+' "$out/server.stdout") || fail start "could not parse URL"
echo "## A3(a) wire gate" >> "$out/receipt.md"
echo "- ready: $(grep -m1 'engine pool ready:' "$out/server.stderr")" >> "$out/receipt.md"
echo "- refcache key: $refkey" >> "$out/receipt.md"

if refjson=$(~/iTools/bin/refcache get "$refkey" single.json 2> "$out/refcache.txt"); then
  cat "$out/refcache.txt" >> "$out/receipt.md"
  cp "$refjson" "$out/single.json"
else
  cat "$out/refcache.txt" >> "$out/receipt.md"
  curl -sf --max-time 240 "$url/v1/completions" -H 'content-type: application/json' \
    -d "{\"prompt\": $prompt_json, \"max_tokens\": 128, \"spec\": false, \"temperature\": 0}" \
    > "$out/single.json" || fail reference "single request failed"
  refjson=$(~/iTools/bin/refcache put "$refkey" single.json "$out/single.json")
  echo "- refcache stored: $refjson" >> "$out/receipt.md"
fi

pids=()
for i in 1 2; do
  curl -sf --max-time 240 "$url/v1/completions" -H 'content-type: application/json' \
    -d "{\"prompt\": $prompt_json, \"max_tokens\": 128, \"spec\": false, \"temperature\": 0}" \
    > "$out/concurrent-$i.json" &
  pids+=("$!")
done
sleep 0.1
curl -sf "$url/health" > "$out/health-busy.json" || fail concurrency "health during requests failed"
for pid in "${pids[@]}"; do wait "$pid" || fail concurrency "request process failed"; done
grep -q 'wire admitted request' "$out/server.stderr" || fail concurrency "no wire-admission receipt"
admit=$(grep -m1 'wire admitted request' "$out/server.stderr")
echo "- admission read-back: $admit" >> "$out/receipt.md"
python3 - "$out" <<'PY' >> "$out/receipt.md"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
ref = json.loads((out / "single.json").read_text())["choices"][0]["tokens"]
assert ref
ref_finish = json.loads((out / "single.json").read_text())["choices"][0]["finish_reason"]
ids = []
for i in (1, 2):
    d = json.loads((out / f"concurrent-{i}.json").read_text())
    c = d["choices"][0]
    assert c["tokens"] == ref, f"request {i} differs from single-request reference"
    assert c["finish_reason"] == ref_finish
    ids.append(d["id"])
assert len(set(ids)) == 2, ids
h = json.loads((out / "health-busy.json").read_text())
print(f"- identity: 2/2 concurrent responses byte-identical in token body to single run")
print(f"- response ids: {ids[0]}, {ids[1]}; busy queue read-back: {h.get('queue')}")
PY
kill -INT "$srv" || fail shutdown "could not signal server"
wait "$srv" || fail shutdown "server exit was non-zero"
trap - EXIT
echo "- clean shutdown: server exit 0" >> "$out/receipt.md"
echo "PASS A3(a) wire gate: $out/receipt.md"
