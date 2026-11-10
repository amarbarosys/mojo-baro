#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/a3-d-admission}
pack=${BARO_PACK:-.work/engine-pack-q4}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  env -u LC_ALL bench/preflight.sh --check
  exec gpu-wait run --priority 70 --timeout 1800 --vram 24 -- env A3_D_JOB=1 GATE_DRYRUN="${GATE_DRYRUN:-0}" LC_ALL= "$0" "$@"
fi

mkdir -p "$out"
rm -f "$out"/request-*.json "$out"/response-*.json "$out"/health.json "$out"/server.stdout "$out"/server.stderr
: > "$out/receipt.md"
fail() { echo "FAIL $1: $2" | tee -a "$out/receipt.md" >&2; exit 1; }

engine_sha=DRYRUN
pack_sha=DRYRUN
if [ "${GATE_DRYRUN:-0}" != 1 ]; then
  [ -x .work/engine ] || fail setup "missing .work/engine"
  [ -x serve/target/release/baro-serve ] || fail setup "missing baro-serve"
  [ -f "$pack/index.txt" ] || fail setup "missing pack index"
  engine_sha=$(sha256sum .work/engine | cut -d' ' -f1)
  pack_sha=$(sha256sum "$pack/index.txt" | cut -d' ' -f1)
fi
cat > "$out/arm.txt" <<EOF
commit=$(git rev-parse HEAD)
dirty=$(git status --short)
engine=.work/engine sha256=$engine_sha
pack=$pack index_sha256=$pack_sha
clients=5
prompt_set=bench/mtp-prompts/p*.tokens first-five-sorted mixed lengths
n=64
temperature=0
BARO_MEGA=0 BARO_SPEC=0 BARO_POOL=1
EOF
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL GPU step: dry-run reached server startup" | tee -a "$out/receipt.md" >&2
  exit 1
fi

python3 - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
files = sorted(pathlib.Path("bench/mtp-prompts").glob("p*.tokens"))[:5]
if len(files) != 5:
    raise SystemExit(f"need at least five exact prompt files, found {len(files)}")
for i, path in enumerate(files):
    toks = [int(x) for x in path.read_text().split()]
    (out / f"request-{i}.json").write_text(json.dumps({
        "prompt": toks, "max_tokens": 64, "spec": False, "temperature": 0,
    }))
PY

BARO_PACK="$pack" BARO_MEGA=0 BARO_SPEC=0 BARO_POOL=1 \
  serve/target/release/baro-serve --engine .work/engine --pack "$pack" --port 0 \
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
url=$(grep -m1 -oE 'http://[0-9.:]+' "$out/server.stdout") || fail start "could not parse URL"

pids=()
for i in 0 1 2 3 4; do
  curl -sf --max-time 240 "$url/v1/completions" -H 'content-type: application/json' \
    --data-binary @"$out/request-$i.json" > "$out/response-$i.json" &
  pids+=("$!")
done
sleep 0.1
curl -sf --max-time 10 "$url/health" > "$out/health.json" || fail admission "health during five requests failed"
for pid in "${pids[@]}"; do wait "$pid" || fail admission "client process failed"; done
grep -q 'wire admitted request' "$out/server.stderr" || fail admission "no wire-admission receipt"
echo "- admission receipt: $(grep -m1 'wire admitted request' "$out/server.stderr")" >> "$out/receipt.md"

python3 - "$out" <<'PY' >> "$out/receipt.md"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
refs = json.loads(pathlib.Path(".work/a3-c-serial/refs.json").read_text())
seen = []
for i in range(5):
    d = json.loads((out / f"response-{i}.json").read_text())
    c = d["choices"][0]
    expected = refs[str(1000 + i)]
    assert c["tokens"] == expected, (i, len(c["tokens"]), len(expected))
    assert c.get("finish_reason") in ("length", "stop")
    seen.append(d["id"])
assert len(set(seen)) == 5, seen
h = json.loads((out / "health.json").read_text())
print(f"- mixed-length identity: 5/5; terminal responses: 5/5")
print(f"- response ids: {', '.join(seen)}; busy queue read-back: {h.get('queue')}")
print("- verdict: PASS admission/id routing; no preemption or throughput claim")
PY

kill -INT "$srv" || fail shutdown "could not signal server"
wait "$srv" || fail shutdown "server exit was non-zero"
trap - EXIT
echo "- clean shutdown: server exit 0" >> "$out/receipt.md"
echo "PASS A3(d) admission gate: $out/receipt.md"
