#!/usr/bin/env bash
# MOEPF gate 3 (bench/moe-prefill-protocol.md): one served ~100k-token chat request through
# baro-serve, tier mode, needle placed in the last 5% of the prompt.
#   bench/moe-prefill-needle.sh ENGINE PACK OUT
# baro-serve itself is the main checkout's shared build, hardcoded absolute
# (this lane does not touch serve/, so it cannot go stale under it).
# env: PORT=8097 (default; never 8099, serve-builder.sh's default port); TIMEOUT_S=60 (startup
#      poll budget); NEEDLE_WORDS=100000 (haystack size, a word-count proxy for "about 100k
#      tokens" -- see bench/moe-prefill-needle-gen.py header, no tokenizer run to verify the ratio).
# PASS = HTTP 200, the streamed answer contains the needle value, and the server's log (engine
# stdout arrives on the server's STDERR with an "engine:" prefix, gate-authoring rule 9) shows a
# non-zero "prefill rows" echo for this request.
# Receipts: OUT/arm.txt, OUT/server.log, OUT/request.json, OUT/response.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3
port=${PORT:-8097}; timeout_s=${TIMEOUT_S:-60}; needle_words=${NEEDLE_WORDS:-100000}
if [ "$port" = 8099 ]; then echo "FAIL args: PORT 8099 is reserved for serve-builder.sh, pick another"; exit 1; fi
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --timeout 1800 -- env PORT="$port" TIMEOUT_S="$timeout_s" NEEDLE_WORDS="$needle_words" "$0" "$@"
fi
mkdir -p "$out"
bench/preflight.sh --check || { echo "FAIL preflight: tree changed since the last passing bench/preflight.sh"; exit 1; }

srv="$HOME/Projects/mojo/mojo-baro/serve/target/release/baro-serve"
[ -x "$srv" ] || { echo "FAIL args: $srv not built or not executable"; exit 1; }

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$port\$"; then
  echo "FAIL port: $port already in use, see \`ss -ltn\`"; exit 1
fi

tmax=$((needle_words + 1024))
{ echo "gate=moe-prefill-needle port=$port timeout_s=$timeout_s needle_words=$needle_words tmax=$tmax"
  echo "eng=$eng sha=$(sha256sum "$eng" | cut -c1-16)"
  echo "pack=$pack index_sha=$(sha256sum "$pack/index.txt" | cut -c1-16) pack_bytes=$(stat -c %s "$pack/pack.bin")"
  echo "modeenv='BARO_MEGA=0 BARO_TMAX=$tmax BARO_PREFILL=1'"
  echo "pcie=$(cat /sys/bus/pci/devices/0000:00:01.1/current_link_speed)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --short | tr '\n' ';')'"; } | tee "$out/arm.txt"

pid=""
cleanup() { [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; }
trap cleanup EXIT

env BARO_MEGA=0 BARO_TMAX="$tmax" BARO_PREFILL=1 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=90 \
  "$srv" --engine "$eng" --pack "$pack" --port "$port" > "$out/server.log" 2>&1 &
pid=$!

ready=0
for _ in $(seq 1 "$timeout_s"); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FAIL server: baro-serve (pid $pid) exited before becoming ready, see $out/server.log"; exit 1
  fi
  if grep -q '^listening on http://' "$out/server.log" 2>/dev/null; then ready=1; break; fi
  sleep 1
done
[ "$ready" = 1 ] || { echo "FAIL server: no 'listening on' line within ${timeout_s}s, see $out/server.log"; exit 1; }
echo "server readback: $(grep '^listening on http://' "$out/server.log")" | tee -a "$out/arm.txt"

python3 bench/moe-prefill-needle-gen.py --words "$needle_words" --out "$out/request.json" --out-value "$out/needle-value.txt"

if ! python3 - "$port" "$out" <<'EOF'
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
req = json.load(open(f"{out}/request.json"))
req["stream"] = True
needle_val = open(f"{out}/needle-value.txt").read().strip()
body = json.dumps(req).encode()
t_send = time.monotonic()
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                            headers={"Content-Type": "application/json"}, method="POST")
try:
    resp = urllib.request.urlopen(r, timeout=300)
except Exception as e:
    print(f"FAIL request: {e}"); sys.exit(1)
if resp.status != 200:
    print(f"FAIL request: HTTP {resp.status}"); sys.exit(1)
t_first = None
answer = []
for raw in resp:
    line = raw.decode(errors="replace").strip()
    if not line.startswith("data:"):
        continue
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        break
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        continue
    delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
    if delta:
        if t_first is None:
            t_first = time.monotonic()
        answer.append(delta)
t_done = time.monotonic()
text = "".join(answer)
open(f"{out}/response.txt", "w").write(text)
print(f"first token: {(t_first - t_send) if t_first else -1:.3f}s  total: {t_done - t_send:.3f}s")
if needle_val not in text:
    print(f"FAIL check: answer does not contain the needle value {needle_val!r}: {text!r}"); sys.exit(1)
print(f"PASS answer contains needle value {needle_val!r}")
EOF
then
  echo "FAIL request: python driver exited non-zero, see $out/response.txt"; exit 1
fi

if ! grep -q 'engine: .*prefill rows: [1-9]' "$out/server.log"; then
  echo "FAIL readback: no non-zero 'prefill rows' echo from the engine in $out/server.log"; exit 1
fi
echo "PASS moe-prefill-needle: see $out/response.txt and $out/server.log"
