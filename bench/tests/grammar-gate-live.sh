#!/usr/bin/env bash
# Live JSON-enforcement gate (bench/grammar-protocol.md gates 1, 2 and 4) on the CURRENT tree's dense engine: starts
# serve/target/release/baro-serve with ENGINE (default .work/preflight/engine-dense, built by bench/preflight.sh) and
# the q4 pack, runs bench/grammar-gate.py against it, and stops server and engine on every exit path. Extra env goes
# to the gate (QUICK=N, TRUNC=1: T>0 requests send top_p 0.9 / top_k 20 / min_p 0.05, the case the 2026-10-12
# removal of the forced no-truncation exercises).
# usage: bench/tests/grammar-gate-live.sh OUT_DIR [ENGINE] [PACK]      (queues itself under gpu-wait)
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$(realpath -m "${1:?usage: grammar-gate-live.sh OUT_DIR [ENGINE] [PACK]}"); eng=${2:-.work/preflight/engine-dense}; pack=${3:-.work/engine-pack-q4}
[ -x "$eng" ] || { echo "FAIL setup: engine $eng not built (bench/preflight.sh)"; exit 1; }
[ -x serve/target/release/baro-serve ] || { echo "FAIL setup: serve/target/release/baro-serve not built"; exit 1; }
# grammar/vocab.mojo reads both; without the meta every response_format request is refused with a 400 (run 1, 0/66)
for f in tokenizer.json tokenizer-meta.json; do [ -s "$pack/$f" ] || { echo "FAIL setup: $pack/$f missing (tools/gguf-tokenizer-json.mojo MODEL PACK)"; exit 1; }; done
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --priority 90 --vram 12 --timeout 1800 -- env QUICK="${QUICK:-0}" TRUNC="${TRUNC:-0}" "$0" "$out" "$eng" "$pack"
fi
rm -rf "$out"; mkdir -p "$out"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
{ echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16)"; echo "pack $pack"; echo "QUICK=${QUICK:-0} TRUNC=${TRUNC:-0}"; } > "$out/arm.txt"
serve/target/release/baro-serve --engine "$eng" --pack "$pack" --port "$port" > "$out/serve.out" 2> "$out/serve.err" & sp=$!
trap 'pkill -TERM -P $sp 2>/dev/null || :; kill $sp 2>/dev/null || :; wait $sp 2>/dev/null || :' EXIT
for _ in $(seq 600); do kill -0 $sp 2>/dev/null || break; curl -sf "127.0.0.1:$port/health" > /dev/null && grep -q "$port" "$out/serve.err" "$out/serve.out" && break; sleep 1; done
curl -sf "127.0.0.1:$port/health" > /dev/null || { echo "FAIL serve: never healthy, see $out/serve.err"; exit 1; }
./.venv/bin/python3 bench/grammar-gate.py "http://127.0.0.1:$port" "$out/serve.err" "$out/gate" > "$out/gate.log" 2>&1 \
  || { tail -5 "$out/gate.log"; echo "FAIL grammar-gate, see $out/gate.log"; exit 1; }
tail -3 "$out/gate.log"
echo "PASS grammar-gate-live"
