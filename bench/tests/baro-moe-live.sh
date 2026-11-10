#!/usr/bin/env bash
# Live check of a MoE bake the way a user meets it: tools/baro serve MODEL.gguf (engine built from the file's own
# sources), one chat through tools/baro chat, then the state API: GET /v1/state, POST /v1/state/export to a file,
# POST /v1/state/import of that file. PASS needs: the engine's BARO_KVQ read-back equal to KVQ (default int8, the
# MoE default since f3220ad), a non-empty reply with a stats line, /v1/state reporting the same kv, an export with
# bytes > 0, and an import that restores the exported prefix (same prefix_hash and pos). An int8 engine answering
# 501 kvq_state_open (docs/P1-STATE-API.md) is a FAIL: the engine refusal it maps was removed.
# Tier mode only (BARO_TIER=64; baro-serve defaults PINNED and ZC): the resident 20 GB pack OOM'd next to the desktop on
# 2026-10-12 and the compositor's ring resets crashed the box. TIERPACK = pack.bin + experts.bin (tools/pack-split-experts.py).
# usage: bench/tests/baro-moe-live.sh MODEL-BARO.gguf TIERPACK [KVQ]      (queues itself under gpu-wait)
set -euo pipefail
export BARO_NEURONS=0
cd "$(dirname "$0")/../.."
model=${1:?usage: baro-moe-live.sh MODEL-BARO.gguf TIERPACK [KVQ]}; tier=${2:?TIERPACK}; kvq=${3:-int8}
[ -f "$tier/experts.bin" ] || { echo "FAIL pack: $tier has no experts.bin; resident MoE serving is refused"; exit 1; }
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --vram 8 --timeout 1200 -- "$0" "$model" "$tier" "$kvq"
fi
t=$PWD/.work/baro-moe-live; rm -rf "$t"; mkdir -p "$t"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
echo "vram used before serve: $(awk '{printf "%.2f GB", $1 / 1e9}' /sys/class/drm/card1/device/mem_info_vram_used)"
BARO_TIER=64 tools/baro serve "$model" --pack "$tier" --port "$port" > "$t/serve.out" 2> "$t/serve.err" & sp=$!
trap 'pkill -TERM -P $sp 2>/dev/null || :; kill $sp 2>/dev/null || :; wait $sp 2>/dev/null || :' EXIT
for _ in $(seq 600); do kill -0 $sp 2>/dev/null || break; curl -sf "127.0.0.1:$port/health" > /dev/null && grep -q "$port" "$t/serve.err" "$t/serve.out" && break; sleep 1; done
curl -sf "127.0.0.1:$port/health" > /dev/null || { echo "FAIL serve: never healthy, see $t/serve.err"; exit 1; }
grep -h 'engine:' "$t/serve.out" | head -2
got=$(grep -ho 'BARO_KVQ: [a-z0-9]*' "$t/serve.out" "$t/serve.err" | head -1 | cut -d' ' -f2)
[ "$got" = "$kvq" ] || { echo "FAIL read-back: BARO_KVQ '$got', want '$kvq' (see $t/serve.*)"; exit 1; }
tools/baro chat --url "127.0.0.1:$port" --max-tokens 24 -p "Name the capital of France in one word." > "$t/reply" 2> "$t/stats" \
  || { echo "FAIL baro chat exit $?: $(cat "$t/stats")"; exit 1; }
echo "reply: $(tr '\n' ' ' < "$t/reply" | cut -c1-120)"; echo "stats: $(tail -1 "$t/stats")"
[ -n "$(tr -d '[:space:]' < "$t/reply")" ] || { echo "FAIL empty reply"; exit 1; }
grep -qE '\[[0-9]+ tok · [0-9.]+ tok/s' "$t/stats" || { echo "FAIL no stats line"; exit 1; }
curl -sf "127.0.0.1:$port/v1/state" > "$t/state.json" || { echo "FAIL GET /v1/state"; exit 1; }
skv=$(jq -r .kv "$t/state.json"); [ "$skv" = "$kvq" ] || { echo "FAIL /v1/state kv '$skv', want '$kvq'"; exit 1; }
code=$(curl -s -o "$t/export.json" -w '%{http_code}' "127.0.0.1:$port/v1/state/export" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"The quick brown fox jumps over the lazy dog. Remember this sentence.\",\"path\":\"$t/state.bin\"}")
[ "$code" = 200 ] || { echo "FAIL export HTTP $code: $(head -c 300 "$t/export.json")"; exit 1; }
eb=$(jq -r .bytes "$t/export.json"); [ "${eb:-0}" -gt 0 ] && [ -s "$t/state.bin" ] || { echo "FAIL export wrote no bytes: $(cat "$t/export.json")"; exit 1; }
code=$(curl -s -o "$t/import.json" -w '%{http_code}' "127.0.0.1:$port/v1/state/import" -H 'Content-Type: application/json' -d "{\"path\":\"$t/state.bin\"}")
[ "$code" = 200 ] || { echo "FAIL import HTTP $code: $(head -c 300 "$t/import.json")"; exit 1; }
for k in prefix_hash pos; do
  [ "$(jq -r .$k "$t/export.json")" = "$(jq -r .$k "$t/import.json")" ] || { echo "FAIL import $k differs: export $(cat "$t/export.json") import $(cat "$t/import.json")"; exit 1; }
done
echo "state: kv $skv, export $eb bytes pos $(jq -r .pos "$t/export.json"), import restore_ms $(jq -r .restore_ms "$t/import.json")"
echo "PASS baro-moe live ($kvq, port $port)"
