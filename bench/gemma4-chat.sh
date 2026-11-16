#!/usr/bin/env bash
# M4 of lane-gemma4-port: Forge-1 E2B through the user path. baro-serve spawns serve/gemma4.mojo (BARO_SERVE=1),
# renders the pack's chat template (tokenizer-meta.json, the GGUF's embedded template) and tokenizes with the pack's
# tokenizer.json; `baro chat -p` asks one greedy question at max_tokens 16000. Pass bar: the reply names Forge and
# says 391 (the llama.cpp receipt in ~/Models/forge-1-gemma-4-e2b-f16/RUN.md), and /tokenize on the 20 held-out
# prompts equals llama-tokenize ids (the server's own tokenizer, not serve/tokenizer.mojo).
# Usage: bench/gemma4-chat.sh GGUF PACK OUT   (queues itself under gpu-wait; builds .work/baro-chat first)
set -euo pipefail
cd "$(dirname "$0")/.."
GGUF=$1; PACK=$2; OUT=$3; PORT=${PORT:-8094}
SERVE=serve/target/release/baro-serve
if [ -z "${G4_INJOB:-}" ]; then
  [ -x .work/gemma4/engine ] && [ -x "$SERVE" ] || { echo "FAIL setup: engine or baro-serve missing"; exit 1; }
  mkdir -p "$OUT"
  ./.venv/bin/mojo build tools/baro-chat.mojo -I . -o .work/baro-chat > "$OUT/chat-build.log" 2>&1 || { echo "FAIL build baro-chat"; exit 1; }
  exec gpu-wait run --timeout 900 --vram 14 -- env G4_INJOB=1 PORT="$PORT" "$PWD/$0" "$GGUF" "$PACK" "$OUT"
fi
setsid "$SERVE" --engine .work/gemma4/engine --pack "$PACK" --port "$PORT" \
  --engine-env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=60 > "$OUT/serve.log" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"ok"\|ready' && break
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL serve: exited, $(tail -3 "$OUT/serve.log")"; exit 1; }
  sleep 1
done
curl -s "http://127.0.0.1:$PORT/health" > "$OUT/health.json"; echo "health $(cat "$OUT/health.json")"
ok=0; n=0
for f in bench/mtp-prompts/*.txt; do
  n=$((n + 1))
  a=$(python3 -I -c 'import json,sys; print(json.dumps({"content": open(sys.argv[1]).read(), "add_special": True}))' "$f" \
      | curl -s "http://127.0.0.1:$PORT/tokenize" -H 'Content-Type: application/json' -d @- \
      | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(str(t) for t in d.get("tokens", d.get("ids", []))))')
  b=$(.work/gemma4/llama-dump/build/bin/llama-tokenize -m "$GGUF" -f "$f" --ids --log-disable 2>/dev/null | tr -d '[]' | tr ',' ' ' | xargs)
  [ "$a" = "$b" ] && ok=$((ok + 1)) || echo "DIFF tokenize $f: server [${a:0:120}] llama [${b:0:120}]"
done
echo "server tokenize parity $ok/$n"
.work/baro-chat --url "127.0.0.1:$PORT" --max-tokens 16000 --temperature 0 -p "What is your name? Then: 17*23=?" \
  > "$OUT/reply.txt" 2> "$OUT/chat-stats.txt" || { echo "FAIL chat: $(tail -3 "$OUT/chat-stats.txt")"; exit 1; }
echo "reply: $(tr '\n' ' ' < "$OUT/reply.txt" | cut -c1-400)"
echo "stats: $(cat "$OUT/chat-stats.txt")"
grep -q Forge "$OUT/reply.txt" && grep -q 391 "$OUT/reply.txt" && [ "$ok" = "$n" ] \
  && echo "PASS gemma4-chat: reply names Forge and 391, tokenize $ok/$n" \
  || { echo "FAIL gemma4-chat: reply or tokenize ($ok/$n)"; exit 1; }
