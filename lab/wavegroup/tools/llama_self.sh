#!/usr/bin/env bash
# llama.cpp against itself (flash attention off vs on) on the prompts where wavegroup and llama.cpp ids diverge: is the split a near-tie?
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --timeout 900 -- "$0" "$@"
fi
mkdir -p .work
srv=$HOME/Projects/imports/llama.cpp/build/bin/llama-server; M=$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf
for fa in off on; do
  port=$((20000 + RANDOM % 20000))
  $srv -m $M -c 16384 -ngl 99 -fa $fa -np 1 --no-webui --host 127.0.0.1 --port $port > .work/llama_self_$fa.log 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  for _ in $(seq 240); do curl -sf localhost:$port/health >/dev/null && break; sleep 0.5; done
  curl -sf localhost:$port/health >/dev/null || { echo "FAIL llama_self $fa: server not healthy, .work/llama_self_$fa.log"; exit 1; }
  for p in p02 p09 p12 p13 p17 p18 p19 p20; do
    curl -sf localhost:$port/completion -d "{\"prompt\": [$(cat tests/prompts/$p.ids)], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"ignore_eos\": true, \"return_tokens\": true}" | jq -r '.tokens|map(tostring)|join(",")' > .work/self_${fa}_$p.ids || { echo "FAIL llama_self $fa $p: completion"; exit 1; }
  done
  kill $pid; wait $pid 2>/dev/null || true
done
