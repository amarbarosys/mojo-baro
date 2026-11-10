#!/usr/bin/env bash
# M4 reference: llama.cpp greedy token ids for a fixed prompt, sent as ids (no server-side tokenisation).
# usage: tools/ref_tokens.sh MODEL.gguf "PROMPT" N > tests/ref_tokens.txt   (line 1 prompt ids, line 2 generated ids)
set -euo pipefail
L=${LLAMA_CPP:-$HOME/Projects/imports/llama.cpp/build/bin}
M=$1 P=$2 N=$3 PORT=${PORT:-18734}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --timeout 300 -- env LLAMA_CPP="$L" PORT="$PORT" "$0" "$@"
fi
ids=$("$L/llama-tokenize" -m "$M" -p "$P" --ids --log-disable 2>/dev/null | tail -1) || { echo "FAIL ref: tokenize"; exit 1; }
"$L/llama-server" -m "$M" --port "$PORT" -ngl 99 -c 16384 --log-disable >/dev/null 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 0.5; done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "FAIL ref: server not healthy"; exit 1; }
out=$(curl -sf "localhost:$PORT/completion" -d "{\"prompt\": $ids, \"n_predict\": $N, \"temperature\": 0, \"top_k\": 1, \"return_tokens\": true, \"cache_prompt\": false, \"samplers\": [\"top_k\"]}") \
  || { echo "FAIL ref: completion"; exit 1; }
echo "$ids" | tr -d '[] '
echo "$out" | jq -r '.tokens | map(tostring) | join(",")'
echo "$out" | jq -r '.content' >&2
