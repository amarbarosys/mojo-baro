#!/usr/bin/env bash
# real_engine_check.sh: baro-mcp against a real baro-serve and a small model (the fake servers of test_mcp.sh
# prove the protocol; this proves the engine path). Needs the GPU: run it as
#   gpu-wait run --timeout 420 -- tools/mcp/real_engine_check.sh [MODEL.gguf]
# Read the whiteboard GPU holds first. Default model: Llama-3.2-1B-Instruct-Q4_K_M (a BARO-stamped file: baro refuses a plain gguf).
# Steps (one baro-mcp process, stdin lines): initialize, set_and_restart BARO_TMAX=16384 on port 8099,
# status, chat (16 tokens). Asserts: restarted true, took_effect true, status 200 with limits.tmax 16384,
# chat content non-empty with a finish_reason, and no brain_ask preamble (prompt_tokens < 100). The engine it started is stopped on exit and the port is checked free.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M-BARO-38e5113.gguf}
PORT=8099
OUT=.work/real-engine-check
fail() { echo "FAIL $1: $2"; exit 1; }
[[ -x .work/baro-mcp ]] || fail binary ".work/baro-mcp missing: build it first (see baro_mcp.mojo header)"
[[ -f "$MODEL" ]] || fail model "$MODEL not found"
[[ ! -s .work/gui-front.pid ]] || fail owned "an engine is already owned by this tree (.work/gui-front.pid)"
if ss -ltn | grep -q ":$PORT "; then fail port "port $PORT is in use"; fi
mkdir -p "$OUT"
stop() { bash face/stop-engine.sh >/dev/null 2>&1 || true; }
trap stop EXIT
CALL='{"jsonrpc":"2.0","id":%s,"method":"tools/call","params":{"name":"%s","arguments":%s}}\n'
{
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}'
  printf "$CALL" 2 set_and_restart "{\"settings\":{\"BARO_TMAX\":16384},\"model\":\"$MODEL\",\"port\":$PORT}"
  printf "$CALL" 3 status '{}'
  printf "$CALL" 4 chat '{"prompt":"Say hello in one short sentence.","max_tokens":16,"temperature":0}'
} | BARO_MCP_REPO="$PWD" BARO_MCP_URL="http://127.0.0.1:$PORT" .work/baro-mcp > "$OUT/replies.jsonl"
body() { jq -c "select(.id==$1) | .result" "$OUT/replies.jsonl"; }
text() { jq -r "select(.id==$1) | .result.content[0].text" "$OUT/replies.jsonl"; }
for id in 2 3 4; do
  [[ "$(body $id | jq -r '.isError // false')" == false ]] || fail "call $id" "isError: $(text $id | cut -c1-300)"
done
[[ "$(text 2 | jq -r '.restarted')" == true ]] || fail set_and_restart "restarted is not true"
[[ "$(text 2 | jq -r '.took_effect.BARO_TMAX')" == true ]] || fail set_and_restart "BARO_TMAX not in the new process environment"
[[ "$(text 3 | jq -r '.health.limits.tmax // .health.body.limits.tmax // empty')" == 16384 ]] || fail status "limits.tmax is not 16384: $(text 3 | cut -c1-300)"
[[ -n "$(text 4 | jq -r '.content // empty')" ]] || fail chat "empty content: $(text 4 | cut -c1-300)"
[[ -n "$(text 4 | jq -r '.finish_reason // empty')" ]] || fail chat "no finish_reason"
[[ "$(text 4 | jq -r '.usage.prompt_tokens')" -lt 100 ]] || fail chat "prompt_tokens >= 100: the front added a tool preamble (BARO_NEURONS?): $(text 4 | cut -c1-200)"
text 4 | jq -c '{content, finish_reason, usage}' | tee "$OUT/chat.txt"
stop
trap - EXIT
if ss -ltn | grep -q ":$PORT "; then fail cleanup "port $PORT still listening after stop"; fi
echo "PASS real-engine-check: $(basename "$MODEL")"
