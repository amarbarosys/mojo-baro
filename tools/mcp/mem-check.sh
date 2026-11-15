#!/usr/bin/env bash
# Memory check for .work/baro-mcp against the fake front (tools/mcp/fake_server.sh).
# Prints: size=BYTES idle_hwm_kb=N chat_hwm_kb=N
#   idle: initialize and tools/list only; the binary's stdin is a FIFO held open, VmHWM read before closing it.
#   chat: then one chat and one settings call in the same session; VmHWM read again.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=.work/baro-mcp
[ -x "$B" ] || { echo "FAIL build: no binary at $B"; exit 1; }
size=$(stat -c %s "$B") || { echo "FAIL size: stat failed"; exit 1; }
[ -n "$size" ] || { echo "FAIL size: empty"; exit 1; }

PORT=$(( 40000 + RANDOM % 20000 ))
FAKE=$PWD/tools/mcp/fake_server.sh
"$FAKE" "$PORT" >/dev/null 2>&1 &
FPID=$!
trap 'kill "$FPID" 2>/dev/null || true; pkill -P "$FPID" 2>/dev/null || true; rm -f "$FIFO" "$OUT"' EXIT

FIFO=.work/mem-fifo.$$
OUT=.work/mem-out.$$
mkfifo "$FIFO"
export BARO_MCP_URL="http://127.0.0.1:$PORT" BARO_MCP_REPO="$PWD"
"$B" < "$FIFO" > "$OUT" 2>/dev/null &
BPID=$!
exec 4> "$FIFO"

wait_lines() { # $1 = number of reply lines to wait for
  local n=$1
  for _ in $(seq 300); do
    [ "$(wc -l < "$OUT" 2>/dev/null || echo 0)" -ge "$n" ] && return 0
    sleep 0.1
  done
  echo "FAIL session: only $(wc -l < "$OUT" 2>/dev/null || echo 0) of $n replies" ; exit 1
}
hwm() { awk '/^VmHWM:/ {print $2}' "/proc/$BPID/status" 2>/dev/null; }

printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' >&4
wait_lines 2
idle=$(hwm)
[ -n "$idle" ] || { echo "FAIL idle_hwm: VmHWM missing for pid $BPID"; exit 1; }

printf '%s\n' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"chat","arguments":{"prompt":"hi"}}}' \
  '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"settings","arguments":{"group":"sampling"}}}' >&4
wait_lines 4
chat=$(hwm)
[ -n "$chat" ] || { echo "FAIL chat_hwm: VmHWM missing for pid $BPID"; exit 1; }

exec 4>&-
wait "$BPID" 2>/dev/null || true
echo "size=$size idle_hwm_kb=$idle chat_hwm_kb=$chat"
