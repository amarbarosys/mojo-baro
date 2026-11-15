#!/usr/bin/env bash
# Fake baro front for test_mcp.sh: canned answers for the routes the request tools call. socat accepts; each
# connection runs this script again with --handle, which reads one HTTP request on stdin and answers on stdout.
#   fake_server.sh PORT          serve on 127.0.0.1:PORT (foreground; the test starts it in the background)
#   fake_server.sh PORT --bug    same, but the chat reply uses the wrong field name contentx (planted defect)
# A chat request whose message content is exactly boom gets HTTP 502 engine_invalid_tokens.
set -euo pipefail
SELF=$(readlink -f "$0")

if [ "${1:-}" = "--handle" ]; then
  BUG=${FAKE_BUG:-0}
  read -r method path _ || exit 0
  path=${path%$'\r'}
  clen=0
  while IFS= read -r h; do
    h=${h%$'\r'}
    [ -z "$h" ] && break
    case "${h,,}" in content-length:*) clen=${h#*:}; clen=${clen// /} ;; esac
  done
  body=""
  if [ "$clen" -gt 0 ] 2>/dev/null; then body=$(head -c "$clen"); fi
  respond() { # $1 status line, $2 body
    local len
    len=$(printf '%s' "$2" | wc -c)
    printf 'HTTP/1.1 %s\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s' "$1" "$len" "$2"
  }
  case "$method $path" in
    "GET /health") respond "200 OK" '{"status":"ok","engine_state":"ready","queue":0,"limits":{"tmax":16384}}' ;;
    "GET /props") respond "200 OK" '{"default_generation_settings":{"n_ctx":16384}}' ;;
    "GET /v1/models") respond "200 OK" '{"object":"list","data":[{"id":"fake","object":"model"}]}' ;;
    "POST /v1/chat/completions")
      if printf '%s' "$body" | grep -q '"content":"boom"'; then
        respond "502 Bad Gateway" '{"error":{"code":"engine_invalid_tokens"}}'
      else
        key=content
        [ "$BUG" = 1 ] && key=contentx
        respond "200 OK" "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"$key\":\"canned answer\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2},\"timings\":{\"fail\":0}}"
      fi ;;
    "POST /v1/embeddings") respond "200 OK" '{"object":"list","data":[{"index":0,"embedding":[0.1,0.2,0.3,0.4,0.5]}]}' ;;
    "GET /v1/checkpoints") respond "200 OK" '{"checkpoints":[{"id":"abc"}]}' ;;
    "POST /v1/checkpoints/abc/fork") respond "200 OK" '{"forked":["abc-1"]}' ;;
    *) respond "404 Not Found" '{"error":"not found"}' ;;
  esac
  exit 0
fi

PORT=${1:?usage: fake_server.sh PORT [--bug]}
export FAKE_BUG=0
[ "${2:-}" = "--bug" ] && export FAKE_BUG=1
exec socat TCP-LISTEN:"$PORT",reuseaddr,fork EXEC:"$SELF --handle",pipes
