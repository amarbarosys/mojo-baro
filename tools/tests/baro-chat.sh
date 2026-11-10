#!/usr/bin/env bash
# baro chat against a canned server (socat): a cut reply is not a finished turn, and finish_reason is shown (audit row C5).
# BARO_CHAT_BIN names a built client; without it the checkout's tools/baro-chat.mojo is built into .work/fix-front-test.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
t=$here/.work/fix-front-test/baro-chat; mkdir -p "$t"
bin=${BARO_CHAT_BIN:-$t/baro-chat}
if [ -z "${BARO_CHAT_BIN:-}" ]; then
  rm -f "$bin"
  "$here/.venv/bin/mojo" build "$here/tools/baro-chat.mojo" -I "$here" -o "$bin" > "$t/build.log" 2>&1 || { echo "FAIL build: $t/build.log" >&2; exit 1; }
fi
fail() { echo "FAIL $1: $2" >&2; exit 1; }
port=$((20000 + RANDOM % 20000))
serve_once() { # serve_once FILE: answer one connection with FILE's bytes
  socat "TCP-LISTEN:$port,reuseaddr,bind=127.0.0.1" "SYSTEM:cat >/dev/null & sleep 0.2; cat $1; sleep 0.4" &
  sleep 0.3
}
head='HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n'

# 1. the stream closes with no [DONE]: the reply is partial, the client must fail by name and exit non-zero
printf "$head"'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n' > "$t/cut.http"
serve_once "$t/cut.http"
if "$bin" --url "127.0.0.1:$port" -p hi > "$t/cut.out" 2> "$t/cut.err"; then fail C5 "a stream with no [DONE] was accepted as a reply (stdout: $(cat "$t/cut.out"))"; fi
grep -q 'without \[DONE\]' "$t/cut.err" || fail C5 "the error does not say the stream was cut: $(cat "$t/cut.err")"
wait || true

# 2. a max_tokens cut ends normally but says so
printf "$head"'data: {"choices":[{"delta":{"content":"hel"}}]}\n\ndata: {"choices":[{"delta":{"content":"lo"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n' > "$t/len.http"
serve_once "$t/len.http"
"$bin" --url "127.0.0.1:$port" -p hi > "$t/len.out" 2> "$t/len.err" || fail C5 "a complete stream failed: $(cat "$t/len.err")"
grep -q 'hello' "$t/len.out" || fail C5 "reply text lost: $(cat "$t/len.out")"
grep -q 'finish length' "$t/len.err" && grep -q 'cut by max_tokens' "$t/len.err" || fail C5 "finish_reason length is not shown: $(cat "$t/len.err")"
wait || true
echo "baro-chat: PASS (C5)"
