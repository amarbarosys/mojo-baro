#!/usr/bin/env bash
# CPU test of tools/baro-chat.mojo against bench/tests/baro-chat-mock-server.py (chunked SSE, lines and UTF-8 split
# across chunks). Cases: one-shot echo round trip (quotes, backslash, tab, newline, control char, non-ASCII, emoji),
# stats line from usage/timings, request fields (max_tokens, temperature, enable_thinking, system), interactive
# history and /reset over stdin, HTTP 400 error, connection refused.
set -euo pipefail
cd "$(dirname "$0")/../.."
t=.work/baro-chat-test; rm -rf "$t"; mkdir -p "$t"
echo "./.venv/bin/mojo build tools/baro-chat.mojo -I . -o .work/baro-chat" | tools/build-many.sh baro-chat > "$t/build.log" 2>&1 || { echo "FAIL build: see $t/build.log"; exit 1; }
export MOCK_LOG=$t/req.log; : > "$MOCK_LOG"
python3 bench/tests/baro-chat-mock-server.py "$t/port" & mp=$!; trap 'kill $mp 2>/dev/null || :' EXIT
for _ in $(seq 50); do [ -s "$t/port" ] && break; sleep 0.1; done
U=127.0.0.1:$(cat "$t/port"); bad=0
fail() { echo "  FAIL $1"; bad=$((bad + 1)); }
msg=$(printf 'say "hi" \\ tab\there\nline2 \001 éß 中文 🙂')
.work/baro-chat --url "$U" --max-tokens 77 --temperature 0.3 --no-think --system "be brief" -p "$msg" > "$t/o1" 2> "$t/e1" || fail "one-shot exit $?"
[ "$(cat "$t/o1")" = "$msg" ] || fail "echo round trip: got $(od -c "$t/o1" | head -3)"
grep -q '\[7 tok · 61.3 tok/s · prefill 0.0 s · cached 4\]' "$t/e1" || fail "stats line: $(cat "$t/e1")"
python3 - "$MOCK_LOG" <<'PY' || bad=$((bad + 1))
import json, sys
r = json.loads(open(sys.argv[1]).read().splitlines()[0])
ok = r["max_tokens"] == 77 and abs(r["temperature"] - 0.3) < 1e-6 and r["chat_template_kwargs"] == {"enable_thinking": False} \
    and r["stream"] is True and r["messages"][0] == {"role": "system", "content": "be brief"} and len(r["messages"]) == 2
if not ok: print("  FAIL request fields", r); sys.exit(1)
PY
: > "$MOCK_LOG"
printf 'one\ntwo\n/reset\nthree\n/exit\n' | .work/baro-chat --url "$U" > "$t/o2" 2> "$t/e2" || fail "interactive exit $?"
n=$(python3 -c "import json,sys; print(' '.join(str(len(json.loads(l)['messages'])) for l in open(sys.argv[1])))" "$MOCK_LOG")
[ "$n" = "1 3 1" ] || fail "history lengths $n, want 1 3 1"
if .work/baro-chat --url "$U" -p ERR > /dev/null 2> "$t/e3"; then fail "400 exited 0"; fi
grep -q '400.*messages is empty' "$t/e3" || fail "400 message: $(cat "$t/e3")"
if .work/baro-chat --url 127.0.0.1:1 -p hi > /dev/null 2> "$t/e4"; then fail "refused exited 0"; fi
grep -q 'cannot connect to 127.0.0.1:1' "$t/e4" || fail "refused message: $(cat "$t/e4")"
[ "$bad" = 0 ] || { echo "FAIL $bad baro-chat checks"; exit 1; }
echo "PASS baro-chat 6/6 checks (echo round trip, stats, fields, history+reset, 400, refused)"
