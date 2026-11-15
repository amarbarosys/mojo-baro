#!/usr/bin/env bash
# Gate: baro-mcp over a pipe with no front running (BARO_MCP_URL points at a closed port).
# initialize, notification (no reply), tools/list (5 tools), settings, explain_setting (ok and unknown),
# capabilities, status (unreachable is isError), unknown method (-32601).
# Planted defect: the same session on a copy where capability kvq says WORKS must FAIL the kvq assertion.
# Template: mojo-neuron/tests/mcp.sh (rpc helper, jq-free python there; jq here for assertions).
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
B=.work/baro-mcp
mkdir -p .work
command -v jq >/dev/null || { echo "FAIL setup: jq missing"; exit 1; }

rm -f "$B"
timeout 240 ./.venv/bin/mojo build tools/mcp/baro_mcp.mojo -I . -I face -I tools/mcp -o "$B" > .work/mcp-build.log 2>&1 \
  || { echo "FAIL build: see .work/mcp-build.log"; exit 1; }
[ -x "$B" ] || { echo "FAIL build: no binary at $B"; exit 1; }

export BARO_MCP_URL=http://127.0.0.1:1   # nothing listens on port 1

session() { # $1 = repo tree, $2 = file for the replies
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
    '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"settings","arguments":{"group":"sampling"}}}' \
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"explain_setting","arguments":{"name":"BARO_TMAX"}}}' \
    '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"explain_setting","arguments":{"name":"NOPE"}}}' \
    '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"capabilities","arguments":{"id":"kvq"}}}' \
    '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"status","arguments":{}}}' \
    '{"jsonrpc":"2.0","id":8,"method":"nope"}' \
    | BARO_MCP_REPO="$1" timeout 60 "$B" > "$2"
}

fails=0
check() { # $1 = step name, rest = command; a false command is a FAIL line, not an exit
  local step=$1; shift
  if "$@"; then echo "ok   $step"; else echo "FAIL $step: assertion false"; fails=$((fails+1)); fi
}
txt() { jq -r --argjson i "$2" 'select(.id==$i) | .result.content[0].text' "$1"; }

REPLIES=.work/mcp-replies.jsonl
session "$REPO" "$REPLIES"

check "initialize echoes 2025-06-18" \
  test "$(jq -r 'select(.id==1) | .result.protocolVersion' "$REPLIES")" = "2025-06-18"
check "notification gets no reply (8 replies for 9 lines, 1 notification)" \
  test "$(wc -l < "$REPLIES")" -eq 8
check "tools/list names the five part-a tools (part b adds more)" \
  bash -c 'jq -e --argjson want "[\"capabilities\",\"explain_setting\",\"models\",\"settings\",\"status\"]" "[.result.tools[].name] as \$n | (\$want - \$n) == []" <(jq -c "select(.id==2)" "$1") >/dev/null' _ "$REPLIES"
check "tools/list names the thirteen tools after part c" \
  test "$(jq -r 'select(.id==2) | .result.tools | length' "$REPLIES")" -eq 13
check "every tool: each required argument is a declared property (schema matches handler)" \
  test "$(jq -r 'select(.id==2) | [.result.tools[] | select((.inputSchema.required // []) - (.inputSchema.properties | keys) | length > 0) | .name] | length' "$REPLIES")" -eq 0
check "settings group=sampling returns >= 8 rows" \
  test "$(txt "$REPLIES" 3 | jq '.settings | length')" -ge 8
check "explain_setting BARO_TMAX: restart true and capabilities contains tmax" \
  test "$(txt "$REPLIES" 4 | jq -r '(.restart == true) and ((.capabilities | index("tmax")) != null)')" = "true"
check "explain_setting NOPE is isError" \
  test "$(jq -r 'select(.id==5) | .result.isError' "$REPLIES")" = "true"
check "capabilities id=kvq has status OFF" \
  test "$(txt "$REPLIES" 6 | jq -r '.[0].status')" = "OFF"
check "status is isError mentioning unreachable" \
  bash -c 'test "$(jq -r "select(.id==7) | .result.isError" "$1")" = true && jq -r "select(.id==7) | .result.content[0].text" "$1" | grep -q unreachable' _ "$REPLIES"
check "unknown method answers -32601" \
  test "$(jq -r 'select(.id==8) | .error.code' "$REPLIES")" = "-32601"

# Planted defect: a copy of docs/ with kvq status WORKS. The kvq assertion must FAIL on it.
PLANT=.work/plant-repo
rm -rf "$PLANT"; mkdir -p "$PLANT/docs"
for e in "$REPO"/docs/*; do ln -s "$e" "$PLANT/docs/$(basename "$e")"; done
rm -f "$PLANT/docs/operate-capabilities.toml"
sed '/^id = "kvq"/,/^status/ s/^status = "OFF"/status = "WORKS"/' "$REPO/docs/operate-capabilities.toml" > "$PLANT/docs/operate-capabilities.toml"
PLANT_REPLIES=.work/mcp-replies-planted.jsonl
session "$(cd "$PLANT" && pwd)" "$PLANT_REPLIES"
if [ "$(txt "$PLANT_REPLIES" 6 | jq -r '.[0].status')" = "OFF" ]; then
  echo "FAIL planted: kvq assertion passed on a WORKS copy (the check is not sensitive)"; fails=$((fails+1))
else
  echo "ok   planted: kvq assertion fails on the WORKS copy"
fi

# Request tools against a fake front (tools/mcp/fake_server.sh, socat). Planted defect: the fake with --bug.
command -v socat >/dev/null || { echo "FAIL setup: socat missing"; exit 1; }
FAKE=$REPO/tools/mcp/fake_server.sh
fake_session() { # $1 = port, $2 = file for the replies, $3 = "--bug" or ""
  local port=$1 out=$2 bug=${3:-} fpid
  "$FAKE" "$port" $bug >/dev/null 2>&1 &
  fpid=$!
  for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"chat","arguments":{"prompt":"hi"}}}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"chat","arguments":{"messages":[{"role":"user","content":"boom"}]}}}' \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"embed","arguments":{"input":"x"}}}' \
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"checkpoint_list","arguments":{}}}' \
    '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"checkpoint_fork","arguments":{"id":"a/b"}}}' \
    '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"checkpoint_fork","arguments":{"id":"abc"}}}' \
    '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"checkpoint_export","arguments":{"id":"abc"}}}' \
    '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"checkpoint_import","arguments":{"name":"../x.baro"}}}' \
    | BARO_MCP_URL="http://127.0.0.1:$port" BARO_MCP_REPO="$REPO" timeout 60 "$B" > "$out"
  kill "$fpid" 2>/dev/null || true
  pkill -P "$fpid" 2>/dev/null || true
  wait "$fpid" 2>/dev/null || true
}
FPORT=$(( 20000 + RANDOM % 20000 ))
FREPLIES=.work/mcp-fake-replies.jsonl
fake_session "$FPORT" "$FREPLIES" ""
check "chat returns canned content with before and after" \
  bash -c 'jq -r "select(.id==1) | .result.content[0].text" "$1" | jq -e ".content == \"canned answer\" and (.before.health_code == 200) and (.after | has(\"queue\"))" >/dev/null' _ "$FREPLIES"
check "chat content 502 is isError containing 502" \
  bash -c 'test "$(jq -r "select(.id==2) | .result.isError" "$1")" = true && jq -r "select(.id==2) | .result.content[0].text" "$1" | grep -q 502' _ "$FREPLIES"
check "embed returns count 1 and dims 5" \
  bash -c 'jq -r "select(.id==3) | .result.content[0].text" "$1" | jq -e ".count == 1 and .dims == 5" >/dev/null' _ "$FREPLIES"
check "checkpoint_list returns the canned list" \
  bash -c 'jq -r "select(.id==4) | .result.content[0].text" "$1" | jq -e ".checkpoints.checkpoints[0].id == \"abc\"" >/dev/null' _ "$FREPLIES"
check "checkpoint_fork a/b is isError bad id" \
  bash -c 'test "$(jq -r "select(.id==5) | .result.isError" "$1")" = true && jq -r "select(.id==5) | .result.content[0].text" "$1" | grep -q "bad id"' _ "$FREPLIES"
check "checkpoint_fork abc returns the body" \
  bash -c 'jq -r "select(.id==6) | .result.content[0].text" "$1" | jq -e ".fork.forked[0] == \"abc-1\"" >/dev/null' _ "$FREPLIES"
check "checkpoint_export is isError binary download not supported" \
  bash -c 'test "$(jq -r "select(.id==7) | .result.isError" "$1")" = true && jq -r "select(.id==7) | .result.content[0].text" "$1" | grep -q "binary download not supported"' _ "$FREPLIES"
check "checkpoint_import rejects a name with a slash" \
  bash -c 'test "$(jq -r "select(.id==8) | .result.isError" "$1")" = true' _ "$FREPLIES"

# Planted defect: the fake answers contentx; the chat content assertion must FAIL on it.
PBUG=.work/mcp-fake-replies-bug.jsonl
fake_session "$((FPORT + 1))" "$PBUG" "--bug"
if jq -r 'select(.id==1) | .result.content[0].text' "$PBUG" | jq -e '.content == "canned answer"' >/dev/null 2>&1; then
  echo "FAIL planted-fake: chat content assertion passed on the contentx fake (the check is not sensitive)"; fails=$((fails+1))
else
  echo "ok   planted-fake: chat content assertion fails on the contentx fake"
fi

# Part c: set_and_restart and preset against a fake owned engine (no GPU, no real model).
# The temp repo holds a fake tools/baro (serve MODEL --port P runs fake_server on P and keeps its environment),
# the real face/stop-engine.sh, a model stub and the two docs. Planted: the fake drops its environment.
make_repo() { # $1 dir, $2 "bug" to plant the environment drop
  local d=$1 bug=${2:-}
  rm -rf "$d"; mkdir -p "$d/tools" "$d/face" "$d/.work" "$d/models" "$d/docs"
  cp "$REPO/face/stop-engine.sh" "$d/face/"
  cp "$REPO/docs/settings.toml" "$REPO/docs/operate-capabilities.toml" "$d/docs/"
  : > "$d/models/fake.gguf"
  cat > "$d/tools/baro" <<FAKE
#!/usr/bin/env bash
# fake engine: serve MODEL --port P
if [ "$bug" = bug ] && [ -z "\${FAKE_CLEARED:-}" ]; then exec env -i PATH="\$PATH" FAKE_CLEARED=1 bash "\$0" "\$@"; fi
port=""
while [ \$# -gt 0 ]; do case "\$1" in --port) port=\$2 ;; esac; shift; done
"$FAKE" "\$port" &
wait
FAKE
  chmod +x "$d/tools/baro"
}
start_owned() { # $1 dir (absolute), $2 port; the pidfile goes into the fake tree, never the real repo
  (cd "$1" && setsid sh -c "echo \$\$ > .work/gui-front.pid; exec env BARO_TMAX=16384 tools/baro serve '$1/models/fake.gguf' --port $2" > "$1/.work/gui-front.log" 2>&1 < /dev/null &)
  for _ in $(seq 100); do (exec 3<>"/dev/tcp/127.0.0.1/$2") 2>/dev/null && return 0; sleep 0.1; done
  return 1
}
SRPORT=$(( 30000 + RANDOM % 20000 ))
GOOD=.work/mcp-part-c-tree
make_repo "$REPO/$GOOD"
start_owned "$REPO/$GOOD" "$SRPORT" || { echo "FAIL setup: fake engine did not answer"; fails=$((fails+1)); }
srsession() { # $1 tree, $2 port, $3 replies file
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"set_and_restart","arguments":{"settings":{"BARO_TMAX":20000}}}}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"set_and_restart","arguments":{"settings":{"BARO_KVQ":"int8"}}}}' \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"set_and_restart","arguments":{"settings":{"NOPE":1}}}}' \
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"set_and_restart","arguments":{"settings":{"BARO_TMAX":"a;b"}}}}' \
    '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"set_and_restart","arguments":{"settings":{"BARO_TMAX":100}}}}' \
    '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"preset","arguments":{"name":"Low VRAM"}}}' \
    '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"preset","arguments":{"name":"Fast","apply":true}}}' \
    | BARO_MCP_URL="http://127.0.0.1:$2" BARO_MCP_REPO="$1" timeout 600 "$B" > "$3"
}
SRREPLIES=.work/mcp-part-c-replies.jsonl
srsession "$REPO/$GOOD" "$SRPORT" "$SRREPLIES"
srtxt() { jq -r --argjson i "$2" 'select(.id==$i) | .result.content[0].text' "$1"; }
check "set_and_restart BARO_TMAX=20000: took_effect true and new running_env has 20000" \
  bash -c 'srtxt() { jq -r --argjson i 1 "select(.id==\$i) | .result.content[0].text" "$1"; }; srtxt "$1" | jq -e ".took_effect.BARO_TMAX == true and .after.running_env.BARO_TMAX == \"20000\"" >/dev/null' _ "$SRREPLIES"
check "set_and_restart BARO_KVQ is isError naming the build flag" \
  bash -c 'test "$(jq -r "select(.id==2) | .result.isError" "$1")" = true && jq -r "select(.id==2) | .result.content[0].text" "$1" | grep -q "build flag"' _ "$SRREPLIES"
check "set_and_restart unknown key is isError" \
  bash -c 'test "$(jq -r "select(.id==3) | .result.isError" "$1")" = true && jq -r "select(.id==3) | .result.content[0].text" "$1" | grep -q "unknown setting"' _ "$SRREPLIES"
check "set_and_restart value a;b is isError (charset)" \
  bash -c 'test "$(jq -r "select(.id==4) | .result.isError" "$1")" = true' _ "$SRREPLIES"
check "set_and_restart BARO_TMAX=100 is isError (floor)" \
  bash -c 'test "$(jq -r "select(.id==5) | .result.isError" "$1")" = true && jq -r "select(.id==5) | .result.content[0].text" "$1" | grep -q 16384' _ "$SRREPLIES"
check "preset Low VRAM lists BARO_KVQ under refused" \
  bash -c 'jq -r "select(.id==6) | .result.content[0].text" "$1" | jq -e ".refused | map(.key) | index(\"BARO_KVQ\") != null" >/dev/null' _ "$SRREPLIES"
check "preset Fast apply restarts and BARO_SPEC is 1 in running_env" \
  bash -c 'jq -r "select(.id==7) | .result.content[0].text" "$1" | jq -e ".result.after.running_env.BARO_SPEC == \"1\"" >/dev/null' _ "$SRREPLIES"
kill "$(cat "$REPO/$GOOD/.work/gui-front.pid")" 2>/dev/null || true

# Planted defect: the fake drops its environment, so BARO_TMAX never reaches running_env: took_effect must be false.
BAD=.work/mcp-part-c-planted
PPORT=$((SRPORT + 1))
make_repo "$REPO/$BAD" bug
start_owned "$REPO/$BAD" "$PPORT" || { echo "FAIL setup: planted fake engine did not answer"; fails=$((fails+1)); }
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"set_and_restart","arguments":{"settings":{"BARO_TMAX":20000}}}}' \
  | BARO_MCP_URL="http://127.0.0.1:$PPORT" BARO_MCP_REPO="$REPO/$BAD" timeout 600 "$B" > .work/mcp-part-c-planted.jsonl
# The planted run must reach the restart (isError false, took_effect present); otherwise the check proves nothing.
if ! jq -e 'select(.id==1) | .result.isError == false and (.result.content[0].text | fromjson | .took_effect | has("BARO_TMAX"))' .work/mcp-part-c-planted.jsonl >/dev/null 2>&1; then
  echo "FAIL planted-env-drop: the planted restart did not complete (no took_effect), so the check is vacuous"; fails=$((fails+1))
elif jq -r 'select(.id==1) | .result.content[0].text' .work/mcp-part-c-planted.jsonl | jq -e '.took_effect.BARO_TMAX == true' >/dev/null 2>&1; then
  echo "FAIL planted-env-drop: took_effect true although the engine dropped its environment"; fails=$((fails+1))
else
  echo "ok   planted-env-drop: took_effect assertion fails on the environment-dropping engine"
fi
kill "$(cat "$REPO/$BAD/.work/gui-front.pid")" 2>/dev/null || true

# Cold start: no engine is up and none is owned; set_and_restart with a model must start one (found by the real-engine run).
COLD=.work/mcp-part-c-cold
CPORT=$((SRPORT + 2))
make_repo "$REPO/$COLD"
printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"set_and_restart\",\"arguments\":{\"settings\":{\"BARO_TMAX\":20000},\"model\":\"$REPO/$COLD/models/fake.gguf\",\"port\":$CPORT}}}" \
  | BARO_MCP_URL="http://127.0.0.1:$CPORT" BARO_MCP_REPO="$REPO/$COLD" timeout 600 "$B" > .work/mcp-part-c-cold.jsonl
check "cold start: set_and_restart with a model starts an engine and BARO_TMAX is in running_env" \
  bash -c 'jq -e "select(.id==1) | .result.isError == false and (.result.content[0].text | fromjson | .restarted == true and .took_effect.BARO_TMAX == true)" "$1" >/dev/null' _ .work/mcp-part-c-cold.jsonl
kill "$(cat "$REPO/$COLD/.work/gui-front.pid")" 2>/dev/null || true

# Dead engine: tools/baro exits at once; set_and_restart must say so with the log tail, not wait 180 s for /health.
DEAD=.work/mcp-part-c-dead
make_repo "$REPO/$DEAD"
printf '#!/usr/bin/env bash\necho "baro: boom" >&2\nexit 1\n' > "$REPO/$DEAD/tools/baro"
printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"set_and_restart\",\"arguments\":{\"settings\":{\"BARO_TMAX\":20000},\"model\":\"$REPO/$DEAD/models/fake.gguf\",\"port\":$((SRPORT + 3))}}}" \
  | BARO_MCP_URL="http://127.0.0.1:$((SRPORT + 3))" BARO_MCP_REPO="$REPO/$DEAD" timeout 60 "$B" > .work/mcp-part-c-dead.jsonl
check "dead engine: set_and_restart is isError within 60 s and quotes the engine log" \
  bash -c 'jq -e "select(.id==1) | .result.isError == true and (.result.content[0].text | contains(\"engine exited\") and contains(\"boom\"))" "$1" >/dev/null' _ .work/mcp-part-c-dead.jsonl

if [ "$fails" -ne 0 ]; then echo "FAIL test_mcp: $fails assertion(s)"; exit 1; fi
echo "PASS test_mcp"
