#!/usr/bin/env bash
# usage: bench/front-real-check.sh OUT    (OUT holds engine + baro-front, built on CPU first)
# Real-engine check of the Mojo front (docs/lanes/2026-11-10-next-steps.md step 0.3): the Rust front
# (serve/target/release/baro-serve) and the Mojo front (OUT/baro-front) serve the SAME engine binary
# and pack in turn, one gpu-wait job. 5 chat prompts at temperature 0, no float tool arguments.
# Checks per arm: /health code seen during load, 5 answers, a dropped stream leaves the server
# answering, SIGTERM exits 0. Then the two arms' final answers (after </think>) must be equal prompt by
# prompt. Full-text equality is counted, not required: on the 27B the Rust front's own reasoning text
# changed between two asks of one prompt once its prefix cache was warm (cached_tokens 0 -> 45).
# Engine served: REAL (P31). Window and cap: BARO_TMAX=16384, max_tokens 16000.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: bench/front-real-check.sh OUT}
PACK=${PACK:-.work/engine-pack-q4}
fail() { echo "FAIL $1: $2"; exit 1; }
if [ -z "${FRONT_REAL_INNER:-}" ]; then
  for f in "$OUT/engine" "$OUT/baro-front" serve/target/release/baro-serve; do [ -x "$f" ] || fail build "missing $f"; done
  exec gpu-wait run --timeout 3600 -- env FRONT_REAL_INNER=1 PACK="$PACK" bash -c "cd $PWD && bench/front-real-check.sh $OUT > $PWD/$OUT/check.log 2>&1"
fi
PORT=18431
P=("Name the capital of France in one word." "What is 17 times 3? Answer with the number only." \
   "Write one short sentence about the sea." "List three primary colors, comma separated." \
   "Spell the word cat backwards.")
ans() { awk 'f; /<\/think>/{f=1}' "$1" | sed '/^[[:space:]]*$/d'; }  # the final answer: text after </think>
h() { curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true; }
ask() { curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"temperature\":0,\"max_tokens\":16000}"; }
arm() {
  local name=$1; shift
  BARO_TMAX=16384 "$@" >"$OUT/$name.server.log" 2>&1 & local pid=$!
  : >"$OUT/$name.health"
  for _ in $(seq 600); do c=$(h); echo "$c" >>"$OUT/$name.health"; [ "$c" = 200 ] && break; kill -0 $pid 2>/dev/null || fail "$name" "server died during load"; sleep 1; done
  [ "$(h)" = 200 ] || fail "$name" "no 200 on /health after 600 s"
  for i in "${!P[@]}"; do
    ask "${P[$i]}" >"$OUT/$name.p$i.json"
    jq -er '.choices[0].message.content' "$OUT/$name.p$i.json" >"$OUT/$name.p$i.txt" || fail "$name" "prompt $i: no content, see $OUT/$name.p$i.json"
    ans "$OUT/$name.p$i.txt" >"$OUT/$name.p$i.ans"
    [ -s "$OUT/$name.p$i.ans" ] || fail "$name" "prompt $i: no final answer after </think>"
  done
  timeout 3 curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 500."}],"stream":true,"temperature":0,"max_tokens":16000}' >"$OUT/$name.drop.sse" || true
  ask "${P[0]}" >"$OUT/$name.after-drop.json"
  jq -er '.choices[0].message.content' "$OUT/$name.after-drop.json" >"$OUT/$name.after-drop.txt" || fail "$name" "no answer after a dropped stream"
  ans "$OUT/$name.after-drop.txt" | cmp -s - "$OUT/$name.p0.ans" || fail "$name" "final answer after a dropped stream differs from prompt 0's"
  kill -TERM $pid; local rc=0; wait $pid || rc=$?
  echo "$name: health codes during load: $(sort "$OUT/$name.health" | uniq -c | tr -s ' \n' ' '); SIGTERM exit $rc"
  [ "$rc" = 0 ] || fail "$name" "SIGTERM exit $rc, see $OUT/$name.server.log"
}
arm rust serve/target/release/baro-serve --engine "$OUT/engine" --pack "$PACK" --port $PORT
arm mojo "$OUT/baro-front" "$OUT/engine" "$PACK" --port $PORT
same=0
for i in "${!P[@]}"; do
  cmp -s "$OUT/rust.p$i.ans" "$OUT/mojo.p$i.ans" || fail compare "prompt $i final answers differ: $OUT/{rust,mojo}.p$i.ans"
  cmp -s "$OUT/rust.p$i.txt" "$OUT/mojo.p$i.txt" && same=$((same + 1))
  echo "prompt $i: path rust=$(jq -r .timings.path "$OUT/rust.p$i.json") mojo=$(jq -r .timings.path "$OUT/mojo.p$i.json")"
done
echo "PASS front-real: 5 prompts equal final answers ($same of 5 with equal full text, reasoning tokens flip on near-ties), Rust front vs Mojo front, real engine, same binary and pack"
