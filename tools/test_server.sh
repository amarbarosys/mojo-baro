#!/usr/bin/env bash
# usage: tools/test_server.sh [OUTDIR]      (needs the GPU; re-execs itself under gpu-wait)
# Gate for serve/src (baro-serve) + serve/engine.mojo BARO_SERVE=1:
#   cargo clippy clean, cargo test (protocol parser), engine + server build,
#   /health, one completion by token ids == ref-tokens-64 (tools/check-tokens.sh),
#   one streamed (SSE) request == ref, two concurrent requests queued == ref,
#   a stop-string ends generation early (M2 control block), a mid-stream
#   cancel stops generation within one window and the next request still
#   matches ref, a rejected over-length request, clean shutdown (SIGINT:
#   server exit 0, engine gone).
# Pack: BARO_PACK (default .work/engine-pack-q4); ref = $pack/ref-tokens-64.txt.
# Set BARO_CHAT_TEMPLATE_FILE to exercise a custom Jinja template file.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --priority 60 --timeout 1800 -- "$0" "$@"
fi
out=${1:-.work/server-test}; mkdir -p "$out"; : > "$out/SUMMARY.txt"
pack=${BARO_PACK:-.work/engine-pack-q4}; ref=$pack/ref-tokens-64.txt
ok() { echo "PASS $1: $2" | tee -a "$out/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$out/SUMMARY.txt"; exit 1; }
[ -f "$ref" ] || die setup "no $ref"
prompt_json=$(python3 -c "import sys; print([int(x) for x in open('$pack/prompt-tokens.txt').read().split()])")

(cd serve && cargo clippy --release --all-targets -- -D warnings) > "$out/clippy.log" 2>&1 || die clippy "$(grep -m1 -E '^(error|warning)' "$out/clippy.log")"
ok clippy "no warnings"
(cd serve && cargo test --release) > "$out/cargo-test.log" 2>&1 || die cargo-test "$(grep -m1 -E 'FAILED|error' "$out/cargo-test.log")"
ok cargo-test "$(grep -m1 -oE '[0-9]+ passed' "$out/cargo-test.log")"
(cd serve && cargo build --release) > "$out/cargo-build.log" 2>&1 || die build "$(grep -m1 error "$out/cargo-build.log")"
if [ ! -x .work/engine ] || [ serve/engine.mojo -nt .work/engine ] || [ serve/registry.mojo -nt .work/engine ]; then
  ./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/engine > "$out/build-engine.log" 2>&1 || die build "$(grep -m1 error: "$out/build-engine.log" | cut -c1-160)"
fi
ok build "engine + baro-serve"

# --- start the server on a free port -------------------------------------------
template_args=()
[ -z "${BARO_CHAT_TEMPLATE_FILE:-}" ] || template_args+=(--chat-template-file "$BARO_CHAT_TEMPLATE_FILE")
BARO_PACK=$pack ./serve/target/release/baro-serve --engine .work/engine --pack "$pack" --port 0 "${template_args[@]}" \
  > "$out/server.stdout" 2> "$out/server.stderr" &
srv=$!
cleanup() { kill -9 "$srv" 2>/dev/null; pkill -9 -P "$srv" 2>/dev/null; }
trap cleanup EXIT
for _ in $(seq 1 600); do grep -q '^listening on' "$out/server.stdout" && break; kill -0 "$srv" 2>/dev/null || die start "server exited: $(tail -3 "$out/server.stderr")"; sleep 0.5; done
url=$(grep -m1 -oE 'http://[0-9.:]+' "$out/server.stdout") || die start "no listening line"
ok start "$url ($(grep -oE 'pack loaded in [0-9.]+ s' "$out/server.stderr" | head -1))"

curl -sf "$url/health" > "$out/health.json" || die health "curl failed"
python3 -c "import json,sys; d=json.load(open('$out/health.json')); sys.exit(0 if d['status']=='ok' and d['limits']['tmax']>0 else 1)" || die health "$(cat "$out/health.json")"
ok health "$(cat "$out/health.json")"
curl -sf "$url/v1/models" > "$out/models.json" || die models "curl failed"
grep -q '"object":"list"' "$out/models.json" || die models "$(cat "$out/models.json")"
ok models "$(python3 -c "import json; print(json.load(open('$out/models.json'))['data'][0]['id'])")"

# --- one completion by token ids: must equal ref-tokens-64 ------------------------
curl -sf "$url/v1/completions" -H 'content-type: application/json' \
  -d "{\"prompt\": $prompt_json, \"max_tokens\": 64, \"spec\": false}" > "$out/cmpl.json" || die completion "curl failed: $(cat "$out/cmpl.json")"
python3 -c "import json; d=json.load(open('$out/cmpl.json')); print('GENERATED:', ' '.join(map(str, d['choices'][0]['tokens'])), '')" > "$out/cmpl.gen" || die completion "$(cat "$out/cmpl.json")"
tools/check-tokens.sh "$ref" "$out/cmpl.gen" > "$out/cmpl.check" 2>&1 || die completion "$(cat "$out/cmpl.check")"
ok completion "$(cat "$out/cmpl.check"); $(python3 -c "import json; t=json.load(open('$out/cmpl.json'))['timings']; print('tok/s_gen', round(t['tok_s_gen'],1), 'prefill_s', round(t['prefill_s'],4))")"

# --- streamed request -------------------------------------------------------------
curl -sfN "$url/v1/completions" -H 'content-type: application/json' \
  -d "{\"prompt\": $prompt_json, \"max_tokens\": 64, \"spec\": false, \"stream\": true}" > "$out/stream.sse" || die stream "curl failed"
python3 - "$out" <<'PY' || die stream "bad SSE (see stream.sse)"
import json, sys
out = sys.argv[1]
datas = [l[6:] for l in open(f"{out}/stream.sse").read().split("\n") if l.startswith("data: ")]
assert datas and datas[-1] == "[DONE]", "no [DONE]"
toks = []; fin = None
for d in datas[:-1]:
    j = json.loads(d); c = j["choices"][0]
    if c["finish_reason"] is None: toks += c["tokens"]
    else: fin = j
assert fin is not None and fin["choices"][0]["finish_reason"] == "length", fin
assert fin["tokens"] == toks and fin["usage"]["completion_tokens"] == len(toks), "final chunk disagrees with deltas"
open(f"{out}/stream.gen", "w").write("GENERATED: " + " ".join(map(str, toks)) + " \n")
print(len(datas) - 2, "token chunks")
PY
tools/check-tokens.sh "$ref" "$out/stream.gen" > "$out/stream.check" 2>&1 || die stream "$(cat "$out/stream.check")"
ok stream "$(cat "$out/stream.check")"

# --- two concurrent requests: both queue, both equal ref --------------------------
qpids=()
for i in 1 2; do
  curl -sf "$url/v1/completions" -H 'content-type: application/json' \
    -d "{\"prompt\": $prompt_json, \"max_tokens\": 64, \"spec\": false}" > "$out/queue$i.json" &
  qpids+=($!)
done
sleep 0.3; curl -sf "$url/health" > "$out/health-busy.json"
wait "${qpids[@]}"
for i in 1 2; do
  python3 -c "import json; d=json.load(open('$out/queue$i.json')); print('GENERATED:', ' '.join(map(str, d['choices'][0]['tokens'])), '')" > "$out/queue$i.gen" || die queue "request $i: $(cat "$out/queue$i.json")"
  tools/check-tokens.sh "$ref" "$out/queue$i.gen" > "$out/queue$i.check" 2>&1 || die queue "request $i: $(cat "$out/queue$i.check")"
done
ok queue "2 concurrent requests both match ref; /health during: $(python3 -c "import json; print('queue', json.load(open('$out/health-busy.json'))['queue'])")"

# --- text endpoints (only when the pack has a tokenizer.json) ----------------------------
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/health.json'))['tokenizer'] else 1)"; then
  curl -sf "$url/detokenize" -H 'content-type: application/json' -d "{\"tokens\": $prompt_json}" > "$out/detok.json" || die detokenize "curl failed"
  python3 - "$out" "$url" "$prompt_json" <<'PY' || die tokenize "roundtrip (see tokenize.json)"
import json, sys, urllib.request
out, url, prompt = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
text = json.load(open(f"{out}/detok.json"))["content"]
req = urllib.request.Request(f"{url}/tokenize", data=json.dumps({"content": text}).encode(), headers={"content-type": "application/json"})
ids = json.load(urllib.request.urlopen(req))["tokens"]
json.dump({"text": text, "tokens": ids, "prompt": prompt}, open(f"{out}/tokenize.json", "w"))
assert ids == prompt, f"tokenize(detokenize(prompt)) = {ids} != {prompt}"
PY
  ok tokenize "detokenize/tokenize roundtrip on the receipt prompt: $(python3 -c "import json; print(repr(json.load(open('$out/tokenize.json'))['text']))")"
  python3 - "$out" "$url" <<'PY' || die completion-text "string prompt (see cmpl-text.json)"
import json, sys, urllib.request
out, url = sys.argv[1], sys.argv[2]
text = json.load(open(f"{out}/detok.json"))["content"]
req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps({"prompt": text, "max_tokens": 64, "spec": False}).encode(), headers={"content-type": "application/json"})
d = json.load(urllib.request.urlopen(req)); json.dump(d, open(f"{out}/cmpl-text.json", "w"))
open(f"{out}/cmpl-text.gen", "w").write("GENERATED: " + " ".join(map(str, d["choices"][0]["tokens"])) + " \n")
PY
  tools/check-tokens.sh "$ref" "$out/cmpl-text.gen" > "$out/cmpl-text.check" 2>&1 || die completion-text "$(cat "$out/cmpl-text.check")"
  ok completion-text "$(cat "$out/cmpl-text.check"); text $(python3 -c "import json; print(repr(json.load(open('$out/cmpl-text.json'))['choices'][0]['text'][:40]))")"
  code=$(curl -s -o "$out/chat.json" -w '%{http_code}' "$url/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "spec": false}')
  [ "$code" = 200 ] || die chat "expected 200, got $code: $(cat "$out/chat.json")"
  python3 -c "import json; d=json.load(open('$out/chat.json')); assert d['choices'][0]['message']['role']=='assistant'; assert 1 <= len(d['choices'][0]['tokens']) <= 16; print(d['choices'][0]['finish_reason'], repr(d['choices'][0]['message']['content'][:60]))" > "$out/chat.check" || die chat "$(cat "$out/chat.json")"
  ok chat "$(cat "$out/chat.check")"
  if [ -n "${BARO_CHAT_TEMPLATE_FILE:-}" ]; then
    grep -q 'CUSTOM_TEMPLATE_MARKER' "$out/server.stderr" || die custom-template "chat request did not use $BARO_CHAT_TEMPLATE_FILE"
    ok custom-template "chat request rendered through $BARO_CHAT_TEMPLATE_FILE"
  fi
  code=$(curl -s -o "$out/overflow.json" -w '%{http_code}' "$url/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 1000000, "spec": false}')
  [ "$code" = 400 ] || die overflow "expected 400, got $code: $(cat "$out/overflow.json")"
  python3 -c "import json; e=json.load(open('$out/overflow.json'))['error']; assert e['type']=='exceed_context_size_error', e; assert e['n_prompt_tokens']>0 and e['n_ctx']>0, e; print(e['type'], e['n_prompt_tokens'], e['n_ctx'])" > "$out/overflow.check" || die overflow "$(cat "$out/overflow.json")"
  ok overflow "$(cat "$out/overflow.check")"
  curl -sfN "$url/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "spec": false, "stream": true}' > "$out/chat.sse" || die chat-stream "curl failed"
  python3 - "$out" <<'PY' > "$out/chat-stream.check" || die chat-stream "$(cat "$out/chat-stream.check")"
import json, sys
out = sys.argv[1]
datas = [l[6:] for l in open(f"{out}/chat.sse").read().split("\n") if l.startswith("data: ")]
assert datas[-1] == "[DONE]"
chunks = [json.loads(d) for d in datas[:-1]]
assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
assert chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop"), chunks[-1]
text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
full = json.load(open(f"{out}/chat.json"))["choices"][0]["message"]["content"]
assert text == full, (text, full)
print(len(chunks) - 1, "delta chunks; streamed text == non-streamed text")
PY
  ok chat-stream "$(cat "$out/chat-stream.check")"

  # --- stop strings end generation early (M2 control block) ------------------------
  python3 - "$out" "$url" "$ref" "$prompt_json" <<'PY' > "$out/stop.check" || die stop "$(cat "$out/stop.check")"
import json, sys, urllib.request
out, url, ref, prompt = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
ref_ids = [int(x) for x in open(ref).read().split()]

def post(path, body):
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))

# A stop string built from the reference continuation's own first k tokens,
# kept only if detokenize->tokenize round-trips exactly (so the engine's
# token-level match is guaranteed to land, not a tokenizer-boundary guess).
stop_k, stop_text = None, None
for k in range(1, 9):
    text = post("/detokenize", {"tokens": ref_ids[:k]})["content"]
    if post("/tokenize", {"content": text})["tokens"] == ref_ids[:k]:
        stop_k, stop_text = k, text
        break
assert stop_k is not None, f"no exact round-trip in the first 8 ref tokens: {ref_ids[:8]}"

d = post("/v1/completions", {"prompt": prompt, "max_tokens": 64, "spec": False, "stop": stop_text})
toks = d["choices"][0]["tokens"]
assert d["choices"][0]["finish_reason"] == "stop", d
assert toks == ref_ids[:stop_k], (toks, ref_ids[:stop_k])
print(f"stopped at token {stop_k} ({stop_text!r}) of max_tokens 64, matches ref prefix, finish_reason=stop")
PY
  ok stop "$(cat "$out/stop.check")"

  # --- cancel mid-generation (M2 control block) -------------------------------------
  : > "$out/cancel.sse"
  curl -sfN "$url/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 256, "spec": false, "stream": true}' > "$out/cancel.sse" &
  cpid=$!
  req_id=""
  for _ in $(seq 1 200); do
    req_id=$(python3 -c "
import json
for l in open('$out/cancel.sse'):
    if l.startswith('data: ') and l.strip() != 'data: [DONE]':
        try:
            print(json.loads(l[6:])['id']); break
        except Exception:
            pass
" 2>/dev/null) || true
    [ -n "$req_id" ] && break
    kill -0 "$cpid" 2>/dev/null || break
    sleep 0.02
  done
  [ -n "$req_id" ] || die cancel "no request id observed in the SSE stream before it ended"
  cancel_resp=$(curl -sf "$url/v1/cancel" -H 'content-type: application/json' -d "{\"id\": \"$req_id\"}") || die cancel "curl to /v1/cancel failed"
  wait "$cpid" || true
  python3 - "$out" "$cancel_resp" <<'PY' > "$out/cancel.check" || die cancel "$(cat "$out/cancel.check")"
import json, sys
out, cancel_resp = sys.argv[1], sys.argv[2]
datas = [l[6:] for l in open(f"{out}/cancel.sse").read().split("\n") if l.startswith("data: ")]
assert datas and datas[-1] == "[DONE]", "no [DONE]"
chunks = [json.loads(d) for d in datas[:-1]]
fin = chunks[-1]["choices"][0]["finish_reason"]
n_tok = len(chunks) - 1
assert json.loads(cancel_resp)["cancelled"] is True, cancel_resp
assert fin == "cancelled", chunks[-1]
assert n_tok < 256, n_tok
print(f"cancelled after {n_tok} of 256 tokens, /v1/cancel -> {cancel_resp.strip()}")
PY
  ok cancel "$(cat "$out/cancel.check")"
  curl -sf "$url/v1/completions" -H 'content-type: application/json' \
    -d "{\"prompt\": $prompt_json, \"max_tokens\": 64, \"spec\": false}" > "$out/postcancel.json" || die cancel-recovery "post-cancel request: curl failed"
  python3 -c "import json; d=json.load(open('$out/postcancel.json')); print('GENERATED:', ' '.join(map(str, d['choices'][0]['tokens'])), '')" > "$out/postcancel.gen" || die cancel-recovery "$(cat "$out/postcancel.json")"
  tools/check-tokens.sh "$ref" "$out/postcancel.gen" > "$out/postcancel.check" 2>&1 || die cancel-recovery "$(cat "$out/postcancel.check")"
  ok cancel-recovery "next request after cancel: $(cat "$out/postcancel.check")"
else
  echo "SKIP text endpoints: no tokenizer.json in $pack" | tee -a "$out/SUMMARY.txt"
fi

# --- rejected over-length request ---------------------------------------------------
code=$(curl -s -o "$out/toolong.json" -w '%{http_code}' "$url/v1/completions" -H 'content-type: application/json' \
  -d "{\"prompt\": $prompt_json, \"max_tokens\": 100000}")
[ "$code" = 400 ] || die reject "expected 400, got $code: $(cat "$out/toolong.json")"
ok reject "400 $(python3 -c "import json; print(json.load(open('$out/toolong.json'))['error']['message'])")"

# --- clean shutdown ------------------------------------------------------------------
eng_pids=$(pgrep -P "$srv" || true)
kill -INT "$srv"; wait "$srv"; rc=$?
[ "$rc" = 0 ] || die shutdown "server exit $rc"
for _ in $(seq 1 20); do alive=0; for p in $eng_pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 0.5; done
[ "$alive" = 0 ] || die shutdown "engine child still alive: $eng_pids"
trap - EXIT
ok shutdown "server exit 0, engine child exited"
echo "ALL PASS" | tee -a "$out/SUMMARY.txt"
