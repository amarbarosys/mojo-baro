#!/usr/bin/env bash
set -euo pipefail
trap 'echo "FAIL line $LINENO: $BASH_COMMAND"' ERR
for tool in curl jq python3 cargo; do
    command -v "$tool" > /dev/null || { echo "FAIL preflight: $tool not installed"; exit 1; }
done

cd "$(dirname "$0")/.."
root=$PWD
out=${1:-.work/team-C/codex/p6-pwa}
pack="$out/fake-pack"
bin=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
tts_bin=${BARO_TTS_BIN:-$root/bench/fixtures/fake-tts.sh}
export BARO_STATE_HMAC_KEY=${BARO_STATE_HMAC_KEY:-p6-gate-state-key}
mkdir -p "$out" "$pack"
: > "$out/SUMMARY.txt"

pass() { echo "PASS $1: $2" | tee -a "$out/SUMMARY.txt"; }
fail() { echo "FAIL $1: $2" | tee -a "$out/SUMMARY.txt" >&2; exit 1; }

cleanup() {
    if [ -n "${srv:-}" ] && kill -0 "$srv" 2>/dev/null; then
        kill "$srv" 2>/dev/null || true
        wait "$srv" 2>/dev/null || true
    fi
    if [ -n "${browser_port:-}" ]; then
        ~/iTools/bin/cdp-headless --kill "$browser_port" >> "$out/cdp-kill.log" 2>&1 || true
    fi
}
trap cleanup EXIT

[ -x bench/fixtures/fake-engine.py ] || fail setup "fake engine is not executable"
[ -x bench/fixtures/fake-tts.sh ] || fail setup "fake TTS runner is not executable"
[ -x "$bin" ] || fail setup "build $bin first"

python3 - "$pack/tokenizer.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
tokenizer = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
    "model": {
        "type": "WordLevel",
    "vocab": {"<unk>": 0, "hello": 1, "world": 2, "!": 3, "<tool_call>": 4, "<function=get_weather>": 5, "<parameter=city>": 6, "Paris": 7, "</parameter>": 8, "</function>": 9, "</tool_call>": 10},
        "unk_token": "<unk>",
    },
}
path.write_text(json.dumps(tokenizer))
PY
printf '{"add_bos":false}\n' > "$pack/tokenizer-meta.json"

BARO_TTS_BIN="$tts_bin" \
BARO_TTS_OUT_DIR="$out/tts" \
"$root/$bin" \
    --engine "$root/bench/fixtures/fake-engine.py" \
    --pack "$pack" --port 0 \
    > "$out/server.stdout" 2> "$out/server.stderr" &
srv=$!
for _ in $(seq 1 200); do
    if grep -q '^listening on ' "$out/server.stdout"; then
        break
    fi
    kill -0 "$srv" 2>/dev/null || fail start "server exited: $(tail -3 "$out/server.stderr")"
    sleep 0.05
done
url=$(grep -m1 -oE 'http://[^ ]+' "$out/server.stdout") || fail start "no listening line"
pass start "$url"

curl -fsS "$url/v1/audio/speech" -H 'content-type: application/json' \
    -d '{"input":"hello from baro","response_format":"wav"}' \
    -o "$out/speech.wav" || fail speech "TTS endpoint failed"
python3 - "$out/speech.wav" <<'PY'
import sys, wave
with wave.open(sys.argv[1], "rb") as wav:
    assert wav.getnchannels() == 1 and wav.getframerate() > 0
    assert wav.getnframes() > 0
PY
pass speech "TTS endpoint returned a valid WAV"

pull_model=$(basename "$pack")
curl -fsS -N "$url/api/pull" -H 'content-type: application/json' \
    -d "{\"model\":\"$pull_model\"}" > "$out/pull-stream.ndjson" || fail pull "loaded model pull failed"
jq -e 'select(.status == "success")' "$out/pull-stream.ndjson" >/dev/null || fail pull "success status missing"
pull_status=$(curl -sS -o "$out/pull-missing.json" -w '%{http_code}' "$url/api/pull" \
    -H 'content-type: application/json' -d '{"model":"missing-model","stream":false}')
[ "$pull_status" = 404 ] || fail pull "missing model status=$pull_status"
pass pull "loaded model pull streams success; unavailable model returns 404"

python3 - "$out/unsigned-state.baro" <<'PY'
import hashlib
import pathlib
import struct
import sys

b = bytearray(256)
b[:4] = b"LAT1"
struct.pack_into("<HBB", b, 4, 1, 1, 1)
b[192:224] = hashlib.sha256(b"").digest()
pathlib.Path(sys.argv[1]).write_bytes(b)
PY
hmac_status=$(curl -sS -o "$out/hmac-reject.json" -w '%{http_code}' \
    -H 'content-type: application/vnd.baro.state' --data-binary "@$out/unsigned-state.baro" \
    "$url/v1/state/import")
[ "$hmac_status" = 409 ] || fail hmac "unsigned import status=$hmac_status"
grep -q '"field":"hmac"' "$out/hmac-reject.json" || fail hmac "unsigned import was not rejected by HMAC: $(cat "$out/hmac-reject.json")"
pass hmac "unsigned state rejected when BARO_STATE_HMAC_KEY is set"

curl -fsS "$url/v1/completions" -H 'content-type: application/json' \
    -d '{"prompt":[1],"max_tokens":3,"spec":false,"hidden":true,"logits_topk":1}' \
    > "$out/latent.json"
jq -e '.hidden | length == 3 and .[0][0] > 0.09 and .[0][1] > 0.19' "$out/latent.json" >/dev/null || fail latent "hidden stream missing: $(cat "$out/latent.json")"
jq -e '.logits_topk | length == 3 and .[0][0].id == 7 and .[0][0].logit == 3.5' "$out/latent.json" >/dev/null || fail latent "logits_topk stream missing: $(cat "$out/latent.json")"
pass latent "HTTP completion returns hidden and logits_topk streams"
curl -fsS -N "$url/v1/completions" -H 'content-type: application/json' \
    -d '{"prompt":[1],"max_tokens":1,"stream":true,"spec":false,"hidden":true,"logits_topk":1}' \
    > "$out/latent-sse.txt"
grep -q '"hidden"' "$out/latent-sse.txt" || fail latent-sse "SSE hidden chunk missing"
grep -q '"logits_topk"' "$out/latent-sse.txt" || fail latent-sse "SSE logits_topk chunk missing"
pass latent-sse "SSE emits hidden and logits_topk chunks"
curl -fsS -N "$url/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"weather"}],"max_tokens":7,"stream":true,"spec":false}' \
    > "$out/tool-sse.txt"
grep -q '"tool_calls"' "$out/tool-sse.txt" || fail tool-sse "SSE tool_calls delta missing"
grep -q '"arguments"' "$out/tool-sse.txt" || fail tool-sse "SSE tool arguments delta missing"
pass tool-sse "SSE emits tool-call name and argument deltas"

curl -fsS "$url/" > "$out/index.html" || fail assets "GET / failed"
python3 - "$url" "$out/index.html" "$out/assets.tsv" <<'PY'
import pathlib
import re
import sys
import urllib.request
from urllib.parse import urljoin

base, html_path, receipt_path = sys.argv[1:]
html = pathlib.Path(html_path).read_text()
refs = sorted(set(re.findall(r'(?:src|href)=["\']([^"\']+)["\']', html)))
if not refs:
    raise SystemExit("index has no referenced assets")
rows = []
for ref in refs:
    if ref.startswith(("#", "http:", "https:", "data:")):
        continue
    target = urljoin(base + "/", ref)
    with urllib.request.urlopen(target, timeout=10) as response:
        body = response.read()
        content_type = response.headers.get("content-type", "")
        if response.status != 200:
            raise SystemExit(f"{ref}: HTTP {response.status}")
        if ref.endswith((".js", ".mjs")) and "javascript" not in content_type:
            raise SystemExit(f"{ref}: wrong content type {content_type}")
        if ref.endswith(".css") and "text/css" not in content_type:
            raise SystemExit(f"{ref}: wrong content type {content_type}")
        if ref.endswith(".webmanifest") and "json" not in content_type:
            raise SystemExit(f"{ref}: wrong content type {content_type}")
        rows.append(f"{ref}\t{response.status}\t{content_type}\t{len(body)}")
pathlib.Path(receipt_path).write_text("\n".join(rows) + "\n")
print(f"{len(rows)} assets: 200 with expected content types")
PY
pass assets "GET / and $(wc -l < "$out/assets.tsv") referenced assets returned 200"

python3 - "$url" "$out/manifest.json" <<'PY'
import json
import pathlib
import sys
import urllib.request

url, output = sys.argv[1:]
with urllib.request.urlopen(url + "/web/manifest.webmanifest", timeout=10) as response:
    manifest = json.load(response)
if not manifest.get("name") or not manifest.get("start_url"):
    raise SystemExit("manifest needs name and start_url")
pathlib.Path(output).write_text(json.dumps(manifest, indent=2))
print(f"manifest name={manifest['name']!r} start_url={manifest['start_url']!r}")
PY
pass manifest "manifest parses"

browser_info=$(~/iTools/bin/cdp-headless --width 400 --height 860 2> "$out/cdp-headless.stderr") || fail browser "cdp-headless failed"
printf '%s\n' "$browser_info" > "$out/cdp-headless.log"
browser_port=$(printf '%s\n' "$browser_info" | tail -n1)
[[ "$browser_port" =~ ^[0-9]+$ ]] || fail browser "invalid CDP port $browser_port"

python3 ~/iTools/browser-cdp/page-shot/page-shot.py "$browser_port" "$url/" "$out/p6-pwa-400x860.png" --size=400x860 --wait=1
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" \
    'navigator.serviceWorker.ready.then(r => r.active ? r.active.scriptURL : "")' \
    --timeout 20 > "$out/sw-url.txt"
grep -q 'sw.js' "$out/sw-url.txt" || fail service-worker "service worker did not become active: $(cat "$out/sw-url.txt")"
pass service-worker "registered $(tr -d '\n' < "$out/sw-url.txt")"

expected=$(curl -fsS "$url/detokenize" -H 'content-type: application/json' \
    -d '{"tokens":[1,2,3]}' | jq -r .content)
printf '%s\n' "$expected" > "$out/fake-sequence.txt"
[ -n "$expected" ] || fail chat "fake sequence detokenized to empty text"

~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    '(() => { const a = document.querySelector("#server-address"); if (a) { a.value = location.origin; a.dispatchEvent(new Event("input", {bubbles:true})); a.dispatchEvent(new Event("change", {bubbles:true})); } const input = document.querySelector("#message-input"); const send = document.querySelector("#send-button"); if (!input || !send) throw new Error("P6 selectors missing"); input.value = "hello"; input.dispatchEvent(new Event("input", {bubbles:true})); send.click(); return "submitted"; })()' \
    > "$out/chat-submit.txt"
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 30 \
    'new Promise((resolve, reject) => { const deadline = Date.now() + 20000; const tick = () => { const nodes = [...document.querySelectorAll("[data-role=assistant-message]")]; const text = nodes.length ? nodes.at(-1).textContent.trim() : ""; if (text) resolve(text); else if (Date.now() > deadline) reject(new Error("assistant reply missing")); else setTimeout(tick, 100); }; tick(); })' \
    > "$out/chat-dom.txt" || fail chat "DOM reply did not arrive"
grep -Fqx "$expected" "$out/chat-dom.txt" || fail chat "DOM reply mismatch: expected $(printf '%q' "$expected"), got $(cat "$out/chat-dom.txt")"
pass chat "DOM assistant reply equals fake sequence $(printf '%q' "$expected")"

~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    '(() => { const f = window.fetch; window.fetch = async (...a) => { const body = a[1] && a[1].body; const file = body && body.get && body.get("file"); if (file) window.__voiceFile = {size:file.size, type:file.type, name:file.name}; const r = await f(...a); if (String(a[0]).includes("/v1/audio/transcriptions")) { window.__voiceStatus = r.status; window.__voiceBody = await r.clone().text(); } return r; }; class C { constructor() { this.sampleRate = 16000; } createMediaStreamSource() { return {connect(){}, disconnect(){}}; } createScriptProcessor() { const p = {onaudioprocess:null, connect(){ setTimeout(() => p.onaudioprocess({inputBuffer:{getChannelData:() => new Float32Array(1600)}}), 100); }, disconnect(){}}; return p; } createGain() { return {gain:{value:0}, connect(){}, disconnect(){}}; } close() {} } window.AudioContext = C; window.webkitAudioContext = C; navigator.mediaDevices.getUserMedia = async () => new MediaStream(); const m = document.querySelector("#mic"); if (!m || m.hidden) throw new Error("voice controls unavailable"); return "ready"; })()' \
    > "$out/voice-ready.txt"
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    'new Promise(resolve => { const m = document.querySelector("#mic"); m.click(); setTimeout(() => { m.click(); setTimeout(() => resolve(JSON.stringify({class: m.className, hasStream: Boolean(window.__voiceStream)})), 500); }, 1000); })' > "$out/voice-stop.txt"
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 45 \
    'new Promise((resolve, reject) => { const end = Date.now() + 40000; const tick = () => { if (window.__voiceStatus) resolve(JSON.stringify({status: window.__voiceStatus, body: window.__voiceBody, file: window.__voiceFile})); else if (Date.now() > end) reject(new Error("transcription request missing")); else setTimeout(tick, 100); }; tick(); })' \
    > "$out/voice-status.txt"
grep -q '"status":200' "$out/voice-status.txt" || fail voice "transcription response $(cat "$out/voice-status.txt")"
pass voice "browser PCM WAV reached /v1/audio/transcriptions with HTTP 200"

~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" \
    'location.reload(); "reloading"' > "$out/reload.txt"
sleep 1
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    '(() => { const nodes = [...document.querySelectorAll("[data-role=assistant-message]")]; return nodes.length ? nodes.at(-1).textContent.trim() : ""; })()' \
    > "$out/reload-dom.txt"
grep -Fqx "$expected" "$out/reload-dom.txt" || fail reload "conversation was not restored"
pass reload "conversation restored from localStorage"

overflow=$(~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" \
    '({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})')
printf '%s\n' "$overflow" > "$out/overflow.json"
python3 - "$out/overflow.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
if d["scrollWidth"] > d["clientWidth"]:
    raise SystemExit(d)
print(f"scrollWidth={d['scrollWidth']} clientWidth={d['clientWidth']}")
PY
pass responsive "400px document has no horizontal overflow: $(tr -d '\n' < "$out/overflow.json")"

python3 ~/iTools/browser-cdp/page-shot/page-shot.py "$browser_port" "$url/" "$out/p6-pwa-1024x768.png" --size=1024x768 --wait=1

kill "$srv"
wait "$srv" 2>/dev/null || true
srv=
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    'location.reload(); "offline reload"' > "$out/offline-reload.txt"
sleep 1
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    '({controlled: Boolean(navigator.serviceWorker.controller), shell: Boolean(document.querySelector("#message-input")), title: document.title})' \
    > "$out/offline.json"
python3 - "$out/offline.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
if not d["controlled"] or not d["shell"]:
    raise SystemExit(d)
print(f"controlled={d['controlled']} shell={d['shell']} title={d['title']!r}")
PY
pass offline "service-worker shell loads with server stopped: $(tr -d '\n' < "$out/offline.json")"

echo "PASS P6-pwa gate: $out/SUMMARY.txt"
