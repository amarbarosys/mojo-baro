#!/usr/bin/env bash
# Vision reference gate (docs/vision.md): llama.cpp (llama-server + mmproj) and our engine on the
# same image and prompt, greedy. Arm A: llama-server answers, its answer is tokenized as the
# reference ids. Arm B: our MoE engine, engine-direct, teacher-forced on those ids
# ("forced agreement: a / b", CLAUDE.md: agreement, never greedy equality). Bar, frozen before the
# run: agreement >= 90% of the forced positions on this image; one image is a receipt, not a claim.
#   gpu-wait run --vram 23 --timeout 900 -- bench/vision-ref.sh IMAGE OUT_DIR
# Needs a prior stub run of bench/vision-smoke.sh on the same image and cap: its requests.jsonl
# carries the exact prompt ids and .bemb path our server sends (STUB_REQ, default the smoke-stub dir).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"; export LC_ALL=C
img=$(realpath "${1:?usage: vision-ref.sh IMAGE OUT_DIR}"); out=$(realpath -m "${2:?OUT_DIR}"); mkdir -p "$out"
L=${LLAMA_CPP:-$HOME/Projects/imports/llama.cpp/build/bin}
gguf=${BARO_TEXT_GGUF:-$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-e340ee1.gguf}
mmproj=${BARO_MMPROJ:-$HOME/Models/RegesCore-1.0-35/mmproj-F16.gguf}
tpl=${TEMPLATE:-$HOME/Models/RegesCore-1.0-35/chat-templates/chat-template-original-fixed.jinja}
eng=${ENGINE:-$here/.work/vision/engine-moe}; pack=${PACK:-$here/.work/moe-ngram/pack}
cap=${BARO_IMAGE_MAX_TOKENS:-256}; gen=${GEN:-64}; q=${PROMPT:-"Describe this image in one sentence."}
req=${STUB_REQ:-$here/.work/vision/smoke-stub/requests.jsonl}
for f in "$img" "$gguf" "$mmproj" "$tpl" "$eng" "$pack/pack.bin" "$req" "$L/llama-server"; do [[ -e $f ]] || { echo "FAIL setup: missing $f"; exit 1; }; done
port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
echo "== arm A: llama-server + mmproj, image cap $cap, greedy $gen"
"$L/llama-server" -m "$gguf" --mmproj "$mmproj" --image-max-tokens "$cap" -ngl 99 -c 4096 --port "$port" --chat-template-file "$tpl" --temp 0 -np 1 > "$out/llama.log" 2>&1 &
lpid=$!; trap 'kill $lpid 2>/dev/null; wait $lpid 2>/dev/null || true' EXIT
for _ in $(seq 600); do curl -sf "localhost:$port/health" > /dev/null 2>&1 && break; kill -0 $lpid 2>/dev/null || { echo "FAIL llama-server exited, see $out/llama.log"; exit 1; }; sleep 1; done
curl -sf "localhost:$port/health" > /dev/null || { echo "FAIL llama-server: no /health in 600 s"; exit 1; }
mime=image/${img##*.}; [[ $mime == image/jpg ]] && mime=image/jpeg
python3 - "$img" "$mime" "$q" "$gen" "$out/llama-req.json" <<'PY'   # base64 of a 1.4 MB render exceeds one argv slot, so no jq --arg
import base64, json, sys
img, mime, q, gen, out = sys.argv[1:6]
url = "data:%s;base64,%s" % (mime, base64.b64encode(open(img, "rb").read()).decode())
json.dump({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}],
           "max_tokens": int(gen), "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, open(out, "w"))
PY
curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' -d @"$out/llama-req.json" > "$out/llama.json" || { echo "FAIL llama request"; tail -5 "$out/llama.log"; exit 1; }
ans=$(jq -r '.choices[0].message.content' "$out/llama.json"); echo "llama answer: $ans"
jq -cn --arg c "$ans" '{content:$c, add_special:false}' | curl -sf "localhost:$port/tokenize" -H 'content-type: application/json' -d @- | jq -c '[.tokens[] | if type=="object" then .id else . end]' > "$out/ref-ids.json"
n_ref=$(jq length "$out/ref-ids.json"); echo "reference ids: $n_ref (llama prompt tokens $(jq .usage.prompt_tokens "$out/llama.json"))"
kill $lpid; wait $lpid 2>/dev/null || true; trap - EXIT; sleep 3
echo "== arm B: our engine, teacher-forced on the reference"
line=$(grep '"embeds"' "$req" | tail -1)   # the smoke's image request, not its text request after it
[[ -n $line ]] || { echo "FAIL setup: no request with embeds in $req"; exit 1; }
n_pad=$(jq -r '[.prompt[] | select(. == 248056)] | length' <<< "$line"); echo "our prompt: $(jq '.prompt | length' <<< "$line") ids, $n_pad image pads, embeds $(jq -r '.embeds[0]' <<< "$line")"
((n_pad > 0)) || { echo "FAIL setup: the request has no image pads"; exit 1; }
jq -c --argjson f "$(cat "$out/ref-ids.json")" --argjson n "$n_ref" '.force = $f | .n = $n | .temperature = 0 | .id = 1' <<< "$line" > "$out/our-req.jsonl"
env BARO_PACK="$pack" BARO_TMAX=8192 BARO_SPEC=0 BARO_MEGA=0 BARO_PREFILL=1 BARO_SERVE=1 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 \
  "$eng" < "$out/our-req.jsonl" > "$out/engine.log" 2>&1 || { echo "FAIL engine exit, see $out/engine.log"; exit 1; }
grep -m1 -oE 'vision: images [0-9]+  image tokens [0-9]+' "$out/engine.log" || { echo "FAIL engine: no vision line"; tail -5 "$out/engine.log"; exit 1; }
ag=$(grep -m1 -oE 'forced agreement: [0-9]+ / [0-9]+' "$out/engine.log") || { echo "FAIL engine: no forced agreement line"; tail -5 "$out/engine.log"; exit 1; }
echo "$ag"; a=$(cut -d' ' -f3 <<< "$ag"); b=$(cut -d' ' -f5 <<< "$ag")
python3 -c "import sys; a,b=int(sys.argv[1]),int(sys.argv[2]); print(f'agreement {100*a/b:.1f}% of {b}'); sys.exit(0 if 100*a/b >= 90 else 1)" "$a" "$b" \
  && echo "PASS vision-ref (bar 90%, one image)" || { echo "FAIL vision-ref: under the 90% bar"; exit 1; }
