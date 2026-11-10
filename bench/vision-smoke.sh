#!/usr/bin/env bash
# Vision served check (docs/vision.md): one chat request with an image through baro-serve.
#   stub (CPU dry run):  ENGINE=bench/stub-engine.py bench/vision-smoke.sh IMAGE OUT_DIR
#   real (under gpu-wait): ENGINE=.work/vision/engine-moe bench/vision-smoke.sh IMAGE OUT_DIR
# Asserts: /tokenize maps the three vision specials to their ids; the engine receives one
# "embeds" file and a <|image_pad|> run of exactly the sidecar's token count (stub mode reads the
# request line; real mode reads the engine's "vision: images 1  image tokens N" line); a text-only
# request on the same server still answers. Real mode also prints the model's answer and FAILs on
# an engine error line. Exit non-zero on any FAIL.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
img=$(realpath "${1:?usage: vision-smoke.sh IMAGE OUT_DIR}"); out=$(realpath -m "${2:?OUT_DIR}"); mkdir -p "$out"
eng=${ENGINE:?ENGINE=engine binary or bench/stub-engine.py}
pack=${PACK:-$here/.work/moe-ngram/pack}
tok=${TOKENIZER:-$here/.work/latent-spec/pack-tier/tokenizer.json}
tpl=${TEMPLATE:-$HOME/Models/RegesCore-1.0-35/chat-templates/chat-template-original-fixed.jinja}
export BARO_MTMD_EMBED=${BARO_MTMD_EMBED:-$here/.work/vision/mtmd-embed}
export BARO_MMPROJ=${BARO_MMPROJ:-$HOME/Models/RegesCore-1.0-35/mmproj-F16.gguf}
export BARO_TEXT_GGUF=${BARO_TEXT_GGUF:-$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-e340ee1.gguf}
export BARO_IMAGE_MAX_TOKENS=${BARO_IMAGE_MAX_TOKENS:-256}
export BARO_VISION_DIR=${BARO_VISION_DIR:-$here/.work/vision/cache}
for f in "$img" "$eng" "$tok" "$tpl" "$BARO_MTMD_EMBED" "$BARO_MMPROJ"; do [[ -e $f ]] || { echo "FAIL setup: missing $f"; exit 1; }; done
stub=0; [[ $eng == *stub-engine.py ]] && stub=1
port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
: > "$out/requests.jsonl"
env FAKE_OUT="$out/requests.jsonl" BARO_PACK="$pack" BARO_TMAX=${BARO_TMAX:-8192} BARO_SPEC=0 BARO_MEGA=${MEGA:-0} BARO_PREFILL=1 \
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=${MEM_PCT:-2} \
  "$here/serve/target/release/baro-serve" --engine "$eng" --pack "$pack" --tokenizer "$tok" --chat-template-file "$tpl" --port "$port" \
  > "$out/server.log" 2>&1 < /dev/null &
spid=$!
trap 'kill $spid 2>/dev/null; wait $spid 2>/dev/null || true' EXIT
for _ in $(seq 600); do
  grep -qF "listening on http://127.0.0.1:$port" "$out/server.log" && curl -sf "localhost:$port/health" > /dev/null && break
  kill -0 $spid 2>/dev/null || { echo "FAIL serve: baro-serve exited, see $out/server.log"; exit 1; }
  sleep 1
done
curl -sf "localhost:$port/health" > /dev/null || { echo "FAIL serve: no /health in 600 s, see $out/server.log"; exit 1; }
ids=$(curl -sf "localhost:$port/tokenize" -H 'content-type: application/json' -d '{"content":"<|vision_start|><|image_pad|><|vision_end|>"}' | jq -c .tokens)
[[ $ids == "[248053,248056,248054]" ]] || { echo "FAIL tokenize: vision specials gave $ids, want [248053,248056,248054]"; exit 1; }
echo "ok tokenize $ids"
q=${PROMPT:-"Describe this image in one sentence."}
body=$(jq -cn --arg img "file://$img" --arg q "$q" --argjson n "${GEN:-64}" '{messages:[{role:"user",content:[{type:"image_url",image_url:{url:$img}},{type:"text",text:$q}]}], max_tokens:$n, temperature:0, chat_template_kwargs:{enable_thinking:false}}')
t0=$(date +%s)
curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' -d "$body" > "$out/image.json" \
  || { echo "FAIL image request:"; head -c 300 "$out/image.json" 2>/dev/null; echo; tail -5 "$out/server.log"; exit 1; }
t1=$(date +%s); echo "image request $((t1 - t0)) s"
# the sidecar's meta for THIS image: same cache key as serve/src/vision.rs (FNV-1a 64 of the bytes, then the cap)
meta=$(python3 -c '
import sys
h = 0xcbf29ce484222325
for b in open(sys.argv[1], "rb").read():
    h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
print("%s/%016x-%s.json" % (sys.argv[2], h, sys.argv[3]))' "$img" "$BARO_VISION_DIR" "$BARO_IMAGE_MAX_TOKENS")
[[ -s $meta ]] || { echo "FAIL sidecar: no meta $meta for this image"; exit 1; }
n_img=$(jq -r '.n_tokens' "$meta")
[[ -n $n_img && $n_img -gt 0 ]] || { echo "FAIL sidecar: no n_tokens in $meta"; exit 1; }
if ((stub)); then
  line=$(tail -1 "$out/requests.jsonl")
  n_emb=$(jq -r '.embeds | length' <<< "$line"); run=$(jq -r '[.prompt[] | select(. == 248056)] | length' <<< "$line")
  [[ $n_emb == 1 ]] || { echo "FAIL request: embeds has $n_emb entries"; exit 1; }
  [[ $run == "$n_img" ]] || { echo "FAIL request: $run image pads in the prompt, sidecar wrote $n_img tokens"; exit 1; }
  jq -r '.embeds[0]' <<< "$line" | { read -r p; [[ -s $p ]] || { echo "FAIL request: embeds file $p missing"; exit 1; }; }
  echo "ok request: 1 embeds file, $run pads = $n_img image tokens"
else
  v=$(grep -m1 -oE 'vision: images 1  image tokens [0-9]+' "$out/server.log" || true)
  [[ $v == *"image tokens $n_img" ]] || { echo "FAIL engine: expected 'vision: images 1  image tokens $n_img', got '${v:-none}'"; tail -8 "$out/server.log"; exit 1; }
  grep -qE '"error"' "$out/image.json" && { echo "FAIL engine error: $(head -c 300 "$out/image.json")"; exit 1; }
  echo "ok engine: $v"
  echo "answer: $(jq -r '.choices[0].message.content' "$out/image.json")"
  echo "usage: $(jq -c '.usage' "$out/image.json")  timings: $(jq -c '.timings' "$out/image.json")"
fi
curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d "$(jq -cn --arg t "${TEXT_PROMPT:-Say hello in three words.}" --argjson n "${GEN:-16}" '{messages:[{role:"user",content:$t}], max_tokens:$n, temperature:0, chat_template_kwargs:{enable_thinking:false}}')" > "$out/text.json" \
  || { echo "FAIL text request after the image request"; exit 1; }
((stub)) || echo "text answer: $(jq -r '.choices[0].message.content' "$out/text.json")"
mode=real; ((stub)) && mode=stub
echo "PASS vision-smoke ($mode engine)"
