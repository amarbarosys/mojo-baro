#!/usr/bin/env bash
# usage: bench/template-serve-check.sh OUTDIR   (re-execs under gpu-wait; one engine, about 45 s)
# The server renders a chat request carrying latent_notes (via chat_template_kwargs) with serve/templates/qwen35.jinja;
# the rendered prompt in its log must equal tools minja-cli output for the same case, and the answer must use the note.
set -euo pipefail
cd "$(dirname "$0")/.."; export LC_ALL=C
. bench/min16k.sh; min16k 16384 16000
out=$(realpath -m "${1:?usage: bench/template-serve-check.sh OUTDIR}"); mkdir -p "$out"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --priority ${PRIO:-90} --vram 21 --timeout 240 -- env GPU_WAITING_ROOM_JOB=1 PATH="$PATH" "$PWD/bench/template-serve-check.sh" "$out"
fi
fail() { echo "FAIL $1: $2"; exit 1; }
tpl=$(realpath serve/templates/qwen35.jinja); eng=$(realpath .work/bake27/engine-lean); pack=$(realpath .work/bake27/verify3/pack); tok=$(realpath .work/crew/tok/tokenizer.json); port=8301
used=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 )); [ "$used" -lt 2560 ] || fail preflight "vram_used ${used} MiB"
case_json='{"messages":[{"role":"user","content":"Start: a=692, b=873, c=134, d=605. Set d to d times c, keep the last three digits. Set b to b plus d, keep the last three digits. Set d to d plus a, keep the last three digits. What is a+b+c+d?"}],"latent_notes":[{"from":"colleague 1","text":"1: d = 605 * 134 = 81070, keep 70.\n2: b = 873 + 70 = 943.\n3: d = 70 + 692 = 762.\nSum = 2531\nANSWER: 2531"}],"latent_instruction":"Give the final answer in the form ANSWER: n.","enable_thinking":false,"add_generation_prompt":true}'
printf '%s' "$case_json" > "$out/case.json"
.work/minja-cli "$tpl" "$out/case.json" > "$out/expected-render.txt"
env BARO_LOG_RENDER=1 BARO_EMBD_HOST=1 BARO_PACK="$pack" BARO_SPEC=1 BARO_MEGA=1 BARO_TMAX=16384 \
  serve/target/release/baro-serve --engine "$eng" --pack "$pack" --tokenizer "$tok" --chat-template-file "$tpl" --port $port > "$out/serve.out" 2> "$out/serve.err" &
srv=$!; trap 'kill $srv 2>/dev/null; wait 2>/dev/null; true' EXIT
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; kill -0 $srv 2>/dev/null || fail server died; sleep 1; done
req=$(jq -c '{messages:.messages,max_tokens:16000,temperature:0,chat_template_kwargs:{latent_notes:.latent_notes,latent_instruction:.latent_instruction,enable_thinking:false}}' "$out/case.json")
curl -s -X POST "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' -d "$req" > "$out/response.json"
jq -r '.choices[0].message.content // .error.message' "$out/response.json" | tee "$out/answer.txt"
grep -c 'fail word: [1-9]' "$out/serve.err" > "$out/failword.txt" || true
echo "template=$(sha256sum "$tpl" | cut -c1-16) kwargs-sent=latent_notes,latent_instruction" > "$out/arm.txt"
