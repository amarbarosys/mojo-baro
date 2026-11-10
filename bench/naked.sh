#!/usr/bin/env bash
# usage: bench/naked.sh OUTDIR   (re-execs under gpu-wait; one int8-KV engine, about 2 minutes)
# Crash check first (n0a: one token with speculation on must now run without speculation and leave the server alive; n0b: one token without; n3: the one-token prompt Hello).
# Caps are 16000 (the maintainer 2026-10-18); a naked model never stops, so each runaway prompt costs about 4 minutes of decode.
# EXPLORATORY, no verdict for the rest: the model with no system prompt and no chat template. Raw /v1/completions with plain text (and one lone
# document-start token). Output ids are decoded with control tokens kept so nothing is hidden.
set -euo pipefail
cd "$(dirname "$0")/.."; export LC_ALL=C
. bench/min16k.sh; min16k 131072 16000
out=$(realpath -m "${1:?usage: bench/naked.sh OUTDIR}"); mkdir -p "$out"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --priority ${PRIO:-90} --vram 19 --timeout 420 -- env GPU_WAITING_ROOM_JOB=1 ENG="${ENG:-}" PATH="$PATH" "$PWD/bench/naked.sh" "$out"
fi
fail() { echo "FAIL $1: $2"; exit 1; }
eng=$(realpath ${ENG:-.work/crew/engine-kv8-fix}); pack=$(realpath .work/bake27/verify3/pack); tok=$(realpath .work/crew/tok/tokenizer.json); port=8306
gguf=$HOME/Models/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored-BARO-e658f85.gguf
used=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 )); [ "$used" -lt 2560 ] || fail preflight "vram_used ${used} MiB"
env BARO_EMBD_HOST=1 BARO_PACK="$pack" BARO_SPEC=1 BARO_MEGA=1 BARO_TMAX=131072 \
  serve/target/release/baro-serve --engine "$eng" --pack "$pack" --tokenizer "$tok" --port $port > "$out/serve.out" 2> "$out/serve.err" &
srv=$!; trap 'kill $srv 2>/dev/null; wait 2>/dev/null; true' EXIT
echo "engine=$eng engine_sha=$(sha256sum "$eng" | cut -c1-16)" > "$out/arm.txt"
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; kill -0 $srv 2>/dev/null || fail server died; sleep 1; done
run() { # name, prompt-json, max_tokens, temperature, spec(true|false)
  req=$(jq -nc --argjson p "$2" --argjson n $3 --argjson t $4 --argjson sp ${5:-true} '{prompt:$p,max_tokens:$n,temperature:$t,top_p:0.95,seed:7,spec:$sp}')
  curl -s -m 120 -X POST "http://127.0.0.1:$port/v1/completions" -H 'content-type: application/json' -d "$req" > "$out/$1.json"
  jq -r '.choices[0].tokens[]?' "$out/$1.json" | tr '\n' ' ' > "$out/$1.ids"
  ./baro-tokenize decode-keep "$out/$1.ids" "$gguf" > "$out/$1.txt" 2>/dev/null || true
  echo "$1: finish=$(jq -r '.choices[0].finish_reason // .error.message' "$out/$1.json" | cut -c1-110) tokens=$(wc -w < "$out/$1.ids") prompt_tokens=$(jq -r '.usage.prompt_tokens // "-"' "$out/$1.json") alive=$(curl -sf -m 3 http://127.0.0.1:$port/health >/dev/null 2>&1 && echo yes || echo NO)" | tee -a "$out/arm.txt"
}
run n0a-onetoken-spec '[248044]' 16000 0 true
run n0b-onetoken-nospec '[248044]' 16000 0 false
run n1-docstart-greedy '[248044,248044]' 16000 0
run n2-docstart-sampled '[248044,248044]' 16000 0.8
run n3-hello '"Hello"' 16000 0
run n4-arith '"What is 17 times 23?"' 16000 0
run n5-capital '"The capital of France is"' 16000 0
run n6-whoareyou '"Who are you?"' 16000 0
run n7-code '"Write a function that reverses a string."' 16000 0
run n8-bufover '"Explain what a buffer overflow is."' 16000 0
echo "failword $(grep -c 'fail word: [1-9]' "$out/serve.err" || true)" | tee -a "$out/arm.txt"
